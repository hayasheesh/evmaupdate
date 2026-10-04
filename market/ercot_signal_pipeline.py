"""Legacy fixed-width SCED activation proxies for simulation.

This path subtracts the first online Base Point and divides by a segment width.
It remains for experiment reproduction, but it is not the raw waveform bank.
For source-faithful Base Point targets and one fixed offset per daily sample,
use ``market.command_waveforms`` / ``tools.build_market_command_waveforms``.

Its source sampling still uses causal five-minute observations, rejects stale
gaps, and does not interpolate or clip the stored proxy.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288
SOURCE_TYPE = "ercot_sced_relative_fixed_width_causal_v2"
TZ = "America/Chicago"
ONLINE_STATUSES = frozenset({"ONL", "ONCLR"})
SPECS = {
    "CLR": (-1.0, {"ONL", "ONCLR"}, {"OUTL"}),
    "ESR": (1.0, {"ON", "ONREG", "ONRR", "ONRRREG"}, {"OFF", "OUT", "OFFNS"}),
    "GEN": (1.0, {"ON", "ONREG", "ONRR", "ONRRREG"}, {"OFF", "OUT", "OFFNS"}),
}
# Public API camelCase and disclosure CSV labels.
ALIASES = {
    "SCEDTimestamp": ("scedtimestamp",),
    "resourceName": ("resourcename",),
    "telResStatus": ("telresstatus", "telemeteredresourcestatus"),
    "basePoint": ("basepoint",),
    "maxPowerConsumption": ("maxpowerconsumption", "maximumpowerconsumption", "mpc"),
    "lowPowerConsumption": ("lowpowerconsumption", "lpc"),
    "HSL": ("hsl", "highsustainedlimit"),
    "LSL": ("lsl", "lowsustainedlimit"),
    "repeatHourFlag": ("repeathourflag", "repeatedhourflag"),
    "resource_kind": ("resourcekind",),
    "source_posting_date": ("sourcepostingdate",),
}


def canonicalize_columns(rows: pd.DataFrame) -> pd.DataFrame:
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in rows.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        found = [lookup[a] for a in aliases if a in lookup]
        if len(found) > 1:
            raise ValueError(f"Ambiguous columns for {canonical}: {found}")
        if found:
            rename[found[0]] = canonical
    return rows.rename(columns=rename)


def _split_for(day: date, train_end: date | None, validation_end: date | None) -> str:
    if train_end is None:
        return "unassigned"
    return "train" if day <= train_end else "validation" if day <= validation_end else "test"


def _timestamps(rows: pd.DataFrame) -> pd.Series:
    explicit_zone = rows["SCEDTimestamp"].astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if explicit_zone.any():
        if not explicit_zone.all():
            raise ValueError("Do not mix timezone-aware and local SCED timestamps")
        return pd.to_datetime(rows["SCEDTimestamp"], errors="coerce", format="mixed", utc=True)
    values = pd.to_datetime(rows["SCEDTimestamp"], errors="coerce", format="mixed")
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC")
    # repeatHourFlag identifies the second occurrence of the fall-back hour.
    flags = rows.get("repeatHourFlag", pd.Series("", index=rows.index)).astype(str).str.upper()
    ambiguity = flags.map({"FALSE": True, "N": True, "0": True,
                           "TRUE": False, "Y": False, "1": False})
    localized = values.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").copy()
    known = ambiguity.notna()
    if known.any():
        localized.loc[known] = values.loc[known].dt.tz_localize(
            TZ, ambiguous=ambiguity.loc[known].to_numpy(bool), nonexistent="NaT"
        )
    return localized.dt.tz_convert("UTC")


def build_ercot_resource_frame(
    *, rows: pd.DataFrame, resource_name: str, resource_kind: str = "CLR",
    train_end: date | None = None, validation_end: date | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """All grid points, including invalid ones, and an observed-segment audit.

    An empty sampling window does not imply offline status or reset a segment.
    Even an offline event between grid points resets the reference. Exclude
    split-crossing segments whole, rather than moving future data into train.
    """
    if (train_end is None) != (validation_end is None):
        raise ValueError("Specify both train_end and validation_end, or neither")
    if train_end is not None and train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    kind = resource_kind.upper()
    sign, on_states, off_states = SPECS[kind]
    upper, lower = ("maxPowerConsumption", "lowPowerConsumption") if kind == "CLR" else ("HSL", "LSL")
    rows = canonicalize_columns(rows)
    required = {"SCEDTimestamp", "resourceName", "telResStatus", "basePoint", upper, lower}
    if missing := required.difference(rows.columns):
        raise ValueError(f"{kind} input missing columns: {sorted(missing)}")
    unit = rows.loc[rows.resourceName.astype(str) == resource_name].copy()
    if "resource_kind" in unit:
        unit = unit.loc[unit.resource_kind.str.upper() == kind].copy()
    if unit.empty:
        raise ValueError(f"Resource not present: {kind}/{resource_name}")
    unit["_timestamp"] = _timestamps(unit)
    if unit._timestamp.isna().any():
        raise ValueError(f"{kind}/{resource_name}: invalid/ambiguous SCED timestamps; fix provenance first")
    # Official later postings supersede the same execution; otherwise fail on
    # conflicting duplicates instead of depending on incidental file order.
    unit["_posting"] = pd.to_datetime(unit.get("source_posting_date", pd.Series(
        "1970-01-01", index=unit.index)), errors="raise", utc=True, format="mixed")
    if unit._posting.isna().any():
        raise ValueError("Missing source posting date; correction precedence is unknown")
    unit = unit.sort_values(["_timestamp", "_posting"], kind="stable")
    newest = unit.groupby("_timestamp")["_posting"].transform("max")
    unit = unit.loc[unit._posting == newest].drop_duplicates(
        ["_timestamp", "telResStatus", "basePoint", upper, lower]
    )
    if unit._timestamp.duplicated().any():
        raise ValueError(f"{kind}/{resource_name}: conflicting duplicate executions")
    unit = unit.reset_index(drop=True)
    status = unit.telResStatus.fillna("").astype(str).str.strip().str.upper()
    online = status.isin(on_states)
    unit["tel_res_status"] = status
    unit["online"] = online
    unit["quality_reason"] = np.where(status.isin(on_states | off_states), "", "unknown_status")
    bp = pd.to_numeric(unit.basePoint, errors="coerce")
    high = pd.to_numeric(unit[upper], errors="coerce")
    low = pd.to_numeric(unit[lower], errors="coerce")
    width = high - low
    bad = online & (~np.isfinite(bp) | ~np.isfinite(high) | ~np.isfinite(low))
    unit.loc[bad, "quality_reason"] = "missing_online_required_value"
    unit.loc[online & (width < 0), "quality_reason"] = "negative_width"
    unit["base_point_mw"] = bp
    unit["response_span_mw"] = width
    unit["reference_base_point_mw"] = np.nan
    unit["reference_response_width_mw"] = np.nan
    unit["relative_base_point_mw"] = 0.0
    unit["signed_activation_raw_up_positive"] = 0.0
    unit["segment_id"] = ""
    unit["segment_split"] = ""
    segment_num = (online & ~online.shift(fill_value=False)).cumsum()
    segments = []
    for _, group in unit.loc[online].groupby(segment_num[online], sort=True):
        idx = group.index
        start, end = group._timestamp.iloc[0], group._timestamp.iloc[-1]
        segment_id = f"{kind}:{resource_name}:{start.isoformat()}"
        first_split = _split_for(start.tz_convert(TZ).date(), train_end, validation_end)
        last_split = _split_for(end.tz_convert(TZ).date(), train_end, validation_end)
        crossing = first_split != last_split
        spans = width.loc[idx]
        valid_widths = spans[np.isfinite(spans) & (spans > 0)]
        reference = float(bp.loc[idx[0]])
        scale = float(valid_widths.median()) if len(valid_widths) else np.nan
        reason = "split_crossing_segment" if crossing else ""
        if not np.isfinite(reference) or not np.isfinite(scale):
            reason = reason or "invalid_segment_reference_or_width"
        unit.loc[idx, "segment_id"] = segment_id
        unit.loc[idx, "segment_split"] = first_split
        unit.loc[idx, "reference_base_point_mw"] = reference
        unit.loc[idx, "reference_response_width_mw"] = scale
        if reason:
            unit.loc[idx, "quality_reason"] = reason
        else:
            unit.loc[idx, "relative_base_point_mw"] = bp.loc[idx] - reference
            unit.loc[idx, "signed_activation_raw_up_positive"] = sign * (bp.loc[idx] - reference) / scale
        segments.append({
            "segment_id": segment_id, "resource_kind": kind, "resource_name": resource_name,
            "start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "observed_hours": (end - start).total_seconds() / 3600,
            "first_observation_online": bool(idx[0] == 0),
            "reference_base_point_mw": reference if np.isfinite(reference) else None,
            "reference_response_width_mw": scale if np.isfinite(scale) else None,
            "positive_width_samples": len(valid_widths), "split": first_split,
            "excluded_reason": reason,
            "negative_width_samples": int((spans < 0).sum()),
            "missing_required_samples": int(bad.loc[idx].sum()),
        })
    local = unit._timestamp.dt.tz_convert(TZ)
    first = pd.Timestamp(local.iloc[0].date(), tz=TZ)
    stop = pd.Timestamp(local.iloc[-1].date() + pd.Timedelta(days=1), tz=TZ)
    grid = pd.DataFrame({"grid_utc": pd.date_range(first, stop, freq="5min", inclusive="left").tz_convert("UTC")})
    frame = pd.merge_asof(grid, unit, left_on="grid_utc", right_on="_timestamp", direction="backward")
    age = (frame.grid_utc - frame._timestamp).dt.total_seconds()
    fresh = age.notna() & (age >= 0) & (age < 300)
    frame.loc[~fresh, "quality_reason"] = "no_fresh_execution"
    frame["source_age_seconds"] = age
    frame["source_sced_timestamp_utc"] = frame._timestamp
    frame["time_ercot"] = frame.grid_utc.dt.tz_convert(TZ)
    frame["source_date"] = frame.time_ercot.dt.date.astype(str)
    frame["scenario_partition"] = frame.time_ercot.dt.date.map(lambda d: _split_for(d, train_end, validation_end))
    online_grid = frame.online.eq(True)
    mismatch = online_grid & fresh & (frame.segment_split != frame.scenario_partition)
    # A row before midnight may actually supply the next day's 00:00 point.
    # Exclude its ENTIRE segment, not just the point on the other side. Use
    # actual as-of selection: an intervening offline row may supersede it.
    crossing_ids = set(frame.loc[mismatch, "segment_id"])
    frame.loc[fresh & frame.segment_id.isin(crossing_ids), "quality_reason"] = "split_crossing_segment"
    for segment in segments:
        if segment["segment_id"] in crossing_ids:
            segment["excluded_reason"] = "split_crossing_segment"
    same_segment = online_grid & frame.segment_id.eq(frame.segment_id.shift()) & fresh & fresh.shift(fill_value=False)
    frame["delta_base_point_mw"] = frame.base_point_mw.diff().where(same_segment, 0.0)
    signed = frame.signed_activation_raw_up_positive
    frame.loc[online_grid & fresh & ~np.isfinite(signed), "quality_reason"] = "nonfinite_normalized_value"
    frame["up_proxy_raw"] = signed.clip(lower=0)
    frame["down_proxy_raw"] = (-signed).clip(lower=0)
    frame["simulation_clip_required"] = signed.abs() > 1
    frame["source_type"] = SOURCE_TYPE
    frame["source_bmu"] = resource_name
    frame["resource_kind"] = kind
    return frame, segments


def _day_reasons(frame: pd.DataFrame, min_change_mw: float) -> list[str]:
    reasons = sorted(set(frame.quality_reason.dropna()) - {""})
    if len(frame) != STEPS_PER_DAY:
        reasons.append("non_288_slot_day")
    if not frame.online.eq(True).any():
        reasons.append("no_online_steps")
    # A flat day is not active solely because of the previous day's last BP.
    within_day = frame.segment_id.eq(frame.segment_id.shift()) & frame.online.eq(True)
    if not (frame.base_point_mw.diff().abs().where(within_day, 0) > min_change_mw).any():
        reasons.append("no_base_point_motion")
    return reasons


EXPORT_COLUMNS = [
    "time_ercot", "grid_utc", "source_sced_timestamp_utc", "source_age_seconds",
    "source_type", "source_bmu", "resource_kind", "source_date", "scenario_partition",
    "tel_res_status", "online", "segment_id", "base_point_mw", "response_span_mw",
    "reference_base_point_mw", "reference_response_width_mw", "delta_base_point_mw",
    "relative_base_point_mw", "up_proxy_raw", "down_proxy_raw",
    "signed_activation_raw_up_positive", "simulation_clip_required",
]


def _export_day(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[EXPORT_COLUMNS].reset_index(drop=True).copy()
    result.insert(0, "step", np.arange(len(result)))
    return result


def _resource_inputs(paths, resource_names, default_kind):
    """Bound memory for all-resource archives; spool compact inputs by resource.

    Temporary files are not scenario samples and disappear on exit. One
    resource's complete history stays together, including corrective postings.
    """
    selected = None if resource_names is None else set(resource_names)
    if selected == set():
        raise ValueError("resource_names must not be empty")
    compact = list(ALIASES)
    accepted_headers = {a for aliases in ALIASES.values() for a in aliases}
    files = {}
    seen_names = set()
    with tempfile.TemporaryDirectory(prefix="ercot_waveform_inputs_") as directory:
        for path in paths:
            for chunk in pd.read_csv(path, chunksize=100_000,
                    usecols=lambda c: re.sub(r"[^a-z0-9]", "", c.lower()) in accepted_headers):
                chunk = canonicalize_columns(chunk)
                if "resourceName" not in chunk or chunk.resourceName.isna().any():
                    raise ValueError(f"Missing resourceName in {path}")
                chunk["resourceName"] = chunk.resourceName.astype(str)
                if selected is not None:
                    chunk = chunk.loc[chunk.resourceName.isin(selected)].copy()
                if chunk.empty:
                    continue
                if "resource_kind" not in chunk:
                    chunk["resource_kind"] = default_kind
                chunk["resource_kind"] = chunk.resource_kind.astype(str).str.upper()
                if unknown := set(chunk.resource_kind) - set(SPECS):
                    raise ValueError(f"Unknown resource kinds: {unknown}")
                if "source_posting_date" not in chunk:
                    chunk["source_posting_date"] = "1970-01-01"
                for key, group in chunk.groupby(["resource_kind", "resourceName"], sort=False):
                    seen_names.add(key[1])
                    filename = hashlib.sha256(repr(key).encode()).hexdigest() + ".csv"
                    target = Path(directory) / filename
                    exists = target.exists()
                    group.reindex(columns=compact).to_csv(target, mode="a", header=not exists, index=False)
                    files[key] = target
        if selected is not None and (missing := selected - seen_names):
            raise ValueError(f"Resources not present: {sorted(missing)}")
        if not files:
            raise ValueError("No source rows selected")
        for (kind, name), path in sorted(files.items()):
            yield kind, name, pd.read_csv(path)


def build_ercot_day_frame(*, resource_name: str, day: date, rows: pd.DataFrame,
                          resource_kind: str = "CLR") -> pd.DataFrame | None:
    """Compatibility helper; pass continuous history, not isolated daily rows."""
    frame, _ = build_ercot_resource_frame(rows=rows, resource_name=resource_name, resource_kind=resource_kind)
    frame = frame.loc[frame.source_date == day.isoformat()]
    return None if frame.empty or _day_reasons(frame, 1e-6) else _export_day(frame)


def build_ercot_scenario_library(
    *, input_csv: str | Path | Iterable[str | Path], output_dir: str | Path,
    resource_names: Iterable[str] | None = None, resource_kind: str = "CLR",
    train_end: date | None = None, validation_end: date | None = None,
    audit_only: bool = False, min_change_mw: float = 1e-6,
) -> dict[str, Any]:
    """New library plus day/segment audit; never delete old banks or outputs."""
    if min_change_mw < 0 or not np.isfinite(min_change_mw):
        raise ValueError("min_change_mw must be finite and nonnegative")
    paths = [Path(input_csv)] if isinstance(input_csv, (str, Path)) else [Path(p) for p in input_csv]
    if not paths:
        raise ValueError("No input CSVs")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output must be empty (existing data preserved): {root}")
    resources, segments, days, written = [], [], [], []
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps({"build_complete": False}), encoding="utf-8")
    for kind, name, group in _resource_inputs(paths, resource_names, resource_kind.upper()):
        frame, audit = build_ercot_resource_frame(
            rows=group, resource_name=str(name), resource_kind=kind,
            train_end=train_end, validation_end=validation_end,
        )
        segments.extend(audit)
        kept, up, down, total, over, peak = 0, 0, 0, 0, 0, 0.0
        for day, day_frame in frame.groupby("source_date", sort=True):
            reasons = _day_reasons(day_frame, min_change_mw)
            days.append({
                "resource_kind": kind, "resource_name": name, "date": day,
                "partition": day_frame.scenario_partition.iloc[0], "reasons": reasons,
                "no_fresh_execution_steps": int((day_frame.quality_reason == "no_fresh_execution").sum()),
                "segment_ids": sorted(set(day_frame.segment_id.dropna()) - {""}),
            })
            if reasons:
                continue
            kept += 1
            signed = day_frame.signed_activation_raw_up_positive.to_numpy(float)
            total += len(signed)
            up += int((signed > 1e-3).sum())
            down += int((signed < -1e-3).sum())
            over += int((np.abs(signed) > 1).sum())
            peak = max(peak, float(np.abs(signed).max()))
            if not audit_only:
                safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))
                suffix = hashlib.sha256(str(name).encode()).hexdigest()[:8]
                filename = f"ercot_sced_{kind}_{safe}_{suffix}_{day}.csv"
                _export_day(day_frame).to_csv(root / filename, index=False, float_format="%.12g")
                written.append(filename)
        resources.append({
            "resource_kind": kind, "resource_name": name, "days": kept,
            "candidate_days": int(frame.source_date.nunique()), "total_5min_steps": total,
            "up_frequency": up / total if total else 0,
            "down_frequency": down / total if total else 0,
            "activation_frequency": (up + down) / total if total else 0,
            "raw_above_unit_frequency": over / total if total else 0,
            "maximum_raw_activation_fraction": peak,
        })
    metadata = {
        "source_type": SOURCE_TYPE, "input_csvs": [str(p.resolve()) for p in paths],
        "offset_required": False, "absolute_dispatch_recoverable": False,
        "semantics": "relative command shapes only; EV-VPP chooses baseline and directional bids",
        "sampling": "latest execution in (t-5min,t]; no interpolation; t starts simulator interval",
        "timezone": TZ, "dst_policy": "reject non-288-slot local days",
        "normalization": "median finite positive width of original executions per observed online segment; CLR MPC-LPC, ESR/GEN HSL-LSL",
        "reference": "first observed online BP; no midnight reset; no mean removal",
        "source_clipping": "none; clip only at simulator input",
        "split_policy": "chronological_segment_disjoint" if train_end else "unassigned",
        "train_end": train_end.isoformat() if train_end else None,
        "validation_end": validation_end.isoformat() if validation_end else None,
        "crossing_segment_policy": "exclude every day touching a crossing segment",
        "min_change_mw": min_change_mw, "audit_only": audit_only,
        "build_complete": True,
        "resources": resources, "written_files": written, "days": days, "segments": segments,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return metadata

"""ERCOT ESR command amounts: SCED Base Point minus the QSE's own plan.

The Secondary Reserve 2 command is an amount added to the resource's own
baseline. For an ERCOT ESR the baseline counterpart is the Current Operating
Plan as it stood one hour before each operating hour (the NP1-301 Adjustment
Period Snapshot). The COP carries no MW schedule for an ESR, only its planned
state of charge at the start of each hour, so the planned output of an hour is
the planned SOC drop over that hour (discharge positive, the Base Point sign).

Sampling reuses ``build_ercot_resource_frame``: the latest SCED execution in
(t-5min, t], stale points rejected, corrections resolved, DST days excluded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from market.command_waveforms import build_command_waveform_day, waveform_to_activation_proxy
from market.ercot_signal_pipeline import STEPS_PER_DAY

SOURCE_TYPE = "ercot_esr_base_point_minus_cop_plan"
COP_COLUMNS = ["Delivery Date", "Resource Name", "Hour Ending", "Hour Beginning Planned SOC"]
# Reasons that belong to the segment-reference normalization of
# build_ercot_resource_frame. This library uses one width per resource and no
# segment reference, so they do not disqualify a point here.
SEGMENT_NORMALIZATION_REASONS = frozenset({
    "invalid_segment_reference_or_width",
    "split_crossing_segment",
    "nonfinite_normalized_value",
})


def load_cop_planned_soc(paths: Iterable[str | Path], names: set[str] | None = None) -> dict[str, pd.Series]:
    """Planned SOC (MWh) by resource, indexed by naive local hour start.

    A resource-hour that appears with two different values is left out rather
    than resolved by file order.
    """
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=COP_COLUMNS)
        if names is not None:
            frame = frame[frame["Resource Name"].isin(names)]
        frames.append(frame)
    if not frames:
        return {}
    cop = pd.concat(frames, ignore_index=True)
    hour = cop["Hour Ending"].astype(str).str.split(":").str[0].str.rstrip("*").astype(int)
    day = pd.to_datetime(cop["Delivery Date"], format="%m/%d/%Y")
    cop["hour_start"] = day + pd.to_timedelta(hour - 1, unit="h")
    cop["soc"] = pd.to_numeric(cop["Hour Beginning Planned SOC"], errors="coerce")
    cop = cop.drop_duplicates(["Resource Name", "hour_start", "soc"])
    conflicting = cop.duplicated(["Resource Name", "hour_start"], keep=False)
    cop = cop[~conflicting]
    return {
        name: group.set_index("hour_start")["soc"].sort_index()
        for name, group in cop.groupby("Resource Name")
    }


def planned_output_mw(planned_soc: pd.Series, hour_starts: pd.DatetimeIndex) -> np.ndarray:
    """Planned output of each hour: SOC at its start minus SOC at the next hour."""
    now = planned_soc.reindex(hour_starts).to_numpy(float)
    later = planned_soc.reindex(hour_starts + pd.Timedelta(hours=1)).to_numpy(float)
    return now - later


def unit_width_mw(rows: pd.DataFrame) -> float:
    """Largest one-sided capability while online: max over max(HSL, -LSL)."""
    online = rows[rows["telResStatus"].astype(str).str.strip().str.upper().isin({"ON", "ONREG", "ONRR", "ONRRREG"})]
    high = pd.to_numeric(online["HSL"], errors="coerce")
    low = pd.to_numeric(online["LSL"], errors="coerce")
    values = np.concatenate([high.to_numpy(float), (-low).to_numpy(float)])
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else 0.0


def day_activation(
    day_frame: pd.DataFrame, planned_soc: pd.Series, *, resource_name: str,
    width_mw: float, partition: str,
) -> tuple[pd.DataFrame | None, str]:
    """Return one day's activation frame, or ``None`` with the reason it is left out."""
    if len(day_frame) != STEPS_PER_DAY:
        return None, "non_288_slot_day"
    quality = day_frame["quality_reason"].fillna("")
    if (~quality.isin(SEGMENT_NORMALIZATION_REASONS | {""})).any():
        return None, "sampling_quality"
    if not day_frame["online"].eq(True).all():
        return None, "not_online_all_day"
    if not np.isfinite(width_mw) or width_mw <= 0.0:
        return None, "no_positive_width"
    local = day_frame["time_ercot"].dt.tz_localize(None)
    plan = planned_output_mw(planned_soc, pd.DatetimeIndex(local.dt.floor("h")))
    if not np.isfinite(plan).all():
        return None, "missing_cop_plan"
    base_point = day_frame["base_point_mw"].to_numpy(float)
    source = pd.DataFrame({
        "step": np.arange(STEPS_PER_DAY),
        "time_ercot": local.dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
        "time_utc": day_frame["grid_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
        "deviation_mw": base_point - plan,
    })
    day = str(day_frame["source_date"].iloc[0])
    waveform = build_command_waveform_day(
        source, market="ERCOT", resource_kind="ESR_PLAN_DEVIATION",
        resource_name=resource_name, source_date=day, step_column="step",
        target_column="deviation_mw", direction_sign=1.0,
        local_time_column="time_ercot", utc_time_column="time_utc",
    )
    activation = waveform_to_activation_proxy(
        waveform, reference_mode="zero", source_type=SOURCE_TYPE,
        scenario_partition=partition, segment_id=f"ercot_plan:{resource_name}:{day}",
        scale_mw=width_mw,
    )
    activation["base_point_mw"] = base_point
    activation["plan_mw"] = plan
    return activation, ""

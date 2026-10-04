"""Build legacy schedule-deviation activation proxies from AEMO dispatch.

This is the historical simulator-proxy path: it subtracts a rolling predispatch
plan, divides by availability, and clips the result. Those operations answer a
different question from source-faithful waveform extraction. For raw
``TOTALCLEARED`` targets and a single fixed per-scenario offset, use
``market.command_waveforms`` / ``tools.build_market_command_waveforms``.

A GB pool built from Bid-Offer Acceptances does not fit: those are
discrete instructions the operator issues to move a unit
off its declared plan.  They leave the median day with an instruction in only
about a tenth of its 288 steps, so the idle band -- the row of the assessment
table that applies when nothing is being asked -- ends up governing almost the
whole day.  Secondary Reserve 2 is an EDC product, and an EDC signal is not
built that way.

The NEM's five-minute dispatch is: every five minutes NEMDE re-solves the whole
economic dispatch and publishes a target MW for every unit, which is the same
mechanism class as EDC.  Subtracting the unit's own schedule gives a deviation
with the same shape as an instruction against a baseline, at exactly the
288-step grid the bidder already uses.

Which schedule matters, and getting it wrong inflates the signal badly.  Japan
closes the gate an hour before delivery: until then a participant revises its
own plan, and only the error left afterwards is what reserve is called for.
Measuring against the schedule filed the previous afternoon instead charges
fifteen hours of intraday re-optimisation to the command.  On two days that
roughly halved every tail statistic -- the largest net energy demand fell from
7.6 to 3.3 band-hours -- while leaving the median density and run length within
a quarter of each other.  So the pool stays the dense one either way, but its
extremes are an artifact unless the baseline is taken at gate closure.  The
rolling pre-dispatch runs, all of which the daily file carries, are what makes
that possible; nothing else published is per-unit at that horizon.

One thing this signal is not: a reserve activation.  Regulation FCAS in the NEM
is delivered by AGC on top of the energy target, and the AGC setpoint is not
published.  What the signal carries is a real five-minute target trajectory for
a fast, energy-limited resource, which is the property the GB pool lacks.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Sequence
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd


NEMWEB = "https://nemweb.com.au"
DISPATCH_REPORT = "/Reports/Current/Next_Day_Dispatch/"
PREDISPATCH_REPORT = "/Reports/Current/Next_Day_PreDispatch/"
USER_AGENT = "EVMA-LOCAL NEM research downloader/1.0"
STEPS_PER_DAY = 288
BLOCKS_PER_DAY = 48
NEM_UTC_OFFSET = timedelta(hours=10)  # AEST, fixed: the NEM does not observe DST
AEMO_TIME = "%Y/%m/%d %H:%M:%S"

# Column positions in the AEMO flat file, counted from the "D" record marker.
DISPATCH_COLUMNS = {
    "settlementdate": 4, "duid": 6, "intervention": 9, "initialmw": 13,
    "totalcleared": 14, "lowerreg": 34, "raisereg": 35, "availability": 36,
    "energy_storage": 70,
}
PREDISPATCH_COLUMNS = {
    "seqno": 4, "duid": 6, "intervention": 9, "totalcleared": 14,
    "issued": 33, "datetime": 34, "availability": 37,
}
# Japan closes the gate one hour before delivery.  Up to that point a
# participant revises its own plan; only what is left afterwards is what
# Secondary Reserve 2 is called for.  So the baseline a command is measured
# against is the plan as it stood an hour out, not the plan filed the previous
# afternoon -- that one carries fifteen hours of intraday re-optimisation the
# Japanese market handles through plan revision rather than through reserve.
BASELINE_LEAD = timedelta(minutes=55)
# The monthly archive keeps one run per period instead of the rolling set,
# issued about 29 minutes out, so a build from it has to ask for less lead.
# Over ten days the two agree to within a tenth on every median statistic.
ARCHIVE_BASELINE_LEAD = timedelta(minutes=25)
TRADING_DAY_START = timedelta(hours=4, minutes=5)
MMSDM = (NEMWEB + "/Data_Archive/Wholesale_Electricity/MMSDM/{year}/"
         "MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
         "PUBLIC_ARCHIVE%23{table}%23FILE01%23{year}{month:02d}010000.zip")


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=600) as response:
        return response.read()


_LISTINGS: dict[str, dict[str, str]] = {}


def _listing(report: str) -> dict[str, str]:
    """Map YYYYMMDD to the report's URL, which carries an opaque serial.

    Cached for the process: the directory pages are large and a library build
    asks for the same two of them once per day it wants.
    """

    if report not in _LISTINGS:
        text = _fetch(NEMWEB + report).decode("utf-8", "replace")
        found: dict[str, str] = {}
        for match in re.finditer(r"(PUBLIC_[A-Z_]+_(\d{8})_\d+\.zip)", text):
            found.setdefault(match.group(2), NEMWEB + report + match.group(1))
        _LISTINGS[report] = found
    return _LISTINGS[report]


def _rows(payload: bytes, table: str) -> Iterable[list[str]]:
    """Yield the data records of one table out of an AEMO flat file in a zip."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.upper().endswith(".CSV"))
        with archive.open(name) as handle:
            stream = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            for row in csv.reader(stream):
                if row and row[0] == "D" and len(row) > 3 and row[2] == table:
                    yield row


def _pick(row: Sequence[str], columns: dict[str, int]) -> dict[str, str]:
    return {key: row[index].strip('"') for key, index in columns.items()}


def _cached_paths(cache_dir: Path, day: date) -> tuple[Path, Path] | None:
    """The cache entry for one day, in either the plain or gzipped form."""

    found = tuple(
        next((c for c in (path, path.with_suffix(".csv.gz")) if c.exists()), None)
        for path in (
            cache_dir / "dispatch" / f"dispatch_{day.isoformat()}.csv",
            cache_dir / "predispatch" / f"predispatch_{day.isoformat()}.csv",
        )
    )
    return found if all(found) else None  # type: ignore[return-value]


def download_nem_day(
    day: date,
    *,
    cache_dir: Path,
    force: bool = False,
) -> tuple[Path, Path] | None:
    """Cache one trading day's dispatch targets and its day-ahead schedule.

    Both reports are hundreds of megabytes uncompressed and almost all of it is
    other units and other tables, so the cache keeps only the storage units'
    rows.  A unit is taken to be storage when it reports an energy level, which
    is how the NEM marks a bidirectional unit; there is no need to pair a
    generator DUID with a load DUID any more.
    """

    stamp = day.strftime("%Y%m%d")
    dispatch_path = cache_dir / "dispatch" / f"dispatch_{day.isoformat()}.csv"
    plan_path = cache_dir / "predispatch" / f"predispatch_{day.isoformat()}.csv"
    # The rolling pre-dispatch runs live in a sixty-day window and cannot be
    # fetched again once a day falls out of it, so that half of the cache is
    # worth keeping compressed rather than deleting.
    cached = _cached_paths(cache_dir, day)
    if not force and cached is not None:
        return cached

    dispatch_urls = _listing(DISPATCH_REPORT)
    plan_urls = _listing(PREDISPATCH_REPORT)
    if stamp not in dispatch_urls or stamp not in plan_urls:
        return None

    dispatch_rows = [
        _pick(row, DISPATCH_COLUMNS)
        for row in _rows(_fetch(dispatch_urls[stamp]), "UNIT_SOLUTION")
        if row[DISPATCH_COLUMNS["energy_storage"]].strip()
    ]
    if not dispatch_rows:
        return None
    units = {row["duid"] for row in dispatch_rows}

    plan_rows = [
        _pick(row, PREDISPATCH_COLUMNS)
        for row in _rows(_fetch(plan_urls[stamp]), "UNIT_SOLUTION")
        if row[PREDISPATCH_COLUMNS["duid"]] in units
        and row[PREDISPATCH_COLUMNS["intervention"]].strip('"') == "0"
    ]
    # Every rolling run is kept.  Which one is the baseline depends on the
    # lead time being modelled, and the file is the only place the earlier
    # runs exist, so the choice is left to the frame builder.

    _write_cache(dispatch_path, dispatch_rows, list(DISPATCH_COLUMNS))
    _write_cache(plan_path, plan_rows, list(PREDISPATCH_COLUMNS))
    return dispatch_path, plan_path


def _write_cache(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, newline="", encoding="utf-8"
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def _trading_day(moment: datetime) -> date:
    """The NEM trading day runs 04:05 to 04:00, stamped at interval end."""

    return (moment - TRADING_DAY_START).date()


def download_nem_month(
    year: int,
    month: int,
    *,
    cache_dir: Path,
    force: bool = False,
) -> list[date]:
    """Fill the daily cache for one month out of the MMSDM archive.

    The daily reports only reach back sixty days.  The archive reaches 2009,
    at the cost of the baseline being the run issued 29 minutes out rather
    than the hour Japan's gate closure would give.  Everything else -- the
    table, the columns, the units -- is identical, so the same cache layout
    and the same frame builder serve both.
    """

    dispatch_by_day: dict[date, list[dict[str, str]]] = {}
    payload = _fetch(MMSDM.format(year=year, month=month, table="DISPATCHLOAD"))
    for row in _rows(payload, "UNIT_SOLUTION"):
        if not row[DISPATCH_COLUMNS["energy_storage"]].strip():
            continue
        picked = _pick(row, DISPATCH_COLUMNS)
        day = _trading_day(datetime.strptime(picked["settlementdate"], AEMO_TIME))
        dispatch_by_day.setdefault(day, []).append(picked)
    del payload
    if not dispatch_by_day:
        return []
    units = {row["duid"] for rows in dispatch_by_day.values() for row in rows}

    plan_by_day: dict[date, list[dict[str, str]]] = {}
    payload = _fetch(MMSDM.format(year=year, month=month, table="PREDISPATCHLOAD"))
    for row in _rows(payload, "UNIT_SOLUTION"):
        if row[PREDISPATCH_COLUMNS["duid"]] not in units:
            continue
        if row[PREDISPATCH_COLUMNS["intervention"]].strip('"') != "0":
            continue
        picked = _pick(row, PREDISPATCH_COLUMNS)
        day = _trading_day(datetime.strptime(picked["datetime"], AEMO_TIME))
        plan_by_day.setdefault(day, []).append(picked)
    del payload

    written: list[date] = []
    for day, rows in sorted(dispatch_by_day.items()):
        plan = plan_by_day.get(day)
        if not plan:
            continue
        dispatch_path = cache_dir / "dispatch" / f"dispatch_{day.isoformat()}.csv"
        plan_path = cache_dir / "predispatch" / f"predispatch_{day.isoformat()}.csv"
        if not force and dispatch_path.exists() and plan_path.exists():
            written.append(day)
            continue
        _write_cache(dispatch_path, rows, list(DISPATCH_COLUMNS))
        _write_cache(plan_path, plan, list(PREDISPATCH_COLUMNS))
        written.append(day)
    return written


def build_nem_day_frame(
    *,
    duid: str,
    day: date,
    dispatch: pd.DataFrame,
    plan: pd.DataFrame,
    baseline_lead: timedelta | None = None,
) -> pd.DataFrame | None:
    """Return one 288-step normalized individual-command scenario."""

    unit = dispatch[dispatch["duid"] == duid].sort_values("settlementdate")
    if len(unit) != STEPS_PER_DAY:
        return None
    if (unit["intervention"].astype(float) != 0.0).any():
        # An intervention run publishes a second solution for the same
        # interval.  Rather than choose between them, drop the day.
        return None

    times = [datetime.strptime(value, AEMO_TIME) for value in unit["settlementdate"]]
    target = unit["totalcleared"].astype(float).to_numpy()
    availability = unit["availability"].astype(float).to_numpy()

    # SETTLEMENTDATE stamps the end of the five-minute interval, and the
    # schedule stamps the end of its thirty-minute block, so the six steps
    # inside a block share the block's planned level.  Among the runs that
    # forecast that block, take the last one issued before the gate closed.
    schedule: dict[datetime, tuple[datetime, float]] = {}
    for row in plan[plan["duid"] == duid].itertuples():
        forecast_for = datetime.strptime(row.datetime, AEMO_TIME)
        issued = datetime.strptime(row.issued, AEMO_TIME)
        if issued > forecast_for - (baseline_lead or BASELINE_LEAD):
            continue
        current = schedule.get(forecast_for)
        if current is None or issued > current[0]:
            schedule[forecast_for] = (issued, float(row.totalcleared))

    baseline = []
    for moment in times:
        block_end = moment if moment.minute % 30 == 0 else (
            moment.replace(minute=(moment.minute // 30) * 30) + timedelta(minutes=30)
        )
        if block_end not in schedule:
            return None
        baseline.append(schedule[block_end][1])
    baseline = np.asarray(baseline, dtype=float)

    delta = target - baseline
    band = np.maximum(availability, 1e-9)
    up = np.clip(np.maximum(delta, 0.0) / band, 0.0, 1.0)
    down = np.clip(np.maximum(-delta, 0.0) / band, 0.0, 1.0)
    utc = [moment - NEM_UTC_OFFSET for moment in times]
    return pd.DataFrame({
        "step": np.arange(STEPS_PER_DAY, dtype=int),
        "time_utc": [
            moment.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            for moment in utc
        ],
        "time_nem": [moment.isoformat() for moment in times],
        "source_type": "aemo_nem_dispatch_minus_gate_closure_plan",
        "source_bmu": str(duid),
        "source_date": day.isoformat(),
        "predispatch_mw": baseline,
        "predispatch_issued": [
            schedule[m if m.minute % 30 == 0 else
                     m.replace(minute=(m.minute // 30) * 30)
                     + timedelta(minutes=30)][0].isoformat()
            for m in times
        ],
        "dispatch_target_mw": target,
        "delta_mw": delta,
        "availability_mw": availability,
        "raise_reg_mw": unit["raisereg"].astype(float).to_numpy(),
        "lower_reg_mw": unit["lowerreg"].astype(float).to_numpy(),
        "energy_storage_mwh": unit["energy_storage"].astype(float).to_numpy(),
        "up_proxy": up,
        "down_proxy": down,
        "signed_activation_up_positive": up - down,
    })


def _nonzero_spell_durations(mask: np.ndarray) -> list[int]:
    spells: list[int] = []
    current = 0
    for flag in mask:
        if flag:
            current += 1
        elif current:
            spells.append(current)
            current = 0
    if current:
        spells.append(current)
    return spells


def build_nem_scenario_library(
    *,
    days: Sequence[date],
    cache_dir: str | Path,
    output_dir: str | Path,
    minimum_availability_mw: float = 20.0,
    force_download: bool = False,
    baseline_lead: timedelta | None = None,
    download: bool = True,
) -> dict[str, Any]:
    """Convert cached NEM dispatch into one CSV per unit/day.

    Units smaller than ``minimum_availability_mw`` are dropped: a two-megawatt
    unit's target moves in whole megawatts, so its normalized command is a
    staircase of a few levels rather than a trajectory.
    """

    cache_root, output_root = Path(cache_dir), Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for stale in output_root.glob("nem_*.csv"):
        stale.unlink()

    written: list[str] = []
    per_unit: dict[str, dict[str, Any]] = {}
    skipped_days: list[str] = []
    for day in days:
        cached = (
            download_nem_day(day, cache_dir=cache_root, force=force_download)
            if download else _cached_paths(cache_root, day)
        )
        if cached is None:
            skipped_days.append(day.isoformat())
            continue
        dispatch = pd.read_csv(cached[0], dtype=str).fillna("")
        plan = pd.read_csv(cached[1], dtype=str).fillna("")
        if plan.empty:
            skipped_days.append(day.isoformat())
            continue
        for duid in sorted(dispatch["duid"].unique()):
            frame = build_nem_day_frame(
                duid=duid, day=day, dispatch=dispatch, plan=plan,
                baseline_lead=baseline_lead,
            )
            if frame is None:
                continue
            if float(frame["availability_mw"].max()) < minimum_availability_mw:
                continue
            name = f"nem_{duid}_{day.isoformat()}.csv"
            frame.to_csv(output_root / name, index=False, float_format="%.9f")
            written.append(name)

            active = (
                frame["up_proxy"].to_numpy() > 1e-3
            ) | (frame["down_proxy"].to_numpy() > 1e-3)
            stats = per_unit.setdefault(duid, {
                "duid": duid, "days": 0, "active_5min_steps": 0,
                "up_steps": 0, "down_steps": 0, "zero_days": 0,
                "spells": [], "max_availability_mw": 0.0,
            })
            stats["days"] += 1
            stats["active_5min_steps"] += int(np.count_nonzero(active))
            stats["up_steps"] += int(np.count_nonzero(frame["up_proxy"] > 1e-3))
            stats["down_steps"] += int(np.count_nonzero(frame["down_proxy"] > 1e-3))
            stats["zero_days"] += int(not active.any())
            stats["spells"].extend(_nonzero_spell_durations(active))
            stats["max_availability_mw"] = max(
                stats["max_availability_mw"], float(frame["availability_mw"].max())
            )

    units = []
    for stats in per_unit.values():
        spells = stats.pop("spells")
        total = stats["days"] * STEPS_PER_DAY
        active = stats["active_5min_steps"]
        units.append({
            **stats,
            "total_5min_steps": total,
            "acceptance_frequency": active / total if total else 0.0,
            "direction_bias_up_minus_down": (
                (stats["up_steps"] - stats["down_steps"]) / active if active else 0.0
            ),
            "mean_nonzero_spell_minutes": float(np.mean(spells)) * 5.0 if spells else 0.0,
            "max_nonzero_spell_minutes": float(np.max(spells)) * 5.0 if spells else 0.0,
            "fraction_zero_days": stats["zero_days"] / stats["days"] if stats["days"] else 0.0,
        })

    metadata = {
        "source": (
            "AEMO Next_Day_Dispatch TOTALCLEARED minus the last Next_Day_PreDispatch "
            f"run issued at least {baseline_lead or BASELINE_LEAD} before the block, "
            "normalized by "
            "the unit's AVAILABILITY"
        ),
        "signal_semantics": "positive delta is EV up; negative delta is EV down",
        "step_minutes": 5,
        "trading_day": "NEM time, 04:05 to 04:00, stamped at interval end",
        "unit_selection": "reports ENERGY_STORAGE, and offers at least "
                          f"{minimum_availability_mw} MW",
        "days_requested": [day.isoformat() for day in days],
        "days_skipped": skipped_days,
        "units": sorted(units, key=lambda row: row["duid"]),
        "written_files": written,
    }
    path = output_root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

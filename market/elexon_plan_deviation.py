"""GB battery command amounts: instructed BOA level minus Final Physical Notification.

In the GB Balancing Mechanism the unit's own Physical Notification becomes its
FPN at gate closure, one hour before the settlement period, and the system
operator's Bid-Offer Acceptances move the unit off it. That is the Secondary
Reserve 2 structure: a baseline fixed at gate closure plus the operator's
command amount. The command amount is sampled at each five-minute point as the
level of the latest acceptance in force minus FPN, and zero when none is in
force. Every acceptance counts whatever its SO flag: dropping one would leave a
level the unit was never instructed to hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market.command_waveforms import build_command_waveform_day, waveform_to_activation_proxy

SOURCE_TYPE = "elexon_battery_boa_minus_fpn"
STEPS_PER_DAY = 288
LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class Segment:
    start: datetime
    end: datetime
    level_from: float
    level_to: float
    acceptance_time: datetime | None = None
    acceptance_number: int = -1
    so_flag: bool = False

    def value_at(self, time: datetime) -> float:
        duration = (self.end - self.start).total_seconds()
        if duration <= 0.0:
            return float(self.level_to)
        fraction = np.clip((time - self.start).total_seconds() / duration, 0.0, 1.0)
        return float(self.level_from + (self.level_to - self.level_from) * fraction)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def to_segments(rows: Iterable[dict[str, Any]], *, boa: bool) -> list[Segment]:
    segments = []
    for row in rows:
        segments.append(Segment(
            start=parse_time(row["timeFrom"]),
            end=parse_time(row["timeTo"]),
            level_from=float(row["levelFrom"]),
            level_to=float(row["levelTo"]),
            acceptance_time=parse_time(row["acceptanceTime"]) if boa else None,
            acceptance_number=int(row.get("acceptanceNumber", -1)) if boa else -1,
            so_flag=bool(row.get("soFlag", False)) if boa else False,
        ))
    return segments


def fpn_at(segments: Sequence[Segment], time: datetime) -> float | None:
    """FPN at ``time``; the latest-starting covering record wins."""
    covering = [s for s in segments if s.start <= time < s.end]
    if not covering:
        return None
    return max(covering, key=lambda s: s.start).value_at(time)


def acceptance_in_force(segments: Sequence[Segment], time: datetime) -> Segment | None:
    """The latest acceptance, issued by ``time``, whose profile covers ``time``."""
    covering = [
        s for s in segments
        if s.start <= time < s.end and s.acceptance_time is not None and s.acceptance_time <= time
    ]
    if not covering:
        return None
    return max(covering, key=lambda s: (s.acceptance_time, s.acceptance_number))


def local_day_grid(day: date) -> list[datetime] | None:
    """Five-minute UTC instants of one London calendar day, or None on a DST day."""
    start = datetime(day.year, day.month, day.day, tzinfo=LONDON)
    end = start + timedelta(days=1)
    start_utc = start.astimezone(timezone.utc)
    end_utc = datetime(end.year, end.month, end.day, tzinfo=LONDON).astimezone(timezone.utc)
    if end_utc - start_utc != timedelta(days=1):
        return None
    return [start_utc + timedelta(minutes=5 * k) for k in range(STEPS_PER_DAY)]


def day_activation(
    *, unit: str, day: date, boa: Sequence[Segment], fpn: Sequence[Segment],
    width_mw: float, partition: str,
) -> tuple[pd.DataFrame | None, str]:
    """Return one day's activation frame, or ``None`` with the reason it is left out."""
    grid = local_day_grid(day)
    if grid is None:
        return None, "non_288_slot_day"
    if not np.isfinite(width_mw) or width_mw <= 0.0:
        return None, "no_positive_width"
    lo, hi = grid[0], grid[-1] + timedelta(minutes=5)
    day_boa = [s for s in boa if s.end > lo and s.start < hi]
    day_fpn = [s for s in fpn if s.end > lo and s.start < hi]
    plan = [fpn_at(day_fpn, t) for t in grid]
    if any(value is None for value in plan):
        return None, "missing_fpn"
    plan_mw = np.asarray(plan, dtype=float)
    chosen = [acceptance_in_force(day_boa, t) for t in grid]
    instructed = np.asarray(
        [plan_mw[i] if c is None else c.value_at(t) for i, (c, t) in enumerate(zip(chosen, grid))],
        dtype=float,
    )
    local = [t.astimezone(LONDON).replace(tzinfo=None) for t in grid]
    source = pd.DataFrame({
        "step": np.arange(STEPS_PER_DAY),
        "time_london": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in local],
        "time_utc": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in grid],
        "deviation_mw": instructed - plan_mw,
    })
    waveform = build_command_waveform_day(
        source, market="GB BM", resource_kind="BATTERY_BOA_MINUS_FPN",
        resource_name=unit, source_date=day.isoformat(), step_column="step",
        target_column="deviation_mw", direction_sign=1.0,
        local_time_column="time_london", utc_time_column="time_utc",
    )
    activation = waveform_to_activation_proxy(
        waveform, reference_mode="zero", source_type=SOURCE_TYPE,
        scenario_partition=partition, segment_id=f"elexon_plan:{unit}:{day.isoformat()}",
        scale_mw=width_mw,
    )
    activation["fpn_mw"] = plan_mw
    activation["instructed_mw"] = instructed
    activation["acceptance_in_force"] = [c is not None for c in chosen]
    activation["so_flag_in_force"] = [bool(c is not None and c.so_flag) for c in chosen]
    return activation, ""


def storage_units(reference: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Battery-like BM Units and their one-sided width max(generation, |demand|).

    The reference data has no battery fuel type. A unit counts when its fuel
    type is OTHER or empty and it can both export and import at comparable
    capacities (0.5x-2x), which excludes pumped storage (PS) and one-way units.
    """
    units = {}
    for row in reference:
        fuel = row.get("fuelType")
        if fuel not in (None, "", "OTHER"):
            continue
        try:
            generation = float(row.get("generationCapacity") or 0.0)
            demand = abs(float(row.get("demandCapacity") or 0.0))
        except ValueError:
            continue
        if generation <= 0.0 or demand <= 0.0 or not 0.5 <= demand / generation <= 2.0:
            continue
        units[str(row["elexonBmUnit"])] = max(generation, demand)
    return units

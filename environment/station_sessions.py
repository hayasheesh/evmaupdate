"""Per-station session pools: arrival rates and the sessions arriving EVs are drawn from.

The tables come from data/input_EVinfo/build_station_sessions.py. For one
station and one day class (weekday / holiday) a pool holds

- the arrival rate, sessions per port and day,
- the share of that day's arrivals falling in each clock hour, and
- the sessions themselves: arrival minute, dwell minutes, delivered kWh and,
  for residential stations, plug-in SoC.

The simulator's per-step arrival probability on one charger is
rate x growth x (hour share) / steps per hour, so a charger sees rate x growth
arrivals a day on average. An arrival at step t draws a session that started in
the same clock hour (widening the window until SESSION_MATCH_MIN_CANDIDATES
sessions qualify), and takes that session's dwell, energy and plug-in SoC
together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from environment.calendars import DAY_CLASSES, WEEKDAY, as_date, day_class

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class SessionClassPool:
    rate_per_port_day: float
    hour_share: np.ndarray          # (24,), sums to 1
    arrival_minute: np.ndarray
    dwell_minutes: np.ndarray
    energy_kwh: np.ndarray
    arrival_soc_pct: np.ndarray     # NaN where the source has none
    candidates_by_hour: tuple       # 24 index arrays into the session arrays


@dataclass(frozen=True)
class StationSessionPool:
    station_id: str
    pools: dict                     # day class -> SessionClassPool

    @property
    def has_arrival_soc(self) -> bool:
        return any(np.isfinite(p.arrival_soc_pct).any() for p in self.pools.values())


def _candidates_by_hour(arrival_minute: np.ndarray, min_candidates: int) -> tuple:
    hours = np.clip((arrival_minute // 60).astype(int), 0, HOURS_PER_DAY - 1)
    out = []
    for hour in range(HOURS_PER_DAY):
        width = 0
        while True:
            window = {(hour + k) % HOURS_PER_DAY for k in range(-width, width + 1)}
            index = np.flatnonzero(np.isin(hours, sorted(window)))
            if index.size >= min_candidates or width >= HOURS_PER_DAY // 2:
                break
            width += 1
        out.append(index)
    return tuple(out)


@lru_cache(maxsize=32)
def load_station_pool(session_dir: str, station_id: str, min_candidates: int) -> StationSessionPool:
    root = Path(session_dir)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    info = meta["stations"][station_id]
    frame = pd.read_csv(root / info["file"])
    pools = {}
    for cls in DAY_CLASSES:
        rows = frame[frame["day_class"] == cls]
        if rows.empty:
            raise ValueError(f"station {station_id!r} has no {cls} sessions")
        minute = rows["arrival_minute"].to_numpy(dtype=float)
        counts = np.bincount(np.clip((minute // 60).astype(int), 0, HOURS_PER_DAY - 1),
                             minlength=HOURS_PER_DAY).astype(float)
        arrays = {
            "arrival_minute": minute,
            "dwell_minutes": rows["dwell_minutes"].to_numpy(dtype=float),
            "energy_kwh": rows["energy_kwh"].to_numpy(dtype=float),
            "arrival_soc_pct": rows["arrival_soc_pct"].to_numpy(dtype=float),
        }
        for values in arrays.values():
            values.setflags(write=False)
        pools[cls] = SessionClassPool(
            rate_per_port_day=float(info["sessions_per_port_day"][cls]),
            hour_share=counts / counts.sum(),
            candidates_by_hour=_candidates_by_hour(minute, int(min_candidates)),
            **arrays,
        )
    return StationSessionPool(station_id=station_id, pools=pools)


def service_day_class(service_date, country: str) -> str:
    """Day class of the simulated service day; no date means a weekday."""
    if service_date is None or str(service_date).strip() == "":
        return WEEKDAY
    return day_class(service_date, country)


def previous_day_classes(service_date, days: int, country: str) -> list[str]:
    """Day classes of the ``days`` calendar days before the service day, oldest first."""
    if service_date is None or str(service_date).strip() == "":
        return [WEEKDAY] * int(days)
    d = as_date(service_date)
    return [day_class(d - timedelta(days=k), country) for k in range(int(days), 0, -1)]


def arrival_probabilities(pool: StationSessionPool, cls: str, *, growth: float,
                          steps: int) -> np.ndarray:
    """Per-step arrival probability on one charger for one day of class ``cls``."""
    class_pool = pool.pools[cls]
    steps_per_hour = steps // HOURS_PER_DAY
    hourly = class_pool.rate_per_port_day * float(growth) * class_pool.hour_share
    probs = np.repeat(hourly / steps_per_hour, steps_per_hour)
    return np.clip(probs, 0.0, 1.0)


def draw_session(pool: StationSessionPool, cls: str, step: int, steps: int, rng) -> int:
    """Index of a session in ``pool.pools[cls]`` matching the arrival hour of ``step``."""
    class_pool = pool.pools[cls]
    hour = min(int(step) * HOURS_PER_DAY // int(steps), HOURS_PER_DAY - 1)
    candidates = class_pool.candidates_by_hour[hour]
    return int(candidates[rng.randrange(len(candidates))])


def session_tables_signature(session_dir: str, station_ids) -> dict:
    """Content hash of the session tables a run used, for bid-bank compatibility."""
    root = Path(session_dir)
    digest = hashlib.sha256()
    for name in ["metadata.json"] + sorted({f"{s}.csv" for s in station_ids}):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return {"station_session_tables_sha256": digest.hexdigest()}

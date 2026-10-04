"""Day-ahead EV arrival forecast for one service day.

The forecast of a station's arrivals is its measured arrival rate and hourly
shape for the service day's class (weekday, or weekend/holiday on the Japanese
calendar), scaled to a future EV population (environment.station_sessions).
The bid draws its EV scenarios from this forecast, and training and evaluation
draw their EVs from the same forecast; no separate forecast error is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from EnvConfig import (
    EPISODE_STEPS,
    EV_PREROLL_DAYS,
    FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER,
    MAX_EV_PER_STATION,
    PER_STATION_SESSION_IDS,
    SERVICE_CALENDAR_COUNTRY,
    SESSION_MATCH_MIN_CANDIDATES,
    STATION_SESSION_DIR,
)
from environment.station_sessions import (
    arrival_probabilities,
    load_station_pool,
    service_day_class,
    session_tables_signature,
)


@dataclass(frozen=True)
class ArrivalScenario:
    service_date: str | None
    arrival_probabilities_by_station: np.ndarray
    day_class: str = "weekday"
    source: str = "station_sessions"


class ArrivalScenarioSampler:
    """Build EVEnv-ready arrival probabilities for a service day."""

    def __init__(self, station_ids: Sequence[str] | None = None):
        self.source = "station_sessions"
        self.station_ids = list(station_ids) if station_ids is not None else list(PER_STATION_SESSION_IDS)
        self.pools = [
            load_station_pool(str(STATION_SESSION_DIR), station, int(SESSION_MATCH_MIN_CANDIDATES))
            for station in self.station_ids
        ]
        self.growth = float(FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER)

    def scenario_for_day(self, service_date: str | None) -> ArrivalScenario:
        cls = service_day_class(service_date, SERVICE_CALENDAR_COUNTRY)
        probs = np.stack([
            arrival_probabilities(pool, cls, growth=self.growth, steps=EPISODE_STEPS)
            for pool in self.pools
        ])
        return ArrivalScenario(
            service_date=None if service_date is None else str(service_date),
            arrival_probabilities_by_station=probs,
            day_class=cls,
            source=self.source,
        )

    def daily_arrival_attempts_by_station(self, cls: str) -> list[float]:
        """Expected arrival attempts per station-day in class ``cls`` (before chargers fill)."""
        return [
            float(pool.pools[cls].rate_per_port_day * self.growth * MAX_EV_PER_STATION)
            for pool in self.pools
        ]

    def describe(self) -> str:
        weekday = self.daily_arrival_attempts_by_station("weekday")
        holiday = self.daily_arrival_attempts_by_station("holiday")
        return (
            f"{self.source}: stations={len(self.station_ids)} "
            f"daily_attempts weekday={min(weekday):.1f}-{max(weekday):.1f} "
            f"holiday={min(holiday):.1f}-{max(holiday):.1f} "
            f"future_growth={self.growth:.2f}"
        )

    def settings_signature(self) -> dict:
        """Stable manifest payload used to reject stale upper-bid banks."""

        return {
            "source": str(self.source),
            "stations": list(self.station_ids),
            **session_tables_signature(str(STATION_SESSION_DIR), self.station_ids),
            "future_growth_multiplier": float(self.growth),
            "max_ev_per_station": int(MAX_EV_PER_STATION),
            "service_calendar": str(SERVICE_CALENDAR_COUNTRY),
            "session_match_min_candidates": int(SESSION_MATCH_MIN_CANDIDATES),
            "preroll_days": int(EV_PREROLL_DAYS),
            "forecast_error": None,
        }

"""Day-ahead EV arrival forecast for one service day.

The forecast of a station's arrivals is its measured arrival rate and hourly
shape for the service day's class (weekday, or weekend/holiday on the Japanese
calendar), scaled to a future EV population (environment.station_sessions).
The bid draws its EV scenarios from this forecast, and training and evaluation
draw their EVs from the same forecast; no separate forecast error is applied.
Calendar and weather features of the day are produced for the observation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from EnvConfig import (
    DAY_CONTEXT_INCLUDE_WEATHER,
    DAY_CONTEXT_WEATHER_CSV,
    EPISODE_STEPS,
    EV_PREROLL_DAYS,
    FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER,
    MAX_EV_PER_STATION,
    PER_STATION_SESSION_IDS,
    SERVICE_CALENDAR_COUNTRY,
    SESSION_MATCH_MIN_CANDIDATES,
    STATION_SESSION_DIR,
)
from environment.observation_config import CALENDAR_CONTEXT_FEATURES, WEATHER_CONTEXT_FEATURES
from environment.station_sessions import (
    arrival_probabilities,
    load_station_pool,
    service_day_class,
    session_tables_signature,
)
from market.day_ahead import N_BLOCKS, calendar_weather_features, load_weather_frame


@dataclass(frozen=True)
class ArrivalScenario:
    service_date: str | None
    arrival_probabilities_by_station: np.ndarray
    day_context: np.ndarray
    day_class: str = "weekday"
    source: str = "station_sessions"


class ArrivalScenarioSampler:
    """Build EVEnv-ready arrival probabilities for a service day."""

    def __init__(
        self,
        weather_csv: str = DAY_CONTEXT_WEATHER_CSV,
        station_ids: Sequence[str] | None = None,
        include_weather: bool = DAY_CONTEXT_INCLUDE_WEATHER,
    ):
        self.source = "station_sessions"
        self.station_ids = list(station_ids) if station_ids is not None else list(PER_STATION_SESSION_IDS)
        self.include_weather = bool(include_weather)
        self.weather = load_weather_frame(str(weather_csv)) if weather_csv else load_weather_frame(None)
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
            day_context=self.context_for_day(service_date),
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

    def context_for_day(self, service_date: str | None) -> np.ndarray:
        dim = len(CALENDAR_CONTEXT_FEATURES) + (len(WEATHER_CONTEXT_FEATURES) if self.include_weather else 0)
        if service_date is None:
            return np.zeros(dim, dtype=np.float32)

        midday = calendar_weather_features(service_date, N_BLOCKS // 2, self.weather)
        values = [
            midday["dow_sin"],
            midday["dow_cos"],
            midday["month_sin"],
            midday["month_cos"],
            midday["is_weekend"],
            midday["is_holiday"],
        ]
        if self.include_weather:
            weather_rows = [calendar_weather_features(service_date, b, self.weather) for b in range(N_BLOCKS)]
            temp = np.asarray([r["temp_c"] for r in weather_rows], dtype=float)
            apparent = np.asarray([r["apparent_temp_c"] for r in weather_rows], dtype=float)
            precip = np.asarray([r["precipitation_mm"] for r in weather_rows], dtype=float)
            rain = np.asarray([r["rain_mm"] for r in weather_rows], dtype=float)
            snow = np.asarray([r["snowfall_cm"] for r in weather_rows], dtype=float)
            wind = np.asarray([r["wind_speed_10m"] for r in weather_rows], dtype=float)
            values.extend([
                _clip_unit((float(np.nanmean(temp)) - 10.0) / 30.0),
                _clip_unit((float(np.nanmean(apparent)) - 10.0) / 30.0),
                _clip_unit(float(np.nansum(precip)) / 20.0),
                _clip_unit(float(np.nansum(rain)) / 20.0),
                _clip_unit(float(np.nansum(snow)) / 10.0),
                _clip_unit(float(np.nanmean(wind)) / 30.0),
            ])
        return np.asarray(values, dtype=np.float32)


def _clip_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))

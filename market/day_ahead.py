"""Calendar and weather features known before the operating day.

This module deliberately avoids using live EV state.  It builds the block-level
context that can exist before the operating day: calendar position, holidays,
and the weather forecast.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from EnvConfig import (
    EPISODE_STEPS,
    SERVICE_CALENDAR_COUNTRY,
)
from environment.calendars import is_public_holiday


STEPS_PER_BLOCK = 6
N_BLOCKS = EPISODE_STEPS // STEPS_PER_BLOCK
DEFAULT_LATITUDE = 40.0150
DEFAULT_LONGITUDE = -105.2705
DEFAULT_TIMEZONE = "America/Denver"

WEATHER_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
]

def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _naive_times(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out


def load_weather_frame(path: str | None) -> pd.DataFrame:
    """Load weather CSV with a ``time`` column; return an empty frame if omitted."""

    if path is None:
        return pd.DataFrame(columns=["time"] + WEATHER_COLUMNS)
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"weather CSV must contain a 'time' column: {path}")
    df["time"] = _naive_times(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def weather_for_block(weather: pd.DataFrame, service_date: str | date, block_index: int) -> dict:
    """Return nearest-hour weather values for a 30-minute block."""

    defaults = {
        "temperature_2m": 0.0,
        "apparent_temperature": 0.0,
        "precipitation": 0.0,
        "rain": 0.0,
        "snowfall": 0.0,
        "cloud_cover": 0.0,
        "wind_speed_10m": 0.0,
    }
    if weather is None or weather.empty:
        return defaults

    day = _coerce_date(service_date)
    block_time = pd.Timestamp(datetime.combine(day, datetime.min.time())) + pd.Timedelta(minutes=30 * block_index)
    w = weather.dropna(subset=["time"]).copy()
    if w.empty:
        return defaults
    idx = (w["time"] - block_time).abs().idxmin()
    delta = abs(w.loc[idx, "time"] - block_time)
    if delta > pd.Timedelta(hours=6):
        return defaults
    row = w.loc[idx]
    out = {}
    for col, default in defaults.items():
        val = row[col] if col in row and pd.notna(row[col]) else default
        out[col] = float(val)
    return out


def calendar_weather_features(
    service_date: str | date | datetime,
    block_index: int,
    weather: pd.DataFrame | None = None,
) -> dict:
    """Feature row available before the operating day."""

    day =_coerce_date(service_date)
    hour =block_index * 0.5
    hour_ang =2.0 * np.pi * hour / 24.0
    dow =day.weekday()
    dow_ang =2.0 * np.pi * dow / 7.0
    month_ang =2.0 * np.pi * (day.month - 1) / 12.0
    wx =weather_for_block(weather, day, block_index)
    return {
        "dow": float(dow),
        "month": float(day.month),
        "is_weekend": float(dow >= 5),
        "is_holiday": float(is_public_holiday(day, SERVICE_CALENDAR_COUNTRY)),
        "block_frac": block_index / N_BLOCKS,
        "hour_sin": float(np.sin(hour_ang)),
        "hour_cos": float(np.cos(hour_ang)),
        "dow_sin": float(np.sin(dow_ang)),
        "dow_cos": float(np.cos(dow_ang)),
        "month_sin": float(np.sin(month_ang)),
        "month_cos": float(np.cos(month_ang)),
        "temp_c": wx["temperature_2m"],
        "apparent_temp_c": wx["apparent_temperature"],
        "precipitation_mm": wx["precipitation"],
        "rain_mm": wx["rain"],
        "snowfall_cm": wx["snowfall"],
        "cloud_cover_pct": wx["cloud_cover"],
        "wind_speed_10m": wx["wind_speed_10m"],
        "has_precipitation": float(wx["precipitation"] > 0.0),
        "has_snow": float(wx["snowfall"] > 0.0),
    }

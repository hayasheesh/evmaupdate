"""Merge Open-Meteo historical-forecast weather into the configured CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY_COLUMNS = (
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--latitude", type=float, default=40.015)
    parser.add_argument("--longitude", type=float, default=-105.2705)
    parser.add_argument("--timezone", default="America/Denver")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/weather/open_meteo_boulder_forecast_2024_demand_days.csv"
        ),
    )
    return parser.parse_args()


def fetch_weather(args: argparse.Namespace) -> pd.DataFrame:
    response = requests.get(
        API_URL,
        params={
            "latitude": float(args.latitude),
            "longitude": float(args.longitude),
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
            "hourly": ",".join(HOURLY_COLUMNS),
            "timezone": str(args.timezone),
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    frame = pd.DataFrame({"time": hourly.get("time", [])})
    for column in HOURLY_COLUMNS:
        frame[column] = hourly.get(column, [])
    if frame.empty or frame.isna().any().any():
        raise RuntimeError("Open-Meteo returned empty or incomplete hourly data")
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    if frame["time"].duplicated().any():
        raise RuntimeError("Open-Meteo returned duplicate local timestamps")
    return frame


def main() -> None:
    args = parse_args()
    incoming = fetch_weather(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        existing = pd.read_csv(output)
        existing["time"] = pd.to_datetime(existing["time"], errors="raise")
        combined = pd.concat([existing, incoming], ignore_index=True)
        # An explicitly fetched interval is the newest source of truth.
        combined = combined.drop_duplicates(subset="time", keep="last")
    else:
        combined = incoming
    combined = combined.sort_values("time").reset_index(drop=True)
    combined["time"] = combined["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    temporary = output.with_suffix(output.suffix + ".part")
    try:
        combined.to_csv(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Merged {len(incoming)} rows for {args.start_date}..{args.end_date} "
        f"into {output} ({len(combined)} total rows)"
    )


if __name__ == "__main__":
    main()

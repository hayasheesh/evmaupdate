"""Build one charging-session table per station from that station's own records.

Each output row is one real session with its arrival time, dwell time and
delivered energy taken together, so the simulator draws all three from the same
session. Residential rows also carry the session's plug-in SoC.

  python data/input_EVinfo/build_station_sessions.py

Sources (all already in this repository):
  Boulder   Electric_Vehicle_Charging_Station_Data_-8671638762898357044.csv
            City of Boulder Open Data, CC0. Local timestamps. Two ports per station.
  ACN-Data  external_sources/acn_data/acn_api_sessions_{jpl,caltech}_2018.json
            ACN-Data API exports (JPL Arroyo Garage, Caltech California Garage).
            UTC timestamps, converted to America/Los_Angeles. A port is a distinct
            EVSE (stationID). The first and last calendar dates of each export are
            partial and are dropped.
  Norway    external_sources/norway_residential/Dataset1_charging_reports.csv and
            Dataset3_session_predictions.csv (Zenodo 10.5281/zenodo.12730566,
            CC BY 4.0). Local timestamps. A port is one user's home charger and is
            exposed from that user's first to last session date. SoC_start is the
            dataset's own value, which it derives by assuming every session ends
            at 95 % SoC.

Day classes: weekday, or holiday (Saturday, Sunday, or a public holiday of the
source country: US federal for Boulder and ACN, Norwegian for Norway).

The arrival rate of a station in a day class is its sessions in that class
divided by its port-days in that class. Sessions with a non-positive dwell or a
negative energy are dropped, and exact duplicate Boulder rows are removed.
"""

from __future__ import annotations

import email.utils
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from environment.calendars import DAY_CLASSES, day_class  # noqa: E402

OUT_DIR = HERE / "station_sessions"
BOULDER_CSV = HERE / "Electric_Vehicle_Charging_Station_Data_-8671638762898357044.csv"
ACN_DIR = HERE / "external_sources" / "acn_data"
NORWAY_DIR = HERE / "external_sources" / "norway_residential"

# station id (used by the simulator), source, source key
STATIONS = (
    ("COMM VITALITY _ 1400 WALNUT1", "boulder", "COMM VITALITY / 1400 WALNUT1"),
    ("COMM VITALITY _ 1104 SPRUCE1", "boulder", "COMM VITALITY / 1104 SPRUCE1"),
    ("COMM VITALITY _ 1500PEARL1", "boulder", "COMM VITALITY / 1500PEARL1"),
    ("ACN_JPL_ARROYO1", "acn", "acn_api_sessions_jpl_2018.json"),
    ("ACN_CALTECH_GARAGE1", "acn", "acn_api_sessions_caltech_2018.json"),
    ("RESIDENTIAL NORWAY _ OSL_S", "norway", "OSL_S"),
    ("RESIDENTIAL NORWAY _ TRO_R", "norway", "TRO_R"),
    ("BOULDER _ CARPENTER PARK1", "boulder", "BOULDER / CARPENTER PARK1"),
    ("BOULDER _ N BOULDER REC 1", "boulder", "BOULDER / N BOULDER REC 1"),
)
BOULDER_PORTS = 2
SOURCES = {
    "boulder": "City of Boulder Open Data, Electric Vehicle Charging Station Data "
               "(dataset 95992b3938be4622b07f0b05eba95d4c_0), CC0 1.0",
    "acn": "ACN-Data (Lee, Li and Low, e-Energy 2019), https://ev.caltech.edu/dataset; "
           "license not stated, citation required",
    "norway": "Zenodo 10.5281/zenodo.12730566 (Norwegian residential EV charging), CC BY 4.0",
}


def _frame(times: pd.Series, dwell_minutes, energy_kwh, session_ids, country: str,
           arrival_soc=None) -> pd.DataFrame:
    times = pd.to_datetime(times)
    frame = pd.DataFrame({
        "source_session_id": [str(s) for s in session_ids],
        "local_date": times.dt.date.astype(str),
        "arrival_minute": (times.dt.hour * 60 + times.dt.minute + times.dt.second / 60.0).round(3),
        "dwell_minutes": np.asarray(dwell_minutes, dtype=float).round(3),
        "energy_kwh": np.asarray(energy_kwh, dtype=float).round(4),
        "arrival_soc_pct": (np.asarray(arrival_soc, dtype=float).round(3)
                            if arrival_soc is not None else np.nan),
    })
    frame["day_class"] = [day_class(d, country) for d in frame["local_date"]]
    keep = (frame["dwell_minutes"] > 0.0) & (frame["energy_kwh"] >= 0.0)
    return frame[keep].reset_index(drop=True)


def _calendar_days(first, last) -> list:
    return list(pd.date_range(first, last, freq="D").date)


def boulder_station(name: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(BOULDER_CSV, usecols=[
        "Station_Name", "Start_Date___Time", "End_Date___Time",
        "Total_Duration__hh_mm_ss_", "Energy__kWh_", "ObjectID",
    ])
    raw = raw[raw["Station_Name"] == name]
    rows = len(raw)
    raw = raw.drop_duplicates(subset=["Start_Date___Time", "End_Date___Time"])
    start = pd.to_datetime(raw["Start_Date___Time"], format="mixed")
    dwell = pd.to_timedelta(raw["Total_Duration__hh_mm_ss_"], errors="coerce").dt.total_seconds() / 60.0
    frame = _frame(start, dwell, raw["Energy__kWh_"], raw["ObjectID"], "US")
    days = _calendar_days(frame["local_date"].min(), frame["local_date"].max())
    exposure = {cls: BOULDER_PORTS * sum(day_class(d, "US") == cls for d in days) for cls in DAY_CLASSES}
    info = {"source": "boulder", "source_key": name, "source_rows": int(rows),
            "duplicates_removed": int(rows - len(raw)), "ports": BOULDER_PORTS,
            "port_definition": "two Level 2 ports per Boulder station",
            "first_date": str(days[0]), "last_date": str(days[-1])}
    return frame, {**info, "port_days": exposure}


def _load_acn_items(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").rstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # The API export ends after the last complete item without closing the list.
        payload = json.loads(re.sub(r",\s*$", "", text) + "]}")
    return list(payload["_items"])


def acn_station(file_name: str) -> tuple[pd.DataFrame, dict]:
    items = _load_acn_items(ACN_DIR / file_name)
    connect = pd.to_datetime([email.utils.parsedate_to_datetime(i["connectionTime"]) for i in items], utc=True)
    disconnect = pd.to_datetime([email.utils.parsedate_to_datetime(i["disconnectTime"]) for i in items], utc=True)
    local = connect.tz_convert("America/Los_Angeles").tz_localize(None)
    dwell = (disconnect - connect).total_seconds() / 60.0
    frame = _frame(pd.Series(local), dwell, [i["kWhDelivered"] for i in items],
                   [i["sessionID"] for i in items], "US")
    all_dates = sorted(set(frame["local_date"]))
    first, last = all_dates[0], all_dates[-1]
    frame = frame[(frame["local_date"] > first) & (frame["local_date"] < last)].reset_index(drop=True)
    days = _calendar_days(pd.Timestamp(first) + pd.Timedelta(days=1), pd.Timestamp(last) - pd.Timedelta(days=1))
    ports = len({i["stationID"] for i in items})
    exposure = {cls: ports * sum(day_class(d, "US") == cls for d in days) for cls in DAY_CLASSES}
    info = {"source": "acn", "source_key": file_name, "source_rows": len(items),
            "site": sorted({i["siteID"] for i in items}), "cluster": sorted({i["clusterID"] for i in items}),
            "ports": ports, "port_definition": "distinct EVSE stationID in the export",
            "first_date": str(days[0]), "last_date": str(days[-1]),
            "partial_edge_dates_dropped": [first, last]}
    return frame, {**info, "port_days": exposure}


def norway_station(location: str) -> tuple[pd.DataFrame, dict]:
    reports = pd.read_csv(NORWAY_DIR / "Dataset1_charging_reports.csv", sep=";", decimal=",")
    predictions = pd.read_csv(NORWAY_DIR / "Dataset3_session_predictions.csv", sep=";", decimal=",")
    reports = reports[reports["location"] == location]
    rows = len(reports)
    merged = reports.merge(predictions[["user_id", "session_id", "SoC_start"]],
                           on=["user_id", "session_id"], how="inner")
    plugin = pd.to_datetime(merged["plugin_time"])
    frame = _frame(plugin, merged["connection_time"].astype(float) * 60.0, merged["energy_session"],
                   merged["session_id"], "NO", arrival_soc=merged["SoC_start"])
    exposure = {cls: 0 for cls in DAY_CLASSES}
    dates = pd.to_datetime(plugin).dt.date
    for _user, user_dates in dates.groupby(merged["user_id"]):
        for d in _calendar_days(user_dates.min(), user_dates.max()):
            exposure[day_class(d, "NO")] += 1
    info = {"source": "norway", "source_key": location, "source_rows": int(rows),
            "rows_without_soc": int(rows - len(merged)),
            "ports": int(merged["user_id"].nunique()),
            "port_definition": "one home charger per user, exposed from the user's first to last session date",
            "first_date": str(dates.min()), "last_date": str(dates.max())}
    return frame, {**info, "port_days": exposure}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {"boulder": boulder_station, "acn": acn_station, "norway": norway_station}
    stations = {}
    for station_id, source, key in STATIONS:
        frame, info = builders[source](key)
        file_name = f"{station_id}.csv"
        frame.to_csv(OUT_DIR / file_name, index=False)
        sessions = {cls: int((frame["day_class"] == cls).sum()) for cls in DAY_CLASSES}
        rates = {cls: (sessions[cls] / info["port_days"][cls] if info["port_days"][cls] else 0.0)
                 for cls in DAY_CLASSES}
        stations[station_id] = {
            **info, "file": file_name, "sessions": sessions,
            "sessions_per_port_day": rates,
            "mean_dwell_hours": {cls: float(frame.loc[frame["day_class"] == cls, "dwell_minutes"].mean() / 60.0)
                                 for cls in DAY_CLASSES},
            "mean_energy_kwh": {cls: float(frame.loc[frame["day_class"] == cls, "energy_kwh"].mean())
                                for cls in DAY_CLASSES},
            "has_arrival_soc": bool(frame["arrival_soc_pct"].notna().any()),
        }
        print(f"{station_id}: {sessions} rate/port-day {rates}", flush=True)
    metadata = {
        "columns": {
            "source_session_id": "session identifier in the source",
            "local_date": "local calendar date of the arrival",
            "day_class": "weekday, or holiday (weekend or public holiday of the source country)",
            "arrival_minute": "local arrival time, minutes after midnight",
            "dwell_minutes": "plug-in to plug-out time",
            "energy_kwh": "energy delivered in the session",
            "arrival_soc_pct": "plug-in SoC, residential stations only (Norway Dataset3 SoC_start)",
        },
        "sources": SOURCES,
        "stations": stations,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

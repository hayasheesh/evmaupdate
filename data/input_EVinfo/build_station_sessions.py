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

Day classes: weekday, or holiday (Saturday, Sunday, or a day off of the source
country: a public holiday or a customary off period, environment.calendars).

Days left out of a station's record (no sessions, no port-days):
  - 14 Mar 2020 to 30 Jun 2021 for the US stations (COVID-19). Weekday sessions
    per port-day at the Boulder stations fell to 3 - 40 % of their 2019 level in
    2020 Q2 - 2021 Q1 and were back in 2021 Q3.
  - the first 180 days of each Boulder station, while use was still growing
    after installation (WALNUT: 0.08 in 2018 Q1 against 0.6 - 0.9 in 2019).
  - outages: a run of days with no session that the station's own rate makes
    implausible, exp(-expected sessions) < 1e-3, where each day of the run
    expects the sessions per day of its own day class over the 90 days on each
    side of the run. Applied to the
    stations with a fixed set of ports (Boulder, ACN); Norway exposes each home
    charger only between its user's first and last session.
Boulder repeats sessions with the start time written in two formats (with and
without seconds) and the duration rounded differently, so duplicates are
matched on the start minute, the duration in whole minutes and the energy.

The arrival rate of a station in a day class is its sessions in that class
divided by its port-days in that class. Sessions with a non-positive dwell or a
negative energy are dropped, and exact duplicate Boulder rows are removed. So
are plug-ins that took under 0.1 kWh and were unplugged within 10 minutes: no
charging took place, and keeping them as arrivals would add EVs that only
occupy a charger for one step. They are dropped before the arrival rate is
counted, so they do not become charging arrivals either.
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
    # The source lists these chargers as COMM VITALITY until early May 2023 and
    # as MUNICIPAL SC from then on, at the same address; both names are one station.
    ("COMM VITALITY _ 1400 WALNUT1", "boulder", ("COMM VITALITY / 1400 WALNUT1", "MUNICIPAL SC / 1400 WALNUT1")),
    ("COMM VITALITY _ 1104 SPRUCE1", "boulder", ("COMM VITALITY / 1104 SPRUCE1", "MUNICIPAL SC / 1104 SPRUCE1")),
    ("COMM VITALITY _ 1500PEARL1", "boulder", ("COMM VITALITY / 1500PEARL1", "MUNICIPAL SC / 1500PEARL1")),
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


EMPTY_PLUGIN_KWH = 0.1
EMPTY_PLUGIN_MINUTES = 10.0
EXCLUDED_PERIODS = {"US": (("2020-03-14", "2021-06-30", "covid19"),)}
BOULDER_RAMP_UP_DAYS = 180
OUTAGE_PROBABILITY = 1e-3
OUTAGE_CONTEXT_DAYS = 90


def _excluded_by_period(days, country: str) -> set:
    out = set()
    for first, last, _why in EXCLUDED_PERIODS.get(country, ()):
        lo, hi = pd.Timestamp(first).date(), pd.Timestamp(last).date()
        out |= {d for d in days if lo <= d <= hi}
    return out


def _outage_days(frame: pd.DataFrame, days: list, excluded: set, country: str) -> list[tuple]:
    """Runs of zero-session days that the station's local rate makes implausible.

    The expected sessions of each day in the run come from days of the same
    day class within OUTAGE_CONTEXT_DAYS on each side, so an empty weekend at
    a workplace is not read as an outage.
    """
    counts = frame.groupby("local_date").size()
    per_day = np.array([0 if d in excluded else int(counts.get(str(d), 0)) for d in days])
    usable = np.array([d not in excluded for d in days])
    classes = np.array([day_class(d, country) for d in days])
    runs, i = [], 0
    while i < len(days):
        if usable[i] and per_day[i] == 0:
            j = i
            while j < len(days) and usable[j] and per_day[j] == 0:
                j += 1
            lo, hi = max(i - OUTAGE_CONTEXT_DAYS, 0), min(j + OUTAGE_CONTEXT_DAYS, len(days))
            ctx = np.r_[lo:i, j:hi]
            ctx = ctx[usable[ctx]]
            expected = 0.0
            for cls in set(classes[i:j]):
                same = ctx[classes[ctx] == cls]
                rate = per_day[same].mean() if len(same) else 0.0
                expected += rate * int((classes[i:j] == cls).sum())
            if expected > 0 and np.exp(-expected) < OUTAGE_PROBABILITY:
                runs.append((days[i], days[j - 1], j - i))
            i = j
        else:
            i += 1
    return runs


def _keep_days(frame: pd.DataFrame, days: list, country: str, *, ramp_up_days: int = 0):
    """Drop excluded periods, the ramp-up and outages; return the frame, kept days and a log."""
    excluded = _excluded_by_period(days, country) | set(days[:ramp_up_days])
    outages = _outage_days(frame, days, excluded, country)
    for first, last, _n in outages:
        excluded |= {d for d in days if first <= d <= last}
    kept = [d for d in days if d not in excluded]
    kept_str = {str(d) for d in kept}
    log = {
        "excluded_periods": [list(p) for p in EXCLUDED_PERIODS.get(country, ())],
        "ramp_up_days_dropped": int(min(ramp_up_days, len(days))),
        "outages": [[str(a), str(b), int(n)] for a, b, n in outages],
        "days_kept": len(kept), "days_in_record": len(days),
        "sessions_dropped_by_day_exclusion": int((~frame["local_date"].isin(kept_str)).sum()),
    }
    return frame[frame["local_date"].isin(kept_str)].reset_index(drop=True), kept, log


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
    no_charge = (frame["energy_kwh"] < EMPTY_PLUGIN_KWH) & (frame["dwell_minutes"] < EMPTY_PLUGIN_MINUTES)
    keep = (frame["dwell_minutes"] > 0.0) & (frame["energy_kwh"] >= 0.0) & ~no_charge
    return frame[keep].reset_index(drop=True)


def _calendar_days(first, last) -> list:
    return list(pd.date_range(first, last, freq="D").date)


def boulder_station(name) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(BOULDER_CSV, usecols=[
        "Station_Name", "Start_Date___Time", "End_Date___Time",
        "Total_Duration__hh_mm_ss_", "Energy__kWh_", "ObjectID",
    ])
    names = (name,) if isinstance(name, str) else tuple(name)
    raw = raw[raw["Station_Name"].isin(names)].copy()
    rows = len(raw)
    raw["_start"] = pd.to_datetime(raw["Start_Date___Time"], format="mixed")
    raw["_start_minute"] = raw["_start"].dt.floor("min")
    raw["_dwell"] = pd.to_timedelta(raw["Total_Duration__hh_mm_ss_"], errors="coerce").dt.total_seconds() / 60.0
    raw["_dwell_minute"] = raw["_dwell"].round()
    raw["_energy"] = pd.to_numeric(raw["Energy__kWh_"], errors="coerce").round(3)
    raw = raw.sort_values("_start").drop_duplicates(subset=["_start_minute", "_dwell_minute", "_energy"])
    start = raw["_start"]
    dwell = raw["_dwell"]
    frame = _frame(start, dwell, raw["Energy__kWh_"], raw["ObjectID"], "US")
    days = _calendar_days(frame["local_date"].min(), frame["local_date"].max())
    frame, kept, log = _keep_days(frame, days, "US", ramp_up_days=BOULDER_RAMP_UP_DAYS)
    exposure = {cls: BOULDER_PORTS * sum(day_class(d, "US") == cls for d in kept) for cls in DAY_CLASSES}
    info = {"source": "boulder", "source_key": list(names), "source_rows": int(rows),
            "duplicates_removed": int(rows - len(raw)), "ports": BOULDER_PORTS,
            "port_definition": "two Level 2 ports per Boulder station",
            "first_date": str(days[0]), "last_date": str(days[-1]), **log}
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
    frame, kept, log = _keep_days(frame, days, "US")
    ports = len({i["stationID"] for i in items})
    exposure = {cls: ports * sum(day_class(d, "US") == cls for d in kept) for cls in DAY_CLASSES}
    info = {"source": "acn", "source_key": file_name, "source_rows": len(items),
            "site": sorted({i["siteID"] for i in items}), "cluster": sorted({i["clusterID"] for i in items}),
            "ports": ports, "port_definition": "distinct EVSE stationID in the export",
            "first_date": str(days[0]), "last_date": str(days[-1]),
            "partial_edge_dates_dropped": [first, last], **log}
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
            "arrival_soc_pct": "Norway Dataset3 SoC_start, residential stations only; EVEnv draws plug-in SoC from the EV Project distribution instead",
        },
        "sources": SOURCES,
        "stations": stations,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

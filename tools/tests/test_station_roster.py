"""EVMA_STATION_ROSTER picks the stations; the default keeps every station type.

EnvConfig reads the variable when it is imported, so each case runs in a fresh
interpreter.
"""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _stations(n, roster=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVMA_")}
    env["EVMA_NUM_STATIONS"] = str(n)
    if roster is not None:
        env["EVMA_STATION_ROSTER"] = roster
    code = "import json, EnvConfig as E; print(json.dumps([E.STATION_ROSTER_NAME, list(E.PER_STATION_SESSION_IDS)]))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_default_roster_starts_with_boulder_and_keeps_seven_distinct_stations():
    name, ids = _stations(7)
    assert name == "all"
    assert len(set(ids)) == 7
    assert ids[0] == "COMM VITALITY _ 1400 WALNUT1"


def test_workplace_residential_alternates_so_even_counts_split_evenly():
    for n in (6, 20):
        name, ids = _stations(n, "workplace_residential")
        assert name == "workplace_residential"
        kinds = Counter("workplace" if s.startswith("ACN_") else "residential" for s in ids)
        assert kinds == {"workplace": n // 2, "residential": n // 2}
        assert not any(s.startswith(("COMM VITALITY", "BOULDER")) for s in ids)


def test_an_unknown_roster_is_refused():
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVMA_")}
    env["EVMA_STATION_ROSTER"] = "commercial"
    out = subprocess.run([sys.executable, "-c", "import EnvConfig"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "EVMA_STATION_ROSTER" in out.stderr

"""How energy is split between EVs inside a station, from the interim-test EV traces.

usage: allocation_metrics.py TEST_EPISODE [TEST_EPISODE ...]

For AB and the shaping run at the same interim tests (same days, same EV
realizations), using the EVs whose SoC the test recorded (long-stay and random
window traces of episodes 4 and 5):
  - share of EV-steps charging / discharging, split by instruction state and by
    whether the EV is at/above or below its target SoC, while the station
    participates
  - net SoC change per step of each group
  - share of station-steps with two or more traced EVs where one charges while
    another discharges
  - departure SoC against target of the long-stay EVs
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\admin\Desktop\EVMALOCALUPDATE\archive"
RUNS = {
    "AB": os.path.join(ROOT, "prod_f500_dense8_AB_7station_20260922_022453", "results"),
    "new": None,
}
new = sorted(glob.glob(os.path.join(ROOT, "prod_f500_dense8_AB_socshape_7station_*")))
RUNS["new"] = os.path.join(new[-1], "results") if new else None


def traces(results: str, test: int) -> pd.DataFrame:
    frames = []
    for ep in (4, 5):
        files = glob.glob(os.path.join(results, f"TEST{test}", f"test_results_ev_soc_*_episode_{ep}.csv"))
        coop_path = os.path.join(results, f"TEST{test}", f"test_results_zz_station_cooperation_full_episode_{ep}.csv")
        if not files or not os.path.exists(coop_path):
            continue
        tr = pd.concat([pd.read_csv(f)[["Step", "EV_ID", "Station", "SoC_%", "Target_%"]] for f in files])
        tr = tr.drop_duplicates(["Step", "EV_ID"]).sort_values(["EV_ID", "Step"])
        tr["d"] = tr.groupby("EV_ID")["SoC_%"].diff().shift(-1)
        coop = pd.read_csv(coop_path).set_index("Step")
        tr = tr.join(coop[["Instruction_State", "Tracking_Enabled"]], on="Step").dropna(subset=["d"])
        tr["ep"] = ep
        frames.append(tr)
    return pd.concat(frames) if frames else pd.DataFrame()


def summarize(tr: pd.DataFrame) -> dict:
    tr = tr.copy()
    tr["dir"] = np.where(tr.d > 0.05, "charge", np.where(tr.d < -0.05, "discharge", "hold"))
    tr["below"] = tr["SoC_%"] < tr["Target_%"]
    on = tr[tr.Tracking_Enabled == 1]
    out = {}
    for state in ("up", "down", "idle"):
        for below, label in ((False, "above"), (True, "below")):
            x = on[(on.Instruction_State == state) & (on.below == below)]
            if len(x):
                out[f"{state}/{label}"] = (len(x), (x.dir == "charge").mean(), (x.dir == "discharge").mean(), x.d.mean())
    g = on.groupby(["ep", "Station", "Step"])
    many = g.EV_ID.count() >= 2
    both = g.dir.agg(lambda s: "charge" in set(s) and "discharge" in set(s))[many]
    out["cross_flow"] = (int(many.sum()), float(both.mean()) if len(both) else float("nan"))
    return out


def long_stay(results: str, test: int) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(results, f"TEST{test}", "test_results_ev_soc_long_stay_*_episode_*.csv"))):
        d = pd.read_csv(f).sort_values("Step")
        for ev, g in d.groupby("EV_ID"):
            rows.append({"ev": int(ev), "target": g["Target_%"].iloc[0], "final": g["SoC_%"].iloc[-1]})
    return pd.DataFrame(rows).set_index("ev") if rows else pd.DataFrame()


def main() -> None:
    tests = [int(a) for a in sys.argv[1:]] or [100]
    for test in tests:
        print(f"===== TEST{test}")
        summaries = {}
        for name, results in RUNS.items():
            if results is None:
                continue
            tr = traces(results, test)
            if tr.empty:
                print(f"{name}: no traces")
                continue
            summaries[name] = summarize(tr)
        keys = sorted(set().union(*[set(s) for s in summaries.values()]) - {"cross_flow"})
        print(f"{'group':12s} | " + " | ".join(f"{n:>30s}" for n in summaries))
        print(f"{'':12s} | " + " | ".join(f"{'n  charge discharge dSoC/step':>30s}" for _ in summaries))
        for k in keys:
            cells = []
            for n in summaries:
                v = summaries[n].get(k)
                cells.append(f"{v[0]:5d} {v[1]:6.2f} {v[2]:9.2f} {v[3]:+8.2f}" if v else f"{'-':>30s}")
            print(f"{k:12s} | " + " | ".join(f"{c:>30s}" for c in cells))
        print("cross-flow (station-steps, share): " + ", ".join(f"{n} {s['cross_flow'][0]} {s['cross_flow'][1]:.2f}" for n, s in summaries.items()))
        ls = {n: long_stay(r, test) for n, r in RUNS.items() if r}
        if all(not v.empty for v in ls.values()):
            joined = pd.concat({n: v.final for n, v in ls.items()}, axis=1)
            joined.insert(0, "target", next(iter(ls.values())).target)
            print("long-stay EVs, departure SoC:")
            print(joined.round(1).to_string())


if __name__ == "__main__":
    main()

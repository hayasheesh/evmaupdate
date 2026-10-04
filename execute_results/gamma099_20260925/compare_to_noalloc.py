"""Compare the GAMMA_GLOBAL=0.99 run with the noalloc main line at the same episodes.

usage: compare_to_noalloc.py [--check] [--until EPISODE]

The worry is that a longer global horizon inflates the global Q and its actor
gradient until the local (SoC) gradient no longer counts. Each 100-episode
window prints both runs side by side. With --check the exit code is 2 when the
latest window crosses an alarm:

  global/local actor-gradient ratio more than 2x noalloc's in the same window
  global Q above 0.8 x the mixer bias cap (150)
  from ep200, tracking + SoC (percent, mean of the last five tests) more than
  10 below noalloc's at the same test
  any non-finite value
With --until the script waits (polling every 5 minutes) until the run has
logged that episode or an alarm fires, then prints and exits.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import sys
import time

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = r"C:\Users\admin\Desktop\EVMALOCALUPDATE"
BASE = os.path.join(ROOT, "archive", "prod_f500_dense8_noalloc_7station_20260921_102504")
NEW_GLOB = os.path.join(ROOT, "archive", "prod_f500_dense8_noalloc_g099_7station_*")
MIXER_B_MAX = 150.0
TAGS = {
    "Q_global": "Q/global",
    "g_global": "Gradient/actor_source_global_raw_mean",
    "g_local": "Gradient/actor_source_local_raw_mean",
    "cos": "Gradient/actor_source_cos_mean",
    "gc_clip": "Clipping/global_critic",
    "gc_loss": "Loss/global_critic",
    "lc_loss": "Loss/local_critic_mean",
}


def scalars(run: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for sub in ("runs", "performance"):
        for path in glob.glob(os.path.join(run, sub, "events.out.tfevents.*")):
            ea = EventAccumulator(path, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                for e in ea.Scalars(tag):
                    out.setdefault(tag, {})[int(e.step)] = float(e.value)
    return out


def actor_clip(data: dict) -> dict[int, float]:
    per = [data[t] for t in data if t.startswith("Clipping/actor_agent")]
    steps = sorted(set().union(*[set(d) for d in per])) if per else []
    return {s: float(np.mean([d[s] for d in per if s in d])) for s in steps}


def window(d: dict[int, float], lo: int, hi: int) -> float:
    vals = [v for k, v in d.items() if lo <= k < hi]
    return float(np.mean(vals)) if vals else float("nan")


def test_rates(run: str) -> dict[int, tuple[float, float]]:
    path = os.path.join(run, "results", "test_history.json")
    if not os.path.exists(path):
        return {}
    h = json.load(open(path, encoding="utf-8"))
    out = {}
    for i, ep in enumerate(h["episodes"]):
        steps = h["surplus_steps"][i] + h["shortage_steps"][i]
        inside = h["surplus_within_narrow"][i] + h["shortage_within_narrow"][i]
        out[int(ep)] = (inside / steps if steps else float("nan"),
                        h["departing_evs_soc_met"][i] / max(h["departing_evs"][i], 1))
    return out


def level(t: dict[int, tuple[float, float]]) -> dict[int, float]:
    eps = sorted(t)
    return {eps[k]: 100.0 * float(np.mean([sum(t[e]) for e in eps[k - 4:k + 1]])) for k in range(4, len(eps))}


def report() -> tuple[int, list[str]]:
    runs = sorted(glob.glob(NEW_GLOB))
    if not runs:
        return 0, ["new run not found"]
    new_run = runs[-1]
    base, new = scalars(BASE), scalars(new_run)
    base["actor_clip"], new["actor_clip"] = actor_clip(base), actor_clip(new)
    last = max((max(d) for d in new.values() if d), default=0)
    print(f"run {os.path.basename(new_run)}  last logged episode {last}")
    print(f"{'window':>11} | {'Q_glob':>13} | {'g_glob/g_loc':>17} | {'cos':>13} | {'actor clip':>11} | {'gcrit clip':>11}")
    alarms: list[str] = []
    latest = None
    for lo in range(0, last + 1, 100):
        hi = lo + 100
        row = {}
        for key in ("Q_global", "g_global", "g_local", "cos", "actor_clip", "gc_clip"):
            tag = TAGS.get(key, key)
            row[key] = (window(base.get(tag, {}), lo, hi), window(new.get(tag, {}), lo, hi))
        if all(math.isnan(v[1]) for v in row.values()):
            continue
        ratio_b = row["g_global"][0] / row["g_local"][0] if row["g_local"][0] else float("nan")
        ratio_n = row["g_global"][1] / row["g_local"][1] if row["g_local"][1] else float("nan")
        print(f"{lo:5d}-{hi - 1:<5d} | {row['Q_global'][0]:5.2f} -> {row['Q_global'][1]:5.2f} | "
              f"{ratio_b:6.2f} -> {ratio_n:6.2f} | {row['cos'][0]:5.2f} -> {row['cos'][1]:5.2f} | "
              f"{row['actor_clip'][0]:4.2f} -> {row['actor_clip'][1]:4.2f} | {row['gc_clip'][0]:4.2f} -> {row['gc_clip'][1]:4.2f}")
        latest = (lo, ratio_b, ratio_n, row["Q_global"][1])
        if any(math.isinf(v[1]) for v in row.values()):
            alarms.append(f"non-finite value in window {lo}")
    if latest:
        lo, ratio_b, ratio_n, q = latest
        if math.isfinite(ratio_b) and math.isfinite(ratio_n) and ratio_n > 2.0 * ratio_b:
            alarms.append(f"window {lo}: global/local gradient ratio {ratio_n:.2f} > 2 x noalloc {ratio_b:.2f}")
        if math.isfinite(q) and q > 0.8 * MIXER_B_MAX:
            alarms.append(f"window {lo}: global Q {q:.1f} near the mixer bias cap")
    tb, tn = test_rates(BASE), test_rates(new_run)
    if tn:
        print(f"{'test ep':>7} | {'tracking':>15} | {'SoC met':>15}")
        for ep in sorted(tn)[-6:]:
            b = tb.get(ep, (float("nan"), float("nan")))
            print(f"{ep:7d} | {b[0]:.3f} -> {tn[ep][0]:.3f} | {b[1]:.3f} -> {tn[ep][1]:.3f}")
        lb, ln = level(tb), level(tn)
        if ln:
            ep = max(ln)
            print(f"tracking + SoC, mean of the last five tests at ep{ep}: noalloc {lb.get(ep, float('nan')):.1f} -> {ln[ep]:.1f}")
            if ep >= 200 and ep in lb and ln[ep] < lb[ep] - 10.0:
                alarms.append(f"test ep {ep}: tracking + SoC level {ln[ep]:.1f} < noalloc {lb[ep]:.1f} - 10")
    for a in alarms:
        print("[alarm]", a)
    return last, alarms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--until", type=int, default=None)
    args = parser.parse_args()
    while args.until is not None:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            last, alarms = report()
        if alarms or last >= args.until:
            break
        time.sleep(300)
    last, alarms = report()
    if (args.check or args.until is not None) and alarms:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

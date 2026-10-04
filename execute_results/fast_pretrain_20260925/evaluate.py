"""Score short noalloc pretrains against the AB run and watch them for divergence.

usage: evaluate.py NAME [NAME ...] [--until EPISODE] [--window 50] [--scale 3]

Performance is tracking + SoC, both in percent, from the interim tests on the
five validation days:
  tracking = market steps inside the narrow band / market steps
  SoC      = departing EVs that met their SoC target / departing EVs
A run's level at a test is the mean of that sum over the last five tests (100
episodes). The target is AB's level at its last test (ep1300). With --scale k
the table also shows noalloc's test and level at k times the episode, the point
a run whose schedule is k times shorter would match if only time were rescaled.

Each training window prints the global and local critic Q, the global/local
actor-gradient ratio, their cosine, actor and critic clip rates and critic
losses. The exit code is 2 when any run crosses an alarm:
  a non-finite scalar
  global Q above 0.8 x the mixer bias cap (50)
  |local Q| above 5 (local rewards stay within about +-0.05 per step)
  global/local actor-gradient ratio outside 0.4-5 (noalloc stayed 1.0-2.5)
  global critic loss above 2 or local critic loss above 0.5 (noalloc: 0.36, 0.10)
  a Traceback in the run's stderr
  under 20 GB free on C:
With --until the script polls every 5 minutes and exits when every run has
logged that episode, when any run crosses an alarm, or when a run's stdout log
has not grown for 30 minutes.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import shutil
import time

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = r"C:\Users\admin\Desktop\EVMALOCALUPDATE"
HERE = os.path.join(ROOT, "execute_results", "fast_pretrain_20260925")
AB = os.path.join(ROOT, "archive", "prod_f500_dense8_AB_7station_20260922_022453")
NOALLOC = os.path.join(ROOT, "archive", "prod_f500_dense8_noalloc_7station_20260921_102504")
LEVEL_TESTS = 5
MIXER_B_MAX = 50.0
TAGS = {
    "Qg": "Q/global",
    "Ql": "Q/local_mean",
    "gG": "Gradient/actor_source_global_raw_mean",
    "gL": "Gradient/actor_source_local_raw_mean",
    "cos": "Gradient/actor_source_cos_mean",
    "gc_clip": "Clipping/global_critic",
    "gc_loss": "Loss/global_critic",
    "lc_loss": "Loss/local_critic_mean",
    "eps": "Training/epsilon",
    "ou": "Training/ou_noise_scale",
}


def run_dir(name: str) -> str | None:
    runs = sorted(glob.glob(os.path.join(ROOT, "archive", f"prod_f500_dense8_noalloc_fast_{name}_2*")))
    return runs[-1] if runs else None


def tests(run: str) -> dict[int, tuple[float, float]]:
    path = os.path.join(run, "results", "test_history.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    out = {}
    for i, ep in enumerate(h["episodes"]):
        steps = h["surplus_steps"][i] + h["shortage_steps"][i]
        inside = h["surplus_within_narrow"][i] + h["shortage_within_narrow"][i]
        out[int(ep)] = (100.0 * inside / steps if steps else float("nan"),
                        100.0 * h["departing_evs_soc_met"][i] / max(h["departing_evs"][i], 1))
    return out


def levels(t: dict[int, tuple[float, float]]) -> dict[int, float]:
    eps = sorted(t)
    out = {}
    for k in range(LEVEL_TESTS - 1, len(eps)):
        out[eps[k]] = float(np.mean([sum(t[e]) for e in eps[k - LEVEL_TESTS + 1:k + 1]]))
    return out


def scalars(run: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for sub in ("runs", "performance"):
        for path in glob.glob(os.path.join(run, sub, "events.out.tfevents.*")):
            ea = EventAccumulator(path, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                for e in ea.Scalars(tag):
                    out.setdefault(tag, {})[int(e.step)] = float(e.value)
    per = [out[t] for t in out if t.startswith("Clipping/actor_agent")]
    steps = sorted(set().union(*[set(d) for d in per])) if per else []
    out["actor_clip"] = {s: float(np.mean([d[s] for d in per if s in d])) for s in steps}
    return out


def window(d: dict[int, float], lo: int, hi: int) -> float:
    vals = [v for k, v in d.items() if lo <= k < hi]
    return float(np.mean(vals)) if vals else float("nan")


AB_TESTS = tests(AB)
NOALLOC_TESTS = tests(NOALLOC)
AB_LEVEL = levels(AB_TESTS)
NOALLOC_LEVEL = levels(NOALLOC_TESTS)
TARGET = AB_LEVEL[max(AB_LEVEL)]


def first_reach(level: dict[int, float]) -> int | None:
    return next((ep for ep in sorted(level) if level[ep] >= TARGET), None)


def report(name: str, win: int, scale: int = 0) -> tuple[int, list[str]]:
    run = run_dir(name)
    if run is None:
        return 0, []
    alarms: list[str] = []
    data = scalars(run)
    last = max((max(d) for d in data.values() if d), default=0)
    print(f"== {name}: {os.path.basename(run)}  last logged episode {last}")
    print(f"{'window':>9} | {'eps':>5} {'noise':>5} | {'Q_glob':>6} {'Q_loc':>6} | {'gG/gL':>5} {'cos':>5} | "
          f"{'a_clip':>6} {'gc_clip':>7} | {'gc_loss':>7} {'lc_loss':>7}")
    for lo in range(0, last + 1, win):
        hi = lo + win
        row = {k: window(data.get(t, {}), lo, hi) for k, t in TAGS.items()}
        row["a_clip"] = window(data["actor_clip"], lo, hi)
        if all(math.isnan(v) for v in row.values()):
            continue
        ratio = row["gG"] / row["gL"] if row["gL"] else float("nan")
        print(f"{lo:4d}-{hi - 1:<4d} | {row['eps']:5.3f} {row['ou']:5.3f} | {row['Qg']:6.2f} {row['Ql']:6.3f} | "
              f"{ratio:5.2f} {row['cos']:5.2f} | {row['a_clip']:6.3f} {row['gc_clip']:7.3f} | "
              f"{row['gc_loss']:7.3f} {row['lc_loss']:7.3f}")
        if any(math.isinf(v) for v in row.values()):
            alarms.append(f"{name} window {lo}: non-finite scalar")
        if row["Qg"] > 0.8 * MIXER_B_MAX:
            alarms.append(f"{name} window {lo}: global Q {row['Qg']:.1f} near the mixer bias cap")
        if abs(row["Ql"]) > 5.0:
            alarms.append(f"{name} window {lo}: local Q {row['Ql']:.2f}")
        if lo > 0 and math.isfinite(ratio) and not 0.4 <= ratio <= 5.0:
            alarms.append(f"{name} window {lo}: global/local gradient ratio {ratio:.2f}")
        if row["gc_loss"] > 2.0 or row["lc_loss"] > 0.5:
            alarms.append(f"{name} window {lo}: critic loss global {row['gc_loss']:.3f} local {row['lc_loss']:.3f}")
    for key, series in data.items():
        if any(not math.isfinite(v) for v in series.values()):
            alarms.append(f"{name}: non-finite values in {key}")
            break
    t = tests(run)
    level = levels(t)
    if t:
        extra = f" | noalloc at {scale}x ep: track SoC  level" if scale else ""
        print(f"{'test ep':>7} | {'track':>5} {'SoC':>5} {'sum':>6} | {'level':>6} | "
              f"{'noalloc lvl':>11} | {'AB lvl':>6}{extra}")
        for ep in sorted(t):
            nl = NOALLOC_LEVEL.get(ep, float("nan"))
            al = AB_LEVEL.get(ep, float("nan"))
            row = (f"{ep:7d} | {t[ep][0]:5.1f} {t[ep][1]:5.1f} {sum(t[ep]):6.1f} | "
                   f"{level.get(ep, float('nan')):6.1f} | {nl:11.1f} | {al:6.1f}")
            if scale:
                nt = NOALLOC_TESTS.get(ep * scale, (float("nan"), float("nan")))
                row += (f" | {ep * scale:5d} {nt[0]:5.1f} {nt[1]:5.1f} "
                        f"{NOALLOC_LEVEL.get(ep * scale, float('nan')):6.1f}")
            print(row)
    reach = first_reach(level)
    print(f"target (AB level at ep{max(AB_LEVEL)}) {TARGET:.1f}; reached at "
          f"{reach if reach is not None else 'not yet'}; noalloc reached at {first_reach(NOALLOC_LEVEL)}")
    err = os.path.join(HERE, f"{name}.stderr.log")
    if os.path.exists(err):
        with open(err, encoding="utf-8", errors="replace") as f:
            if "Traceback" in f.read():
                alarms.append(f"{name}: Traceback in stderr")
    tested = max(t) if t else 0
    return tested, alarms


def stalled(name: str) -> bool:
    log = os.path.join(HERE, f"{name}.stdout.log")
    return os.path.exists(log) and time.time() - os.path.getmtime(log) > 1800


def check_all(names: list[str], win: int, scale: int = 0) -> tuple[list[int], list[str]]:
    tested, alarms = [], []
    for name in names:
        ep, a = report(name, win, scale)
        tested.append(ep)
        alarms += a
        if stalled(name):
            alarms.append(f"{name}: stdout log has not grown for 30 minutes")
    free_gb = shutil.disk_usage("C:\\").free / 1e9
    if free_gb < 20:
        alarms.append(f"C: has {free_gb:.1f} GB free")
    for a in alarms:
        print("[alarm]", a)
    return tested, alarms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--until", type=int, default=None)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--scale", type=int, default=0)
    args = parser.parse_args()
    while args.until is not None:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            tested, alarms = check_all(args.names, args.window, args.scale)
        if alarms or min(tested) >= args.until:
            break
        time.sleep(300)
    _, alarms = check_all(args.names, args.window, args.scale)
    if alarms:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

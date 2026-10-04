"""Stop the socshape run once its tracking + SoC level stops improving.

usage: stop_on_stall.py [--from-episode 800] [--window 200] [--margin 1.0]

The level at a test is the mean of tracking + SoC (percent) over the last five
tests. From --from-episode on, after every new test: if the best level of the
last --window episodes is not at least --margin above the best level before
them, the run is asked to save its exact state and stop (pre_train.py
--request-stop), and this script exits. It also exits when the run's stdout log
has not grown for 30 minutes (finished or dead). Polls every 5 minutes.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\admin\Desktop\EVMALOCALUPDATE"
HERE = os.path.join(ROOT, "execute_results", "soc_shaping_20260926")
sys.path.insert(0, os.path.join(ROOT, "execute_results", "fast_pretrain_20260925"))
from evaluate import levels, tests  # noqa: E402

RUN_GLOB = os.path.join(ROOT, "archive", "prod_f500_dense8_AB_socshape_7station_*")
LOG = os.path.join(HERE, "pretrain.stdout.log")


def stalled_at(level: dict[int, float], ep: int, window: int, margin: float) -> tuple[bool, float, float]:
    before = [v for e, v in level.items() if e <= ep - window]
    recent = [v for e, v in level.items() if ep - window < e <= ep]
    if not before or not recent:
        return False, float("nan"), float("nan")
    return max(recent) < max(before) + margin, max(recent), max(before)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-episode", type=int, default=800)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()
    run = sorted(glob.glob(RUN_GLOB))[-1]
    checked = 0
    while True:
        level = levels(tests(run))
        latest = max(level, default=0)
        if latest >= args.from_episode and latest > checked:
            checked = latest
            stall, recent, before = stalled_at(level, latest, args.window, args.margin)
            print(f"ep{latest}: best of last {args.window} eps {recent:.1f}, best before {before:.1f}", flush=True)
            if stall:
                out = subprocess.run([sys.executable, os.path.join(ROOT, "pre_train.py"), "--request-stop", run],
                                     cwd=ROOT, capture_output=True, text=True)
                print(f"[stall] level did not rise {args.margin} in {args.window} episodes; stop requested at ep{latest}")
                print(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:])
                return
        if os.path.exists(LOG) and time.time() - os.path.getmtime(LOG) > 1800:
            print(f"stdout log idle for 30 minutes; last level ep{latest}")
            return
        time.sleep(300)


if __name__ == "__main__":
    main()

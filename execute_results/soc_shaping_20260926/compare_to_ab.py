"""Compare the AB + summed/surplus SoC shaping run with AB at the same episodes.

usage: compare_to_ab.py [--run socshape|soclax|force|newbank|perev|perevpot|rule|perevpotrule|rulefloor] [--base ab|newbank|perev|perevpot|rule] [--until EPISODE] [--window 100]

Prints, per window, both runs' global and local Q, the global/local actor
gradient ratio, the local critic's clip rate and losses, then the tests:
tracking, SoC, their sum and the mean of the sum over the last five tests.
The shaping change is expected to raise the local Q and gradient, so those are
reported, not alarmed. The exit code is 2 on:
  a non-finite scalar
  global Q above 0.8 x the mixer bias cap (50)
  |local Q| above 50, global critic loss above 2, local critic loss above 5
  a Traceback in the run's stderr
  the run's stdout log not growing for 30 minutes (with --until)
With --until the script polls every 5 minutes until the run has a test at that
episode or an alarm fires.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import os
import sys
import time

import numpy as np

ROOT = r"C:\Users\admin\Desktop\EVMALOCALUPDATE"
HERE = os.path.join(ROOT, "execute_results", "soc_shaping_20260926")
sys.path.insert(0, os.path.join(ROOT, "execute_results", "fast_pretrain_20260925"))
from evaluate import levels, scalars, tests, window  # noqa: E402

AB = os.path.join(ROOT, "archive", "prod_f500_dense8_AB_7station_20260922_022453")
RUN_GLOBS = {
    "socshape": os.path.join(ROOT, "archive", "prod_f500_dense8_AB_socshape_7station_*"),
    "soclax": os.path.join(ROOT, "archive", "prod_f500_dense8_AB_soclax_7station_*"),
    "force": os.path.join(ROOT, "archive", "prod_f500_dense8_AB_force_7station_*"),
    # AB's settings on the AEMO plan-deviation bank: a different bank and
    # command library, so AB's numbers are a reference, not a like-for-like pair.
    "newbank": os.path.join(ROOT, "archive", "prod_aemoplan_AB_7station_*"),
    # The newbank run with per-EV local critics; compare with --base newbank.
    "perev": os.path.join(ROOT, "archive", "prod_aemoplan_AB_perev_7station_*"),
    # perev with EVMA_LOCAL_REWARD_MODE=potential; compare with --base newbank or perev.
    "perevpot": os.path.join(ROOT, "archive", "prod_aemoplan_AB_perev_pot_7station_*"),
    # The station rule split (rule_alloc_20260929) on base ab / potential.
    "rule": os.path.join(ROOT, "archive", "prod_aemoplan_AB_rule_7station_*"),
    "perevpotrule": os.path.join(ROOT, "archive", "prod_aemoplan_AB_perev_pot_rule_7station_*"),
    # The rule with the SoC floor and no local critic (rule_alloc_20260929 -SocFloor).
    "rulefloor": os.path.join(ROOT, "archive", "prod_aemoplan_AB_rule_floor_7station_*"),
}
BASES = {
    "ab": AB,
    "newbank": os.path.join(ROOT, "archive", "prod_aemoplan_AB_7station_20260926_210314"),
    "perev": os.path.join(ROOT, "archive", "prod_aemoplan_AB_perev_7station_20260928_123439"),
    "perevpot": os.path.join(ROOT, "archive", "prod_aemoplan_AB_perev_pot_7station_20260928_192612"),
    "rule": os.path.join(ROOT, "archive", "prod_aemoplan_AB_rule_7station_20260929_011453"),
}
LOGS = {
    "socshape": os.path.join(HERE, "pretrain"),
    "soclax": os.path.join(HERE, "pretrain_lax"),
    "force": os.path.join(HERE, "pretrain_force"),
    "newbank": os.path.join(ROOT, "execute_results", "newbank_AB_20260926", "pretrain"),
    "perev": os.path.join(ROOT, "execute_results", "perev_20260927", "pretrain"),
    "perevpot": os.path.join(ROOT, "execute_results", "perev_potential_20260928", "pretrain"),
    "rule": os.path.join(ROOT, "execute_results", "rule_alloc_20260929", "pretrain_ab"),
    "perevpotrule": os.path.join(ROOT, "execute_results", "rule_alloc_20260929", "pretrain_potential"),
    "rulefloor": os.path.join(ROOT, "execute_results", "rule_alloc_20260929", "pretrain_ab_floor"),
}
TAGS = {
    "Qg": "Q/global", "Ql": "Q/local_mean",
    "gG": "Gradient/actor_source_global_raw_mean", "gL": "Gradient/actor_source_local_raw_mean",
    "lc_loss": "Loss/local_critic_mean", "gc_loss": "Loss/global_critic",
    "r_loc": "Reward/local", "r_glob": "Reward/global",
}


def lc_clip(d: dict) -> dict[int, float]:
    per = [d[t] for t in d if t.startswith("Clipping/local_critic_agent")]
    steps = sorted(set().union(*[set(x) for x in per])) if per else []
    return {s: float(np.mean([x[s] for x in per if s in x])) for s in steps}


def report(win: int, name: str = "socshape", base: str = AB) -> tuple[int, list[str]]:
    runs = sorted(glob.glob(RUN_GLOBS[name]))
    if not runs:
        return 0, []
    run = runs[-1]
    new, ab = scalars(run), scalars(base)
    new["lc_clip"], ab["lc_clip"] = lc_clip(new), lc_clip(ab)
    last = max((max(d) for d in new.values() if d), default=0)
    alarms: list[str] = []
    print(f"run {os.path.basename(run)}  last logged episode {last}")
    print(f"{'window':>9} | {'Q_glob AB->new':>14} | {'Q_loc AB->new':>14} | {'gG/gL AB->new':>13} | "
          f"{'lc_clip':>11} | {'lc_loss':>13} | {'r_local':>13}")
    for lo in range(0, last + 1, win):
        hi = lo + win
        a = {k: window(ab.get(t, {}), lo, hi) for k, t in TAGS.items()}
        n = {k: window(new.get(t, {}), lo, hi) for k, t in TAGS.items()}
        a["lc_clip"], n["lc_clip"] = window(ab["lc_clip"], lo, hi), window(new["lc_clip"], lo, hi)
        if all(np.isnan(v) for v in n.values()):
            continue
        ra = a["gG"] / a["gL"] if a["gL"] else float("nan")
        rn = n["gG"] / n["gL"] if n["gL"] else float("nan")
        print(f"{lo:4d}-{hi - 1:<4d} | {a['Qg']:5.2f} -> {n['Qg']:5.2f} | {a['Ql']:5.2f} -> {n['Ql']:6.2f} | "
              f"{ra:4.2f} -> {rn:5.2f} | {a['lc_clip']:4.2f}->{n['lc_clip']:4.2f} | "
              f"{a['lc_loss']:5.3f}->{n['lc_loss']:6.3f} | {a['r_loc']:+.3f}->{n['r_loc']:+.3f}")
        if any(np.isinf(v) for v in n.values()):
            alarms.append(f"window {lo}: non-finite scalar")
        if n["Qg"] > 40:
            alarms.append(f"window {lo}: global Q {n['Qg']:.1f} near the mixer bias cap")
        if abs(n["Ql"]) > 50:
            alarms.append(f"window {lo}: local Q {n['Ql']:.1f}")
        if n["gc_loss"] > 2.0 or n["lc_loss"] > 5.0:
            alarms.append(f"window {lo}: critic loss global {n['gc_loss']:.3f} local {n['lc_loss']:.3f}")
    for key, series in new.items():
        if any(not np.isfinite(v) for v in series.values()):
            alarms.append(f"non-finite values in {key}")
            break
    tn, ta = tests(run), tests(base)
    ln, la = levels(tn), levels(ta)
    if tn:
        print(f"{'test ep':>7} | {'track AB->new':>13} | {'SoC AB->new':>13} | {'sum AB->new':>15} | {'5-test mean AB->new':>19}")
        for ep in sorted(tn):
            a = ta.get(ep, (float("nan"), float("nan")))
            print(f"{ep:7d} | {a[0]:5.1f} -> {tn[ep][0]:5.1f} | {a[1]:5.1f} -> {tn[ep][1]:5.1f} | "
                  f"{sum(a):6.1f} -> {sum(tn[ep]):6.1f} | {la.get(ep, float('nan')):8.1f} -> {ln.get(ep, float('nan')):6.1f}")
    err = f"{LOGS[name]}.stderr.log"
    if os.path.exists(err) and "Traceback" in open(err, encoding="utf-8", errors="replace").read():
        alarms.append("Traceback in stderr")
    for a in alarms:
        print("[alarm]", a)
    return (max(tn) if tn else 0), alarms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", type=int, default=None)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--run", choices=sorted(RUN_GLOBS), default="socshape")
    parser.add_argument("--base", choices=sorted(BASES), default="ab", help="the run shown on the left of each ->")
    args = parser.parse_args()
    log = f"{LOGS[args.run]}.stdout.log"
    while args.until is not None:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            tested, alarms = report(args.window, args.run, BASES[args.base])
        stalled = os.path.exists(log) and time.time() - os.path.getmtime(log) > 1800
        if alarms or stalled or tested >= args.until:
            if stalled:
                print("[alarm] stdout log has not grown for 30 minutes")
            break
        time.sleep(300)
    _, alarms = report(args.window, args.run, BASES[args.base])
    if alarms:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

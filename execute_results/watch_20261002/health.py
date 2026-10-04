"""学習中のMARL実行の勾配・Q・損失と途中テストを100エピソードごとにまとめる。

usage: health.py RUN_DIR [--window 100] [--last-windows 6]

TensorBoard の記録（runs/、performance/）と results/test_history.json を読むだけで、
実行には触れない。Windows と研究室PC（コンテナ内）の両方で動く。

列の意味
  Qg        大域criticのQ（Q/global）
  gfrac     行動器の勾配に占める大域側の割合（GradHealth/actor_source_global_ratio）
  gG/gL     大域・局所の生の勾配ノルムの比
  cos       大域・局所の勾配の向きの一致度
  a_clip    行動器の勾配がクリップされた割合
  gc_clip   大域criticの勾配がクリップされた割合
  gc_raw    大域criticのクリップ前の勾配ノルム（上限5）
  gc_loss / lc_loss / td  大域・局所criticの損失、大域のTD誤差
警告（[alarm]）は数値の発散の兆しだけに出す。
  非有限値、|Qg| > 40（mixerのバイアス上限50の0.8倍）、gc_loss > 2、lc_loss > 5、
  直近の窓の gc_raw がその前の窓の3倍超
収束の目安として、直近400エピソードのテスト和の傾き（100エピソードあたり）を出す。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "Qg": "Q/global",
    "gfrac": "GradHealth/actor_source_global_ratio",
    "gG": "GradHealth/actor_source_global_raw",
    "gL": "GradHealth/actor_source_local_raw",
    "cos": "GradHealth/actor_source_local_global_cos",
    "gc_clip": "Clipping/global_critic",
    "gc_raw": "GradHealth/global_critic_raw",
    "gc_loss": "GradHealth/global_critic_loss",
    "lc_loss": "GradHealth/local_critic_loss",
    "td": "GlobalCritic/td_error_abs_mean",
}


def scalars(run: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for sub in ("runs", "performance"):
        for path in glob.glob(os.path.join(run, sub, "**", "events.out.tfevents.*"), recursive=True):
            ea = EventAccumulator(path, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                for e in ea.Scalars(tag):
                    out.setdefault(tag, {})[int(e.step)] = float(e.value)
    per = [out[t] for t in out if t.startswith("Clipping/actor_agent")]
    steps = sorted(set().union(*[set(d) for d in per])) if per else []
    out["a_clip"] = {s: float(np.mean([d[s] for d in per if s in d])) for s in steps}
    return out


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


def window(d: dict[int, float], lo: int, hi: int) -> float:
    vals = [v for k, v in d.items() if lo <= k < hi]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--last-windows", type=int, default=6)
    args = parser.parse_args()
    data = scalars(args.run)
    last = max((max(d) for d in data.values() if d), default=-1)
    print(f"run {os.path.basename(os.path.normpath(args.run))}  last logged episode {last}")
    alarms: list[str] = []
    rows = []
    for lo in range(0, last + 1, args.window):
        hi = lo + args.window
        row = {k: window(data.get(t, {}), lo, hi) for k, t in TAGS.items()}
        row["a_clip"] = window(data.get("a_clip", {}), lo, hi)
        row["ratio"] = row["gG"] / row["gL"] if row["gL"] else float("nan")
        rows.append((lo, hi, row))
    print(f"{'window':>9} | {'Qg':>6} | {'gfrac':>5} {'gG/gL':>5} {'cos':>6} | {'a_clip':>6} {'gc_clip':>7} {'gc_raw':>6} | "
          f"{'gc_loss':>7} {'lc_loss':>7} {'td':>6}")
    for lo, hi, r in rows[-args.last_windows:]:
        print(f"{lo:4d}-{hi - 1:<4d} | {r['Qg']:6.2f} | {r['gfrac']:5.3f} {r['ratio']:5.2f} {r['cos']:6.3f} | "
              f"{r['a_clip']:6.3f} {r['gc_clip']:7.3f} {r['gc_raw']:6.2f} | {r['gc_loss']:7.3f} {r['lc_loss']:7.3f} {r['td']:6.3f}")
    for key, series in data.items():
        if any(not math.isfinite(v) for v in series.values()):
            alarms.append(f"non-finite values in {key}")
    for lo, hi, r in rows[-2:]:
        if abs(r["Qg"]) > 40:
            alarms.append(f"window {lo}: |Qg| {r['Qg']:.1f}")
        if r["gc_loss"] > 2.0 or r["lc_loss"] > 5.0:
            alarms.append(f"window {lo}: critic loss global {r['gc_loss']:.3f} local {r['lc_loss']:.3f}")
    if len(rows) >= 2 and rows[-2][2]["gc_raw"] > 0 and rows[-1][2]["gc_raw"] > 3 * rows[-2][2]["gc_raw"]:
        alarms.append(f"global critic raw gradient jumped {rows[-2][2]['gc_raw']:.2f} -> {rows[-1][2]['gc_raw']:.2f}")
    t = tests(args.run)
    eps = sorted(t)
    if eps:
        print(f"{'test ep':>7} | {'track':>5} {'SoC':>5} {'sum':>6} | {'5-test mean':>11}")
        for i, ep in enumerate(eps):
            if i < len(eps) - 6:
                continue
            five = eps[max(0, i - 4):i + 1]
            mean5 = float(np.mean([sum(t[e]) for e in five])) if len(five) == 5 else float("nan")
            print(f"{ep:7d} | {t[ep][0]:5.1f} {t[ep][1]:5.1f} {sum(t[ep]):6.1f} | {mean5:11.1f}")
        sel = [e for e in eps if e > eps[-1] - 400]
        if len(sel) >= 5:
            slope = float(np.polyfit(sel, [sum(t[e]) for e in sel], 1)[0]) * 100.0
            print(f"slope of the test sum over the last 400 episodes: {slope:+.2f} per 100 episodes "
                  f"({len(sel)} tests, sd {float(np.std([sum(t[e]) for e in sel])):.1f})")
    for a in alarms:
        print("[alarm]", a)


if __name__ == "__main__":
    main()

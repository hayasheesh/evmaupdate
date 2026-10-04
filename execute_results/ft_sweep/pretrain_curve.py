"""pretrain の成績を「生のMARL出力」で並べる。

  usage: python pretrain_curve.py <run_dir> [<run_dir> ...]

成績は results/test_performance_metrics.csv の Dispatch_Tracking_Rate_%
（補正前の学習器そのものの追従率）と Raw_Actor_MAE_kW から読む。
CentralEV/ と System/ は補正後なので制御器の比較には使わない。

診断量は TensorBoard から読む:
  target_q  … Q のスケール。gamma を上げれば r/(1-gamma) で上がるはず
  actor_clip… クリップ頻度。上がりすぎると勾配の大きさの情報が落ちる
  cos/ratio … local と global の勾配の和解。この構成で一番壊れやすい量
"""
from __future__ import annotations
import sys, os, csv, glob, json
from collections import defaultdict
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB_TAGS = {
    "target_q":   "GlobalCritic/target_q_abs_mean",
    "td_err":     "GlobalCritic/td_error_abs_mean",
    "actor_clip": "GradHealth/actor_clip_count",
    "cos":        "GradHealth/actor_source_local_global_cos",
    "ratio":      "GradHealth/actor_source_global_ratio",
}
CSV_COLS = {
    "raw_track": "Dispatch_Tracking_Rate_%",
    "raw_mae":   "Raw_Actor_MAE_kW",
    "soc_hit":   "SoC_Hit_Rate_%",
    "deficit":   "Avg_SoC_Deficit_kWh",
}
ORDER = ["raw_track", "raw_mae", "soc_hit", "deficit",
         "target_q", "td_err", "actor_clip", "cos", "ratio"]
SEP = "/" + chr(92)


def basename(path):
    return os.path.basename(path.rstrip(SEP))


def test_interval(run_dir):
    """results/TEST<ep>/ の並びから中間テストの間隔を読む。

    test_performance_metrics.csv の Episode 列はテスト番号であって学習
    エピソードではない。この間隔を掛けて学習エピソード軸に直す。
    """
    eps = []
    for d in glob.glob(os.path.join(run_dir, "results", "TEST*")):
        tail = os.path.basename(d)[4:]
        if tail.isdigit():
            eps.append(int(tail))
    if len(eps) < 2:
        return 1
    eps.sort()
    return max(1, eps[0])


def load_csv(run_dir):
    path = os.path.join(run_dir, "results", "test_performance_metrics.csv")
    if not os.path.exists(path):
        return {}
    interval = test_interval(run_dir)
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ep = int(float(row["Episode"])) * interval
            except (KeyError, ValueError, TypeError):
                continue
            for key, col in CSV_COLS.items():
                v = row.get(col, "")
                if v not in ("", None):
                    try:
                        out[key][ep] = float(v)
                    except ValueError:
                        pass
    return out


def load_tb(run_dir):
    out = defaultdict(dict)
    for ev in glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"), recursive=True):
        try:
            ea = EventAccumulator(ev, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        avail = set(ea.Tags().get("scalars", []))
        for key, tag in TB_TAGS.items():
            if tag in avail:
                for s in ea.Scalars(tag):
                    out[key][int(s.step)] = float(s.value)
    return out


def near(series, ep, half=25):
    vals = [v for k, v in series.items() if abs(k - ep) <= half]
    return float(np.mean(vals)) if vals else None


def report(run_dir):
    d = load_csv(run_dir)
    d.update(load_tb(run_dir))
    if not d:
        print("  (データなし) " + run_dir)
        return None
    eps = sorted({k for s in d.values() for k in s})
    if not eps:
        return None
    last = eps[-1]
    # 中間テストが打たれたエピソードを軸にする。なければ等間隔。
    marks = sorted(d.get("raw_track", {}))
    if not marks:
        marks = [e for e in eps if e % 200 == 0]
    if last not in marks:
        marks.append(last)
    if len(marks) > 12:
        step = len(marks) // 11
        marks = marks[::step] + ([last] if last not in marks[::step] else [])
    print()
    print("### " + basename(run_dir) + "   最終ep=" + str(last))
    print(f"{'ep':>6s}" + "".join(f"{k:>11s}" for k in ORDER))
    rows = {}
    for ep in marks:
        row = {k: near(d.get(k, {}), ep) for k in ORDER}
        rows[ep] = row
        cells = "".join(
            (f"{row[k]:11.3f}" if row[k] is not None else f"{'-':>11s}") for k in ORDER)
        print(f"{ep:6d}" + cells)
    return rows


if __name__ == "__main__":
    runs = sys.argv[1:] or [
        "archive/direct_bid_256cmd_25d_evcount_12hbid_7station_20260916_002104"]
    all_rows = {}
    for r in runs:
        rows = report(r)
        if rows:
            all_rows[basename(r)] = rows
    if len(runs) == 1 and all_rows:
        ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_curve.json")
        with open(ref, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=1)
        print()
        print("保存: " + ref)

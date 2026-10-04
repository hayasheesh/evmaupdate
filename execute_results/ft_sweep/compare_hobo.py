"""hobo100100 と現行runを、同じ名前のスカラーで並べる。

Q のスケール、勾配の大きさとクリップ頻度、local と global の和解（cos）、
報酬を、同じ学習エピソードで突き合わせる。
成績は results/test_performance_metrics.csv の生の追従率から読む。
"""
from __future__ import annotations
import sys, os, csv, glob
from collections import defaultdict
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = {
    "Q_global":    "Q/global",
    "Q_local":     "Q/local_mean",
    "R_global":    "Reward/global",
    "R_local":     "Reward/local",
    "g_local":     "Gradient/actor_source_local_raw_mean",
    "g_global":    "Gradient/actor_source_global_raw_mean",
    "ratio":       "Gradient/actor_source_global_ratio_mean",
    "cos":         "Gradient/actor_source_cos_mean",
    "gcrit_raw":   "Gradient/global_critic_raw",
    "gcrit_clip":  "Clipping/global_critic",
    "loss_gcrit":  "Loss/global_critic",
}
MARKS = (100, 200, 300, 400, 500, 600)
SEP = "/" + chr(92)


def load_tb(run):
    out = defaultdict(dict)
    actor_clip = defaultdict(list)
    for ev in glob.glob(os.path.join(run, "**", "events.out.tfevents*"), recursive=True):
        try:
            ea = EventAccumulator(ev, size_guidance={"scalars": 0}); ea.Reload()
        except Exception:
            continue
        avail = set(ea.Tags().get("scalars", []))
        for k, tag in TB.items():
            if tag in avail:
                for s in ea.Scalars(tag):
                    out[k][int(s.step)] = float(s.value)
        # actor のクリップは agent 別にしかないので、台数で平均する
        for tag in avail:
            if tag.startswith("Clipping/actor_agent"):
                for s in ea.Scalars(tag):
                    actor_clip[int(s.step)].append(float(s.value))
    out["actor_clip"] = {k: float(np.mean(v)) for k, v in actor_clip.items()}
    return out


def test_interval(run):
    eps = [int(os.path.basename(d)[4:]) for d in glob.glob(os.path.join(run, "results", "TEST*"))
           if os.path.basename(d)[4:].isdigit()]
    return max(1, min(eps)) if len(eps) >= 2 else 1


def load_csv(run):
    path = os.path.join(run, "results", "test_performance_metrics.csv")
    if not os.path.exists(path):
        return {}
    iv = test_interval(run)
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ep = int(float(r["Episode"])) * iv
            except (KeyError, ValueError, TypeError):
                continue
            for k, col in (("raw_track", "Dispatch_Tracking_Rate_%"),
                           ("soc_hit", "SoC_Hit_Rate_%")):
                v = r.get(col, "")
                if v not in ("", None):
                    try:
                        out[k][ep] = float(v)
                    except ValueError:
                        pass
    return out


def near(series, ep, half=30):
    v = [x for k, x in series.items() if abs(k - ep) <= half]
    return float(np.mean(v)) if v else None


ORDER = ["raw_track", "soc_hit", "Q_global", "Q_local", "R_global", "R_local",
         "g_local", "g_global", "ratio", "cos", "actor_clip", "gcrit_clip"]

runs = sys.argv[1:]
data = {}
for r in runs:
    d = load_tb(r); d.update(load_csv(r))
    data[os.path.basename(r.rstrip(SEP))] = d

for ep in MARKS:
    print(f"\n--- ep{ep} ---")
    print(f"{'':28s}" + "".join(f"{k:>11s}" for k in ORDER))
    for name, d in data.items():
        cells = ""
        for k in ORDER:
            v = near(d.get(k, {}), ep)
            cells += (f"{v:11.3f}" if v is not None else f"{'-':>11s}")
        print(f"{name[:28]:28s}" + cells)

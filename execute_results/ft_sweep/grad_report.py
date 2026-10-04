"""勾配の健全性を hobo100100 と同一エピソードで比べる。
   usage: grad_report.py <run_dir> [--check]
   --check: 崩壊なら終了コード2。
     崩壊 = 勾配の hobo 比が 3.0 倍超、またはクリップ 0.02 超が2点連続。
   hobo は ep200 以降 1.1 前後で平坦、全1821epの最大 3.255、クリップは常に 0。"""
import glob, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load(root):
    acc = {}
    for p in glob.glob(root + "/**/events.out.tfevents.*", recursive=True):
        ea = EventAccumulator(p, size_guidance={"scalars": 0}); ea.Reload()
        for t in ea.Tags()["scalars"]:
            for e in ea.Scalars(t):
                acc.setdefault(t, {})[e.step] = e.value
    return acc

def at(d, ep, tol=12):
    ks = [k for k in d if abs(k - ep) <= tol]
    return float(np.mean([d[k] for k in ks])) if ks else None

run = sys.argv[1]
cur, hob = load(run), load("docs/hobo100100")
steps = sorted({s for d in cur.values() for s in d})
if not steps:
    print("スカラーなし"); raise SystemExit(0)
top = max(steps)
marks = [m for m in range(50, top + 1, 50)][-8:]

print(f"run = {run}\n到達 ep = {top}\n")
print(f"{'ep':>6s} {'勾配':>8s} {'hobo':>8s} {'比':>6s} {'クリップ':>8s} {'Q':>8s} {'報酬':>8s}")
ratios, clips = [], []
for m in marks:
    g = at(cur.get("Gradient/global_critic_raw", {}), m)
    h = at(hob.get("Gradient/global_critic_raw", {}), m)
    c = at(cur.get("Clipping/global_critic", {}), m)
    q = at(cur.get("Q/global", {}), m)
    r = at(cur.get("Reward/global", {}), m)
    ratio = (g / h) if (g is not None and h) else None
    if ratio is not None: ratios.append((m, ratio))
    if c is not None: clips.append(c)
    f = lambda v, w=8, p=4: (f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-")
    print(f"{m:6d} {f(g)} {f(h)} {(f'{ratio:6.2f}' if ratio is not None else '     -')} {f(c)} {f(q)} {f(r)}")

print()
if ratios:
    m, r = ratios[-1]
    print(f"直近の hobo 比: {r:.2f} 倍（ep{m}）")
    if r > 2.0: print("  警戒: 2.0 倍を超えています")
    else: print("  hobo の範囲内")

if "--check" in sys.argv:
    reasons = []
    if ratios and ratios[-1][1] > 3.0:
        reasons.append(f"勾配の hobo 比が {ratios[-1][1]:.2f} 倍（閾値 3.0）")
    if len(clips) >= 2 and clips[-1] > 0.02 and clips[-2] > 0.02:
        reasons.append(f"クリップが2点連続で 0.02 超（{clips[-2]:.4f} -> {clips[-1]:.4f}）")
    if reasons:
        print("")
        print("!!! 崩壊と判定 !!!")
        for x in reasons: print("  -", x)
        raise SystemExit(2)

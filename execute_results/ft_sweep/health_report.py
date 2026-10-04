"""学習が壊れたかを、学習そのものの目的で判定する。
   usage: health_report.py <log> [run_dir] [--check]
   --check: 崩壊なら終了コード2。

   勾配の大きさでは判定しない。center=1.0 の run は ep120→ep640 で
   テスト MAE 119.9→46.3 kW、追従 68.8→88.6% と単調に良くなりながら、
   勾配は hobo 比 1.4→3.3 倍まで伸びた。批評家の勾配が伸びること自体は
   方策が良くなっている最中にも起きる。逆に center=0.0 の run は勾配が
   ほぼ同じ伸び方をしたが、追従は ep240 で 1.26% に凍りついたままだった。
   見るべきはテスト時の追従であって勾配ではない。

   崩壊の定義:
     全滅 = ep200 以降で追従率 30% 未満（健全な run は ep200 で 71.9%）
     悪化 = 最良の追従率から 10pt 以上落ちた点が2回連続
     停滞 = 直近10点で追従率が 1pt も伸びず、かつ 80% 未満
   勾配とクリップは参考として表示するだけで、停止の根拠にしない。"""
import re, sys

log = sys.argv[1]
run = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None

PAT = re.compile(
    r"^\[TEST(\d+) summary\].*?MARL/Central=([0-9.]+)/([0-9.]+)%"
    r" \| MAE=([0-9.]+)->([0-9.]+)kW"
)
pts = []
with open(log, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = PAT.match(line)
        if m:
            pts.append((int(m.group(1)), float(m.group(2)), float(m.group(4))))

if not pts:
    print("テスト集計なし"); raise SystemExit(0)

print(f"log = {log}\nテスト点 = {len(pts)} 個、最終 ep = {pts[-1][0]}\n")
print(f"{'ep':>6s} {'追従%':>8s} {'MAE kW':>9s} {'最良比':>8s}")
best = 0.0
for ep, trk, mae in pts[-10:]:
    pass
run_best = 0.0
marks = []
for ep, trk, mae in pts:
    run_best = max(run_best, trk)
    marks.append((ep, trk, mae, trk - run_best))
for ep, trk, mae, gap in marks[-10:]:
    print(f"{ep:6d} {trk:8.2f} {mae:9.1f} {gap:+8.2f}")

trks = [t for _, t, _ in pts]
best = max(trks)
print(f"\n最良の追従率 {best:.2f}%（ep{pts[trks.index(best)][0]}）、直近 {trks[-1]:.2f}%")

reasons = []
eps = [e for e, _, _ in pts]
if eps[-1] >= 200 and trks[-1] < 30.0:
    reasons.append(f"ep{eps[-1]} で追従率 {trks[-1]:.2f}%（下限 30%）")
if len(trks) >= 2:
    # その時点までの最良と比べる。後から来た改善で過去を悪化扱いしない。
    drops, bb = [], 0.0
    for t in trks:
        bb = max(bb, t)
        drops.append(bb - t)
    if drops[-1] > 10.0 and drops[-2] > 10.0:
        reasons.append(
            f"追従率が最良から2点連続で 10pt 超の低下（{drops[-2]:.1f}pt -> {drops[-1]:.1f}pt）"
        )
if len(trks) >= 10 and trks[-1] < 80.0:
    window = trks[-10:]
    if max(window) - window[0] < 1.0:
        reasons.append(
            f"直近10点で追従率が {window[0]:.2f}% から {max(window) - window[0]:+.2f}pt しか伸びず、"
            "80% に届いていない"
        )

if run:
    print("\n--- 参考（停止の根拠にはしない） ---")
    try:
        import glob
        import numpy as np
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = {}
        for p in glob.glob(run + "/**/events.out.tfevents.*", recursive=True):
            ea = EventAccumulator(p, size_guidance={"scalars": 0}); ea.Reload()
            for t in ea.Tags()["scalars"]:
                for e in ea.Scalars(t):
                    acc.setdefault(t, {})[e.step] = e.value
        def at(tag, ep, tol=12):
            d = acc.get(tag, {})
            v = [d[k] for k in d if abs(k - ep) <= tol]
            return float(np.mean(v)) if v else float("nan")
        top = max((s for d in acc.values() for s in d), default=0)
        print(f"{'ep':>6s} {'勾配':>8s} {'クリップ':>9s} {'批評家損失':>10s} {'Q':>9s}")
        for m in [x for x in range(100, top + 1, 100)][-6:]:
            print(f"{m:6d} {at('Gradient/global_critic_raw', m):8.4f}"
                  f" {at('Clipping/global_critic', m):9.4f}"
                  f" {at('Loss/global_critic', m):10.4f} {at('Q/global', m):9.4f}")
    except Exception as exc:
        print("  取得できず:", exc)

if "--check" in sys.argv:
    if reasons:
        print("\n!!! 崩壊と判定 !!!")
        for x in reasons: print("  -", x)
        raise SystemExit(2)

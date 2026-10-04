"""本番 run の進捗を hobo100100 と並べて出す。

usage: python progress_report.py <run_dir> [ep]
  ep を省略すると到達している最後の 100 の節目まで出す。
"""
from __future__ import annotations

import glob
import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HOBO = "docs/hobo100100"
WINDOW = 25   # 各 ep 点の前後をならす幅


def load(pattern: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for path in sorted(glob.glob(pattern)):
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for event in ea.Scalars(tag):
                out.setdefault(tag, {})[event.step] = event.value
    return out


def at(series: dict[str, dict[int, float]], tag: str, ep: int) -> float | None:
    values = [v for s, v in series.get(tag, {}).items() if ep - WINDOW < s <= ep]
    return statistics.mean(values) if values else None


def fmt(value: float | None, width: int = 9, digits: int = 4) -> str:
    return f"{value:>{width}.{digits}f}" if value is not None else f"{'-':>{width}}"


def ratio(run: float | None, hobo: float | None) -> str:
    if run is None or hobo is None or hobo == 0:
        return f"{'-':>7}"
    return f"{run / hobo:>7.2f}"


def table(title: str, rows, H, R, ep: int) -> None:
    print(f"### {title}")
    print(f"{'項目':<22} {'hobo':>9} {'本番':>9} {'比':>7}")
    for tag, label in rows:
        h, r = at(H, tag, ep), at(R, tag, ep)
        print(f"{label:<22} {fmt(h)} {fmt(r)} {ratio(r, h)}")
    print()


def main() -> int:
    run_dir = Path(sys.argv[1])
    H = load(f"{HOBO}/performance/*")
    R = load(f"{run_dir}/performance/*")
    steps = sorted(R.get("Q/global", {}))
    if not steps:
        print("まだスカラーがない")
        return 1
    reached = steps[-1]
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else (reached // 100) * 100
    print(f"run = {run_dir}")
    print(f"到達 ep = {reached}   報告 ep = {ep}   （各値は ep-{WINDOW} 〜 ep の平均）")
    print()
    table("生の性能 local 側", [
        ("Reward/local", "報酬 local"),
        ("Metrics/soc_hit_rate", "SoC 達成率 %"),
        ("Q/local_mean", "Q local"),
    ], H, R, ep)
    table("生の性能 global 側", [
        ("Reward/global", "報酬 global"),
        ("Metrics/surplus_absorption_rate", "余剰吸収率 %"),
        ("Metrics/supply_cooperation_rate", "供給協力率 %"),
        ("Q/global", "Q global"),
    ], H, R, ep)
    table("勾配 global critic", [
        ("Gradient/global_critic_raw", "生ノルム"),
        ("Clipping/global_critic", "クリップ率"),
        ("Loss/global_critic", "loss"),
    ], H, R, ep)
    table("勾配 local critic / actor", [
        ("Gradient/local_critic_raw_agent1", "lcrit 生ノルム"),
        ("Clipping/local_critic_agent1", "lcrit クリップ率"),
        ("Loss/local_critic_mean", "lcrit loss"),
        ("Gradient/actor_raw_agent1", "actor 生ノルム"),
        ("Clipping/actor_agent1", "actor クリップ率"),
    ], H, R, ep)

    print("### 正常化の判定")
    clip = at(R, "Clipping/global_critic", ep)
    q = at(R, "Q/global", ep)
    grad = at(R, "Gradient/global_critic_raw", ep)
    loss_now = at(R, "Loss/global_critic", ep)
    loss_first = at(R, "Loss/global_critic", WINDOW)
    print(f"  1. global critic クリップ率 = {fmt(clip, 0, 4)} "
          f"（hobo は全 1821 ep で 0.0000）")
    if loss_first and loss_now:
        print(f"  2. global critic loss {loss_first:.4f} → {loss_now:.4f} "
              f"（{loss_first / loss_now:.2f} 倍に低下）")
    if q and grad:
        print(f"  3. Q global = {q:.3f}（上限 20）  勾配/Q = {grad / q:.3f}  "
              f"Q が上限に達したときの勾配見込み = {grad / q * 20:.1f}（閾値 20）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

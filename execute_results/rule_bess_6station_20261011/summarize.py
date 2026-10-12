"""run_rule_bess.py の結果を、市場と蓄電池の大きさごとにまとめる。1エピソードが1日なので、不適合のコマ数はそのまま1日の不適合の回数になる。"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
print('| 市場 | 蓄電池 | エピソード | 5分ごとの合格率 前→後 | 30分コマの合格率 前→後 | 1日の不適合 平均/最大（後） | 不適合0の日（後） | SoC 平均/最低 | 蓄電池の電力量÷調整の電力量 | 出力上限に当たった時刻 |')
print('|---|---|---|---|---|---|---|---|---|---|')
for task in ('aemo', 'ercot'):
    p = HERE / f'result_{task}_rule_bess_4cmd_2ev.jsonl'
    if not p.exists():
        continue
    g = defaultdict(list)
    for line in p.open(encoding='utf-8'):
        r = json.loads(line)
        g[r['bess_kw']].append(r)
    for kw in sorted(g):
        rs = g[kw]
        n = len(rs)
        mean = lambda k: sum(r[k] for r in rs) / n
        fails = [r['post_blocks_failed'] for r in rs]
        share = sum(r['bess_abs_energy_tracking_kwh'] for r in rs) / max(sum(r['regulation_energy_kwh'] for r in rs), 1e-9)
        print(f"| {task.upper()} | {kw:.0f} kW / {rs[0]['bess_kwh']:.0f} kWh | {n} | "
              f"{mean('pre_step_pass_pct'):.1f}% → {mean('post_step_pass_pct'):.1f}% | "
              f"{mean('pre_block_pass_pct'):.1f}% → {mean('post_block_pass_pct'):.1f}% | "
              f"{sum(fails) / n:.2f} / {max(fails)} | {sum(f == 0 for f in fails)}/{n} | "
              f"{mean('soc_hit_pct'):.1f}% / {min(r['soc_hit_pct'] for r in rs):.1f}% | {100 * share:.1f}% | "
              f"{sum(r['bess_power_limit_steps'] for r in rs)} |")

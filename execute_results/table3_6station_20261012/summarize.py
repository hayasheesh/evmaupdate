"""表3の結果（result_<task>_<mode>_4cmd_2ev.jsonl）を市場と方法ごとにまとめる。  python summarize.py [結果のフォルダ]"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
print('| 市場 | 方法 | 組 | 成り立つ／1日すべて帯の中 | 30分コマの合格率 | 1日の不適合 平均/最大 | 不適合0の日 | 5分ごとの合格率 | SoC達成率 | 平均誤差率 |')
print('|---|---|---|---|---|---|---|---|---|---|')
for t in ('aemo', 'ercot'):
    for m in ('lp', 'rule', 'marl_force', 'marl'):
        p = d / f'result_{t}_{m}_4cmd_2ev.jsonl'
        if not p.exists():
            continue
        rs = [json.loads(l) for l in p.open(encoding='utf-8')]
        n = len(rs)
        if m == 'lp':
            ok = sum(r['feasible'] is True for r in rs); unk = sum(r['feasible'] is None for r in rs)
            print(f"| {t.upper()} | {m} | {n} | {ok}/{n}（判定できず {unk}） | | | | | | |")
            continue
        mean = lambda k: sum(r[k] for r in rs) / n
        f = [r['blocks_failed'] for r in rs]
        ep = rs[0].get('episode', '')
        print(f"| {t.upper()} | {m}{' ep' + str(ep) if ep else ''} | {n} | {sum(r['all_steps_in_band'] for r in rs)}/{n} | {mean('block_pass_pct'):.1f}% | "
              f"{sum(f)/n:.2f} / {max(f)} | {sum(x == 0 for x in f)}/{n} | {mean('step_pass_pct'):.1f}% | {mean('soc_hit_pct'):.1f}% | {mean('mean_error_to_award_pct'):.2f}% |")

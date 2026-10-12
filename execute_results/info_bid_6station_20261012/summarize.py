"""info_bid.py の結果を、市場と条件ごとに1行にまとめる（入札量は1日あたり kW・コマ、組の平均）。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {'oracle': '指令とEVを知る（両方の向き）', 'oracle_used': '指令とEVを知る（指令が使う向きだけ）',
         'cmd': '指令だけを知る', 'ev': 'EVだけを知る'}
print('| 市場 | 条件 | 組 | 入札量（kW・コマ／日） | 本線の入札量 | 本線との比 | 当日の指令とEVで成り立つ組 |')
print('|---|---|---|---|---|---|---|')
for task in ('aemo', 'ercot'):
    for mode in NAMES:
        f = HERE / f'result_{task}_{mode}.jsonl'
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.read_text(encoding='utf-8').splitlines()]
        if not rows:
            continue
        cap = sum(r['capacity_kw_block'] for r in rows) / len(rows)
        main = sum(r['main_capacity_kw_block'] for r in rows) / len(rows)
        ok = sum(r['feasible'] is True for r in rows)
        unk = sum(r['feasible'] is None for r in rows)
        print(f"| {task.upper()} | {NAMES[mode]} | {len(rows)} | {cap:,.0f} | {main:,.0f} | {cap / main:.2f} | {ok}/{len(rows)}"
              + (f'（判定できず {unk}）' if unk else '') + ' |')

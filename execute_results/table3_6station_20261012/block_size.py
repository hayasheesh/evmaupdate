"""blocks6/ の結果（run_blocks.py）から、帯の幅（約定量の10%）ごとの30分コマの合格率を出す。

帯の幅は、そのコマの6つの5分の帯のうち最も狭いもの（band_kw）。最低入札量70 kWのコマで7 kW。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGES = [0, 10, 15, 20, 30, 1e9]
LABEL = ['10 kW未満', '10〜15 kW', '15〜20 kW', '20〜30 kW', '30 kW以上']
d = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'blocks6'
print('| 市場 | 方法 | ' + ' | '.join(LABEL) + ' |')
print('|---|---|' + '---|' * len(LABEL))
for task in ('aemo', 'ercot'):
    for mode in ('rule', 'marl'):
        f = d / f'result_{task}_{mode}_4cmd_2ev.jsonl'
        if not f.exists():
            continue
        cells = [[0, 0] for _ in LABEL]
        for line in f.read_text(encoding='utf-8').splitlines():
            for b in json.loads(line).get('blocks', []):
                i = next(i for i in range(len(LABEL)) if EDGES[i] <= b['band_kw'] < EDGES[i + 1])
                cells[i][0] += 1
                cells[i][1] += b['pass']
        print(f'| {task.upper()} | {mode} | ' + ' | '.join(f'{100 * p / n:.1f}%（{n}）' if n else '—' for n, p in cells) + ' |')

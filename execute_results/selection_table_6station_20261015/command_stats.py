"""設計用（train 区分）の過去の指令から、窓の長さ L ごとに「同じ向きに出させた累積エネルギー」の分位点を出す（2026-10-16）。

    python command_stats.py <aemo|ercot>      command_stats_<task>.json を書く

指令は本線の学習用の入札（25日 × 設計用の指令128本）に入っている設計用の指令で、重複を除いて数える。
1本の指令について、上げの発動率から下げの発動率を引いた値を5分ごとに足し、長さ L の窓の中の和の最大（上げ）と
最小の符号を変えたもの（下げ）を約定量×時間で出す。そのうえで、指令の間の分位点 q を取る。
table.py の stat_q<q> は、この値を超えない範囲で一番厳しい指令を作る。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
WINDOW_HOURS = [0.5, 1, 2, 3, 4, 6, 8, 12, 24]
QUANTILES = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]


def main(task: str) -> None:
    bank = ROOT / 'execute_results/bid_banks' / f'wr_cap77wd_min70_margin7_train_25_6station_128cmd_3ev_{task}_plan_deviation'
    seen, net, sources = set(), [], []
    for path in sorted((bank / 'days').glob('*/fixed_bid.pkl')):
        bid = pickle.loads(path.read_bytes())
        for x in bid['design_activation_scenario_payload']:
            key = (x.get('source_date'), x.get('source_bmu'), x.get('name'))
            if key in seen:
                continue
            seen.add(key)
            sources.append(str(x.get('source')))
            net.append(np.asarray(x['up_proxy'], float)[:288] - np.asarray(x['down_proxy'], float)[:288])
    net = np.asarray(net)
    cumulative = np.concatenate([np.zeros((len(net), 1)), np.cumsum(net, axis=1)], axis=1) * (5.0 / 60.0)
    energy = {'up': {}, 'down': {}}
    for hours in WINDOW_HOURS:
        steps = int(round(12 * hours))
        window = cumulative[:, steps:] - cumulative[:, :-steps]
        for direction, values in (('up', window.max(axis=1)), ('down', -window.min(axis=1))):
            energy[direction][f'{hours:g}'] = [float(np.quantile(values, q)) for q in QUANTILES]
    out = {'task': task, 'bank': bank.name, 'partition': 'train (design commands of the 25 training days)',
           'commands': int(len(net)), 'window_hours': WINDOW_HOURS, 'quantiles': QUANTILES,
           'unit': 'award x hours', 'energy': energy}
    (HERE / f'command_stats_{task}.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(task, len(net), 'commands')


if __name__ == '__main__':
    main(sys.argv[1])

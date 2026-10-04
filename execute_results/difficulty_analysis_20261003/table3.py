"""制御の前提ごとの差を、AEMO・ERCOT・GB の3市場で並べる（force_comparison_20261002 の表に GB を足したもの）。

市場ごとに360組（検証日5日 × 未見指令24本 × EV実現3本）。3市場で日付と EV 実現は同じで、違うのは指令と入札だけ。
「1日を通して合格」= 評価対象の5分がすべて帯の中、かつその日に出発する EV が全部 SoC 目標に届く。
不確実性による低下 = 100 − 完全情報LPで成立する割合、制御による低下 = 完全情報LPで成立する割合 − 1日合格。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
E = HERE.parent
sys.path.insert(0, str(E / 'force_comparison_20261002'))
from summarize import KEY, MARKETS, metrics, pct  # noqa: E402

GB_EP = json.loads((E / 'unseen_decomposition_gb_20261003/status.json').read_text(encoding='utf-8-sig'))['gbEpisode']
G = E / 'unseen_decomposition_gb_20261003'
MARKETS = dict(MARKETS)
MARKETS['GB'] = {
    'lp': G / 'lp_certify_cases.csv',
    'methods': {
        f'MARL + force（GBで学習、ep{GB_EP}）': G / f'results_gb_ep{GB_EP}_marl_force/all_rollouts.csv',
        'ルールベース（中央観測、forceあり）': G / 'results_rule/all_rollouts.csv',
        '中央LP + force': G / 'results_central_lp_marl_force/all_rollouts.csv',
        f'参考: MARL（forceなし、GBで学習、ep{GB_EP}）': G / f'results_gb_ep{GB_EP}/all_rollouts.csv',
    },
}


def main() -> None:
    out = {}
    for market, spec in MARKETS.items():
        lp = pd.read_csv(spec['lp'])
        lp['feasible'] = lp['feasible'].map({'True': True, 'False': False, True: True, False: False})
        n = len(lp)
        feas = int((lp['feasible'] == True).sum())  # noqa: E712
        unk = int(lp['feasible'].isna().sum())
        rows = {}
        for name, path in spec['methods'].items():
            if not path.exists():
                rows[name] = None
                continue
            df = pd.read_csv(path).merge(lp[KEY + ['realized_seed', 'feasible']], on=KEY, suffixes=('', '_lp'), validate='one_to_one')
            assert len(df) == 360 and (df['realized_seed'] == df['realized_seed_lp']).all(), (market, name)
            r = metrics(df)
            r['on_lp_feasible'] = metrics(df[df['feasible'] == True])  # noqa: E712
            r['missed_reachable_share_pct'] = pct(df['missed_though_reachable'].sum(), df['missed_steps'].sum())
            rows[name] = r
        out[market] = {'cases': n, 'lp_feasible_pct_range': [pct(feas, n), pct(feas + unk, n)], 'lp_unknown': unk, 'methods': rows}
    (HERE / 'table3.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for market, r in out.items():
        lo, hi = r['lp_feasible_pct_range']
        print(f"== {market}: 完全情報LP 成立 {lo:.1f}〜{hi:.1f}%")
        for name, m in r['methods'].items():
            if m is None:
                print(f"  {name}: (未完了)")
                continue
            print(f"  {name}: 1日合格 {m['day_pass_pct']:.1f} 追従 {m['tracking_pct']:.1f} SoC {m['soc_pct']:.1f} コマ {m['block_pass_pct']:.1f} "
                  f"force {m['forced_ev_steps_mean']:.0f} | LP成立組: 1日合格 {m['on_lp_feasible']['day_pass_pct']:.1f} 追従 {m['on_lp_feasible']['tracking_pct']:.1f} "
                  f"| 外れのうちその時点で届いた {m['missed_reachable_share_pct']:.0f}%")


if __name__ == '__main__':
    main()

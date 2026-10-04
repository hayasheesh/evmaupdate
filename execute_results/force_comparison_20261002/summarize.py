"""force 充電を入れて SoC をほぼ100%にした条件で、制御の方法 × 市場の表を作る。

市場ごとに同じ360組（検証日5日 × 未見指令24本 × EV実現3本）。AEMO と ERCOT は日付・EV実現が同じで、
違うのは指令と入札だけ。完全情報LPは入札段階と同じ判定で、SoC目標は必ず満たす条件。
「1日を通して合格」= 評価対象の5分がすべて帯の中、かつその日に出発するEVが全部SoC目標に届く。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
E = HERE.parent
KEY = ['bid_day_index', 'scenario', 'seed']
MARKETS = {
    'AEMO': {
        'lp': E / 'unseen_decomposition_aemo_20261002/lp_certify_cases.csv',
        'methods': {
            'MARL + force（AEMOで学習）': E / 'unseen_decomposition_aemo_20261002/results_marl_force/all_rollouts.csv',
            'ルールベース（中央観測、forceあり）': E / 'unseen_decomposition_aemo_20261002/results_rule/all_rollouts.csv',
            '中央LP + force': E / 'unseen_decomposition_aemo_20261002/results_central_lp_force/all_rollouts.csv',
            '参考: MARL（forceなし）': E / 'marl_unseen_aemo_ab2200_20261002/results/all_rollouts.csv',
        },
    },
    'ERCOT': {
        'lp': E / 'unseen_decomposition_ercot_20261002/lp_certify_cases.csv',
        'methods': {
            'MARL + force（ERCOTで学習）': E / 'unseen_decomposition_ercot_20261002/results_ercot_marl_force/all_rollouts.csv',
            'MARL + force（AEMOで学習）': E / 'unseen_decomposition_ercot_20261002/results_aemo_marl_force/all_rollouts.csv',
            'ルールベース（中央観測、forceあり）': E / 'unseen_decomposition_ercot_20261002/results_rule/all_rollouts.csv',
            '中央LP + force': E / 'unseen_decomposition_ercot_20261002/results_central_lp_marl_force/all_rollouts.csv',
            '参考: MARL（forceなし、ERCOTで学習）': E / 'unseen_decomposition_ercot_20261002/results_ercot/all_rollouts.csv',
        },
    },
}


def pct(a, b):
    return 100.0 * a / b if b else float('nan')


def metrics(df: pd.DataFrame) -> dict:
    tracking_ok = df['missed_steps'] == 0
    soc_ok = df['departing_evs_soc_met'] == df['departing_evs']
    return {
        'cases': int(len(df)),
        'day_pass_pct': pct((tracking_ok & soc_ok).sum(), len(df)),
        'tracking_pct': pct(df['assessed_steps'].sum() - df['missed_steps'].sum(), df['assessed_steps'].sum()),
        'soc_pct': pct(df['departing_evs_soc_met'].sum(), df['departing_evs'].sum()),
        'block_pass_pct': pct(df['blocks_assessed'].sum() - df['blocks_failed'].sum(), df['blocks_assessed'].sum()),
        'forced_ev_steps_mean': float(df['forced_ev_step_overrides'].mean()) if 'forced_ev_step_overrides' in df else float('nan'),
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
            rows[name] = r
        out[market] = {'lp_feasible_pct_range': [pct(feas, n), pct(feas + unk, n)], 'methods': rows}
    (HERE / 'table.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for market, r in out.items():
        lo, hi = r['lp_feasible_pct_range']
        print(f"== {market}: 完全情報LP 成立 {lo:.1f}〜{hi:.1f}%")
        print(f"{'方法':<34}{'1日合格':>8}{'5分追従':>8}{'SoC達成':>8}{'コマ合格':>8}{'force回数':>9} | LP成立組の5分追従")
        for name, m in r['methods'].items():
            if m is None:
                print(f"{name:<34} (未完了)")
                continue
            print(f"{name:<34}{m['day_pass_pct']:8.1f}{m['tracking_pct']:8.1f}{m['soc_pct']:8.1f}{m['block_pass_pct']:8.1f}"
                  f"{m['forced_ev_steps_mean']:9.1f} | {m['on_lp_feasible']['tracking_pct']:.1f}")


if __name__ == '__main__':
    main()

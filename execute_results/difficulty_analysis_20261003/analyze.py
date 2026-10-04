"""指令と入札の特徴（features_*.csv）と、制御の結果（unseen_decomposition_* の360組）を突き合わせる。

結果は EV 実現3本の平均にして、(市場, 日, 指令) の1組を1行にする（市場ごとに120行）。
EV 実現は3市場で同じ seed なので、市場の差は指令と入札だけから来る。
出力
  cases.csv          1組1行の特徴と結果
  market_summary.csv 市場ごとの特徴の中央値と結果の平均
  spearman.csv       市場ごとの順位相関（特徴 × 結果）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E = HERE.parent
KEY = ['bid_day_index', 'scenario']
GB_EP = int((E / 'unseen_decomposition_gb_20261003/status.json').exists() and
            json.loads((E / 'unseen_decomposition_gb_20261003/status.json').read_text(encoding='utf-8-sig')).get('gbEpisode', 1240))
SOURCES = {
    'aemo': {
        'lp': E / 'unseen_decomposition_aemo_20261002/lp_certify_cases.csv',
        'marl': E / 'marl_unseen_aemo_ab2200_20261002/results/all_rollouts.csv',
        'marl_force': E / 'unseen_decomposition_aemo_20261002/results_marl_force/all_rollouts.csv',
        'rule': E / 'unseen_decomposition_aemo_20261002/results_rule/all_rollouts.csv',
        'central_lp_force': E / 'unseen_decomposition_aemo_20261002/results_central_lp_force/all_rollouts.csv',
    },
    'ercot': {
        'lp': E / 'unseen_decomposition_ercot_20261002/lp_certify_cases.csv',
        'marl': E / 'unseen_decomposition_ercot_20261002/results_ercot/all_rollouts.csv',
        'marl_force': E / 'unseen_decomposition_ercot_20261002/results_ercot_marl_force/all_rollouts.csv',
        'rule': E / 'unseen_decomposition_ercot_20261002/results_rule/all_rollouts.csv',
        'central_lp_force': E / 'unseen_decomposition_ercot_20261002/results_central_lp_marl_force/all_rollouts.csv',
    },
    'gb': {
        'lp': E / 'unseen_decomposition_gb_20261003/lp_certify_cases.csv',
        'marl': E / f'unseen_decomposition_gb_20261003/results_gb_ep{GB_EP}/all_rollouts.csv',
        'marl_force': E / f'unseen_decomposition_gb_20261003/results_gb_ep{GB_EP}_marl_force/all_rollouts.csv',
        'rule': E / 'unseen_decomposition_gb_20261003/results_rule/all_rollouts.csv',
        'central_lp_force': E / 'unseen_decomposition_gb_20261003/results_central_lp_marl_force/all_rollouts.csv',
    },
}
FEATURES = [
    'bid_up_kw_mean', 'bid_down_kw_mean', 'bid_baseline_kw_mean', 'bid_min_award_share',
    'up_active_frac', 'down_active_frac', 'idle_frac', 'up_util_when_active', 'down_util_when_active',
    'direction_switches', 'longest_run_steps', 'mean_run_steps', 'util_ramp_mean',
    'reg_abs_mean_kw', 'reg_abs_max_kw', 'reg_ramp_mean_kw',
    'energy_up_kwh', 'energy_down_kwh', 'energy_net_kwh', 'cum_min_kwh', 'cum_max_kwh', 'cum_range_kwh',
    'tol_mean_kw', 'tol_over_reg_active', 'in_band_at_baseline_share', 'need_move_kw_mean',
    'need_up_kwh', 'need_down_kwh', 'need_cum_min_kwh', 'need_cum_max_kwh', 'need_cum_range_kwh',
]


def outcome_frame(path: Path, prefix: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    assert len(df) == 360, (path, len(df))
    df['tracking'] = 100.0 * (df['assessed_steps'] - df['missed_steps']) / df['assessed_steps']
    df['soc'] = 100.0 * df['departing_evs_soc_met'] / df['departing_evs']
    df['day_pass'] = 100.0 * ((df['missed_steps'] == 0) & (df['departing_evs_soc_met'] == df['departing_evs']))
    df['up_step_pass'] = 100.0 * df['up_step_pass_rate']
    df['down_step_pass'] = 100.0 * df['down_step_pass_rate']
    df['missed_reachable_share'] = 100.0 * df['missed_though_reachable'] / df['missed_steps'].where(df['missed_steps'] > 0)
    cols = ['tracking', 'soc', 'day_pass', 'up_step_pass', 'down_step_pass', 'missed_reachable_share', 'forced_ev_step_overrides']
    g = df.groupby(KEY)[cols].mean().reset_index()
    return g.rename(columns={c: f'{prefix}_{c}' for c in cols})


def main() -> None:
    frames = []
    for market, src in SOURCES.items():
        f = pd.read_csv(HERE / f'features_{market}.csv')
        lp = pd.read_csv(src['lp']) if src['lp'].exists() else None
        if lp is not None and len(lp) == 360:
            lp['lp_feasible'] = 100.0 * lp['feasible'].map({'True': 1, 'False': 0, True: 1, False: 0})
            g = lp.groupby(KEY)['lp_feasible'].mean().reset_index()
            assert (lp.merge(f[KEY + ['command_source']], on=KEY, suffixes=('', '_f')).eval('command_source == command_source_f')).all()
            f = f.merge(g, on=KEY, how='left', validate='one_to_one')
        for name in ('marl', 'marl_force', 'rule', 'central_lp_force'):
            o = outcome_frame(src[name], name)
            if o is not None:
                f = f.merge(o, on=KEY, how='left', validate='one_to_one')
        frames.append(f)
    cases = pd.concat(frames, ignore_index=True)
    cases.to_csv(HERE / 'cases.csv', index=False)
    outcomes = [c for c in cases.columns if c == 'lp_feasible' or c.split('_')[0] in ('marl', 'rule', 'central')]
    summary = pd.concat({
        'feature_median': cases.groupby('market')[FEATURES].median().T,
        'outcome_mean': cases.groupby('market')[outcomes].mean().T,
    })
    summary.to_csv(HERE / 'market_summary.csv')
    rows = []
    for market, g in cases.groupby('market'):
        for o in outcomes:
            if g[o].notna().sum() < 20 or g[o].nunique() < 2:
                continue
            for feat in FEATURES:
                rho = g[[feat, o]].rank().corr().iloc[0, 1]
                rows.append({'market': market, 'outcome': o, 'feature': feat, 'rho': rho, 'n': int(g[[feat, o]].dropna().shape[0])})
    pd.DataFrame(rows).to_csv(HERE / 'spearman.csv', index=False)
    print(summary.round(2).to_string())


if __name__ == '__main__':
    main()

"""ERCOT の360組で、2つのモデルと完全情報LPの結果を突き合わせ、未見指令の合格率を分解する表を作る。

「1日を通して合格」= 評価対象の5分すべてが帯の中、かつその日に出発するEVが全部SoC目標に届く。
完全情報LPでは certify が成立を返した組（帯とSoC目標を同時に満たす充放電が存在する）。
分解
  不確実性による低下 = 100% - 完全情報LPで成立する割合
  制御による低下     = 完全情報LPで成立する割合 - その制御で合格した割合
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = ['bid_day_index', 'scenario', 'seed']
SOURCES = {
    'AEMOで学習したAB（ep2200）': HERE / 'results_aemo/all_rollouts.csv',
    'ERCOTで学習したAB（ep1760）': HERE / 'results_ercot/all_rollouts.csv',
}


def pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else float('nan')


def metrics(df: pd.DataFrame) -> dict:
    tracking_ok = df['missed_steps'] == 0
    soc_ok = df['departing_evs_soc_met'] == df['departing_evs']
    return {
        'cases': int(len(df)),
        'day_pass_pct': pct((tracking_ok & soc_ok).sum(), len(df)),
        'day_pass_tracking_only_pct': pct(tracking_ok.sum(), len(df)),
        'day_pass_soc_only_pct': pct(soc_ok.sum(), len(df)),
        'tracking_pct': pct(df['assessed_steps'].sum() - df['missed_steps'].sum(), df['assessed_steps'].sum()),
        'soc_pct': pct(df['departing_evs_soc_met'].sum(), df['departing_evs'].sum()),
        'block_pass_pct': pct(df['blocks_assessed'].sum() - df['blocks_failed'].sum(), df['blocks_assessed'].sum()),
    }


def main() -> None:
    lp = pd.read_csv(HERE / 'lp_certify_cases.csv')
    lp['feasible'] = lp['feasible'].map({'True': True, 'False': False, True: True, False: False})
    assert len(lp) == 360 and not lp.duplicated(KEY).any()
    n = len(lp)
    feas = int((lp['feasible'] == True).sum())  # noqa: E712
    unknown = int(lp['feasible'].isna().sum())
    lp_lo, lp_hi = pct(feas, n), pct(feas + unknown, n)
    out = {'cases': n, 'lp_feasible_pct_range': [lp_lo, lp_hi], 'lp_unknown': unknown,
           'lp_band_max_diff_kw': float(lp[['band_lower_max_diff_kw', 'band_upper_max_diff_kw']].max().max()),
           'lp_assessed_mask_equal_all': bool(lp['assessed_mask_equal'].all()), 'methods': {}}
    checks = {}
    for name, path in SOURCES.items():
        df = pd.read_csv(path)
        assert len(df) == 360 and not df.duplicated(KEY).any(), name
        df = df.merge(lp[KEY + ['realized_seed', 'feasible', 'evs_target_required']],
                      on=KEY, suffixes=('', '_lp'), validate='one_to_one')
        assert (df['realized_seed'] == df['realized_seed_lp']).all(), name
        checks[name] = {
            'departing_evs_equals_lp_target_required': int((df['departing_evs'] == df['evs_target_required']).sum()),
        }
        row = metrics(df)
        row['control_loss_pt_range'] = [lp_lo - row['day_pass_pct'], lp_hi - row['day_pass_pct']]
        row['on_lp_feasible'] = metrics(df[df['feasible'] == True])  # noqa: E712
        row['on_lp_infeasible'] = metrics(df[df['feasible'] == False])  # noqa: E712
        out['methods'][name] = row
    marl = pd.read_csv(SOURCES['ERCOTで学習したAB（ep1760）'])
    for name, path in SOURCES.items():
        other = pd.read_csv(path).merge(marl[KEY + ['departing_evs']], on=KEY, suffixes=('', '_marl'))
        checks[name]['departing_evs_equal_to_marl'] = int((other['departing_evs'] == other['departing_evs_marl']).sum())
    out['ev_match_checks'] = checks
    (HERE / 'decomposition.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f"完全情報LPで成立: {lp_lo:.1f}〜{lp_hi:.1f}%（{feas}/{n}、時間切れ {unknown}）"
          f" → 不確実性による低下 {100 - lp_hi:.1f}〜{100 - lp_lo:.1f} pt")
    print(f"{'方法':<22}{'1日合格':>8}{'制御による低下':>16}{'追従だけ':>8}{'SoCだけ':>8}{'5分追従':>8}{'SoC達成':>8}{'コマ合格':>8}")
    for name, r in out['methods'].items():
        lo, hi = r['control_loss_pt_range']
        print(f"{name:<22}{r['day_pass_pct']:8.1f}{f'{hi:.1f}〜{lo:.1f}':>16}{r['day_pass_tracking_only_pct']:8.1f}"
              f"{r['day_pass_soc_only_pct']:8.1f}{r['tracking_pct']:8.1f}{r['soc_pct']:8.1f}{r['block_pass_pct']:8.1f}")
    print('LPで成立する組 / しない組 での 5分追従・SoC達成・1日合格')
    for name, r in out['methods'].items():
        a, b = r['on_lp_feasible'], r['on_lp_infeasible']
        print(f"{name:<22} 成立{a['cases']:>4}組: {a['tracking_pct']:5.1f} {a['soc_pct']:5.1f} {a['day_pass_pct']:5.1f}"
              f"  不成立{b['cases']:>4}組: {b['tracking_pct']:5.1f} {b['soc_pct']:5.1f} {b['day_pass_pct']:5.1f}")
    print('EVの一致:', json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    main()

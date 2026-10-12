"""保存済みbest/late NPZの指令特徴と帯外率を再集計する。"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / 'command_cohort_analysis.json'
EPS = 1e-5


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0}
    return {'n': len(x), 'min': float(x.min()), 'mean': float(x.mean()),
            'p50': float(np.quantile(x, .5)), 'p90': float(np.quantile(x, .9)),
            'p99': float(np.quantile(x, .99)), 'max': float(x.max())}


def fraction(n, d):
    return {'numerator': int(n), 'denominator': int(d),
            'pct': float(100 * n / d) if d else None}


def cohort(a, mask):
    mask = mask & a['active']
    bad = mask & a['bad']
    n = int(mask.sum())
    return {'outside_band': fraction(bad.sum(), n),
            'outside_band_strict': fraction((mask & a['bad_strict']).sum(), n),
            'outside_above_request': fraction((bad & (a['residual'] < 0)).sum(), n),
            'outside_below_request': fraction((bad & (a['residual'] > 0)).sum(), n),
            'outside_rated_interval': fraction((mask & a['rated_impossible']).sum(), n),
            'instant_infeasible': fraction((mask & ~a['instant']).sum(), n),
            'deadline_any_EV_lower_bound_exceeds_upper': fraction((mask & a['invalid_floor']).sum(), n),
            'outside_band_instant_feasible': fraction((bad & a['instant']).sum(), bad.sum()),
            'outside_band_deadline_safe_feasible': fraction((bad & a['safe']).sum(), bad.sum()),
            'request_kw': stats(a['request'][mask]), 'abs_request_over_connected_rated': stats(a['ratio'][mask]),
            'band_halfwidth_kw': stats(a['band'][mask]), 'band_fullwidth_kw': stats(2 * a['band'][mask]),
            'band_halfwidth_over_connected_rated': stats(a['band_ratio'][mask]),
            'absolute_residual_kw': stats(abs(a['residual'][mask]))}


def load(root, row, meta):
    path = root / (row['mode'] + '_' + row['id'] + '.npz')
    with np.load(path, allow_pickle=False) as raw:
        e = raw['ev'].astype(np.float64)
        z = raw['step'].astype(np.float64)
    ec = {name: i for i, name in enumerate(meta['ev_columns'])}
    sc = {name: i for i, name in enumerate(meta['step_columns'])}
    assert np.array_equal(z[:, sc['step']], np.arange(1, 289))
    ix = e[:, ec['step']].astype(int) - 1
    assert np.all((ix >= 0) & (ix < 288))
    rated = np.bincount(ix, weights=e[:, ec['rated_kw']], minlength=288)
    low = np.bincount(ix, weights=e[:, ec['instant_min_kw']], minlength=288)
    high = np.bincount(ix, weights=e[:, ec['instant_max_kw']], minlength=288)
    floor = np.bincount(ix, weights=e[:, ec['deadline_min_kw']], minlength=288)
    invalid_floor = np.bincount(ix, weights=(e[:, ec['deadline_min_kw']] > e[:, ec['instant_max_kw']] + EPS), minlength=288) > 0
    request, band, residual = (z[:, sc[k]] for k in ('request_kw', 'band_kw', 'residual_kw'))
    active = z[:, sc['tracking_enabled']] > 0
    instant = z[:, sc['instant_feasible']] > 0
    safe = z[:, sc['deadline_safe_feasible']] > 0
    # NPZのEV列はfloat32なので、境界近傍では保存フラグと再判定が異なりうる。
    instant_rebuilt = (low <= request + band + EPS) & (high >= request - band - EPS)
    safe_rebuilt = ~invalid_floor & (floor <= request + band + EPS) & (high >= request - band - EPS)
    departed = e[e[:, ec['remaining_steps']] == 1]
    keys = departed[:, [ec[k] for k in ('station', 'id', 'arrival_step')]]
    assert len(np.unique(keys, axis=0)) == len(departed) == row['depart_count']
    deficit = np.maximum(departed[:, ec['target']] - departed[:, ec['soc_after']], 0) * departed[:, ec['capacity_kwh']] / 100
    official_miss_float = len(departed) * (100 - row['soc']) / 100
    official_miss = int(round(official_miss_float))
    assert abs(official_miss - official_miss_float) < 1e-7
    a = {'active': active, 'request': request, 'band': band, 'residual': residual,
         'bad': active & (abs(residual) > band + EPS), 'bad_strict': active & (abs(residual) > band),
         'instant': instant, 'safe': safe, 'rated': rated, 'invalid_floor': invalid_floor,
         'rated_impossible': (request - band > rated + EPS) | (request + band < -rated - EPS),
         'ratio': np.divide(abs(request), rated, out=np.full(288, np.nan), where=rated > 0),
         'band_ratio': np.divide(band, rated, out=np.full(288, np.nan), where=rated > 0),
         'position': np.arange(288) % 6 + 1,
         'request_delta': np.r_[np.nan, np.diff(request)],
         'previous_active': np.r_[False, active[:-1]],
         'date': np.repeat(row['date'], 288)}
    assert a['bad'].sum() == row['failed_steps']
    rebuilt_errors = {
        'instant_flag_disagreement': fraction((instant != instant_rebuilt).sum(), 288),
        'deadline_flag_disagreement': fraction((safe != safe_rebuilt).sum(), 288),
        'tracking_epsilon_changes': fraction((a['bad'] != a['bad_strict']).sum(), active.sum()),
        'ev_count_mismatch_steps': int((np.bincount(ix, minlength=288) != z[:, sc['EV_count']]).sum()),
        'trace_depart_miss_soc_points_0_001': int((departed[:, ec['soc_after']] < departed[:, ec['target']] - .001).sum()),
        'summary_depart_miss_soc_points_0_001': row['depart_miss']}
    dep = {'departure_count': len(departed),
           'official_soc_miss_from_summary': fraction(official_miss, len(departed)),
           'trace_strict_soc_miss': fraction((deficit > 0).sum(), len(departed)),
           'diagnostic_deficit_gt_0_001_kwh': fraction((deficit > .001).sum(), len(departed)),
           'trace_deficit_between_0_and_0_001_kwh': fraction(((deficit > 0) & (deficit <= .001)).sum(), len(departed)),
           'trace_total_deficit_kwh': float(deficit.sum()), 'trace_deficit_kwh': stats(deficit)}
    return a, dep, rebuilt_errors, {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, keys


def all_cohorts(a):
    masks = {'all': np.ones(len(a['active']), bool)}
    masks.update({'request_sign/' + label: mask for label, mask in
                  [('negative', a['request'] < -1e-6), ('zero', abs(a['request']) <= 1e-6), ('positive', a['request'] > 1e-6)]})
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1), (1, float('inf'))]:
        masks[f'abs_request_over_rated/[{lo},{hi})'] = (a['ratio'] >= lo) & (a['ratio'] < hi)
    masks['connected_rated_zero'] = a['rated'] == 0
    for lo, hi in [(0, 25), (25, 50), (50, 100), (100, 200), (200, float('inf'))]:
        masks[f'band_halfwidth_kw/[{lo},{hi})'] = (a['band'] >= lo) & (a['band'] < hi)
    for position in range(1, 7):
        masks[f'block_position/{position}'] = a['position'] == position
    masks['block_boundary/first_or_last'] = (a['position'] == 1) | (a['position'] == 6)
    masks['block_boundary/interior'] = (a['position'] > 1) & (a['position'] < 6)
    masks['tracking_start'] = ~a['previous_active']
    # 指令差は同じケースの直前時刻との比較。ケース間の差は作らない。
    for lo, hi in [(0, 100), (100, 300), (300, float('inf'))]:
        masks[f'abs_request_delta_kw/[{lo},{hi})'] = (abs(a['request_delta']) >= lo) & (abs(a['request_delta']) < hi)
    return {name: cohort(a, mask) for name, mask in masks.items()}


def concat(arrays):
    return {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}


def compare(best, late):
    assert np.array_equal(best['active'], late['active'])
    for k in ('request', 'band', 'rated', 'date'):
        assert np.array_equal(best[k], late[k])
    n = best['active'].sum()
    best_cohorts, late_cohorts = all_cohorts(best), all_cohorts(late)
    return {'both_outside': fraction((best['bad'] & late['bad']).sum(), n),
            'best_outside_late_inside': fraction((best['bad'] & ~late['bad']).sum(), n),
            'best_inside_late_outside': fraction((~best['bad'] & late['bad']).sum(), n),
            'both_inside': fraction((best['active'] & ~best['bad'] & ~late['bad']).sum(), n),
            'outside_rate_delta_percentage_points': float(100 * (late['bad'].sum() - best['bad'].sum()) / n),
            'cohort_outside_rate_delta_percentage_points': {
                k: (v['outside_band']['pct'] - best_cohorts[k]['outside_band']['pct'])
                if v['outside_band']['denominator'] else None for k, v in late_cohorts.items()}}


def performance(rows):
    return {'case_denominator': len(rows),
            'case_mean_tracking_pct': float(np.mean([r['tracking'] for r in rows])),
            'case_mean_official_soc_achievement_pct': float(np.mean([r['soc'] for r in rows])),
            'case_mean_tracking_plus_soc': float(np.mean([r['tracking'] + r['soc'] for r in rows]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, help='metadata.json、summary.json、NPZを含む1市場のフォルダ。省略時は既存5日の2市場。')
    parser.add_argument('--input-base', type=Path, help='2市場のフォルダを含むディレクトリ。省略時はスクリプト所在ディレクトリ。')
    parser.add_argument('--folder-prefix', default='trace_plateau20', help='フォルダ名の接頭辞。例: calendar20。')
    parser.add_argument('--date-tag', default='20261008', help='フォルダ名末尾の日付。')
    parser.add_argument('--output', type=Path, default=OUT, help='出力JSON。入力のscopeと件数を出力に保存する。')
    args = parser.parse_args()
    if args.input_root and args.input_base:
        parser.error('--input-rootと--input-baseは同時に指定できません。')
    input_base = (args.input_base or BASE).resolve()
    roots = [args.input_root.resolve()] if args.input_root else [input_base / f'{args.folder_prefix}_{m}_{args.date_tag}' for m in ('aemo', 'ercot')]
    result = {'provenance': '[raw再集計] [コード確認]',
              'definitions': {
                  'scope': '各input_rootを独立して集計。入力metadataのscopeと日数・指令数・ケース数を市場ごとに保存する。曜日と指令・接続状況が交絡するため曜日の因果効果はこの集計だけでは不明。',
                  'request_sign': 'request_kwはbaselineを含むEV合計電力。正は充電、負は放電。regulation=request-baselineの符号はNPZにbaselineが無いため未集計。',
                  'denominator': '帯外率はtracking_enabledの5分時刻数。同一の日付・指令をEVseed間で重複して数える。date_command_countも併記する。',
                  'band': 'band_kwは半幅。帯は[request-band,request+band]。全幅は2*band。',
                  'outside_band': 'abs(raw_actor_residual_kw)>band_kw+1e-5。公式の厳密判定abs(residual)>bandも併記。',
                  'block_position': '(step-1)%6+1。30分コマの最初と最後を境界、2〜5を内部として集計。',
                  'connected_rated': 'その時刻のEV行のrated_kw合計。absrequest/定格>1でも帯と定格範囲が重なる場合がある。SoC制約と出発目標は定格合計に含まれない。',
                  'instant_feasible': '現在SoCの0〜100%と定格のみ。未来の指令と出発目標を無視。保存済みフラグを使う。',
                  'deadline_safe_feasible': '以後は定格充電できると仮定する1時刻の下限。全EVでdeadline_min<=instant_maxを要求するため、1台の回復不能でもfalseになる。falseを指令の物理的不可の原因とは断定できない。未来の指令を満たす保証は無い。',
                  'physical_interpretation': '単時刻で不可でも、そのSoCを手前で作った原因は特定できない。全日LPが不可の時刻・MARLの過去SoC起因・現在行動起因の3分類はこの集計だけでは不明。',
                  'departure_key': '各ケース内でstation,id,arrival_step。remaining_steps==1の行。',
                  'soc': '公式はraw_reward_soc_tensorの厳密final_soc>=target。公式未達数はsummary.socと出発数から復元。trace.soc_afterはexecuted_kwから再構成後float32保存であり公式状態と一致する保証は無い。',
                  'diagnostic_threshold': '不足>0.001kWhは診断用。trace生成側depart_missは不足>0.001SoC百分率点であり別の閾値。',
                  'aggregation': '公式追従率とSoC達成率の和はケース平均。帯外率は時刻をプールした率。日別はその日の全指令・EVseedを集計。'},
              'code_evidence': [
                  {'path': 'EnvConfig.py:140', 'fact': '5分、POWER_TO_ENERGY=5/60。'},
                  {'path': 'environment/EVEnv.py:1098', 'fact': 'current_net_demand=raw_net_demand。regulation=request-baseline。'},
                  {'path': 'environment/EVEnv.py:1388', 'fact': '追従公式判定はdeviation<=tol_narrow_metrics。'},
                  {'path': 'environment/EVEnv.py:1218', 'fact': '公式SoC用raw_new_socsはfloat32演算でraw_powerから計算。実行電力から再構成するtraceと演算順序が異なる。'},
                  {'path': 'environment/EVEnv.py:1519', 'fact': 'soc_achieved_metric=departing_final_socs>=departing_target_socs。'},
                  {'path': 'execute_results/watch_codex_20261008/trace_plateau20.py:64', 'fact': '単時刻上下限とdeadline下限を計算。'},
                  {'path': 'execute_results/watch_codex_20261008/trace_plateau20.py:97', 'fact': 'soc_after再構成、depart_missの0.001百分率点閾値。107行でEV列float32、step列float64に保存。'},
                  {'path': 'execute_results/watch_codex_20261008/trace_screen_helpers.py:91', 'fact': 'trace環境は中央残差分配と蓄電池を無効化。raw actorの帯外率を評価。'}],
              'markets': {}}
    for root in roots:
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
        meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
        market = meta['market']
        rows = {mode: sorted([r for r in summary['rows'] if r['mode'] == mode], key=lambda r: r['id']) for mode in ('best', 'late')}
        assert len(rows['best']) == len(rows['late']) > 0
        assert len({r['id'] for r in rows['best']}) == len(rows['best'])
        assert len({r['id'] for r in rows['late']}) == len(rows['late'])
        assert all(r['market'] == market for mode_rows in rows.values() for r in mode_rows)
        if 'cases' in meta:
            assert len(rows['best']) == meta['cases']
        arrays, deps, audits, sources, keys = {}, {}, {}, [], {}
        sources.extend({'path': str((root / name).resolve()), 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in ('metadata.json', 'summary.json'))
        cases = []
        for mode in ('best', 'late'):
            arrays[mode], deps[mode], audits[mode], keys[mode] = [], [], [], []
            for row in rows[mode]:
                a, dep, audit, source, ev_keys = load(root, row, meta)
                arrays[mode].append(a); deps[mode].append(dep); audits[mode].append(audit); sources.append(source); keys[mode].append(ev_keys)
                cases.append({'mode': mode, 'id': row['id'], 'date': row['date'],
                              'weekday': date.fromisoformat(row['date']).weekday(),
                              'command': row['command'], 'ev_seed': row['ev_seed'],
                              'ev_hash': row['ev_hash'], 'cohorts': all_cohorts(a), 'departure': dep,
                              'validation': audit, 'performance': performance([row])})
        for i, (b, l) in enumerate(zip(rows['best'], rows['late'])):
            for k in ('id', 'date', 'command', 'ev_seed', 'ev_hash'):
                assert b[k] == l[k]
            assert np.array_equal(keys['best'][i], keys['late'][i])
        pooled = {m: concat(arrays[m]) for m in arrays}
        by_day = []
        for day in sorted(set(r['date'] for r in rows['best'])):
            entry = {'date': day, 'weekday': date.fromisoformat(day).weekday()}
            day_arrays = {}
            for mode in arrays:
                indexes = [i for i, r in enumerate(rows[mode]) if r['date'] == day]
                assert indexes
                day_arrays[mode] = concat([arrays[mode][i] for i in indexes])
                entry[mode] = {'cohorts': all_cohorts(day_arrays[mode]), 'performance': performance([rows[mode][i] for i in indexes])}
            entry['best_to_late'] = compare(day_arrays['best'], day_arrays['late'])
            by_day.append(entry)
        modes = {}
        for mode in arrays:
            d = sum(x['departure_count'] for x in deps[mode])
            modes[mode] = {'episode': meta[mode + '_episode'], 'cohorts': all_cohorts(pooled[mode]), 'performance': performance(rows[mode]),
                           'departures': {'departure_count': d,
                             **{k: fraction(sum(x[k]['numerator'] for x in deps[mode]), d) for k in
                                ('official_soc_miss_from_summary', 'trace_strict_soc_miss', 'diagnostic_deficit_gt_0_001_kwh', 'trace_deficit_between_0_and_0_001_kwh')},
                             'trace_total_deficit_kwh': sum(x['trace_total_deficit_kwh'] for x in deps[mode])},
                           'validation': {k: sum(x[k]['numerator'] if isinstance(x[k], dict) else x[k] for x in audits[mode]) for k in audits[mode][0]}}
        day_commands = {(r['date'], r['command']) for r in rows['best']}
        result['markets'][market] = {'input_root': str(root.resolve()), 'input_scope': meta.get('scope'),
                                     'raw_metadata': meta,
                                     'case_count_per_mode': len(rows['best']), 'date_count': len(by_day),
                                     'date_command_count': len(day_commands),
                                     'distinct_command_count': len({r['command'] for r in rows['best']}),
                                     'weekday_date_counts': {str(w): sum(date.fromisoformat(d['date']).weekday() == w for d in by_day) for w in range(7)},
                                     'cases_per_date': {d['date']: sum(r['date'] == d['date'] for r in rows['best']) for d in by_day},
                                     'modes': modes, 'best_to_late': compare(pooled['best'], pooled['late']),
                                     'by_day': by_day, 'cases': cases, 'sources': sources}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for market, r in result['markets'].items():
        print(market)
        for mode, x in r['modes'].items():
            print(mode, json.dumps({'outside': x['cohorts']['all']['outside_band'], 'performance': x['performance'], 'departures': x['departures'], 'validation': x['validation']}))
        print('by_day', [(d['date'], d['best']['cohorts']['all']['outside_band'], d['late']['cohorts']['all']['outside_band']) for d in r['by_day']])


if __name__ == '__main__':
    main()

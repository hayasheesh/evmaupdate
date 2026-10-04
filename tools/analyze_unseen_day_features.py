"""保存済み未見LP判定・入札・指令を集計する。EV生成や最適化は行わない。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def longest_minutes(mask):
    padded = np.pad(np.asarray(mask, dtype=int), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(max(ends - starts, default=0) * 5)


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) and np.std(y) else None


def direction_reversals(delta, enabled):
    previous, reversals = 0, 0
    for power, live in zip(delta, enabled):
        if not live:
            previous = 0
            continue
        direction = 1 if power > 1e-6 else -1 if power < -1e-6 else 0
        if direction:
            reversals += int(previous != 0 and previous != direction)
            previous = direction
    return reversals


def avg(rows, key):
    return float(np.mean([r[key] for r in rows]))


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'execute_results/bid_unseen_realized_ev_aemo_256_20260930')
    parser.add_argument('--bank', type=Path, default=ROOT / 'execute_results/bid_banks/train_25_minmedmax_3of128ev_256cmd_all_commands_aemo_plan_deviation')
    parser.add_argument('--output', type=Path, default=ROOT / 'execute_results/unseen_day_features_20260930')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bank / 'manifest.json').read_text(encoding='utf-8'))
    day_rows, seed_rows, command_rows, trial_rows = [], [], [], []
    for entry in manifest['entries']:
        date = entry['service_date']
        result = json.loads((args.results / f'AEMO_{date}.json').read_text(encoding='utf-8'))
        bid_path = args.bank / entry['bid_path']
        raw = bid_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == result['bid_sha256']
        bid = pickle.loads(raw)
        base = np.asarray(bid['baseline_plan'], dtype=float)
        up = np.asarray(bid['up_plan'], dtype=float)
        down = np.asarray(bid['down_plan'], dtype=float)
        live = (up + down) > 1e-6
        b, u, d, enabled = [np.repeat(x, 6) for x in (base, up, down, live)]
        commands = {(str(c['source_date']), str(c['source_bmu'])): c for c in bid['activation_scenario_payload']}
        assert len(commands) == result['unseen_commands']
        trials = [t for r in result['rows'] for t in r['trials']]
        passed = sum(t['feasible'] is True for t in trials)
        failed = sum(t['feasible'] is False for t in trials)
        unknown = sum(t['feasible'] is None for t in trials)
        counts = [int(n) for n in result['ev_counts']]
        selection = {x['label']: x['ev_count'] for x in bid['ev_scenario_selection']}
        row = {'date': date, 'month': int(date[5:7]), 'pass_pct': 100 * passed / len(trials),
               'passed': passed, 'failed': failed, 'unknown': unknown,
               'ev_min_eval': min(counts), 'ev_mean_eval': float(np.mean(counts)), 'ev_max_eval': max(counts),
               'ev_design_min': selection['minimum'], 'ev_design_median': selection['median'], 'ev_design_max': selection['maximum'],
               'ev_eval_to_design_median': float(np.mean(counts)) / selection['median'],
               'ev_eval_outside_design_count_range': sum(n < selection['minimum'] or n > selection['maximum'] for n in counts),
               'mean_up_kw': float(np.mean(up)), 'mean_down_kw': float(np.mean(down)),
               'mean_width_kw': float(np.mean(up + down)), 'mean_live_width_kw': float(np.mean((up + down)[live])),
               'live_hours': float(np.sum(live) * .5), 'baseline_net_kwh': float(np.sum(base[live]) * .5),
               'baseline_charge_kwh': float(np.sum(np.maximum(base[live], 0)) * .5),
               'baseline_discharge_kwh': float(np.sum(np.maximum(-base[live], 0)) * .5),
               'max_abs_baseline_kw': float(np.max(np.abs(base[live]))),
               'baseline_total_variation_kw': float(np.sum(np.abs(np.diff(base))[live[1:] & live[:-1]])),
               'mean_width_per_design_ev_kw': float(np.mean(up + down)) / selection['median']}
        seed_lookup = dict(zip(result['ev_seeds'], counts))
        for seed, count in seed_lookup.items():
            tt = [t for t in trials if t['ev_seed'] == seed]
            seed_rows.append({'date': date, 'seed': seed, 'ev_count': count,
                              'ev_count_delta_from_day_mean': count - float(np.mean(counts)),
                              'ev_count_ratio_to_design_median': count / selection['median'],
                              'pass_pct': 100 * sum(t['feasible'] is True for t in tt) / len(tt),
                              'failed': sum(t['feasible'] is False for t in tt),
                              'unknown': sum(t['feasible'] is None for t in tt)})
        daily_commands = []
        for scored in result['rows']:
            command = commands[(scored['source_date'], scored['source_bmu'])]
            us = np.asarray(command['up_proxy'], dtype=float)
            ds = np.asarray(command['down_proxy'], dtype=float)
            aup = us * (u > 1e-6)
            adn = ds * (d > 1e-6)
            delta = d * ds - u * us
            target = b + delta
            target[~enabled] = 0
            features = {'date': date, 'source_date': scored['source_date'], 'source_bmu': scored['source_bmu'],
                        'up_normalized_hours': float(np.sum(aup) / 12),
                        'down_normalized_hours': float(np.sum(adn) / 12),
                        'net_normalized_hours': float(np.sum(adn - aup) / 12),
                        'absolute_normalized_hours': float(np.sum(adn + aup) / 12),
                        'absolute_net_normalized_hours': float(abs(np.sum(adn - aup) / 12)),
                        'active_instruction_points': int(np.count_nonzero((aup + adn) > 1e-6)),
                        'direction_reversals': direction_reversals(delta, enabled),
                        'up_activation_kwh': float(np.sum(u * us) / 12),
                        'down_activation_kwh': float(np.sum(d * ds) / 12),
                        'absolute_activation_kwh': float(np.sum(np.abs(delta)) / 12),
                        'net_activation_kwh': float(np.sum(delta) / 12),
                        'net_target_kwh': float(np.sum(target) / 12),
                        'longest_up_minutes': longest_minutes(aup > 1e-6),
                        'longest_down_minutes': longest_minutes(adn > 1e-6),
                        'longest_up_over_10pct_minutes': longest_minutes(aup >= .1),
                        'longest_down_over_10pct_minutes': longest_minutes(adn >= .1),
                        'max_abs_target_kw': float(np.max(np.abs(target))),
                        'max_abs_activation_kw': float(np.max(np.abs(delta))),
                        'max_target_change_kw': float(np.max(np.abs(np.diff(target))[enabled[1:] & enabled[:-1]], initial=0)),
                        'target_total_variation_kw': float(np.sum(np.abs(np.diff(target))[enabled[1:] & enabled[:-1]])),
                        'failed_trials': sum(t['feasible'] is False for t in scored['trials']),
                        'passed_trials': sum(t['feasible'] is True for t in scored['trials']),
                        'unknown_trials': sum(t['feasible'] is None for t in scored['trials'])}
            daily_commands.append(features)
            command_rows.append(features)
            for t in scored['trials']:
                trial_rows.append({**features, 'seed': t['ev_seed'], 'ev_count': seed_lookup[t['ev_seed']], 'feasible': t['feasible']})
        for key in ('up_normalized_hours', 'down_normalized_hours', 'absolute_normalized_hours',
                    'net_target_kwh', 'max_abs_activation_kw', 'longest_up_minutes', 'longest_down_minutes'):
            row['command_mean_' + key] = avg(daily_commands, key)
        row['commands_any_failure'] = sum(r['failed_trials'] > 0 for r in daily_commands)
        row['commands_all_3_failed'] = sum(r['failed_trials'] == 3 for r in daily_commands)
        row['commands_all_3_passed'] = sum(r['passed_trials'] == 3 for r in daily_commands)
        day_rows.append(row)
        print(date, f"{row['pass_pct']:.2f}%", 'EV', counts, 'design', list(selection.values()),
              'width', round(row['mean_width_kw']), 'base-E', round(row['baseline_net_kwh']),
              'base-max', round(row['max_abs_baseline_kw']),
              'all3fail', row['commands_all_3_failed'], flush=True)
    # Keep all associations descriptive. Trial outcomes share daily bids and EV draws.
    group_fields = ['pass_pct', 'ev_mean_eval', 'ev_eval_to_design_median', 'mean_width_kw',
                    'mean_width_per_design_ev_kw', 'live_hours', 'baseline_net_kwh', 'baseline_charge_kwh',
                    'baseline_discharge_kwh', 'max_abs_baseline_kw', 'baseline_total_variation_kw',
                    'command_mean_up_normalized_hours', 'command_mean_down_normalized_hours',
                    'command_mean_absolute_normalized_hours', 'command_mean_max_abs_activation_kw',
                    'command_mean_longest_up_minutes', 'command_mean_longest_down_minutes']
    groups = {'low_below_85pct': [r for r in day_rows if r['pass_pct'] < 85],
              'high_at_least_95pct': [r for r in day_rows if r['pass_pct'] >= 95]}
    summary = {'groups': {name: {'days': len(rows), 'dates': [r['date'] for r in rows],
                                'means': {k: avg(rows, k) for k in group_fields}} for name, rows in groups.items()},
               'daily_correlations_with_pass_pct': {k: correlation([r[k] for r in day_rows], [r['pass_pct'] for r in day_rows]) for k in group_fields[1:]},
               'ev_draws_outside_design_count_range': sum(r['ev_eval_outside_design_count_range'] for r in day_rows)}
    summary['by_month'] = {str(month): {'days': len(rr), 'pass_pct': 100 * sum(r['passed'] for r in rr) / sum(r['passed'] + r['failed'] + r['unknown'] for r in rr)}
                           for month in sorted({r['month'] for r in day_rows})
                           for rr in [[r for r in day_rows if r['month'] == month]]}
    day_pass = {r['date']: r['pass_pct'] for r in day_rows}
    summary['ev_count_correlation_with_seed_pass_pct_within_day'] = correlation(
        [r['ev_count_delta_from_day_mean'] for r in seed_rows],
        [r['pass_pct'] - day_pass[r['date']] for r in seed_rows])
    command_fields = ['up_normalized_hours', 'down_normalized_hours', 'net_normalized_hours',
                      'absolute_normalized_hours', 'absolute_net_normalized_hours', 'active_instruction_points',
                      'direction_reversals', 'absolute_activation_kwh', 'net_activation_kwh', 'net_target_kwh',
                      'longest_up_minutes', 'longest_down_minutes', 'longest_up_over_10pct_minutes',
                      'longest_down_over_10pct_minutes', 'max_abs_activation_kw', 'max_target_change_kw', 'target_total_variation_kw']
    command_day_mean = {r['date']: {k: avg([c for c in command_rows if c['date'] == r['date']], k) for k in command_fields} for r in day_rows}
    known = [r for r in trial_rows if r['feasible'] is not None]
    summary['command_features_by_trial_outcome'] = {}
    for label, flag in [('passed', True), ('failed', False)]:
        rr = [r for r in known if r['feasible'] is flag]
        summary['command_features_by_trial_outcome'][label] = {'trials': len(rr),
            'means': {k: avg(rr, k) for k in command_fields},
            'mean_difference_from_daily_pool': {k: float(np.mean([r[k] - command_day_mean[r['date']][k] for r in rr])) for k in command_fields}}
    summary['within_day_feature_correlation_with_failure'] = {k: correlation(
        [r[k] - command_day_mean[r['date']][k] for r in known],
        [float(not r['feasible']) - (1 - day_pass[r['date']] / 100) for r in known]) for k in command_fields}
    summary['within_day_command_quartiles'] = {}
    for key in ('absolute_normalized_hours', 'absolute_activation_kwh', 'max_abs_activation_kw', 'target_total_variation_kw',
                'longest_down_minutes', 'longest_up_minutes', 'absolute_net_normalized_hours',
                'active_instruction_points', 'direction_reversals'):
        selected = {'smallest_quarter': [], 'largest_quarter': []}
        for day in day_rows:
            rr = [r for r in command_rows if r['date'] == day['date']]
            order = sorted(rr, key=lambda r: abs(r['net_normalized_hours']) if key == 'absolute_net_normalized_hours' else r[key])
            quarter = len(order) // 4
            selected['smallest_quarter'].extend(order[:quarter])
            selected['largest_quarter'].extend(order[-quarter:])
        summary['within_day_command_quartiles'][key] = {
            name: {'commands': len(rr), 'trials': len(rr) * 3,
                   'infeasible': sum(r['failed_trials'] for r in rr),
                   'unknown': sum(r['unknown_trials'] for r in rr),
                   'failure_pct': 100 * sum(r['failed_trials'] for r in rr) / (len(rr) * 3),
                   'feature_mean': float(np.mean([abs(r['net_normalized_hours']) if key == 'absolute_net_normalized_hours' else r[key] for r in rr]))}
            for name, rr in selected.items()}
    summary['command_failure_count_histogram'] = {str(i): sum(r['failed_trials'] == i for r in command_rows) for i in range(4)}
    summary['failed_trials_where_another_ev_draw_passed'] = sum(r['failed_trials'] for r in command_rows if r['passed_trials'] > 0)
    by_bmu = defaultdict(list)
    for r in known:
        by_bmu[r['source_bmu']].append(r)
    summary['by_source_bmu'] = {name: {'trials': len(rr), 'failed': sum(not r['feasible'] for r in rr),
                                     'failure_pct': 100 * sum(not r['feasible'] for r in rr) / len(rr)} for name, rr in by_bmu.items()}
    save_csv(args.output / 'days.csv', sorted(day_rows, key=lambda r: r['pass_pct']))
    save_csv(args.output / 'ev_seeds.csv', seed_rows)
    save_csv(args.output / 'commands.csv', command_rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Each seed is one EV realization tested on 256 commands, not 256 EV draws.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Meiryo'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    colors = {4: '#0072B2', 8: '#D55E00', 10: '#009E73', 12: '#CC79A7'}
    for month, color in colors.items():
        rr = [r for r in day_rows if r['month'] == month]
        axes[0].scatter([r['mean_width_kw'] for r in rr], [r['pass_pct'] for r in rr], color=color, label=f'{month}月', s=45)
        axes[1].scatter([r['max_abs_baseline_kw'] for r in rr], [r['pass_pct'] for r in rr], color=color, s=45)
        ss = [r for r in seed_rows if int(r['date'][5:7]) == month]
        axes[2].scatter([r['ev_count_delta_from_day_mean'] for r in ss], [r['pass_pct'] - day_pass[r['date']] for r in ss], color=color, s=25, alpha=.7)
    axes[0].set(xlabel='全48コマ平均の上げ＋下げ入札幅 [kW]', ylabel='未見通過率 [%]', title='入札幅と日別通過率')
    axes[1].set(xlabel='基準電力の絶対値の最大 [kW]', ylabel='未見通過率 [%]', title='基準電力と日別通過率')
    axes[2].set(xlabel='同日の3EV実現の平均台数との差 [台]', ylabel='同日の平均通過率との差 [ポイント]', title='EV台数と通過率：同日内で比較')
    axes[2].axhline(0, color='gray', linewidth=.8)
    axes[2].axvline(0, color='gray', linewidth=.8)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('AEMO 256指令入札：25日・75EV実現の保存データ集計', fontsize=14)
    fig.savefig(args.output / 'daily_features.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()

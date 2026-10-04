"""保存したMARL検証時系列から、帯外誤差と指令強度の関係を集計する。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(rows):
    failures = [r for r in rows if r['failed']]
    def quantiles(key):
        return np.quantile([r[key] for r in failures], [.25, .5, .75, .9, 1]).tolist() if failures else []
    return {
        'points': len(rows), 'failures': len(failures),
        'pass_rate_pct': 100 * (1 - len(failures) / len(rows)) if rows else None,
        'excess_kw_q25_q50_q75_q90_max': quantiles('excess_kw'),
        'error_to_band_q25_q50_q75_q90_max': quantiles('error_to_band'),
        'within_10kw_of_band': sum(r['excess_kw'] <= 10 for r in failures),
        'within_25pct_of_band': sum(r['error_to_band'] <= 1.25 for r in failures),
        'over_twice_band': sum(r['error_to_band'] > 2 for r in failures),
        'direction_underdelivery': sum(r['underdelivery'] for r in failures),
        'band_only_rescued': sum(r['band_only_pass'] for r in failures),
    }


def load_run(root):
    metadata = json.loads((root / 'diagnostic_summary.json').read_text(encoding='utf-8'))
    if not metadata['reference_matches']:
        raise ValueError(f'検証集計が元の結果と一致しない: {root}')
    rows = []
    for file in sorted(root.glob('episode_*.csv')):
        if file.stem == 'episode_soc_summary':
            continue
        with file.open(encoding='utf-8-sig', newline='') as handle:
            episode = list(csv.DictReader(handle))
        previous = None
        for source in episode:
            r = {k: (v if k == 'service_date' else float(v)) for k, v in source.items()}
            r['delta_kw'] = r['target_kw'] - r['baseline_kw']
            idle_eps = max(1e-3, 1e-6 * abs(r['baseline_kw']))
            r['direction'] = 'up' if r['delta_kw'] < -idle_eps else 'down' if r['delta_kw'] > idle_eps else 'idle'
            r['signed_error_kw'] = r['actual_kw'] - r['target_kw']
            r['abs_error_kw'] = abs(r['signed_error_kw'])
            r['excess_kw'] = max(0.0, r['abs_error_kw'] - r['tolerance_kw'])
            r['failed'] = r['excess_kw'] > 1e-4
            r['error_to_band'] = r['abs_error_kw'] / r['tolerance_kw']
            capacity = r['up_kw'] if r['direction'] == 'up' else r['down_kw']
            r['utilization'] = abs(r['delta_kw']) / capacity if r['direction'] != 'idle' and capacity > 0 else 0
            r['underdelivery'] = (r['direction'] == 'up' and r['signed_error_kw'] > 0) or (r['direction'] == 'down' and r['signed_error_kw'] < 0)
            r['band_only_tolerance_kw'] = 0.1 * (r['up_kw'] + r['down_kw'])
            r['band_only_pass'] = r['abs_error_kw'] <= r['band_only_tolerance_kw'] + 1e-4
            r['target_jump_kw'] = abs(r['target_kw'] - previous['target_kw']) if previous else 0
            r['within_previous_band'] = bool(previous and previous['tracking_enabled'] and abs(r['actual_kw'] - previous['target_kw']) <= previous['tolerance_kw'] + 1e-4)
            r['target_changed'] = bool(previous and previous['tracking_enabled'] and r['target_jump_kw'] > 1e-3)
            previous = r
            if r['tracking_enabled']:
                rows.append(r)
    summary = {'checkpoint_episode': metadata['checkpoint_episode'], 'soc_hit_pct': metadata['soc_hit'],
               'tracking_plus_soc': metadata['dispatch_rate'] + metadata['soc_hit'], 'all': summarize(rows)}
    if abs(summary['all']['pass_rate_pct'] - metadata['dispatch_rate']) > 1e-5:
        raise ValueError('時系列の帯内率が検証集計と一致しない')
    summary['by_direction'] = {direction: summarize([r for r in rows if r['direction'] == direction]) for direction in ('idle', 'up', 'down')}
    active = [r for r in rows if r['direction'] != 'idle']
    summary['by_utilization'] = {label: summarize([r for r in active if lo <= r['utilization'] < hi])
                               for label, lo, hi in [('[0,0.25)', 0, .25), ('[0.25,0.5)', .25, .5), ('[0.5,0.8)', .5, .8), ('[0.8,1]', .8, 1.001)]}
    summary['by_command_delta_kw'] = {label: summarize([r for r in active if lo <= abs(r['delta_kw']) < hi])
                                     for label, lo, hi in [('[0,100)', 0, 100), ('[100,200)', 100, 200), ('[200,400)', 200, 400), ('[400,inf)', 400, float('inf'))]}
    summary['by_day'] = {date: summarize([r for r in rows if r['service_date'] == date]) for date in sorted({r['service_date'] for r in rows})}
    failures = [r for r in rows if r['failed']]
    summary['excess_kw_bins'] = {label: sum(lo < r['excess_kw'] <= hi for r in failures)
                               for label, lo, hi in [('(0,10]', 0, 10), ('(10,25]', 10, 25), ('(25,50]', 25, 50), ('(50,100]', 50, 100), ('(100,inf)', 100, float('inf'))]}
    summary['band_only_sensitivity'] = {'fixed_targets_and_outputs': True,
                                       'rescued_failures': sum(r['band_only_pass'] for r in failures),
                                       'remaining_failures': sum(not r['band_only_pass'] for r in failures),
                                       'pass_rate_pct': 100 * sum(r['band_only_pass'] for r in rows) / len(rows)}
    summary['failures_at_changed_target'] = sum(r['target_changed'] for r in failures)
    summary['failures_within_previous_band'] = sum(r['within_previous_band'] for r in failures)
    summary['worst_failures'] = sorted(failures, key=lambda r: r['excess_kw'], reverse=True)[:12]
    return summary, rows


def draw(output, datasets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Meiryo'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for j, (name, (_, rows)) in enumerate(datasets.items()):
        ax = axes[0, j]
        for direction, label, color in [('idle', '基準電力維持', '#777777'), ('up', '上げ', '#0072B2'), ('down', '下げ', '#D55E00')]:
            selected = [r for r in rows if r['direction'] == direction]
            ax.scatter([abs(r['delta_kw']) for r in selected], [r['error_to_band'] for r in selected], s=14, color=color, alpha=.6, label=label)
        ax.axhline(1, color='black', linewidth=1, label='帯の境界')
        ax.axhline(2, color='black', linestyle=':', linewidth=.8)
        ax.set(title=f'{name}：指令量と誤差', xlabel='基準電力からの指令量の絶対値 [kW]', ylabel='絶対誤差 / 許容幅')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=.18)
        ax = axes[1, j]
        labels, near, middle, large = [], [], [], []
        for direction, label in [('idle', '基準電力維持'), ('up', '上げ'), ('down', '下げ')]:
            selected = [r for r in rows if r['direction'] == direction and r['failed']]
            labels.append(label)
            near.append(sum(r['error_to_band'] <= 1.25 for r in selected))
            middle.append(sum(1.25 < r['error_to_band'] <= 2 for r in selected))
            large.append(sum(r['error_to_band'] > 2 for r in selected))
        x = np.arange(3)
        ax.bar(x, near, label='帯の1.25倍以内', color='#56B4E9')
        ax.bar(x, middle, bottom=near, label='1.25〜2倍', color='#E69F00')
        ax.bar(x, large, bottom=np.array(near) + middle, label='2倍超', color='#D55E00')
        ax.set(xticks=x, xticklabels=labels, ylabel='失敗した5分時点の数', title=f'{name}：失敗の大きさ')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('同じ検証5日・918時点の追従失敗（蓄電池による補正なし）', fontsize=14)
    fig.savefig(output / 'failure_patterns.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), constrained_layout=True)
    dates = sorted({r['service_date'] for _, rows in datasets.values() for r in rows})
    for ax, date in zip(axes, dates):
        first = [r for r in next(iter(datasets.values()))[1] if r['service_date'] == date]
        x = [(r['step'] - 1) / 12 for r in first]
        target = [r['target_kw'] for r in first]
        band = [r['tolerance_kw'] for r in first]
        # NaNs at nonparticipating gaps prevent joining separate assessment periods.
        grid_target = np.full(288, np.nan)
        grid_band = np.full(288, np.nan)
        indices = np.asarray([int(r['step']) - 1 for r in first])
        grid_target[indices], grid_band[indices] = target, band
        clock = np.arange(288) / 12
        ax.fill_between(clock, grid_target - grid_band, grid_target + grid_band, color='#777777', alpha=.16, label='許容帯')
        ax.plot(clock, grid_target, color='black', linewidth=1, label='総電力指令')
        for (name, (_, rows)), color in zip(datasets.items(), ['#0072B2', '#D55E00']):
            selected = [r for r in rows if r['service_date'] == date]
            power = np.full(288, np.nan)
            power[[int(r['step']) - 1 for r in selected]] = [r['actual_kw'] for r in selected]
            ax.plot(clock, power, color=color, linewidth=1, label=name)
        ax.set(title=date, ylabel='総電力 [kW]', xlim=(0, 24), xticks=np.arange(0, 25, 2))
        ax.grid(alpha=.2)
    axes[0].legend(ncols=4, fontsize=9)
    axes[-1].set_xlabel('時刻 [時]')
    fig.savefig(output / 'tracking_5days.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    datasets = {label: load_run(args.root / folder) for label, folder in [('AB 1940回', 'AB1940'), ('ルール配分 860回', 'floor860')]}
    summary = {name: result[0] for name, result in datasets.items()}
    (args.root / 'failure_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, (_, rows) in datasets.items():
        folder = 'AB1940' if name.startswith('AB') else 'floor860'
        with (args.root / folder / 'failure_points.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(r for r in rows if r['failed'])
    draw(args.root, datasets)
    for name, result in summary.items():
        print(name)
        print(json.dumps({k: v for k, v in result.items() if k != 'worst_failures'}, ensure_ascii=False, indent=2))
        print('worst failures:')
        for r in result['worst_failures'][:5]:
            print({k: r[k] for k in ('service_date', 'step', 'direction', 'delta_kw', 'target_kw', 'actual_kw', 'tolerance_kw', 'excess_kw', 'error_to_band', 'utilization')})


if __name__ == '__main__':
    main()

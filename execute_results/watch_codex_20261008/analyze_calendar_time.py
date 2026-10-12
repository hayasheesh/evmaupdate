"""保存済みbest/lateの暦・時刻別集計。環境の実行は行わない。"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('calendar_only', ROOT / 'environment/calendars.py')
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
WEEKDAYS = ['月', '火', '水', '木', '金', '土', '日']


def counts(step, ev):
    active = step[:, 1] > 0
    failed = active & (np.abs(step[:, 4]) > step[:, 3])
    departed = ev[ev[:, 4] == 1].astype(np.float64)
    debt = np.maximum((departed[:, 7] - departed[:, 6]) * departed[:, 8] / 100, 0)
    return {
        'tracking_n': int(active.sum()), 'tracking_fail_n': int(failed.sum()),
        'departure_n': len(departed), 'soc_miss_n': int((debt > 0.001).sum()),
        'departure_shortfall_kwh': float(debt.sum()),
    }


def rates(c):
    c = dict(c)
    c['tracking_success_pct'] = 100 * (1 - c['tracking_fail_n'] / c['tracking_n']) if c['tracking_n'] else None
    c['soc_miss_pct'] = 100 * c['soc_miss_n'] / c['departure_n'] if c['departure_n'] else None
    c['tracking_plus_soc_success'] = c['tracking_success_pct'] + 100 - c['soc_miss_pct'] if c['tracking_n'] and c['departure_n'] else None
    return c


def sum_counts(items):
    return rates({k: sum(i[k] for i in items) for k in ['tracking_n', 'tracking_fail_n', 'departure_n', 'soc_miss_n', 'departure_shortfall_kwh']})


def time_bins(records, width):
    out = []
    total = sum_counts([r['counts'] for r in records])
    for start in range(0, 288, width):
        parts = []
        for r in records:
            s, e = r['step'], r['ev']
            parts.append(counts(s[(s[:, 0] > start) & (s[:, 0] <= start + width)], e[(e[:, 0] > start) & (e[:, 0] <= start + width)]))
        c = sum_counts(parts)
        minute = start * 5
        end = (start + width) * 5
        c.update(interval=f'{minute // 60:02d}:{minute % 60:02d}-{end // 60:02d}:{end % 60:02d}', start_step=start + 1, end_step=start + width)
        c['share_of_all_tracking_failures_pct'] = 100 * c['tracking_fail_n'] / total['tracking_fail_n'] if total['tracking_fail_n'] else None
        c['share_of_all_soc_misses_pct'] = 100 * c['soc_miss_n'] / total['soc_miss_n'] if total['soc_miss_n'] else None
        out.append(c)
    return out


def aggregate(records):
    groups = {}
    total = sum_counts([r['counts'] for r in records])
    for field in ['date', 'dayclass', 'weekday']:
        grouped = defaultdict(list)
        for r in records:
            grouped[r[field]].append(r)
        groups[field] = {}
        for name, rr in sorted(grouped.items()):
            c = sum_counts([r['counts'] for r in rr])
            c.update(case_n=len(rr), unique_date_n=len({r['date'] for r in rr}))
            c['share_of_all_tracking_failures_pct'] = 100 * c['tracking_fail_n'] / total['tracking_fail_n'] if total['tracking_fail_n'] else None
            c['share_of_all_soc_misses_pct'] = 100 * c['soc_miss_n'] / total['soc_miss_n'] if total['soc_miss_n'] else None
            c['hourly'] = time_bins(rr, 12)
            groups[field][name] = c
    return {'overall': total, 'groups': groups, 'five_minute': time_bins(records, 1), 'hourly': time_bins(records, 12), 'three_hour': time_bins(records, 36)}


def delta(best, late):
    keys = ['tracking_n', 'tracking_fail_n', 'departure_n', 'soc_miss_n', 'departure_shortfall_kwh', 'tracking_success_pct', 'soc_miss_pct', 'tracking_plus_soc_success']
    return {k: late[k] - best[k] if best[k] is not None and late[k] is not None else None for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-base', '--input-root', dest='input_base', type=Path, default=BASE,
                        help='市場フォルダの親、またはmetadata.jsonとsummary.jsonを含む単一市場フォルダ')
    parser.add_argument('--folder-prefix', default='trace_plateau20',
                        help='市場フォルダの接頭辞。フォルダ名はPREFIX_{market}_20261008')
    parser.add_argument('--folder-template', default=None,
                        help='親フォルダからの相対フォルダ名。{market}を市場名に置換')
    parser.add_argument('--markets', nargs='+', choices=['aemo', 'ercot'], default=['aemo', 'ercot'])
    parser.add_argument('--output', type=Path, default=BASE / 'calendar_time_analysis.json')
    parser.add_argument('--scope-label', default=None, help='入力範囲の説明。例: train25+validation5。集計は入力全体で行う')
    args = parser.parse_args()
    input_root = args.input_base.resolve()
    output = args.output.resolve()
    template = args.folder_template or (args.folder_prefix + '_{market}_20261008')
    folders = {market: input_root / template.format(market=market) for market in args.markets}
    if (input_root / 'metadata.json').is_file():
        market = json.loads((input_root / 'metadata.json').read_text(encoding='utf-8'))['market']
        if market not in args.markets:
            parser.error('単一市場フォルダのmarketが--marketsに含まれていません')
        folders = {market: input_root}
    for folder in folders.values():
        if output.is_relative_to(folder.resolve()):
            parser.error('--outputは入力トレースフォルダの外に指定してください')
    result = {
        'source': '[raw再集計]',
        'input_root': str(input_root),
        'input_base': str(input_root),
        'folder_template': template,
        'scope_label': args.scope_label,
        'definitions': {
            'tracking': 'tracking_enabled時刻のみ。abs(residual_kw)<=band_kwを成功。raw_actor_residualの追従であり蓄電池補償後の率ではない。',
            'soc': 'remaining_steps==1の出発EVのみ。(target-soc_after)*capacity_kwh/100>0.001kWhを実質未達。公式の厳密判定と異なる。',
            'time': 'step=1は00:00-00:05、step=288は23:55-24:00。出発はその5分区間の制御後に判定。',
            'pooling': '時刻または出発EVの母数で加重した率。case率の単純平均も別記。',
            'dayclass': 'environment/calendars.pyのday_class(date, JP)。土日または日本の祝日はholiday。',
            'delta': 'late-best。率の差はpercentage points。',
        },
        'limitations': [
            '日数、曜日数、休日数、日別の指令数・EVseed数は各市場のscopeに記録する。',
            '同日のseed反復は曜日・日付・指令の反復ではない。',
            '日付差には指令とEV構成の差が含まれるため、曜日・休日の因果効果は判別できない。',
            '時間帯の差には指令、約定による評価時間、滞在EV、出発数の差が含まれる。時刻の因果効果は判別できない。',
            '入力範囲全体の記述統計。学習日と検証日の区別はsource_metadataまたはscope_labelにある範囲情報を参照する。',
        ], 'markets': {},
    }
    calendar = {}
    for market, folder in folders.items():
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        assert meta['market'] == market
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        expected_case_n = int(meta['cases'])
        expected_dates = sorted({row['date'] for row in summary['rows'] if row['mode'] in ['best', 'late']})
        assert expected_case_n > 0 and expected_dates
        records = {'best': [], 'late': []}
        checks = []
        for row in summary['rows']:
            if row['mode'] not in records:
                continue
            with np.load(folder / f"{row['mode']}_{row['id']}.npz", allow_pickle=False) as data:
                e = data['ev'].copy()
                s = data['step'].copy()
            assert meta['ev_columns'][4:9] == ['remaining_steps', 'soc_before', 'soc_after', 'target', 'capacity_kwh']
            assert meta['step_columns'][:5] == ['step', 'tracking_enabled', 'request_kw', 'band_kw', 'residual_kw']
            assert np.array_equal(s[:, 0], np.arange(1, 289))
            d = date.fromisoformat(row['date'])
            weekday, dc = WEEKDAYS[d.weekday()], cal.day_class(d, 'JP')
            calendar[row['date']] = dict(weekday=weekday, dayclass=dc, jp_public_holiday=cal.is_public_holiday(d, 'JP'))
            c = counts(s, e)
            assert c['departure_n'] == row['depart_count'] == int(s[:, 13].sum())
            bad_eps = int(((s[:, 1] > 0) & (np.abs(s[:, 4]) > s[:, 3] + 1e-5)).sum())
            assert bad_eps == row['failed_steps']
            assert abs(rates(c)['tracking_success_pct'] - row['tracking']) < 1e-6
            checks.append(dict(id=row['id'], mode=row['mode'], strict_band_fail_n=c['tracking_fail_n'], trace_epsilon_fail_n=bad_eps, saved_official_soc_success_pct=row['soc']))
            records[row['mode']].append(dict(id=row['id'], date=row['date'], weekday=weekday, dayclass=dc, command=row['command'], ev_seed=row['ev_seed'], ev_hash=row['ev_hash'], step=s, ev=e, counts=c))
        ids = {mode: [r['id'] for r in rr] for mode, rr in records.items()}
        assert all(ii and len(ii) == len(set(ii)) for ii in ids.values()), 'best/lateの空集合または重複case'
        assert set(ids['best']) == set(ids['late']), 'best/lateのcase集合不一致'
        assert all(len(rr) == expected_case_n for rr in records.values()), 'metadataの期待ケース数と不一致'
        assert all(sorted({r['date'] for r in rr}) == expected_dates for rr in records.values()), 'summaryの期待日付集合と不一致'
        dates = sorted({r['date'] for r in records['best']})
        weekday_days = {w: sum(WEEKDAYS[date.fromisoformat(d).weekday()] == w for d in dates) for w in WEEKDAYS}
        per_date = {}
        for d in dates:
            rr = [r for r in records['best'] if r['date'] == d]
            per_date[d] = {
                'case_n_per_model': len(rr), 'command_n': len({r['command'] for r in rr}),
                'ev_seed_n': len({r['ev_seed'] for r in rr}),
                'ev_seed_n_per_command': {command: len({r['ev_seed'] for r in rr if r['command'] == command}) for command in sorted({r['command'] for r in rr})},
            }
        m = {
            'episodes': {'best': meta['best_episode'], 'late': meta['late_episode']},
            'validation': checks, 'source_directory': str(folder.resolve()), 'source_metadata': meta,
            'scope': {
                'scope_label': args.scope_label, 'date_n': len(dates), 'dates': dates,
                'expected_case_n_per_model': expected_case_n, 'expected_case_n_source': 'metadata.json:cases',
                'expected_date_n': len(expected_dates), 'expected_date_n_source': 'summary.json:rows best/late unique dates',
                'case_n_per_model': len(records['best']), 'rollout_n': sum(len(rr) for rr in records.values()),
                'weekday_date_n': weekday_days,
                'dayclass_date_n': {dc: sum(cal.day_class(d, 'JP') == dc for d in dates) for dc in ['weekday', 'holiday']},
                'jp_public_holiday_date_n': sum(cal.is_public_holiday(d, 'JP') for d in dates),
                'per_date': per_date,
                'weekday_replication': '全曜日に複数の日付がある' if all(n >= 2 for n in weekday_days.values()) else '0日または1日しかない曜日がある',
                'missing_weekdays': [w for w, n in weekday_days.items() if n == 0],
                'single_date_weekdays': [w for w, n in weekday_days.items() if n == 1],
            },
        }
        for mode, rr in records.items():
            m[mode] = aggregate(rr)
            for bins in ['five_minute', 'hourly', 'three_hour']:
                for key in ['tracking_n', 'tracking_fail_n', 'departure_n', 'soc_miss_n']:
                    assert sum(c[key] for c in m[mode][bins]) == m[mode]['overall'][key]
            m[mode]['cases'] = [{**{k: r[k] for k in ['id', 'date', 'weekday', 'dayclass', 'command', 'ev_seed', 'ev_hash']}, **rates(r['counts'])} for r in rr]
            m[mode]['case_mean_tracking_success_pct'] = float(np.mean([rates(r['counts'])['tracking_success_pct'] for r in rr]))
            m[mode]['case_mean_soc_miss_pct'] = float(np.mean([rates(r['counts'])['soc_miss_pct'] for r in rr]))
        paired = []
        late_by_id = {r['id']: r for r in records['late']}
        for b in records['best']:
            l = late_by_id[b['id']]
            assert all(b[k] == l[k] for k in ['date', 'command', 'ev_seed', 'ev_hash'])
            assert np.array_equal(b['step'][:, :4], l['step'][:, :4])
            bd = b['ev'][b['ev'][:, 4] == 1]
            ld = l['ev'][l['ev'][:, 4] == 1]
            assert np.array_equal(bd[:, [0, 1, 2, 7, 8]], ld[:, [0, 1, 2, 7, 8]])
            bm = (bd[:, 7].astype(float) - bd[:, 6]) * bd[:, 8] / 100 > 0.001
            lm = (ld[:, 7].astype(float) - ld[:, 6]) * ld[:, 8] / 100 > 0.001
            bf = (b['step'][:, 1] > 0) & (np.abs(b['step'][:, 4]) > b['step'][:, 3])
            lf = (l['step'][:, 1] > 0) & (np.abs(l['step'][:, 4]) > l['step'][:, 3])
            paired.append(dict(id=b['id'], date=b['date'], delta=delta(rates(b['counts']), rates(l['counts'])), tracking_new_fail_n=int((~bf & lf).sum()), tracking_recovered_n=int((bf & ~lf).sum()), soc_new_miss_n=int((~bm & lm).sum()), soc_recovered_n=int((bm & ~lm).sum())))
        m['late_minus_best'] = {'overall': delta(m['best']['overall'], m['late']['overall']), 'paired_cases': paired, 'groups': {}, 'hourly': []}
        for field in ['date', 'dayclass', 'weekday']:
            m['late_minus_best']['groups'][field] = {k: delta(v, m['late']['groups'][field][k]) for k, v in m['best']['groups'][field].items()}
        m['late_minus_best']['hourly'] = [{'interval': b['interval'], **delta(b, l)} for b, l in zip(m['best']['hourly'], m['late']['hourly'])]
        result['markets'][market] = m
    result['calendar'] = calendar
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: {mode: m[mode]['overall'] for mode in ['best', 'late']} for k, m in result['markets'].items()}, ensure_ascii=False, indent=2))
    print('検証完了:', output)


if __name__ == '__main__':
    main()

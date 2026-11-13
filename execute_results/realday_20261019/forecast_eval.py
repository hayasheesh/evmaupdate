"""元データで、局ごとの1日・1時間ごとの到着台数（セッション数）の前日予測を作り、外れを数える（2026-10-19）。

    python forecast_eval.py

前日の昼に予測するとして、その2日前（d−2）までの記録だけを使う。予測の作り方：
  all_mean        記録の全平日の平均（本線の到着率と同じ。未来の日も含む）
  prior_mean      d−2 までの平日の平均
  last_week       前週の同じ曜日（d−7）
  same_wd_4w      直近4週の同じ曜日（d−7、d−14、d−21、d−28 のうち記録のある日）の平均
  weekdays_4w     直近4週（d−28〜d−2）の平日の平均
対象の日は、各データ源の平日のうち、28日より前に記録があり、上のどの予測も作れる日。セッションが1件以上ある日を記録のある日とする。
物差し（局ごと）：1日の台数の WAPE（Σ|予測−実際| ÷ Σ実際）、1時間ごとの WAPE（日ごとに24時間の |予測−実際| を足し、
実際の日の合計で割り、全日で合わせる）。本番のテストの3パターンの日付の値も出す。結果は forecast_eval.json。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realday as rd  # noqa: E402

METHODS = ('all_mean', 'prior_mean', 'last_week', 'same_wd_4w', 'weekdays_4w')


def hourly_table(frame):
    out = {}
    for date, rows in frame.groupby('local_date'):
        c = np.bincount(np.clip((rows['arrival_minute'].to_numpy() // 60).astype(int), 0, 23), minlength=24).astype(float)
        out[date] = (c, str(rows['day_class'].iloc[0]))
    return out


def forecasts(table, d):
    D = dt.date.fromisoformat(d)
    wk = {k: v for k, (v, cls) in table.items() if cls == 'weekday'}
    prior = {k: v for k, v in wk.items() if dt.date.fromisoformat(k) <= D - dt.timedelta(days=2)}
    out = {'all_mean': np.mean(list(wk.values()), axis=0)}
    if len(prior) < 10:
        return None
    out['prior_mean'] = np.mean(list(prior.values()), axis=0)
    lw = str(D - dt.timedelta(days=7))
    if lw not in wk:
        return None
    out['last_week'] = wk[lw]
    same = [wk[str(D - dt.timedelta(days=7 * k))] for k in range(1, 5) if str(D - dt.timedelta(days=7 * k)) in wk]
    if len(same) < 2:
        return None
    out['same_wd_4w'] = np.mean(same, axis=0)
    w4 = [v for k, v in prior.items() if dt.date.fromisoformat(k) >= D - dt.timedelta(days=28)]
    if len(w4) < 5:
        return None
    out['weekdays_4w'] = np.mean(w4, axis=0)
    return out


def main():
    frames = rd._frames()
    days = json.loads((HERE / 'days.json').read_text(encoding='utf-8'))
    test_dates = {}
    for pat, rows in days.items():
        for r in rows:
            test_dates.setdefault(r['source'], {})[r['date']] = pat
    result = {}
    for sid in sorted(set(rd.ROSTER)):
        _info, frame = frames[sid]
        table = hourly_table(frame)
        acc = {m: {'abs_day': 0.0, 'abs_hour': 0.0, 'actual': 0.0, 'n': 0} for m in METHODS}
        per_test = {}
        for d, (actual, cls) in sorted(table.items()):
            if cls != 'weekday':
                continue
            f = forecasts(table, d)
            if f is None:
                continue
            for m in METHODS:
                acc[m]['abs_day'] += abs(f[m].sum() - actual.sum())
                acc[m]['abs_hour'] += np.abs(f[m] - actual).sum()
                acc[m]['actual'] += actual.sum()
                acc[m]['n'] += 1
            if d in test_dates.get(sid, {}):
                per_test[d] = {'pattern': test_dates[sid][d], 'actual': float(actual.sum()),
                               **{m: {'day_total': float(f[m].sum()),
                                      'day_err_pct': float(100 * (f[m].sum() - actual.sum()) / actual.sum()),
                                      'hour_wape_pct': float(100 * np.abs(f[m] - actual).sum() / actual.sum())} for m in METHODS}}
        result[sid] = {'days': acc['all_mean']['n'],
                       'methods': {m: {'day_wape_pct': 100 * a['abs_day'] / a['actual'], 'hour_wape_pct': 100 * a['abs_hour'] / a['actual']}
                                   for m, a in acc.items()},
                       'test_dates': per_test}
        print(f'== {sid}（評価した平日 {acc["all_mean"]["n"]}日）', flush=True)
        for m in METHODS:
            r = result[sid]['methods'][m]
            print(f'  {m:12s} 1日の台数 WAPE {r["day_wape_pct"]:5.1f}%   1時間ごと WAPE {r["hour_wape_pct"]:5.1f}%', flush=True)
        for d, t in sorted(per_test.items()):
            print(f'  本番のテスト {t["pattern"]:7s} {d} 実際 {t["actual"]:.0f}件  ' +
                  '  '.join(f'{m} {t[m]["day_err_pct"]:+.0f}%/{t[m]["hour_wape_pct"]:.0f}%' for m in METHODS), flush=True)
    (HERE / 'forecast_eval.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

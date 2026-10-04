"""PJM の RegD 信号から、AGC 型の比較用の指令ライブラリを2つ作る。

元データは PJM が公開している RTO Regulation Signal Data（data/input.demand_fromPJM/"MM YYYY.xlsx"、
2秒ごと、Dynamic シートが RegD）。RegD は資源の調整幅で正規化された信号（−1〜1、上げが正）なので、
固定幅は 1 とし、ほかの集合と同じく 5分点ごとの値をそのまま使う（平均しない、clip しない）。

  python tools/build_pjm_regd_library.py --mode real
      実指令（pjm_regd）。各日の 00:00, 00:05, …, 23:55 ちょうどの2秒値。夏時間の切り替え日と、
      5分点の値が欠けた日は除く。暦月ごとに月の初めから連続した 60/20/20 の日で train / validation / test に分ける。
      validation と test の日だけを、学習の途中テストと最終評価に使う。

  python tools/build_pjm_regd_library.py --mode phase_shift
      疑似指令（pjm_regd_phase_shift）。実指令の train の日だけから作る。5分点を取る時刻を 0, 10, …, 290 秒
      ずらして（1日30本）、同じく 288 点を取る。入札と学習はこちらだけを使う。
      疑似指令の中でも、入札の設計に使う指令と学習で引く指令の日が重ならないように、元の train の日を
      暦月ごとに連続した 60/20/20 で分け直す（train は入札の設計、validation と test は学習で引く指令）。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from market.command_waveforms import month_balanced_partitions, waveform_to_activation_proxy  # noqa: E402

SAMPLES_PER_DAY = 43200      # 2秒 × 43200 = 24時間
SAMPLES_PER_STEP = 150       # 5分 = 2秒 × 150
OFFSET_STEP_SAMPLES = 5      # 疑似指令のずらし幅 10秒（2秒 × 5）
DST_CHANGE_DAYS = {date(2024, 3, 10), date(2024, 11, 3)}
SOURCE = 'https://www.pjm.com/-/media/DotCom/markets-ops/ancillary/rto-regulation-signal-data.ashx'
DEFAULT_OUTPUT = {
    'real': ROOT / 'data/pjm/command_libraries/regd_calendar_day',
    'phase_shift': ROOT / 'data/pjm/command_libraries/regd_phase_shift_train_calendar_day',
}


def number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def read_month(path: Path) -> dict[date, np.ndarray]:
    """Dynamic シートを読み、日ごとに 43200 個の2秒値を返す（欠けた値は nan）。"""
    month, year = path.stem.split()
    first = date(int(year), int(month), 1)
    workbook = openpyxl.load_workbook(path, read_only=True)
    rows = workbook['Dynamic'].iter_rows(values_only=True)
    header = next(rows)
    days = sum(1 for cell in header[1:] if cell is not None)
    if not isinstance(header[1], str):
        assert pd.Timestamp(header[1]).date() == first, (path, header[1])
    values = []
    for row in rows:
        if len(values) == SAMPLES_PER_DAY:
            break
        values.append([number(v) for v in row[1:1 + days]])
    signal = np.asarray(values, dtype=float)
    assert signal.shape == (SAMPLES_PER_DAY, days), (path, signal.shape)
    return {first + timedelta(days=d): signal[:, d] for d in range(days)}


def real_days(signals: dict[date, np.ndarray]) -> tuple[list[date], dict, Counter]:
    """実指令に使う日（5分点がすべてそろった、夏時間の切り替え日でない日）と、その区分。"""
    reasons = Counter()
    kept = []
    for day in sorted(signals):
        if day in DST_CHANGE_DAYS:
            reasons['dst_change_day'] += 1
        elif not np.isfinite(signals[day][::SAMPLES_PER_STEP]).all():
            reasons['missing_five_minute_point'] += 1
        else:
            kept.append(day)
    return kept, month_balanced_partitions(kept), reasons


def write_day(out: Path, *, day: date, points: np.ndarray, source_type: str, name: str, resource: str,
              partition: str, segment: str) -> int:
    times = [datetime(day.year, day.month, day.day) + timedelta(minutes=5 * k) for k in range(288)]
    waveform = pd.DataFrame({
        'market': 'PJM', 'resource_kind': 'REGD', 'resource_name': resource,
        'source_date': day.isoformat(), 'step': np.arange(288),
        'source_time_local': [t.isoformat() for t in times],
        'target_mw': points, 'direction_sign': 1,
    })
    activation = waveform_to_activation_proxy(
        waveform, reference_mode='zero', source_type=source_type,
        scenario_partition=partition, segment_id=segment, scale_mw=1.0,
    )
    activation['source_bmu'] = resource
    activation.to_csv(out / name, index=False, float_format='%.12g')
    return int(np.sum(np.abs(activation['signed_activation_up_positive_raw']) > 1.0))


def date_ranges(days, partitions) -> dict:
    ranges: dict[str, dict[str, list[str]]] = {}
    for d in days:
        month = ranges.setdefault(d.strftime('%Y-%m'), {})
        span = month.setdefault(partitions[d], [d.isoformat(), d.isoformat()])
        span[1] = d.isoformat()
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('real', 'phase_shift'), required=True)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'data/input.demand_fromPJM')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir or DEFAULT_OUTPUT[args.mode]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty output: {out}')

    signals: dict[date, np.ndarray] = {}
    for path in sorted(args.source_dir.glob('* ????.xlsx')):
        signals.update(read_month(path))
        print(f'[read] {path.name}', flush=True)
    kept, partitions, reasons = real_days(signals)
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    beyond = 0
    skipped_offsets = 0

    if args.mode == 'real':
        source_type = 'pjm_regd'
        for day in kept:
            beyond += write_day(out, day=day, points=signals[day][::SAMPLES_PER_STEP], source_type=source_type,
                                name=f'pjm_regd_{day.isoformat()}.csv', resource='PJM_RegD',
                                partition=partitions[day], segment=f'{source_type}:{day.isoformat()}')
            counts[partitions[day]] += 1
        days_used, parts_used = kept, partitions
        extra = {
            'sampling': 'value of the 2-second signal at each five-minute point of the local calendar day; DST change days excluded',
            'use': 'validation and test days only for interim tests and the final evaluation; train days feed pjm_regd_phase_shift',
        }
    else:
        source_type = 'pjm_regd_phase_shift'
        train_days = [d for d in kept if partitions[d] == 'train']
        parts_used = month_balanced_partitions(train_days)
        days_used = train_days
        offsets = list(range(0, SAMPLES_PER_STEP, OFFSET_STEP_SAMPLES))
        for day in train_days:
            for offset in offsets:
                points = signals[day][offset::SAMPLES_PER_STEP][:288]
                if points.size != 288 or not np.isfinite(points).all():
                    skipped_offsets += 1
                    continue
                seconds = 2 * offset
                beyond += write_day(out, day=day, points=points, source_type=source_type,
                                    name=f'pjm_regd_ps{seconds:03d}s_{day.isoformat()}.csv',
                                    resource=f'PJM_RegD_offset{seconds:03d}s', partition=parts_used[day],
                                    segment=f'{source_type}:{day.isoformat()}:{seconds:03d}')
                counts[parts_used[day]] += 1
        extra = {
            'sampling': 'value of the 2-second signal at 00:00+s, 00:05+s, ... for offsets s = 0, 10, ..., 290 seconds',
            'source_days': 'train days of pjm_regd only; validation and test days of pjm_regd are never read',
            'partition_note': 'source train days re-split by calendar month into consecutive 60/20/20 blocks: '
                              'train = bid design commands, validation and test = commands the lower controller draws',
            'offset_seconds': [2 * o for o in offsets],
            'skipped_series_with_missing_points': skipped_offsets,
        }
    metadata = {
        'build_complete': True,
        'bank_ready': all(counts[p] >= 128 for p in ('train', 'validation', 'test')),
        'regime': source_type,
        'source_kind': 'PJM RTO regulation signal RegD (Dynamic), normalized to each resource regulation capability',
        'source': SOURCE,
        'plan': 'none; the signal is the deviation from the resource regulation midpoint',
        'reference': 'zero',
        'normalization': 'fixed width 1 (the signal is already in units of the resource regulation capability)',
        'clipping': False,
        'interpolation_or_fill': 'none; series with a missing five-minute point are excluded',
        'partition_policy': 'whole local calendar dates; within each calendar month consecutive 60/20/20 blocks',
        'partition_date_ranges': date_ranges(days_used, parts_used),
        'files': dict(counts),
        'excluded_days': dict(reasons),
        'steps_beyond_width': beyond,
        **extra,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'mode': args.mode, 'files': dict(counts), 'excluded_days': dict(reasons),
                      'skipped': skipped_offsets, 'steps_beyond_width': beyond}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

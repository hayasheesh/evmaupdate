"""IESO（オンタリオ）の調整力信号から、AGC 型の比較用の指令ライブラリ（ieso_regulation）を作る。

元データは IESO が公開している normalised regulation signal（data/ieso/raw/ の zip、
2025-01-01〜2026-06-30、1秒ごと、東部標準時）。信号は系統全体の調整力の合計容量で正規化され、
−1（最大の下げ）〜 +1（最大の上げ）。固定幅は 1 とし、ほかの集合と同じく 5分点ごとの値をそのまま使う
（平均しない、clip しない）。

  python tools/build_ieso_regulation_library.py

1. 各日（東部標準時の暦日）の 00:00, 00:05, …, 23:55 で有効な値（その時刻以前で最後の1秒値）を取る。
   その値が5分点から10秒より前のものなら、その5分点は欠けたものとして、その日を除く。
2. 東部標準時は夏時間で動かないので、23時間・25時間の日はない。
3. 暦月ごとに、月の初めから連続した 60/20/20 の日で train / validation / test に分ける。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from market.command_waveforms import month_balanced_partitions, waveform_to_activation_proxy  # noqa: E402

SOURCE_TYPE = 'ieso_regulation'
ZIP_NAME = 'regulation-signal-normalised-Jan1-2025-Jun30-2026.zip'
MAX_AGE_S = 10


def read_signal(zip_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """秒単位の時刻（1970年からの秒、東部標準時の壁時計）と信号を返す。"""
    times, values = [], []
    with zipfile.ZipFile(zip_path) as archive:
        name = archive.infolist()[0].filename
        with archive.open(name) as handle:
            for chunk in pd.read_csv(handle, chunksize=5_000_000, header=0, names=['timestamp', 'signal'],
                                     dtype={'timestamp': str, 'signal': np.float64}):
                stamp = pd.to_datetime(chunk['timestamp'], format='%Y-%m-%d %H:%M:%S')
                times.append(stamp.to_numpy('datetime64[s]').astype(np.int64))
                values.append(chunk['signal'].to_numpy(np.float64))
                print(f'[read] {stamp.iloc[-1]}', flush=True)
    t = np.concatenate(times)
    v = np.concatenate(values)
    order = np.argsort(t, kind='stable')
    return t[order], v[order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, default=ROOT / 'data/ieso/raw' / ZIP_NAME)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/ieso/command_libraries/regulation_calendar_day')
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty output: {out}')

    t, v = read_signal(args.zip)
    duplicates = int(np.sum(np.diff(t) == 0))
    first = datetime.utcfromtimestamp(int(t[0])).date()
    last = datetime.utcfromtimestamp(int(t[-1])).date()
    reasons = Counter()
    points: dict[date, np.ndarray] = {}
    day = first
    while day <= last:
        start = int((np.datetime64(day.isoformat(), 's')).astype(np.int64))
        grid = start + 300 * np.arange(288)
        idx = np.searchsorted(t, grid, side='right') - 1
        ok = (idx >= 0) & (t[np.clip(idx, 0, None)] >= grid - MAX_AGE_S)
        values = np.where(ok, v[np.clip(idx, 0, None)], np.nan)
        if not ok.all():
            reasons['missing_five_minute_point'] += 1
        elif not np.isfinite(values).all():
            reasons['non_finite_value'] += 1
        else:
            points[day] = values
        day += timedelta(days=1)
    kept = sorted(points)
    partitions = month_balanced_partitions(kept)

    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    beyond = 0
    for day in kept:
        times = [datetime(day.year, day.month, day.day) + timedelta(minutes=5 * k) for k in range(288)]
        waveform = pd.DataFrame({
            'market': 'IESO', 'resource_kind': 'REGULATION', 'resource_name': 'IESO_regulation',
            'source_date': day.isoformat(), 'step': np.arange(288),
            'source_time_local': [x.isoformat() for x in times],
            'target_mw': points[day], 'direction_sign': 1,
        })
        activation = waveform_to_activation_proxy(
            waveform, reference_mode='zero', source_type=SOURCE_TYPE,
            scenario_partition=partitions[day], segment_id=f'{SOURCE_TYPE}:{day.isoformat()}', scale_mw=1.0,
        )
        activation['source_bmu'] = 'IESO_regulation'
        activation.to_csv(out / f'ieso_regulation_{day.isoformat()}.csv', index=False, float_format='%.12g')
        counts[partitions[day]] += 1
        beyond += int(np.sum(np.abs(activation['signed_activation_up_positive_raw']) > 1.0))
    ranges: dict[str, dict[str, list[str]]] = {}
    for d in kept:
        month = ranges.setdefault(d.strftime('%Y-%m'), {})
        span = month.setdefault(partitions[d], [d.isoformat(), d.isoformat()])
        span[1] = d.isoformat()
    metadata = {
        'build_complete': True,
        'bank_ready': all(counts[p] >= 128 for p in ('train', 'validation', 'test')),
        'regime': SOURCE_TYPE,
        'source_kind': 'IESO aggregated frequency regulation signal, normalised to the total regulation capacity provided in real time',
        'source': 'https://www.ieso.ca/-/media/Files/IESO/Document-Library/ancillary-services/' + ZIP_NAME,
        'plan': 'none; the signal is the regulation request relative to the energy schedule (basepoint)',
        'sampling': f'value in force at each five-minute point of the Eastern Standard Time calendar day (last one-second value no older than {MAX_AGE_S} s)',
        'reference': 'zero',
        'normalization': 'fixed width 1 (the signal is already normalised to -1..+1, +1 is maximum upward regulation)',
        'clipping': False,
        'interpolation_or_fill': 'none; days with a missing five-minute point are excluded',
        'partition_policy': 'whole EST calendar dates; within each calendar month consecutive 60/20/20 blocks',
        'partition_date_ranges': ranges,
        'files': dict(counts),
        'excluded_days': dict(reasons),
        'duplicate_timestamps': duplicates,
        'steps_beyond_width': beyond,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': dict(counts), 'excluded_days': dict(reasons), 'duplicates': duplicates,
                      'steps_beyond_width': beyond}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""評価に使う PJM の指令（5分点の疑似指令）を、元の2秒ごとの RegD に戻す（2026-10-17）。

疑似指令 pjm_regd_phase_shift は、ある日の2秒ごとの RegD を、00:00 から offset 秒ずらした5分点で取り出したもの
（tools/build_pjm_regd_library.py）。指令の source（例 pjm_regd_ps070s_2024-03-01.csv）から日と offset を読み、
その5分点から次の5分点までの2秒ごとの値（150個）を5分ごとに並べて返す。最後の5分が日をまたぐときは翌日の値を使う。
元データは data/input.demand_fromPJM/"MM YYYY.xlsx"（2秒ごと、Dynamic シート）。月ごとに読んだ値は cache/ に保存する。
"""
from __future__ import annotations

from datetime import date, timedelta
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / 'data' / 'input.demand_fromPJM'
CACHE = HERE / 'cache'
SAMPLES_PER_STEP = 150
STEPS = 288
_MONTHS: dict[tuple[int, int], dict[date, np.ndarray]] = {}


def _month(year: int, month: int) -> dict[date, np.ndarray]:
    key = (year, month)
    if key in _MONTHS:
        return _MONTHS[key]
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f'regd_{year:04d}_{month:02d}.npz'
    if cached.exists():
        data = np.load(cached)
        days = {date.fromisoformat(k): data[k] for k in data.files}
    else:
        sys.path.insert(0, str(ROOT))
        from tools.build_pjm_regd_library import read_month
        days = read_month(SOURCE_DIR / f'{month:02d} {year:04d}.xlsx')
        np.savez_compressed(cached, **{d.isoformat(): v for d, v in days.items()})
    _MONTHS[key] = days
    return days


def day_signal(day: date) -> np.ndarray | None:
    try:
        return _month(day.year, day.month).get(day)
    except FileNotFoundError:
        return None


def parse_source(source: str) -> tuple[date, int]:
    """指令の source から（日、ずらした秒）を読む。"""
    m = re.search(r'pjm_regd_ps(\d{3})s_(\d{4}-\d{2}-\d{2})\.csv', source)
    if not m:
        raise ValueError(f'not a phase-shifted RegD command: {source}')
    return date.fromisoformat(m.group(2)), int(m.group(1))


def seconds_signal(source: str) -> np.ndarray:
    """形 (288, 150) の2秒ごとの RegD。[t, 0] が指令の5分点 t の値になる。"""
    day, seconds = parse_source(source)
    offset = seconds // 2
    today = day_signal(day)
    if today is None:
        raise FileNotFoundError(day)
    nxt = day_signal(day + timedelta(days=1))
    tail = nxt[:offset + 1] if nxt is not None else np.full(offset + 1, today[-1])
    series = np.concatenate([today, tail])[offset:offset + STEPS * SAMPLES_PER_STEP]
    series = series.reshape(STEPS, SAMPLES_PER_STEP).astype(float)
    # 欠けた2秒値は、直前の値で埋める（5分点はライブラリを作るときに欠けがないことを確かめてある）
    flat = series.reshape(-1)
    for i in np.flatnonzero(~np.isfinite(flat)):
        flat[i] = flat[i - 1] if i else 0.0
    return flat.reshape(STEPS, SAMPLES_PER_STEP)

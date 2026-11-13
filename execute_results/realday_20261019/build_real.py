"""実データの1日（パターンごと）の本当の状況、誤差率、その日の入札、finetune 用の束を作る（Windows、2026-10-19）。

    python build_real.py <busy|quiet|average> [--workers 10]

1. 本当の状況（realday.real_realization）を real_<pattern>.pkl に書く。
2. その日の世界（入札と同じ確率 + 前の2日の確率で0時のEV）の候補128通り（本線の候補と同じ種）を作る。
3. 誤差率（5分ごとの接続台数、6局の合計）：その日の世界の候補の平均と本当の状況、本線の世界の候補の平均と本当の状況。
   接続台数は error_sources_20261019/fastcount.connected_fast で数える。
4. その日の入札：候補から本線と同じ下2上1で3通りを選び、本線と同じ入札器で1本作る。入札の到着の確率はその日のもの
   （学習の reset はこの確率でくじを引く）。prev_<pattern>.pkl に、0時のEVを流す前の2日の確率を書く。
5. 束：本線の学習用と検証用の目録を写し、どの日の項目にもその日の入札を置く（realday_<pattern>_aemo_*）。
結果の要約は build_<pattern>.json。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'error_sources_20261019'))
import realday as rd  # noqa: E402

world = rd.world


def bank_name(pattern: str, split: str) -> str:
    name = {'train': 'train_25', 'validation': 'validation_5'}[split]
    return f'realday_{pattern}_aemo_{name}_6station_128cmd_3ev_aemo_plan_deviation'


def base_pattern(pattern: str) -> str:
    """fc_<日>（前日の昼の単純な予測で作る入札と学習の世界）と ml_<日>（機械学習の水準の予測、realday.ALPHA_ML）の、
    本当の状況のパターン。"""
    return pattern[3:] if pattern.startswith(('fc_', 'ml_')) else pattern


def load_pattern(pattern):
    return rd.load_days(base_pattern(pattern))


def profiles(pattern):
    """入札と学習の世界の到着の確率と前の2日。fc_<日> は前日の昼の予測（realday.forecast_profiles）、ほかはその日の実際の件数。"""
    days = load_pattern(pattern)
    if pattern.startswith(('fc_', 'ml_')):
        probs, prev, used = rd.forecast_profiles(days, alpha=rd.ALPHA_ML if pattern.startswith('ml_') else 0.0)
        return probs, prev, used
    probs, prev = rd.day_profiles(days)
    return probs, prev, None


def _cands(args):
    """候補 seeds の分。world_kind が 'day' ならその日の世界、'main' なら本線の世界。
    返すのは pickle の bytes（EVSpec の一覧）と接続台数の配列。"""
    pattern, world_kind, seeds, want_specs = args
    import fastcount as fc
    world.configure(rd.TASK)
    base = rd.target_base()
    date = base.get('service_date')
    if world_kind == 'day':
        probs, prev, _used = profiles(pattern)
        orig = rd.install_preroll(prev)
    else:
        probs = base['arrival_probabilities_by_station']
        orig = None
    try:
        conn, specs = [], []
        for s in seeds:
            env, events, initial = fc.realization(s, probs, date)
            conn.append(fc.connected_fast(events, initial, slots=int(env.max_ev_per_station)).tolist())
            if want_specs:
                specs.append(world.capture_seeded(s, probs, date))
    finally:
        if orig is not None:
            from environment.EVEnv import EVEnv
            EVEnv._preroll_initial_evs = orig
    return conn, pickle.dumps(specs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern', choices=rd.ALL_PATTERNS + tuple(f'{k}_{p}' for k in ('fc', 'ml') for p in rd.ALL_PATTERNS))
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    import multiprocessing as mp
    import numpy as np
    import fastcount as fc
    pattern = args.pattern
    world.configure(rd.TASK)
    base = rd.target_base()
    fs, date = int(base['forecast_seed']), base.get('service_date')
    days = load_pattern(pattern)
    probs, prev, used = profiles(pattern)
    (HERE / f'prev_{pattern}.pkl').write_bytes(pickle.dumps(prev))
    # 1. 本当の状況（fc_<日> と ml_<日> は、同じ日の本当の状況があればそのまま使う。なければ作る。中身は確率によらない）
    env, _e, _i = fc.realization(fs, probs, date)
    real_path = HERE / f'real_{base_pattern(pattern)}.pkl'
    if pattern.startswith(('fc_', 'ml_')) and real_path.exists():
        events, initial = pickle.loads(real_path.read_bytes())
    else:
        events, initial = rd.real_realization(env, days)
        real_path.write_bytes(pickle.dumps((events, initial)))
    real_conn = fc.connected_fast(events, initial, slots=int(env.max_ev_per_station))
    n_arr = [sum(1 for t in events for ev in t[st] if ev is not None) for st in range(len(events[0]))]
    n_ini = [len(x) for x in initial]
    # 2. 候補
    seeds = [fs + 10007 * i for i in range(128)]
    chunks = [seeds[k:k + 8] for k in range(0, 128, 8)]
    mp.set_start_method('spawn', force=True)
    with mp.Pool(args.workers, maxtasksperchild=1) as pool:
        day_parts = pool.map(_cands, [(pattern, 'day', c, True) for c in chunks], chunksize=1)
        main_parts = pool.map(_cands, [(pattern, 'main', c, False) for c in chunks], chunksize=1)
    day_conn = np.array([c for part in day_parts for c in part[0]])
    main_conn = np.array([c for part in main_parts for c in part[0]])
    cands = [c for part in day_parts for c in pickle.loads(part[1])]
    assert len(cands) == 128
    # 3. 誤差率
    def wape(pred):
        return float(100 * np.abs(pred - real_conn).sum() / real_conn.sum())
    vol = day_conn.sum(axis=1)
    lo, hi = np.percentile(vol, [5, 95])
    summary = {'pattern': pattern, 'days': days, 'forecast_used': used, 'real_arrivals_by_station': n_arr, 'real_midnight_by_station': n_ini,
               'real_connected_volume': float(real_conn.sum()),
               'day_forecast_volume': float(vol.mean()), 'main_forecast_volume': float(main_conn.sum(axis=1).mean()),
               'wape_day_forecast_vs_real': wape(day_conn.mean(axis=0)),
               'wape_main_forecast_vs_real': wape(main_conn.mean(axis=0)),
               'day_forecast_p5_pct': float(100 * (lo / vol.mean() - 1)), 'day_forecast_p95_pct': float(100 * (hi / vol.mean() - 1)),
               'real_vs_day_forecast_volume_pct': float(100 * (real_conn.sum() / vol.mean() - 1))}
    print(json.dumps({k: v for k, v in summary.items() if k != 'days'}, ensure_ascii=False), flush=True)
    # 4. 入札
    os.environ['EVMA_BID_SOLVE_CACHE'] = '0'
    from training import lower_bid_training as lbt
    from training.bid_bank import save_fixed_bid
    evs, meta = world.select_low2max(cands)
    for x in meta:
        x['realday'] = {'pattern': pattern}
    lbt._selected_ev_scenario_bank = lambda **k: (evs, meta, None)
    orig = rd.install_preroll(prev)
    try:
        fixed = lbt.build_fixed_upper_bid_for_day(
            np.asarray(base['base_series'], float), date,
            arrival_scenario=types.SimpleNamespace(arrival_probabilities_by_station=probs),
            forecast_seed=fs, assessment_band_fraction=float(base['assessment_band_fraction']), scenario_workers=3)
    finally:
        from environment.EVEnv import EVEnv
        EVEnv._preroll_initial_evs = orig
    bid = dict(base)
    bid.update(fixed)
    bid['arrival_probabilities_by_station'] = probs
    bid['realday'] = {'pattern': pattern, 'days': days}
    (HERE / f'bid_{pattern}.pkl').write_bytes(pickle.dumps(bid))
    summary['capacity_kw_block'] = float(np.sum(bid['up_plan']) + np.sum(bid['down_plan']))
    summary['main_capacity_kw_block'] = float(np.sum(base['up_plan']) + np.sum(base['down_plan']))
    summary['picks'] = [int(x['candidate_index']) for x in meta]
    # 5. 束
    for split in ('train', 'validation'):
        src = world.main_bank(rd.TASK, split)
        dst = world.BANKS / bank_name(pattern, split)
        if dst.exists():
            raise FileExistsError(dst)
        dst.mkdir(parents=True)
        shutil.copy2(src / 'manifest.json', dst / 'manifest.json')
        manifest = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['entries']:
            save_fixed_bid(dst / Path(str(entry['bid_path']).replace('\\', '/')), bid)
        (dst / 'REALDAY.json').write_text(json.dumps({'pattern': pattern, 'days': days}, ensure_ascii=False), encoding='utf-8')
    (HERE / f'build_{pattern}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('capacity_kw_block', 'main_capacity_kw_block', 'picks')}), flush=True)


if __name__ == '__main__':
    main()

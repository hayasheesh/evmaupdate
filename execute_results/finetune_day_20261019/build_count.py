"""その日の到着台数を中心にした世界（countworld）の入札と、finetune 用の束を作る（Windows、2026-10-19）。

    python build_count.py

作り方は build_day.py と同じ。候補128通りを countworld.candidates と同じ種で、この世界（k は K）から16並列で作り、本線と同じ下2上1で3通りを選び、
本線と同じ入札器で入札を1本作って day_bid_count.pkl に書く。束（count_bank_name）は本線の学習用と検証用の目録を写し、
どの日の項目にもその入札を置く。
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import countworld as cw  # noqa: E402

dw = cw.dw
K = 1.0
BID = HERE / 'day_bid_count.pkl'


def count_bank_name(split: str) -> str:
    name = {'train': 'train_25', 'validation': 'validation_5'}[split]
    return f'ftcount_aemo20240201_{name}_6station_128cmd_3ev_aemo_plan_deviation'


def _cands(chunk):
    """候補の8通り分（countworld.candidates と同じ種）。親へは pickle の bytes で返す。"""
    dw.world.configure(dw.TASK)
    base = dw.target_base()
    n_day = cw.day_counts(dw.center(base))
    fs, probs, date = int(base['forecast_seed']), base['arrival_probabilities_by_station'], base.get('service_date')
    with cw.count_world(n_day, K):
        return pickle.dumps([dw.world.capture_seeded(fs + 10007 * i, probs, date) for i in range(8 * chunk, 8 * chunk + 8)])


def main():
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    with mp.Pool(16, maxtasksperchild=1) as pool:
        parts = pool.map(_cands, range(16), chunksize=1)
    dw.world.configure(dw.TASK)
    os.environ['EVMA_BID_SOLVE_CACHE'] = '0'
    import numpy as np
    from training import lower_bid_training as lbt
    from training.bid_bank import save_fixed_bid
    base = dw.target_base()
    cands = [c for part in parts for c in pickle.loads(part)]
    assert len(cands) == 128
    evs, meta = dw.world.select_low2max(cands)
    for x in meta:
        x['count_world'] = dict(k=K, center_j=dw.CENTER_J, day_index=dw.DAY_INDEX)
    lbt._selected_ev_scenario_bank = lambda **k: (evs, meta, None)
    fixed = lbt.build_fixed_upper_bid_for_day(
        np.asarray(base['base_series'], float), base.get('service_date'),
        arrival_scenario=types.SimpleNamespace(arrival_probabilities_by_station=base['arrival_probabilities_by_station']),
        forecast_seed=int(base['forecast_seed']), assessment_band_fraction=float(base['assessment_band_fraction']),
        scenario_workers=3)
    bid = dict(base)
    bid.update(fixed)
    BID.write_bytes(pickle.dumps(bid))
    cap = float(np.sum(bid['up_plan']) + np.sum(bid['down_plan']))
    main_cap = float(np.sum(base['up_plan']) + np.sum(base['down_plan']))
    info = {'service_date': str(bid.get('service_date')), 'capacity_kw_block': cap, 'main_capacity_kw_block': main_cap,
            'picks': [int(x['candidate_index']) for x in meta], 'k': K}
    (HERE / 'day_bid_count.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False), flush=True)
    for split in ('train', 'validation'):
        src = dw.world.main_bank(dw.TASK, split)
        dst = dw.world.BANKS / count_bank_name(split)
        if dst.exists():
            raise FileExistsError(dst)
        dst.mkdir(parents=True)
        shutil.copy2(src / 'manifest.json', dst / 'manifest.json')
        manifest = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['entries']:
            save_fixed_bid(dst / Path(str(entry['bid_path']).replace('\\', '/')), bid)
        (dst / 'FTCOUNT.json').write_text(json.dumps({'day_bid': str(BID), **info}, ensure_ascii=False), encoding='utf-8')
        print('bank', dst.name, len(manifest['entries']), 'entries', flush=True)


if __name__ == '__main__':
    main()

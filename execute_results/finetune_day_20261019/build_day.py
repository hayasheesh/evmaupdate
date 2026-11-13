"""その日の世界の入札と、finetune 用の束を作る（Windows、2026-10-19）。

    python build_day.py

1. dayworld の世界（kappa、tau は CHOSEN）から入札の候補128通りを作り、本線と同じ下2上1で3通りを選び、本線と同じ入札器
   （training.lower_bid_training.build_fixed_upper_bid_for_day、指令128本、日の中の並列3）で、その日の入札を1本作る。
   入札の解の保存は使わない。本線の検証用の束のその日の項目に、作った入札を重ねて day_bid.pkl に書く。
2. finetune 用の束：本線の学習用（25日）と検証用（5日）の束の目録をそのまま写し、どの日の項目にも day_bid.pkl を置く
   （学習は、どの日付を引いてもその日の入札で行う）。目録の条件は本線と同じなので、学習の照合を通る。
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
import dayworld as dw  # noqa: E402

CHOSEN = {'kappa': 2.0, 'tau': 0.0}
BANKS = dw.world.BANKS


def fake_bank_name(split: str) -> str:
    name = {'train': 'train_25', 'validation': 'validation_5'}[split]
    return f'ftday_aemo20240201_{name}_6station_128cmd_3ev_aemo_plan_deviation'


def main():
    dw.world.configure(dw.TASK)
    os.environ['EVMA_BID_SOLVE_CACHE'] = '0'
    import numpy as np
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank, save_fixed_bid
    base = dw.target_base()
    D = dw.center(base)
    cands = dw.candidates(base, D, CHOSEN['kappa'], CHOSEN['tau'])
    evs, meta = dw.world.select_low2max(cands)
    for x in meta:
        x['day_world'] = dict(CHOSEN, center_j=dw.CENTER_J, day_index=dw.DAY_INDEX)
    lbt._selected_ev_scenario_bank = lambda **k: (evs, meta, None)
    fixed = lbt.build_fixed_upper_bid_for_day(
        np.asarray(base['base_series'], float), base.get('service_date'),
        arrival_scenario=types.SimpleNamespace(arrival_probabilities_by_station=base['arrival_probabilities_by_station']),
        forecast_seed=int(base['forecast_seed']), assessment_band_fraction=float(base['assessment_band_fraction']),
        scenario_workers=3)
    bid = dict(base)
    bid.update(fixed)
    (HERE / 'day_bid.pkl').write_bytes(pickle.dumps(bid))
    cap = float(np.sum(bid['up_plan']) + np.sum(bid['down_plan']))
    main_cap = float(np.sum(base['up_plan']) + np.sum(base['down_plan']))
    info = {'service_date': str(bid.get('service_date')), 'capacity_kw_block': cap, 'main_capacity_kw_block': main_cap,
            'picks': [int(x['candidate_index']) for x in meta], **CHOSEN}
    (HERE / 'day_bid.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False), flush=True)
    for split in ('train', 'validation'):
        src = dw.world.main_bank(dw.TASK, split)
        dst = BANKS / fake_bank_name(split)
        if dst.exists():
            raise FileExistsError(dst)
        dst.mkdir(parents=True)
        shutil.copy2(src / 'manifest.json', dst / 'manifest.json')
        manifest = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['entries']:
            save_fixed_bid(dst / Path(str(entry['bid_path']).replace('\\', '/')), bid)
        (dst / 'FTDAY.json').write_text(json.dumps({'day_bid': str(HERE / 'day_bid.pkl'), **info}, ensure_ascii=False), encoding='utf-8')
        print('bank', dst.name, len(manifest['entries']), 'entries', flush=True)


if __name__ == '__main__':
    main()

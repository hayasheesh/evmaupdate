"""いろいろな実データの日で一から学習するための束（学習用25日・中間テスト用5日）を作る（研究室、2026-10-19）。

    python build_multi.py [--parallel 6]

1. realday.choose_multi_days で組んだ日を30日選ぶ（局ごとに平日からランダム、本番のテストの日付は除く）。multi_days.json に書く。
2. 組んだ日 k ごとに、本線の束の同じ番号の項目（学習用は0〜24、中間テスト用は0〜4）を元にし（指令と種と日付はその項目のもの）、
   その日の件数から作った到着の確率と前の2日の確率で候補128通りを引き、本線と同じ下2上1と入札器で入札を作る（build_real.py と同じ）。
3. 束 realday_multi_aemo_{train_25,validation_5} を作る（目録は本線の束から写し、項目ごとにその日の入札を置く）。
   0時のEVを流すための前の2日は、到着の確率の鍵（realday.probs_key）ごとに prev_multi.pkl に書く。
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
import realday as rd  # noqa: E402

world = rd.world
OUT = HERE / 'multi_bids'


def bank_name(split: str) -> str:
    name = {'train': 'train_25', 'validation': 'validation_5'}[split]
    return f'realday_multi_aemo_{name}_6station_128cmd_3ev_aemo_plan_deviation'


def _build(args):
    split, index, day = args
    out = OUT / f'{split}_{index:02d}.pkl'
    if out.exists():
        return str(out)
    world.configure(rd.TASK)
    os.environ['EVMA_BID_SOLVE_CACHE'] = '0'
    import numpy as np
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank
    bank = BidBank(world.main_bank(rd.TASK, split))
    base = bank.load_entry(bank.entries[index])
    fs, date = int(base['forecast_seed']), base.get('service_date')
    probs, prev = rd.day_profiles(day)
    orig = rd.install_preroll(prev)
    try:
        cands = [world.capture_seeded(fs + 10007 * i, probs, date) for i in range(128)]
        evs, meta = world.select_low2max(cands)
        for x in meta:
            x['realday_multi'] = {'split': split, 'index': index}
        lbt._selected_ev_scenario_bank = lambda **k: (evs, meta, None)
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
    bid['realday'] = {'multi': split, 'index': index, 'days': day}
    OUT.mkdir(exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_bytes(pickle.dumps((bid, prev)))
    tmp.replace(out)
    print(f'{split} {index}: {float(np.sum(bid["up_plan"]) + np.sum(bid["down_plan"])):,.0f} kW', flush=True)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parallel', type=int, default=6)
    ap.add_argument('--one', nargs=2, metavar=('SPLIT', 'INDEX'))
    args = ap.parse_args()
    path = HERE / 'multi_days.json'
    if path.exists():
        days = json.loads(path.read_text(encoding='utf-8'))
    else:
        days = rd.choose_multi_days(30)
        path.write_text(json.dumps(days, ensure_ascii=False, indent=1), encoding='utf-8')
    if args.one:
        split, index = args.one[0], int(args.one[1])
        _build((split, index, days[index] if split == 'train' else days[25 + index]))
        return
    # 入札器が日の中で子プロセスを作るので、日ごとに別のプロセスとして起動する
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    jobs = [('train', i) for i in range(25)] + [('validation', j) for j in range(5)]
    (HERE / 'logs').mkdir(exist_ok=True)

    def run(job):
        split, index = job
        if (OUT / f'{split}_{index:02d}.pkl').exists():
            return job, 0
        with (HERE / 'logs' / f'multi_{split}_{index:02d}.log').open('a', encoding='utf-8') as f:
            code = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--one', split, str(index)],
                                  stdout=f, stderr=subprocess.STDOUT).returncode
        return job, code

    with ThreadPoolExecutor(args.parallel) as ex:
        for job, code in ex.map(run, jobs):
            print('done', job, 'exit', code, flush=True)
            if code != 0:
                raise RuntimeError(f'{job} の入札が失敗した')
    world.configure(rd.TASK)
    from training.bid_bank import save_fixed_bid
    by_probs = {}
    for split, n in (('train', 25), ('validation', 5)):
        src = world.main_bank(rd.TASK, split)
        dst = world.BANKS / bank_name(split)
        if dst.exists():
            raise FileExistsError(dst)
        dst.mkdir(parents=True)
        shutil.copy2(src / 'manifest.json', dst / 'manifest.json')
        manifest = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
        assert len(manifest['entries']) == n
        for i, entry in enumerate(manifest['entries']):
            bid, prev = pickle.loads((OUT / f'{split}_{i:02d}.pkl').read_bytes())
            by_probs[rd.probs_key(bid['arrival_probabilities_by_station'])] = prev
            save_fixed_bid(dst / Path(str(entry['bid_path']).replace(chr(92), '/')), bid)
        (dst / 'REALDAY.json').write_text(json.dumps({'multi': split}, ensure_ascii=False), encoding='utf-8')
        print('bank', dst.name, n, 'entries', flush=True)
    assert len(by_probs) == 30, '組んだ日の到着の確率が重なった'
    (HERE / 'prev_multi.pkl').write_bytes(pickle.dumps(by_probs))


if __name__ == '__main__':
    main()

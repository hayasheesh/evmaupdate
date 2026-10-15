"""EVの予測精度を85%・95%に振ったとき、入札量と当日に成り立つ組がどう変わるかを測る（6 station、検証用の40組）。

    python accuracy_bid.py actual                    # 40組の当日のEVを取り出して保存する（AEMO と ERCOT で同じ）
    python accuracy_bid.py calibrate                 # 精度が目標になる強さ（95% の k、85% の σ）を探す。入札はしない
    python accuracy_bid.py bid <aemo|ercot> <95|85> <k または σ> [workers]

予測の精度は docs/EVデータ.md の「予測の精度」と同じ物差しで測る。入札器が見る128通りの平均を予測とし、
5分ごとの接続台数（6 station の合計）の WAPE を当日と比べ、精度 = 100% − WAPE とする。今の予測（分布だけ）は約91%。

入札器が見る128通りを、組ごとに次のように作り直す。
  95%  今の128通りの各実現に、当日のEVを1台ずつ確率 k で入れ、元の実現のEVは確率 1−k で残す（当日の情報を一部知る予測）。
  85%  組ごとに1つの倍率 f = 1+ε（ε ~ N(0, σ)、128通りで共通）を到着の量に掛ける。f < 1 なら各EVを確率 f で残し、
       f > 1 なら別の実現のEVを確率 f−1 で足す（予測がその日だけ多めか少なめに外れる）。
どちらも、1 station の充電器10口を超える同時接続は、後から来たEVを外す。EVの名前は混ぜる元ごとに付け直す。
入札器の条件（EVの選び方、設計用の指令128本、最低入札量70 kW、余裕7 kW など）は本線と同じで、作り直した128通りから
同じ規則で3通りを選ぶ。判定は info_bid.py と同じく、その組の当日の指令とEVで全情報のLPが成り立つか。
結果は result_<task>_<level>.jsonl（1行1組）。
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Windows では空の文字列だと変数が消え、GPU が見えてしまう
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'info_bid_6station_20261012'))
import info_bid as ib  # noqa: E402

t3 = ib.t3
N_DAY, N_CMD, N_EV, PORTS, N_ST = 5, 4, 2, 10, 6
ACTUAL = HERE / 'actual_evs.pkl'


def sets():
    return [(d, c, e) for d in range(N_DAY) for c in range(N_CMD) for e in range(N_EV)]


def candidates(task, day):
    """入札器が使った128通り（ev_candidate_cache）。保存ファイルの名前は入札データを作ったときの装置（cuda）で決まる。"""
    import Config
    import torch
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank
    base = dict(BidBank(t3._CACHE['test_bank']).load_entry(BidBank(t3._CACHE['test_bank']).entries[day]))
    seed = int(base.get('forecast_seed', 0))
    count = int(lbt.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES)
    payloads = [(seed + 10007 * i, base.get('arrival_probabilities_by_station'), base.get('service_date')) for i in range(count)]
    cpu = Config.DEVICE
    Config.DEVICE = torch.device('cuda')
    try:
        path = lbt._ev_candidate_cache_path(payloads)
    finally:
        Config.DEVICE = cpu
    return pickle.loads(path.read_bytes())


def relabel(evs, tag):
    return [dataclasses.replace(ev, ev_id=f'{tag}:{ev.ev_id}') for ev in evs]


def port_limit(evs):
    """1 station の同時接続を口数以内にする（到着の早い順に入れ、埋まっていれば外す）。"""
    kept = []
    for st in range(N_ST):
        active = []
        for ev in sorted((x for x in evs if int(x.station_id) == st), key=lambda x: (x.arrival_t, x.departure_t)):
            active = [d for d in active if d > ev.arrival_t]
            if len(active) < PORTS:
                kept.append(ev)
                active.append(ev.departure_t)
    return kept


def mix_95(cands, actual, k, rng):
    out = []
    for cand in cands:
        a = [ev for ev in relabel(actual, 'a') if rng.random() < k]
        b = [ev for ev in relabel(cand, 'c') if rng.random() < 1 - k]
        out.append(port_limit(a + b))
    return out


def shift_85(cands, sigma, rng):
    f = float(min(max(1 + rng.normal(0.0, sigma), 0.3), 2.0))
    out = []
    n = len(cands)
    for i, cand in enumerate(cands):
        if f <= 1:
            out.append(port_limit([ev for ev in relabel(cand, 'c') if rng.random() < f]))
        else:
            extra = [ev for ev in relabel(cands[(i + 37) % n], 'x') if rng.random() < f - 1]
            out.append(port_limit(relabel(cand, 'c') + extra))
    return out, f


def connected(evs, steps=288):
    import numpy as np
    c = np.zeros(steps)
    for ev in evs:
        c[int(ev.arrival_t):min(int(ev.departure_t), steps)] += 1
    return c


def wape(cands, actual):
    import numpy as np
    mean = np.mean([connected(x) for x in cands], axis=0)
    a = connected(actual)
    return 100.0 * float(np.abs(mean - a).sum() / max(a.sum(), 1e-9))


def forecast(task, level, strength, day, c, e, act):
    import numpy as np
    rng = np.random.default_rng(1_300_000 + 1000 * day + 10 * c + e + (0 if level == '95' else 500_000))
    cands = candidates(task, day)
    if level == '95':
        return mix_95(cands, act, strength, rng), None
    return shift_85(cands, strength, rng)


def cmd_actual():
    t3._setup('aemo')
    out = {}
    for d, c, e in sets():
        out[(d, c, e)] = ib.actual_evs('aemo', d, c, e)
        print(d, c, e, len(out[(d, c, e)]), flush=True)
    ACTUAL.write_bytes(pickle.dumps(out))


def cmd_calibrate():
    import numpy as np
    t3._setup('aemo')
    act = pickle.loads(ACTUAL.read_bytes())
    base = [wape(candidates('aemo', d), act[(d, c, e)]) for d, c, e in sets()]
    print(f'now: WAPE {np.mean(base):.2f}% (accuracy {100 - np.mean(base):.1f}%)', flush=True)
    for k in (0.3, 0.4, 0.45, 0.5, 0.6):
        w = [wape(forecast('aemo', '95', k, d, c, e, act[(d, c, e)])[0], act[(d, c, e)]) for d, c, e in sets()]
        print(f'95: k={k} WAPE {np.mean(w):.2f}% (accuracy {100 - np.mean(w):.1f}%)', flush=True)
    for s in (0.08, 0.12, 0.16, 0.2, 0.25):
        w = [wape(forecast('aemo', '85', s, d, c, e, act[(d, c, e)])[0], act[(d, c, e)]) for d, c, e in sets()]
        print(f'85: sigma={s} WAPE {np.mean(w):.2f}% (accuracy {100 - np.mean(w):.1f}%)', flush=True)


def run_job(args):
    task, level, strength, day, c, e = args
    t3._setup(task)
    import numpy as np
    import types
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank
    from training.blockwise_bid import build_blockwise_bid_for_day
    t0 = time.time()
    act = pickle.loads(ACTUAL.read_bytes())[(day, c, e)]
    cands, f = forecast(task, level, strength, day, c, e, act)
    w = wape(cands, act)
    ev_bank, selection = lbt._select_ev_scenarios(cands)
    bank = BidBank(t3._CACHE['test_bank'])
    base = dict(bank.load_entry(bank.entries[day]))
    original = lbt._selected_ev_scenario_bank
    lbt._selected_ev_scenario_bank = lambda **kw: (ev_bank, selection, None)
    try:
        bid = build_blockwise_bid_for_day(
            np.asarray(base['base_series'], float), base.get('service_date'),
            arrival_scenario=types.SimpleNamespace(arrival_probabilities_by_station=base.get('arrival_probabilities_by_station')),
            forecast_seed=int(base.get('forecast_seed', 0)),
            assessment_band_fraction=float(base['assessment_band_fraction']), scenario_workers=1)
    finally:
        lbt._selected_ev_scenario_bank = original
    feasible, reason = ib.judge(task, bid, day, c, e, evs=act)
    return {'task': task, 'level': level, 'strength': strength, 'day': day, 'command': c, 'ev_seed': e,
            'service_date': str(base.get('service_date')), 'wape_pct': w, 'shift_factor': f,
            'capacity_kw_block': float(np.sum(bid['up_plan']) + np.sum(bid['down_plan'])),
            'main_capacity_kw_block': float(np.sum(base['up_plan']) + np.sum(base['down_plan'])),
            'feasible': feasible, 'reason': reason, 'solve_s': round(time.time() - t0, 1)}


def cmd_bid(task, level, strength, workers):
    out = HERE / f'result_{task}_{level}.jsonl'
    done = set()
    if out.exists():
        done = {(r['day'], r['command'], r['ev_seed']) for r in map(json.loads, out.read_text(encoding='utf-8').splitlines())}
    jobs = [(task, level, strength, d, c, e) for d, c, e in sets() if (d, c, e) not in done]
    print(f'{task} {level}: {len(jobs)} bids', flush=True)
    with mp.get_context('spawn').Pool(workers) as pool, out.open('a', encoding='utf-8') as fh:
        for row in pool.imap_unordered(run_job, jobs):
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            print(row['day'], row['command'], row['ev_seed'], round(row['capacity_kw_block']), row['feasible'], round(row['wape_pct'], 1), row['solve_s'], flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'actual':
        cmd_actual()
    elif mode == 'calibrate':
        cmd_calibrate()
    else:
        cmd_bid(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 10)

"""EVの選び方・指令の本数・予測精度を変えた検証用の入札（6 station、5日）を作り、未見通過率を測る。

    python table.py calibrate                       # 予測精度 85・91・95% の誤差の大きさ σ を決め、calib.json に書く
    python table.py build <aemo|ercot> <variant>    # 1つの変種の入札を作る（run.py が子プロセスで呼ぶ）
    python table.py unseen <aemo|ercot> <variant> <workers>   # 未見通過率を測る
    python table.py report                          # 表と図

入札器のほかの条件（最低入札量70 kW、余裕7 kW、基準値の段差の罰）は本線（resbid_6station_20261012 の stage_worker）と同じ。
EVの候補128通りは本線と同じもの（bid_loss_6station_20261013/data）を使い、指令は128本の先頭からN本を使う。

未見通過率は、検証用の5日 × 設計に使っていない指令128本 × 入札と重ならないEVの乱数3通り ＝ 1,920組で、
全情報のLPが成り立つ割合。判定できない組は20秒 → 120秒 → 600秒と延ばして解き直す。

予測精度は、128通りの平均の接続台数（5分ごと）と当日との誤差率（WAPE）で測る（docs/EVデータ.md と同じ）。
3水準とも同じ作り方にする。予測の128通りは、当日のEVを1台ずつ入れ、その到着と出発の時刻に
標準偏差 σ ステップの誤差を足して作る（perturb）。σ が小さいほど当日に近い。σ は誤差率が 5%・9%・15% になるように決める。
本線の予測（同じ分布から引いた128通り）は、当日のEVの情報を持たないので、この作り方の外にある参考の値として並べる。
当日のEVは評価の組ごとに違うので、精度を変えた入札は（日 × 評価のEVの乱数）の15本ずつ作り、その乱数の組だけで判定する。
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
import types
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Windows では空の文字列だと変数が消え、GPU が見えてしまう
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SW = ROOT / 'execute_results/resbid_6station_20261012'
POOL = ROOT / 'execute_results/bid_loss_6station_20261013/data'
BANKS = ROOT / 'execute_results/bid_banks'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SW))

EVAL_SEEDS = 3
ACCURACY_TARGETS = {'95': 5.0, '91': 9.0, '85': 15.0}

# 変種の名前 → (指令の本数, 選び方, 精度)。選び方は low<K>max（接続の少ない K 通り＋台数最多）、low<K>（接続の少ない K 通りだけ）、
# minmedmax、low1medmax、random<r>。精度が None なら予測はそのまま（入札は日ごとに5本）。
VARIANTS = {
    'main': (128, 'low2max', None),
    'cmd8': (8, 'low2max', None), 'cmd16': (16, 'low2max', None), 'cmd32': (32, 'low2max', None),
    'cmd64': (64, 'low2max', None), 'cmd256': (256, 'low2max', None),
    'sel_minmedmax': (128, 'minmedmax', None), 'sel_low1medmax': (128, 'low1medmax', None),
    'sel_random0': (128, 'random0', None), 'sel_random1': (128, 'random1', None), 'sel_random2': (128, 'random2', None),
    'cnt1': (128, 'low1', None), 'cnt2': (128, 'low1max', None), 'cnt4': (128, 'low3max', None),
    'cnt6': (128, 'low5max', None),
}
for _acc in ACCURACY_TARGETS:
    for _sel in ('low2max', 'random0'):
        VARIANTS[f'acc{_acc}_{_sel}'] = (128, _sel, _acc)

# 既に作ってある入札（本線の検証用、指令256本、SoCの需要の最小・中央・最大）
EXISTING_TAG = {'main': '128cmd_3ev', 'cmd256': '256cmd_low2max', 'sel_minmedmax': '128cmd_minmedmax'}


def base_name(task: str) -> str:
    return f'wr_cap77wd_min70_margin7_validation_5_6station_128cmd_3ev_{task}_plan_deviation'


def bank_name(task: str, variant: str, seed_index: int | None = None) -> str:
    n, sel, acc = VARIANTS[variant]
    tag = EXISTING_TAG.get(variant, f'{n}cmd_{sel}' if acc is None else f'{n}cmd_acc{acc}_{sel}_s{seed_index}')
    return base_name(task).replace('_128cmd_3ev_', f'_{tag}_')


def bank_names(task: str, variant: str) -> list[str]:
    if VARIANTS[variant][2] is None:
        return [bank_name(task, variant)]
    return [bank_name(task, variant, j) for j in range(EVAL_SEEDS)]


# ---------------- EV ----------------

def relabel(evs, prefix):
    import dataclasses
    return [dataclasses.replace(ev, ev_id=f'{prefix}{i}_{ev.ev_id}') for i, ev in enumerate(evs)]


def port_limit(evs):
    """同じ局の同時接続が充電口の数を超えたら、後から来たEVを外す（bid_loss_6station_20261013 と同じ）。"""
    from EnvConfig import MAX_EV_PER_STATION
    kept, active = [], {}
    for ev in sorted(evs, key=lambda e: (int(e.arrival_t), str(e.ev_id))):
        st = int(ev.station_id)
        now = [d for d in active.get(st, []) if d > int(ev.arrival_t)]
        if len(now) < int(MAX_EV_PER_STATION):
            kept.append(ev)
            now.append(int(ev.departure_t))
        active[st] = now
    return kept


def jitter_ev(ev, tau, rng, eta_ch):
    """到着と出発の時刻に、標準偏差 tau ステップの正規の誤差を足す。0時に停まっている車の到着は動かさない。"""
    import dataclasses
    steps = 288
    arrival = int(ev.arrival_t)
    if arrival > 0:
        arrival = int(min(max(arrival + round(rng.normal(0.0, tau)), 1), steps - 1))
    departure = int(max(int(ev.departure_t) + round(rng.normal(0.0, tau)), arrival + 1))
    # 滞在が短くなって目標に届かなくなるEVは、目標を定格で届く分までに下げる（物理的に成り立たない候補を作らない）
    reach = float(ev.initial_soc) + float(ev.max_charge_kw) * eta_ch * (departure - arrival) * (5.0 / 60.0) / float(ev.capacity_kwh)
    target = min(float(ev.target_soc), reach)
    return dataclasses.replace(ev, arrival_t=arrival, departure_t=departure, target_soc=target,
                               target_required=bool(departure <= steps))


def perturb(cands, actual, tau, rng):
    """当日のEVから128通りの予測を作る。各候補は、当日のEVを1台ずつ入れ、到着と出発の時刻に標準偏差 tau ステップの
    誤差を足したもの。tau が小さいほど当日に近い。台数は当日と同じにする（台数を増やすと、満車に近い局で
    充電口の上限に外され、予測が少なめに偏るため）。
    cands（本線の予測の128通り）は使わない（3水準とも同じ作り方にするため）。"""
    from market.physical_lp_bidding.data_classes import BiddingLPConfig
    eta_ch = float(BiddingLPConfig().eta_ch)
    out = []
    for ci in range(len(cands)):
        evs = []
        for k, ev in enumerate(actual):
            evs.append(jitter_ev(dataclasses_rename(ev, f'f{ci}_{k}'), tau, rng, eta_ch))
        out.append(port_limit(evs))
    return out


def dataclasses_rename(ev, tag):
    import dataclasses
    return dataclasses.replace(ev, ev_id=f'{tag}_{ev.ev_id}')


def connected(evs, steps=288):
    import numpy as np
    out = np.zeros(steps)
    for ev in evs:
        out[int(ev.arrival_t):min(int(ev.departure_t), steps)] += 1
    return out


def wape(cands, actual):
    import numpy as np
    mean = np.mean([connected(x) for x in cands], axis=0)
    a = connected(actual)
    return 100.0 * float(np.abs(mean - a).sum() / max(a.sum(), 1e-9))


def select(sel, cands, day=0):
    """128通りから選ぶ。本線と同じ関数を使い、K 通りを変えるときは接続の少ない側を前から足す（入れ子になる）。"""
    import numpy as np
    from training import lower_bid_training as lbt
    if sel == 'minmedmax':
        return lbt._select_ev_scenarios_by_count(cands)
    if sel == 'low1medmax':
        return lbt._select_ev_scenarios_low_connection(cands, low_picks=1, count_picks=('median', 'maximum'))
    if sel.startswith('random'):
        r = int(sel[len('random'):])
        picks = [int(i) for i in np.random.default_rng(2_026_101_500 + 1000 * r + day).choice(len(cands), 3, replace=False)]
        # 台数の順位は本線の選び方と同じ並べ方（台数、同数なら候補の番号）で数える
        rank = {i: k for k, i in enumerate(sorted(range(len(cands)), key=lambda i: (len(cands[i]), i)))}
        return [cands[i] for i in picks], [{'label': f'random_{j + 1}', 'candidate_index': i, 'rank_zero_based': rank[i],
                                            'candidate_count': len(cands), 'ev_count': len(cands[i])}
                                           for j, i in enumerate(picks)]
    if sel.endswith('max'):
        return lbt._select_ev_scenarios_low_connection(cands, low_picks=int(sel[3:-3]), count_picks=('maximum',))
    return lbt._select_ev_scenarios_low_connection(cands, low_picks=int(sel[3:]), count_picks=())


# ---------------- 入力の準備 ----------------

def configure(task):
    import stage_worker
    if 'Config' in sys.modules:
        raise RuntimeError('configure は新しいプロセスで1回だけ呼ぶ')
    sys.modules['Config'] = types.ModuleType('Config')  # 本物の Config は条件を上書きしてから読む
    stage_worker.configure(task)
    del sys.modules['Config']


def load_inputs(task):
    """本線の検証用の入札の5日と、日ごとの候補128通り。本線の3通りが候補から同じに選ばれることを確かめる。"""
    from training.bid_bank import BidBank
    from training import lower_bid_training as lbt
    bank = BidBank(BANKS / base_name(task))
    days = []
    for d, entry in enumerate(bank.entries):
        base = bank.load_entry(entry)
        cands = pickle.loads((POOL / f'candidates_day{d}.pkl').read_bytes())
        if len(cands) != 128:
            raise ValueError('候補は128通りのはず')
        _, meta = lbt._select_ev_scenarios_low_connection(cands, low_picks=2, count_picks=('maximum',))
        if [int(x['candidate_index']) for x in meta] != [int(x['candidate_index']) for x in base['ev_scenario_selection']]:
            raise ValueError(f'{task} day{d}: 本線の3通りが候補から再現できない')
        days.append(dict(base=base, cands=cands))
    if len(days) != 5:
        raise ValueError('検証用は5日のはず')
    return days


def actual_evs(base):
    """未見の評価で使う当日のEV（tools/evaluate_unseen_bid_feasibility.py と同じ seed と作り方）。"""
    import numpy as np
    from training.lower_bid_training import _sample_ev_scenario_bank
    return _sample_ev_scenario_bank(count=EVAL_SEEDS, seed=int(base['forecast_seed']), seed_offset=10_000_000,
                                    arrival_probs=np.asarray(base['arrival_probabilities_by_station']),
                                    label='independent evaluation', service_date=base.get('service_date'))


def forecast(cands, actual, acc, calib, day, j):
    import numpy as np
    rng = np.random.default_rng(1_500_000 + 1000 * day + 10 * j + int(acc))
    return perturb(cands, actual, calib[acc]['sigma'], rng)


# ---------------- calibrate ----------------

def cmd_calibrate():
    import numpy as np
    configure('aemo')
    days = load_inputs('aemo')
    actual = {d: actual_evs(x['base']) for d, x in enumerate(days)}
    raw = [wape(x['cands'], actual[d][j]) for d, x in enumerate(days) for j in range(EVAL_SEEDS)]

    def mean_wape(sigma, acc):
        vals = []
        for d, x in enumerate(days):
            for j in range(EVAL_SEEDS):
                rng = np.random.default_rng(1_500_000 + 1000 * d + 10 * j + int(acc))
                vals.append(wape(perturb(x['cands'], actual[d][j], sigma, rng), actual[d][j]))
        return float(np.mean(vals)), vals

    out = {'method': 'actual_sessions_time_jitter', 'model_wape_mean': float(np.mean(raw)), 'model_wape_per_pair': raw}
    for acc, target in ACCURACY_TARGETS.items():
        floor, _ = mean_wape(0.0, acc)
        if floor > target:
            out[acc] = {'sigma': 0.0, 'target_wape': target, 'achieved_wape': floor, 'reachable': False}
            print(acc, 'σ=0 でも誤差率', round(floor, 2), '%', flush=True)
            continue
        lo, hi = 0.0, 6.0
        while mean_wape(hi, acc)[0] < target:
            hi *= 2
            if hi > 400:
                raise RuntimeError('σ を大きくしても目標の誤差率に届かない')
        for _ in range(18):
            mid = (lo + hi) / 2
            if mean_wape(mid, acc)[0] < target:
                lo = mid
            else:
                hi = mid
        sigma = (lo + hi) / 2
        achieved, vals = mean_wape(sigma, acc)
        out[acc] = {'sigma': sigma, 'target_wape': target, 'achieved_wape': achieved, 'per_pair': vals,
                    'reachable': True, 'floor_wape_sigma0': floor}
        print(acc, 'σ', round(sigma, 4), '誤差率', round(achieved, 2), '%', flush=True)
    (HERE / 'calib.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


# ---------------- build ----------------

def cmd_build(task, variant, seed_index=None):
    n, sel, acc = VARIANTS[variant]
    if variant in EXISTING_TAG:
        raise ValueError(f'{variant} は既にある入札を使う')
    configure(task)
    out = BANKS / bank_name(task, variant, seed_index)
    os.environ['EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS'] = str(n)
    os.environ['EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR'] = str(out)
    # 入札の解の保存は鍵に EV の選び方の差し替えを含まないので、使うと本線の入札がそのまま返る。読みも書きもしない。
    os.environ['EVMA_BID_SOLVE_CACHE'] = '0'
    from training import lower_bid_training as lbt
    days = load_inputs(task)
    by_date = {str(x['base']['service_date']): (d, x) for d, x in enumerate(days)}
    calib = json.loads((HERE / 'calib.json').read_text(encoding='utf-8')) if acc else None

    def selected(*, count, seed, arrival_probs, label, service_date, workers=1):
        d, x = by_date[str(service_date)]
        if int(seed) != int(x['base']['forecast_seed']):
            raise ValueError('EVの予測の seed が本線と違う')
        cands = x['cands']
        if acc:
            cands = forecast(cands, actual_evs(x['base'])[seed_index], acc, calib, d, seed_index)
        evs, meta = select(sel, cands, d)
        return evs, meta, None

    lbt._selected_ev_scenario_bank = selected
    activation = lbt._activation_scenarios_for_day

    def first_n(*args, **kwargs):
        # 設計の指令は128本の先頭 n 本（本数を変えても入れ子になる）
        if kwargs.get('n_scenarios') == n and n < 128:
            payloads, mode = activation(*args, **dict(kwargs, n_scenarios=128))
            return payloads[:n], mode
        return activation(*args, **kwargs)

    lbt._activation_scenarios_for_day = first_n
    import tools.build_training_bid_bank as builder
    sys.argv = ['build_training_bid_bank.py', '--split', 'test', '--days', '5', '--train-split-count', '25',
                '--paired-train-days', '25', '--paired-test-days', '5', '--workers', '1', '--scenario-workers', '3',
                '--output-dir', str(out)]
    started = time.time()
    builder.main()
    check_bank(task, variant, seed_index)
    (out / 'BUILD_SECONDS').write_text(f'{time.time() - started:.1f}\n')


def check_bank(task, variant, seed_index=None):
    """入札の指令とEVの実現が意図どおりかを確かめる。"""
    import numpy as np
    from training.bid_bank import BidBank
    n, sel, acc = VARIANTS[variant]
    bank = BidBank(BANKS / bank_name(task, variant, seed_index))
    base = BidBank(BANKS / base_name(task))
    if len(bank.entries) != 5:
        raise ValueError('5日そろっていない')
    for e, b in zip(bank.entries, base.entries):
        bid, ref = bank.load_entry(e), base.load_entry(b)
        design, ref_design = bid['design_activation_scenario_payload'], ref['design_activation_scenario_payload']
        if len(design) != n:
            raise ValueError(f'設計の指令が {len(design)} 本')
        for x, y in zip(design, ref_design[:min(n, 128)]):
            for key in ('up_proxy', 'down_proxy'):
                np.testing.assert_array_equal(x[key], y[key])
        for key in ('forecast_seed', 'service_date', 'assessment_band_fraction', 'assessment_i_control_margin_kw'):
            if bid[key] != ref[key]:
                raise ValueError(f'本線と違う: {key}')


# ---------------- unseen ----------------

CHUNK = 16  # 未見の指令128本を16本ずつに分けて並列にする


def _eval_job(args):
    task, name, entry_index, seeds, chunk = args
    import numpy as np
    from market.physical_lp_bidding.colgen_feasibility import certify
    from market.physical_lp_bidding.data_classes import BiddingLPConfig
    from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
    from training.bid_bank import BidBank
    from training.lower_bid_training import FEEDBACK_COMMAND_SEED_OFFSET, _activation_scenarios_for_day
    bank = BidBank(BANKS / name)
    bid = bank.load_entry(bank.entries[entry_index])
    feedback, _ = _activation_scenarios_for_day(bid.get('service_date'), int(bid['forecast_seed']) + FEEDBACK_COMMAND_SEED_OFFSET,
                                                n_scenarios=128, scenario_partition='feedback')
    design_ids = {(r.get('source_date'), r.get('source_bmu')) for r in bid['design_activation_scenario_payload']}
    if design_ids & {(r.get('source_date'), r.get('source_bmu')) for r in feedback}:
        raise ValueError('設計の指令と未見の指令が重なる')
    evs_all = actual_evs(bid)
    cfg = BiddingLPConfig(assessment_band_fraction=float(bid['assessment_band_fraction']),
                          apply_transition_band=bool(bid['apply_transition_band']))
    base, up, down = (np.asarray(bid[k], float) for k in ('baseline_plan', 'up_plan', 'down_plan'))
    rows, started = [], time.time()
    for ci, command in enumerate(feedback):
        if ci // CHUNK != chunk:
            continue
        _, _, lower, upper = fixed_bid_tracking_bands(cfg, base, up, down, np.asarray(command['up_proxy'], float),
                                                      np.asarray(command['down_proxy'], float),
                                                      apply_transition_band=cfg.apply_transition_band)
        for j in seeds:
            feasible, limit = None, None
            for limit in (20.0, 120.0, 600.0):
                feasible, _, _ = certify(evs_all[j], lower, upper, steps=cfg.steps, dt=cfg.dt_hours, eta_ch=cfg.eta_ch,
                                         return_dispatch=False, time_limit_s=limit)
                if feasible is not None:
                    break
            rows.append({'command': ci, 'ev_seed_index': j, 'feasible': feasible, 'time_limit_s': limit})
    return {'bank': name, 'day': entry_index, 'chunk': chunk, 'service_date': str(bid['service_date']),
            'offered_capacity_kw_block': float(np.sum(up + down)), 'seconds': time.time() - started, 'rows': rows}


def cmd_unseen(task, variant, workers):
    import multiprocessing as mp
    out = HERE / f'unseen_{task}_{variant}.jsonl'
    done = set()
    if out.exists():
        done = {(r['bank'], r['day'], r['chunk']) for r in map(json.loads, out.read_text(encoding='utf-8').splitlines())}
    acc = VARIANTS[variant][2]
    jobs = []
    for j, name in enumerate(bank_names(task, variant)):
        for d in range(5):
            for chunk in range(128 // CHUNK):
                if (name, d, chunk) not in done:
                    jobs.append((task, name, d, [j] if acc else list(range(EVAL_SEEDS)), chunk))
    mp.set_start_method('spawn', force=True)
    with mp.Pool(workers, initializer=configure, initargs=(task,)) as pool, out.open('a', encoding='utf-8') as f:
        for r in pool.imap_unordered(_eval_job, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            print(variant, r['bank'][-40:], r['day'], r['chunk'], sum(x['feasible'] is True for x in r['rows']), '/', len(r['rows']), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'calibrate':
        cmd_calibrate()
    elif cmd == 'build':
        cmd_build(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == 'unseen':
        cmd_unseen(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise ValueError(cmd)

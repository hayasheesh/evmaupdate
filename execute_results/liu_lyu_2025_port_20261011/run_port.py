"""Liu/Lyu 2025 の方式を、職場＋住宅の6 station・検証用の平日5日で動かし、30分コマの判定とSoC達成率を測る（学習はしない）。

    python run_port.py <aemo|ercot> <variant> [commands_per_day] [ev_seeds] [workers] [force]

variant:
  fixed   入札は本研究の入札器のもの（検証用の入札）。計画と配分を Liu/Lyu の方式にする
  theirs  入札も Liu/Lyu の日前の入札（上下対称の調整容量、価格は元のコードの NYISO のデータ）にする。ランプの機会制約は外す
  theirs_ramp  同じく、ランプの機会制約を元のとおり入れた入札
  bid / bid_ramp  上の2つの入札を先に解いて their_bids/ に置く
  rule    比較用：リポジトリの中央観測ルールベース（rule_based_central）
force=1 で、本研究の force 充電（出発に間に合わなくなるEVを充電する層）を Liu/Lyu の配分の後に重ねる（既定0）。

指令は soc_miss_analysis.py と同じ holdout の指令（評価用）、EVの乱数も同じ式。BESS は使わない。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SW_DIR = HERE.parent / 'wr_cap77wd_6station_20261010'


def _setup(task):
    sys.path.insert(0, str(SW_DIR))
    sys.path.insert(0, str(HERE))
    import stage_worker
    stage_worker.configure(task)
    import torch
    torch.set_num_threads(1)
    return stage_worker


def _forecast_evs(fixed_bid, after_step):
    """入札器が前提にしたEVの3通りの実現のうち、after_step より後に到着するEV（重み1/3）。"""
    bank = fixed_bid['bid_ev_scenario_bank']
    w = 1.0 / len(bank)
    out = []
    for r, evs in enumerate(bank):
        for ev in evs:
            if int(ev.arrival_t) <= after_step:
                continue
            out.append(dict(key=('f', r, str(ev.ev_id)), w=w, P=float(ev.max_charge_kw), cap=float(ev.capacity_kwh),
                            E0=float(ev.initial_soc) * float(ev.capacity_kwh), arr=int(ev.arrival_t), dep=int(ev.departure_t),
                            target_kwh=float(ev.target_soc) * float(ev.capacity_kwh), target_required=bool(ev.target_required),
                            present=False))
    return out


def _present_evs(env, step):
    out = []
    for st in range(env.num_stations):
        idx = env._get_sorted_active_evs(st)
        ids = env.ev_ids[st, idx].tolist()
        arr = env.arrival_step[st, idx].tolist()
        soc = env.soc[st, idx].tolist()
        tgt = env.target[st, idx].tolist()
        dep = env.depart[st, idx].tolist()
        P = env.ev_max_power_kw[st, idx].tolist()
        cap = env.ev_capacity_kwh[st, idx].tolist()
        for j in range(len(ids)):
            out.append(dict(key=(st, int(ids[j]), int(arr[j])), w=1.0, P=float(P[j]), cap=float(cap[j]),
                            E0=soc[j] / 100.0 * cap[j], E=soc[j] / 100.0 * cap[j], arr=step, dep=int(dep[j]),
                            target_kwh=tgt[j] / 100.0 * cap[j], target_required=int(dep[j]) <= 288, present=True,
                            station=st, slot=j))
    return out


def _their_prices():
    """元のコードの data_prepare_main.m と同じ読み方（NYISO N.Y.C.、2024-04-13）。1時間の値を30分コマ2つに使う。"""
    import csv
    import numpy as np
    d = HERE / 'their_data'
    def rows(name):
        with open(d / name, encoding='utf-8-sig') as f:
            return list(csv.reader(f))
    dam = rows('20240413damasp.csv')
    hdr, body = dam[0], dam[1:]
    nyc = body[8::11]
    price_cap = np.array([float(r[7]) for r in nyc])      # 8列目（1始まり）
    rt = rows('20240413rtasp.csv')[1:]
    rt_nyc = rt[8::11]
    mil = np.array([float(r[8]) for r in rt_nyc])
    mil = np.concatenate([mil[:157], mil[159:]])
    mil_h = mil.reshape(-1, 12).mean(axis=1)
    lb = rows('20240413damlbmp_zone.csv')[1:]
    lb_nyc = lb[9::15]
    price_e = np.array([float(r[3]) + float(r[4]) for r in lb_nyc])
    return np.repeat(price_cap, 2), np.repeat(mil_h, 2), np.repeat(price_e, 2)


def _their_bid(fixed_bid, task, ramp):
    import numpy as np
    import liu_lyu_port as L
    from training.blockwise_bid import minimum_bid_quantity_kw
    payloads = fixed_bid['design_activation_scenario_payload']
    probs = L.scenario_probabilities(payloads)
    eps_up, eps_dn = L.ramp_quantiles(payloads)
    price_cap, price_mil, price_e = _their_prices()
    ys = np.stack([np.asarray(p['down_proxy'], float)[:288] - np.asarray(p['up_proxy'], float)[:288] for p in payloads])
    mileage = np.abs(np.diff(ys, axis=1)).reshape(ys.shape[0], -1)
    mileage_blk = np.array([np.abs(np.diff(ys[:, k * 6:(k + 1) * 6], axis=1)).sum(axis=1).mean() for k in range(48)])
    price_reg = price_cap + mileage_blk * price_mil
    evs = _forecast_evs(fixed_bid, -1)
    for ev in evs:
        ev['present'] = False
    bid = L.day_ahead_bid(evs, probs, price_e, price_reg, price_mil, eps_up, eps_dn, minimum_bid_quantity_kw(), ramp=ramp)
    return bid


def run_episode(args):
    task, variant, force, day, c, e, n_cmd = args
    stage_worker = _setup(task)
    import numpy as np
    import torch
    import liu_lyu_port as L
    from Config import EPISODE_STEPS
    from environment.EVEnv import EVEnv
    from environment.normalize import use_instruction_scale
    from market.bid_participation import participation_by_block
    from tools.evaluator import set_env_seed
    from training.bid_bank import BidBank
    from training.lower_bid_training import _activation_scenarios_for_day, build_fixed_upper_bid_training_episode
    from training.system_controller import apply_force_charging

    _train, test_bank = stage_worker.banks(task)
    bank = BidBank(test_bank)
    base_bid = dict(bank.load_entry(bank.entries[day]))
    payloads, mode = _activation_scenarios_for_day(base_bid.get('service_date'), int(base_bid.get('forecast_seed', 0)) + 512_209,
                                                   n_scenarios=n_cmd, scenario_partition='holdout')
    fixed_bid = dict(base_bid, activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=mode)
    bid_info = {}
    if variant in ('theirs', 'theirs_ramp'):
        cache = HERE / 'their_bids' / f'{task}_day{day}_{"ramp" if variant == "theirs_ramp" else "noramp"}.json'
        if cache.exists():
            tb = json.loads(cache.read_text(encoding='utf-8'))
        else:
            raise RuntimeError(f'run the bid stage first: {cache}')
        for k in ('baseline_plan', 'up_plan', 'down_plan'):
            fixed_bid[k] = np.asarray(tb[k], float)
        bid_info = {k: tb[k] for k in ('h_R',)}
    baseline = np.asarray(fixed_bid['baseline_plan'], float)
    up = np.asarray(fixed_bid['up_plan'], float)
    down = np.asarray(fixed_bid['down_plan'], float)
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, c)
    tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
    set_env_seed(4_000_000 + 1000 * day + 10 * c + e)
    use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.use_central_ev_residual_allocator = (variant == 'rule')
    env.use_residual_bess = False
    env.reset(net_demand_series=target, tol_narrow_series=tol, tracking_enabled_series=tracking,
              market_context_series=arr.get('market_context_series'),
              arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
              service_date=arr.get('service_date'), baseline_series=arr.get('baseline_series'))
    probs = L.scenario_probabilities(fixed_bid['design_activation_scenario_payload'])
    alloc = L.Allocator()
    disp = np.zeros(EPISODE_STEPS)
    resp = np.zeros(EPISODE_STEPS)
    plan_stats = []
    t0 = time.time()
    for t in range(EPISODE_STEPS):
        env.begin_step()
        zeros = env.soc.new_zeros((env.num_stations, env.max_ev_per_station))
        if variant == 'rule':
            actions, _f, _ = apply_force_charging(zeros, env, slack_kwh=0.1)
        else:
            present = _present_evs(env, t)
            if t % L.STEPS_PER_BLOCK == 0:
                b = t // L.STEPS_PER_BLOCK
                evs = present + _forecast_evs(fixed_bid, t)
                plan, st = L.plan_block(evs, b, baseline, up, down, probs)
                alloc.set_plan(plan, b)
                plan_stats.append(st)
            p = alloc.allocate(present, t, -float(env.current_net_demand), last_block=(t // L.STEPS_PER_BLOCK == L.N_BLOCKS - 1))
            actions = zeros.clone()
            for ev, pw in zip(present, p):
                actions[ev['station'], ev['slot']] = float(np.clip(pw / ev['P'], -1.0, 1.0))
            if force:
                actions, _f, _ = apply_force_charging(actions, env, slack_kwh=0.1)
        _o, _lr, _gr, _d, info = env.apply_action(actions)
        disp[t] = float(env.current_net_demand)
        resp[t] = float(disp[t] - float(info['pre_bess_residual_kw']))
    m = env.metrics
    err = np.abs(disp - resp)
    step_pass = err <= np.asarray(tol, float)[:EPISODE_STEPS] + 1e-6
    part = participation_by_block(up, down)
    blk_pass = []
    for k in range(48):
        if not part[k]:
            continue
        blk_pass.append(float(np.mean(step_pass[k * 6:(k + 1) * 6])) >= 0.90)
    reg = disp - np.repeat(baseline, 6)
    award = np.where(reg < -1e-6, np.repeat(up, 6), np.repeat(down, 6))
    pmask = np.repeat(part, 6)
    dep = int(m['departing_evs'])
    hit = int(m['central_departing_evs_soc_met'] if variant == 'rule' else m['departing_evs_soc_met'])
    return {
        'task': task, 'variant': variant, 'force': int(force), 'day': day, 'command': c, 'ev_seed': e,
        'service_date': str(base_bid.get('service_date')),
        'step_pass_pct': 100.0 * float(np.mean(step_pass[tracking])),
        'block_pass_pct': 100.0 * float(np.mean(blk_pass)) if blk_pass else float('nan'),
        'blocks_assessed': len(blk_pass), 'blocks_failed': int(len(blk_pass) - sum(blk_pass)),
        'mean_abs_error_kw': float(np.mean(err[pmask])) if pmask.any() else float('nan'),
        'mean_error_to_award_pct': 100.0 * float(np.sum(err[pmask]) / max(np.sum(award[pmask]), 1e-9)),
        'soc_hit_pct': 100.0 * hit / max(dep, 1), 'departed_evs': dep,
        'offered_capacity_kw_block': float(np.sum(up + down)),
        'plan_slack_current_block_kw_mean': float(np.mean([s['slack_current_block_kw'] for s in plan_stats])) if plan_stats else 0.0,
        'plan_vars_max': int(max([s['n_vars'] for s in plan_stats])) if plan_stats else 0,
        'seconds': time.time() - t0, **bid_info,
    }


def bid_stage(args):
    task, day, ramp = args
    stage_worker = _setup(task)
    import numpy as np
    from training.bid_bank import BidBank
    _train, test_bank = stage_worker.banks(task)
    bank = BidBank(test_bank)
    fixed_bid = dict(bank.load_entry(bank.entries[day]))
    t0 = time.time()
    bid = _their_bid(fixed_bid, task, ramp)
    out = {k: (np.asarray(v).tolist() if hasattr(v, '__len__') else v) for k, v in bid.items()}
    out['seconds'] = time.time() - t0
    out['service_date'] = str(fixed_bid.get('service_date'))
    out['our_offered_capacity_kw_block'] = float(np.sum(fixed_bid['up_plan'] + fixed_bid['down_plan']))
    out['their_offered_capacity_kw_block'] = float(np.sum(bid['up_plan'] + bid['down_plan']))
    p = HERE / 'their_bids' / f'{task}_day{day}_{"ramp" if ramp else "noramp"}.json'
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return {'day': day, 'ramp': ramp, 'seconds': out['seconds'], 'their_capacity': out['their_offered_capacity_kw_block'],
            'our_capacity': out['our_offered_capacity_kw_block'], 'h_R': out['h_R']}


def main():
    task, variant = sys.argv[1], sys.argv[2]
    n_cmd = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_ev = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    force = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    mp.set_start_method('spawn', force=True)
    if variant in ('bid', 'bid_ramp'):
        with mp.Pool(min(workers, 5)) as pool:
            for r in pool.imap_unordered(bid_stage, [(task, d, variant == 'bid_ramp') for d in range(5)]):
                print(json.dumps(r, ensure_ascii=False), flush=True)
        return
    jobs = [(task, variant, force, d, c, e, n_cmd) for d in range(5) for c in range(n_cmd) for e in range(n_ev)]
    out = HERE / f'result_{task}_{variant}_force{force}_{n_cmd}cmd_{n_ev}ev.jsonl'
    with mp.Pool(workers) as pool, open(out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(run_episode, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps({k: r[k] for k in ('day', 'command', 'ev_seed', 'step_pass_pct', 'block_pass_pct', 'soc_hit_pct', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()

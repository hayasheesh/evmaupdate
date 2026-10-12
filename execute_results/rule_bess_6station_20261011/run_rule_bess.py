"""中央観測ルールベースに連系点の蓄電池を足し、蓄電池の大きさごとに30分コマの判定とSoC達成率を測る（学習はしない）。

    python run_rule_bess.py <aemo|ercot> <bess_kw,...> [commands_per_day] [ev_seeds] [workers]

入札、指令、EVの乱数は liu_lyu_2025_port_20261011/run_port.py の rule と同じ（検証用の平日5日、holdout の指令）。
蓄電池の容量は EnvConfig の既定と同じ式で出力から決める（目標SoC 50%から下限10%まで、30分を定格で出せる量）。
EnvConfig は 500 局に広げたときの上限（局あたり 2 MW 未満）で出力を検査するので、環境を作った後に出力と容量を上書きする。
誤差は蓄電池の前（EVだけ）と後（連系点）の両方を記録する。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = HERE.parent / 'liu_lyu_2025_port_20261011'


def run_episode(args):
    task, bess_kw, day, c, e, n_cmd = args
    sys.path.insert(0, str(PORT))
    import run_port
    stage_worker = run_port._setup(task)
    import numpy as np
    from Config import EPISODE_STEPS
    from EnvConfig import BESS_DISCHARGE_EFFICIENCY, BESS_MIN_SOC_PCT, BESS_TARGET_SOC_PCT
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
    baseline = np.asarray(fixed_bid['baseline_plan'], float)
    up = np.asarray(fixed_bid['up_plan'], float)
    down = np.asarray(fixed_bid['down_plan'], float)
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, c)
    tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
    set_env_seed(4_000_000 + 1000 * day + 10 * c + e)
    use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.use_central_ev_residual_allocator = True
    env.use_residual_bess = bess_kw > 0
    if bess_kw > 0:
        env.bess_power_limit_kw = float(bess_kw)
        env.bess_energy_capacity_kwh = float(bess_kw) * 0.5 / (
            max((BESS_TARGET_SOC_PCT - BESS_MIN_SOC_PCT) / 100.0, 1e-9) * BESS_DISCHARGE_EFFICIENCY)
    env.reset(net_demand_series=target, tol_narrow_series=tol, tracking_enabled_series=tracking,
              market_context_series=arr.get('market_context_series'),
              arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
              service_date=arr.get('service_date'), baseline_series=arr.get('baseline_series'))
    disp = np.zeros(EPISODE_STEPS)
    pre = np.zeros(EPISODE_STEPS)
    post = np.zeros(EPISODE_STEPS)
    bp = np.zeros(EPISODE_STEPS)
    plim = np.zeros(EPISODE_STEPS, bool)
    elim = np.zeros(EPISODE_STEPS, bool)
    t0 = time.time()
    for t in range(EPISODE_STEPS):
        env.begin_step()
        zeros = env.soc.new_zeros((env.num_stations, env.max_ev_per_station))
        actions, _f, _ = apply_force_charging(zeros, env, slack_kwh=0.1)
        _o, _lr, _gr, _d, info = env.apply_action(actions)
        disp[t] = float(env.current_net_demand)
        pre[t] = float(info['pre_bess_residual_kw'])
        post[t] = float(info['post_bess_residual_kw'])
        bp[t] = float(info['bess_power_kw'])
        plim[t] = bool(info.get('bess_power_limit_hit', False))
        elim[t] = bool(info.get('bess_energy_limit_hit', False))
    m = env.metrics
    tol = np.asarray(tol, float)[:EPISODE_STEPS]
    part = participation_by_block(up, down)
    pmask = np.repeat(part, 6)
    reg = disp - np.repeat(baseline, 6)
    award = np.where(reg < -1e-6, np.repeat(up, 6), np.repeat(down, 6))

    def judge(res):
        err = np.abs(res)
        step_pass = err <= tol + 1e-6
        blk = [float(np.mean(step_pass[k * 6:(k + 1) * 6])) >= 0.90 for k in range(48) if part[k]]
        return {
            'step_pass_pct': 100.0 * float(np.mean(step_pass[tracking])),
            'block_pass_pct': 100.0 * float(np.mean(blk)) if blk else float('nan'),
            'blocks_assessed': len(blk), 'blocks_failed': int(len(blk) - sum(blk)),
            'mean_error_to_award_pct': 100.0 * float(np.sum(err[pmask]) / max(np.sum(award[pmask]), 1e-9)),
        }

    dt_h = 5.0 / 60.0
    dep = int(m['departing_evs'])
    out = {
        'task': task, 'bess_kw': float(bess_kw), 'bess_kwh': float(env.bess_energy_capacity_kwh) if bess_kw > 0 else 0.0,
        'day': day, 'command': c, 'ev_seed': e, 'service_date': str(base_bid.get('service_date')),
        'soc_hit_pct': 100.0 * int(m['central_departing_evs_soc_met']) / max(dep, 1), 'departed_evs': dep,
        'offered_capacity_kw_block': float(np.sum(up + down)),
        'regulation_energy_kwh': float(np.sum(np.abs(reg[pmask])) * dt_h),
        'bess_abs_energy_kwh': float(np.sum(np.abs(bp)) * dt_h),
        'bess_abs_energy_tracking_kwh': float(np.sum(np.abs(bp[pmask])) * dt_h),
        'bess_max_abs_kw': float(np.max(np.abs(bp))),
        'bess_power_limit_steps': int(plim[pmask].sum()), 'bess_energy_limit_steps': int(elim[pmask].sum()),
        'seconds': time.time() - t0,
    }
    out.update({f'pre_{k}': v for k, v in judge(pre).items()})
    out.update({f'post_{k}': v for k, v in judge(post).items()})
    return out


def main():
    """大きさを小さい順に回し、全エピソードで蓄電池の後の不適合が0になった大きさで止める。"""
    task = sys.argv[1]
    sizes = [float(x) for x in sys.argv[2].split(',')]
    n_cmd = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_ev = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    mp.set_start_method('spawn', force=True)
    out = HERE / f'result_{task}_rule_bess_{n_cmd}cmd_{n_ev}ev.jsonl'
    # 済んだエピソードは飛ばし、その不適合も数に入れる（途中で止めた後の再開）。
    done = {}
    if out.exists():
        for line in out.open(encoding='utf-8'):
            r = json.loads(line)
            done[(float(r['bess_kw']), r['day'], r['command'], r['ev_seed'])] = int(r['post_blocks_failed'])
    with mp.Pool(workers) as pool, open(out, 'a', encoding='utf-8') as f:
        for s in sizes:
            keys = [(s, d, c, e) for d in range(5) for c in range(n_cmd) for e in range(n_ev)]
            jobs = [(task, *k, n_cmd) for k in keys if k not in done]
            failed = sum(done.get(k, 0) for k in keys)
            for r in pool.imap_unordered(run_episode, jobs):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
                failed += int(r['post_blocks_failed'])
            print(json.dumps({"task": task, "bess_kw": s, "episodes": len(keys), 'post_blocks_failed': failed}), flush=True)
            if failed == 0:
                break


if __name__ == '__main__':
    main()

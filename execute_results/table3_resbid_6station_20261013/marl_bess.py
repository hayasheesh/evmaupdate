"""本線のMARL（6 station、既定は AEMO、ep2000、行動器だけ。市場は環境変数 MARL_BESS_TASK）に連系点の蓄電池を足し、蓄電池の大きさごとに30分コマの判定を測る（実験D3）。

    python marl_bess.py [bess_kw,...] [workers]

組と入札は表3と同じ40組（table3_6station_20261012/run_table3.py の _episode_inputs）。行動器の読み込みも run_table3 と同じ。
蓄電池の大きさの決め方は rule_bess_6station_20261011/run_rule_bess.py と同じ（目標SoC 50%から下限10%まで、30分を定格で出せる容量）。
誤差は蓄電池の前（EVだけ）と後（連系点）の両方を記録する。結果は result_aemo_marl_bess.jsonl、済んだ組は飛ばす。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.update(TABLE3_SW_DIR=str(HERE.parent / 'resbid_6station_20261012'), TABLE3_OUT_DIR=str(HERE),
                  TABLE3_EPISODE='2000', CUDA_VISIBLE_DEVICES='-1')
sys.path.insert(0, str(HERE.parent / 'table3_6station_20261012'))
import run_table3 as t3  # noqa: E402

TASK = os.environ.get('MARL_BESS_TASK', 'aemo')
OUT = HERE / f'result_{TASK}_marl_bess.jsonl'


def run_episode(args):
    bess_kw, day, c, e = args
    t3._setup(TASK)
    import numpy as np
    import torch
    from Config import EPISODE_STEPS
    from EnvConfig import BESS_DISCHARGE_EFFICIENCY, BESS_MIN_SOC_PCT, BESS_TARGET_SOC_PCT
    from environment.EVEnv import EVEnv
    from environment.normalize import load_observation_normalization_for_archive, normalize_observation, use_instruction_scale
    from market.bid_participation import participation_by_block
    from tools.evaluator import set_env_seed
    from training.system_controller import build_agent, find_model_path_and_episode

    fixed_bid, _command, target, tol, tracking, arr = t3._episode_inputs(TASK, day, c, 4)
    up = np.asarray(fixed_bid['up_plan'], float)
    down = np.asarray(fixed_bid['down_plan'], float)
    set_env_seed(4_000_000 + 1000 * day + 10 * c + e)
    use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.use_central_ev_residual_allocator = False
    env.use_residual_bess = bess_kw > 0
    if bess_kw > 0:
        env.bess_power_limit_kw = float(bess_kw)
        env.bess_energy_capacity_kwh = float(bess_kw) * 0.5 / (
            max((BESS_TARGET_SOC_PCT - BESS_MIN_SOC_PCT) / 100.0, 1e-9) * BESS_DISCHARGE_EFFICIENCY)
    env.reset(net_demand_series=target, tol_narrow_series=tol, tracking_enabled_series=tracking,
              market_context_series=arr.get('market_context_series'),
              arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
              service_date=arr.get('service_date'), baseline_series=arr.get('baseline_series'))
    d, ep = t3.model_dir_and_episode(TASK)
    load_observation_normalization_for_archive(d)
    agent = build_agent(env)
    ckpt, ep = find_model_path_and_episode(str(d / 'results' / f'TEST{ep}'), ep)
    agent.load_actors(ckpt, ep, map_location='cpu')
    agent.set_test_mode(True)
    pre = np.zeros(EPISODE_STEPS)
    post = np.zeros(EPISODE_STEPS)
    bp = np.zeros(EPISODE_STEPS)
    t0 = time.time()
    for t in range(EPISODE_STEPS):
        raw_obs = env.begin_step()
        agent.update_active_evs(env)
        with torch.no_grad():
            actions = torch.as_tensor(agent.act(normalize_observation(raw_obs), env=env, noise=False), dtype=torch.float32)
        _o, _lr, _gr, _d, info = env.apply_action(actions)
        pre[t] = float(info['pre_bess_residual_kw'])
        post[t] = float(info['post_bess_residual_kw'])
        bp[t] = float(info['bess_power_kw'])
    part = participation_by_block(up, down)
    pmask = np.repeat(part, 6)

    def judge(res):
        step_pass = np.abs(res) <= tol + 1e-6
        blk = [float(np.mean(step_pass[k * 6:(k + 1) * 6])) >= 0.90 for k in range(48) if part[k]]
        return {'step_pass_pct': 100.0 * float(np.mean(step_pass[tracking])),
                'block_pass_pct': 100.0 * float(np.mean(blk)) if blk else float('nan'),
                'blocks_assessed': len(blk), 'blocks_failed': int(len(blk) - sum(blk))}

    m = env.metrics
    out = {'task': TASK, 'bess_kw': float(bess_kw), 'bess_kwh': float(env.bess_energy_capacity_kwh) if bess_kw > 0 else 0.0,
           'day': day, 'command': c, 'ev_seed': e, 'model': d.name, 'episode': int(ep),
           'soc_hit_pct': 100.0 * int(m.get('central_departing_evs_soc_met', m['departing_evs_soc_met'])) / max(int(m['departing_evs']), 1),
           'bess_abs_energy_tracking_kwh': float(np.sum(np.abs(bp[pmask])) * 5 / 60),
           'bess_max_abs_kw': float(np.max(np.abs(bp))), 'seconds': time.time() - t0}
    out.update({f'pre_{k}': v for k, v in judge(pre).items()})
    out.update({f'post_{k}': v for k, v in judge(post).items()})
    return out


def main():
    sizes = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    done = set()
    if OUT.exists():
        done = {(r['bess_kw'], r['day'], r['command'], r['ev_seed']) for r in map(json.loads, OUT.open(encoding='utf-8'))}
    jobs = [(s, d, c, e) for s in sizes for d in range(5) for c in range(4) for e in range(2) if (s, d, c, e) not in done]
    mp.set_start_method('spawn', force=True)
    with mp.Pool(workers) as pool, OUT.open('a', encoding='utf-8') as f:
        for r in pool.imap_unordered(run_episode, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
    rows = [json.loads(x) for x in OUT.open(encoding='utf-8')]
    for s in sorted({r['bess_kw'] for r in rows}):
        rs = [r for r in rows if r['bess_kw'] == s]
        print(json.dumps({'bess_kw': s, 'sets': len(rs), 'post_blocks_failed': sum(r['post_blocks_failed'] for r in rs),
                          'post_block_pass_pct': round(sum(r['post_block_pass_pct'] for r in rs) / len(rs), 2)}), flush=True)


if __name__ == '__main__':
    main()

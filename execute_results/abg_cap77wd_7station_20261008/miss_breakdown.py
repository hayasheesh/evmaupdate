"""保存した行動器を検証用の入札で動かし、帯を外した時刻（BESSの前、入札したコマ）を状況ごとに分ける（学習はしない）。

    python miss_breakdown.py <aemo|ercot> [commands_per_day] [ev_seeds]

各時刻について、その時刻の初めに EV 群が出せる電力の範囲（step_dispatch_envelope）、指令の変わり幅、
SoC の上下限で切られた電力（行動器の出力 − 実際の電力）を記録する。
2026-10-09 夕方より前の版は、切られた電力を枠の並びの定格で計算していて誤り（miss_breakdown_*_ep1300.json の soc_clip の項目は使わない）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_worker  # noqa: E402

task = sys.argv[1]
n_cmd = int(sys.argv[2]) if len(sys.argv) > 2 else 8
n_ev = int(sys.argv[3]) if len(sys.argv) > 3 else 2
config = stage_worker.configure(task)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Config import EPISODE_STEPS  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.normalize import load_observation_normalization_for_archive, normalize_observation, use_instruction_scale  # noqa: E402
from environment.physical_capability import step_dispatch_envelope  # noqa: E402
from tools.evaluator import set_env_seed  # noqa: E402
from training.bid_bank import BidBank  # noqa: E402
from training.lower_bid_training import _activation_scenarios_for_day, build_fixed_upper_bid_training_episode  # noqa: E402
from training.system_controller import build_agent, find_model_path_and_episode  # noqa: E402

ROOT = HERE.parents[1]
model_dir = sorted((ROOT / 'archive').glob(f'{stage_worker.TASKS[task][1]}_2*'))[-1]
episode = max(int(p.name[4:]) for p in (model_dir / 'results').glob('TEST*') if p.name[4:].isdigit())
_train, test_bank = stage_worker.banks(task)
bank = BidBank(test_bank)
load_observation_normalization_for_archive(model_dir)
agent = None
rows = []
for day, entry in enumerate(bank.entries):
    base_bid = dict(bank.load_entry(entry))
    payloads, mode = _activation_scenarios_for_day(base_bid.get('service_date'), int(base_bid.get('forecast_seed', 0)) + 512_209,
                                                   n_scenarios=n_cmd, scenario_partition='holdout')
    fixed_bid = dict(base_bid, activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=mode)
    for c in range(len(payloads)):
        target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, c)
        tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
        for e in range(n_ev):
            set_env_seed(4_000_000 + 1000 * day + 10 * c + e)
            use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
            env = EVEnv()
            env.use_central_ev_residual_allocator = False
            env.use_residual_bess = False
            env.reset(net_demand_series=target, tol_narrow_series=tol, tracking_enabled_series=tracking,
                      market_context_series=arr.get('market_context_series'),
                      arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
                      service_date=arr.get('service_date'), baseline_series=arr.get('baseline_series'))
            if agent is None:
                agent = build_agent(env)
                ckpt, ep = find_model_path_and_episode(str(model_dir / 'results' / f'TEST{episode}'), episode)
                agent.load_actors(ckpt, ep, map_location='cpu')
                agent.set_test_mode(True)
            prev_req = None
            for t in range(EPISODE_STEPS):
                obs = normalize_observation(env.begin_step())
                on = bool(env.current_tracking_enabled)
                b = float(env.tol_narrow_metrics)
                req = float(env.current_net_demand)
                pmin, pmax, _obl = step_dispatch_envelope(env)
                n_ev_now = int(env.ev_mask.sum().item())
                agent.update_active_evs(env)
                with torch.no_grad():
                    actions = agent.act(obs, env=env, noise=False)
                # Actions and the env's EV powers are both in each station's
                # departure-sorted active order, so the ratings must be too.
                order = [env._get_sorted_active_evs(st).clone() for st in range(env.num_stations)]
                a_t = torch.as_tensor(actions, dtype=torch.float32)
                _, _, _, _, info = env.apply_action(actions)
                actual = torch.as_tensor(info['raw_actor_ev_power_kw'], dtype=torch.float32)
                clip = 0.0
                for st, idx in enumerate(order):
                    k = int(idx.numel())
                    if k:
                        rated = env.ev_max_power_kw[st, idx]
                        nominal = torch.clamp(a_t[st, :k].to(rated.device), -1.0, 1.0) * rated
                        clip += float((nominal - actual[st, :k].to(rated.device)).abs().sum().item())
                if on:
                    rows.append((float(env.last_raw_actor_residual_kw), b, req, pmin, pmax,
                                 abs(req - prev_req) if prev_req is not None else 0.0,
                                 t % 6, t // 12, clip, n_ev_now))
                prev_req = req if on else None
a = np.asarray(rows, float)
r, b, req, pmin, pmax, dreq, sib, hour, clip, nev = a.T
miss = np.abs(r) > b
infeasible = (req - b > pmax) | (req + b < pmin)
edge = ~infeasible & ((req + b > pmax - 0) & (req - b > pmax - 2 * b) | (req - b < pmin + 0) & (req + b < pmin + 2 * b))
near_edge = ~infeasible & ((pmax - req < 2 * b) | (req - pmin < 2 * b))
big_jump = dreq > 2 * b
clip_big = clip > b
first_step = sib == 0
short = r > 0   # EV power below the command (too little import)


def share(mask):
    return float(100 * mask.mean()) if mask.size else float('nan')


m = miss
out = {
    'task': task, 'model': model_dir.name, 'episode': episode, 'tracked_steps': int(len(a)),
    'step_pass_pct': share(~m), 'miss_steps': int(m.sum()),
    'of_misses_pct': {
        'infeasible_at_step_start': share(infeasible[m]),
        'feasible_command_within_2_bands_of_fleet_limit': share((near_edge & ~infeasible)[m]),
        'feasible_command_jump_over_2_bands': share((big_jump & ~infeasible)[m]),
        'feasible_first_step_of_block': share((first_step & ~infeasible)[m]),
        'feasible_soc_clip_over_band': share((clip_big & ~infeasible)[m]),
        'feasible_none_of_these': share((~infeasible & ~near_edge & ~big_jump & ~first_step & ~clip_big)[m]),
        'ev_power_below_command': share(short[m]),
    },
    'miss_rate_pct_by_condition': {
        'all': share(m),
        'infeasible_at_step_start': share(m[infeasible]),
        'near_fleet_limit': share(m[near_edge]),
        'not_near_limit': share(m[~near_edge & ~infeasible]),
        'command_jump_over_2_bands': share(m[big_jump]),
        'no_big_jump': share(m[~big_jump]),
        'first_step_of_block': share(m[first_step]),
        'other_steps_of_block': share(m[~first_step]),
        'soc_clip_over_band': share(m[clip_big]),
        'soc_clip_within_band': share(m[~clip_big]),
    },
    'condition_share_pct_of_tracked_steps': {
        'infeasible': share(infeasible), 'near_fleet_limit': share(near_edge),
        'command_jump_over_2_bands': share(big_jump), 'soc_clip_over_band': share(clip_big),
    },
    'miss_rate_pct_by_hour': {int(h): share(m[hour == h]) for h in range(24) if (hour == h).any()},
    'abs_error_kw_misses_median': float(np.median(np.abs(r[m]))) if m.any() else None,
    'band_kw_misses_median': float(np.median(b[m])) if m.any() else None,
    'band_kw_all_median': float(np.median(b)),
}
path = HERE / f'miss_breakdown_{task}_ep{episode}.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))

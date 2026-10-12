"""帯を外した時刻を、割り当て直せば帯に入ったかで分ける（学習はしない）。

    python miss_allocation.py <aemo|ercot> [commands_per_day] [ev_seeds]

行動器の出力と環境の EV 電力は、どちらも各 station の在車 EV を出発の早い順に詰めた並び
（EVEnv._get_sorted_active_evs）なので、上限も同じ並びで作る。上限は step_dispatch_envelope と同じ式。
帯を外した時刻について、帯に入るのに足りない電力 d（|誤差| − 帯）を次の順に分ける。
  同じ station 内：行動器が求めたのに上限で切られた電力のうち、同じ station の余裕のある EV へ移せる分の合計が d 以上
  他の station も含む：群全体の余裕は d 以上だが、上の移し替えでは届かない（求めた電力そのものが足りない）
  群全体でも無理：その時刻の初めの群全体の余裕が d に届かない
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
from EnvConfig import POWER_TO_ENERGY  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.normalize import load_observation_normalization_for_archive, normalize_observation, use_instruction_scale  # noqa: E402
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
dt = float(POWER_TO_ENERGY)
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
            for t in range(EPISODE_STEPS):
                obs = normalize_observation(env.begin_step())
                on = bool(env.current_tracking_enabled)
                b = float(env.tol_narrow_metrics)
                order = [env._get_sorted_active_evs(st).clone() for st in range(env.num_stations)]
                caps = []
                for st, idx in enumerate(order):
                    soc = env.soc[st, idx]
                    kwh = env.ev_kwh_per_soc_pct[st, idx]
                    rated = env.ev_max_power_kw[st, idx]
                    caps.append((torch.minimum(rated, (100.0 - soc) * kwh / dt).clamp(min=0.0),
                                 torch.minimum(rated, soc * kwh / dt).clamp(min=0.0), rated))
                agent.update_active_evs(env)
                with torch.no_grad():
                    actions = agent.act(obs, env=env, noise=False)
                actions = torch.as_tensor(actions, dtype=torch.float32)
                _, _, _, _, info = env.apply_action(actions)
                if not on:
                    continue
                raw = torch.as_tensor(info['raw_actor_ev_power_kw'], dtype=torch.float32)
                st_rows = []
                for st, idx in enumerate(order):
                    k = int(idx.numel())
                    if k == 0:
                        st_rows.append((0.0, 0.0, 0.0, 0.0))
                        continue
                    cap_up, cap_dn, rated = caps[st]
                    nominal = torch.clamp(actions[st, :k].to(rated.device), -1.0, 1.0) * rated
                    actual = raw[st, :k].to(rated.device)
                    st_rows.append((
                        float((nominal - actual).clamp(min=0).sum()),   # asked import that was cut
                        float((actual - nominal).clamp(min=0).sum()),   # asked export that was cut
                        float((cap_up - actual).clamp(min=0).sum()),    # import room left
                        float((actual + cap_dn).clamp(min=0).sum()),    # export room left
                    ))
                rows.append((float(env.last_raw_actor_residual_kw), b, st_rows))

n_miss = 0
cat = {'within_station': 0, 'other_stations': 0, 'fleet_short': 0}
clip_over_band = 0
clip_over_band_all = 0
for r, b, st_rows in rows:
    clip_total = sum(s[0] + s[1] for s in st_rows)
    clip_over_band_all += clip_total > b
    if abs(r) <= b:
        continue
    n_miss += 1
    clip_over_band += clip_total > b
    need = abs(r) - b
    if r > 0:   # EV power below the command: more import was needed
        movable = sum(min(s[0], s[2]) for s in st_rows)
        fleet_room = sum(s[2] for s in st_rows)
    else:
        movable = sum(min(s[1], s[3]) for s in st_rows)
        fleet_room = sum(s[3] for s in st_rows)
    if fleet_room < need:
        cat['fleet_short'] += 1
    elif movable >= need:
        cat['within_station'] += 1
    else:
        cat['other_stations'] += 1
short = sum(1 for r, b, _ in rows if abs(r) > b and r > 0)
out = {
    'task': task, 'model': model_dir.name, 'episode': episode, 'tracked_steps': len(rows), 'miss_steps': n_miss,
    'step_pass_pct': 100.0 * (1 - n_miss / max(len(rows), 1)),
    'of_misses_pct': {k: 100.0 * v / max(n_miss, 1) for k, v in cat.items()},
    'misses_with_soc_cut_over_band_pct': 100.0 * clip_over_band / max(n_miss, 1),
    'all_steps_with_soc_cut_over_band_pct': 100.0 * clip_over_band_all / max(len(rows), 1),
    'misses_ev_power_below_command_pct': 100.0 * short / max(n_miss, 1),
}
path = HERE / f'miss_allocation_{task}_ep{episode}.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))

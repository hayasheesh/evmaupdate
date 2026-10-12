"""保存した行動器を検証用の入札で動かし、SoC の目標を外した EV を1台ずつ調べる（学習はしない）。

    python soc_miss_analysis.py <aemo|ercot> [commands_per_day] [ev_seeds] [actor|rule]

actor は保存した行動器（中央の配分も BESS もなし）。rule はリポジトリのルールベース（rule_based_central：
行動は0、出発に間に合わない EV だけ充電を強制し、中央が station ごとの範囲を見て指令を配る）。どちらも EV の実際の
電力と BESS 前のずれで数える。

EV ごとに、到着時の SoC と目標、滞在、定格、目標までに要る量 ÷ 滞在中ずっと定格で充電した量（必要な充電時間の割合）、
出発時の SoC を記録する。各時刻について、EV が定格より絞られていたか（目標未達のまま実際の電力 < 定格）、
放電していたかと、そのときの全体のずれの向き（EV 群の電力が指令より少ないか）、追従の有無を記録する。
全体が指令より少ない（もっと充電してよい）時刻に目標未達の EV を絞るのは、追従にも SoC にも得のない絞り方である。
出発時の判定は環境と同じく、出発時の SoC が目標以上なら達成（環境の集計と一致するかも確かめる）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_worker  # noqa: E402

task = sys.argv[1]
n_cmd = int(sys.argv[2]) if len(sys.argv) > 2 else 8
n_ev = int(sys.argv[3]) if len(sys.argv) > 3 else 2
run_mode = sys.argv[4] if len(sys.argv) > 4 else 'actor'
assert run_mode in ('actor', 'rule')
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
from training.system_controller import apply_force_charging, build_agent, find_model_path_and_episode  # noqa: E402

ROOT = HERE.parents[1]
model_dir = sorted((ROOT / 'archive').glob(f'{stage_worker.TASKS[task][1]}_2*'))[-1]
episode = max(int(p.name[4:]) for p in (model_dir / 'results').glob('TEST*') if p.name[4:].isdigit())
_train, test_bank = stage_worker.banks(task)
bank = BidBank(test_bank)
load_observation_normalization_for_archive(model_dir)
dt = float(POWER_TO_ENERGY)
agent = None
evs = []            # one dict per departed EV
env_hit = env_dep = 0
tracked = passed = 0
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
            env.use_central_ev_residual_allocator = (run_mode == 'rule')
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
            live = {}
            for t in range(EPISODE_STEPS):
                obs = normalize_observation(env.begin_step())
                on = bool(env.current_tracking_enabled)
                b = float(env.tol_narrow_metrics)
                order = [env._get_sorted_active_evs(st).clone() for st in range(env.num_stations)]
                before = []
                for st, idx in enumerate(order):
                    ids = env.ev_ids[st, idx].tolist()
                    soc = env.soc[st, idx].tolist()
                    tgt = env.target[st, idx].tolist()
                    dep = env.depart[st, idx].tolist()
                    rated = env.ev_max_power_kw[st, idx].tolist()
                    spk = env.ev_soc_pct_per_kwh[st, idx].tolist()
                    for k, i in enumerate(ids):
                        if i not in live:
                            stay = int(dep[k]) - int(env.step_count) + 1
                            need = max(tgt[k] - soc[k], 0.0)
                            full = stay * rated[k] * dt * spk[k]
                            live[i] = dict(station=st, arrive_step=int(env.step_count), init=soc[k], target=tgt[k], rated=rated[k],
                                           stay=stay, duty=need / full if full > 0 else 0.0,
                                           held=0, held_fleet_short=0, held_fleet_short_on=0, held_on=0, held_off=0,
                                           discharge=0, discharge_fleet_short=0, min_slack=1e9)
                        rec = live[i]
                        left = int(dep[k]) - int(env.step_count) + 1
                        need_steps = max(tgt[k] - soc[k], 0.0) / max(rated[k] * dt * spk[k], 1e-9)
                        rec['min_slack'] = min(rec['min_slack'], left - need_steps)
                    before.append((ids, soc, tgt, rated, spk, dep))
                if run_mode == 'actor':
                    agent.update_active_evs(env)
                    with torch.no_grad():
                        actions = torch.as_tensor(agent.act(obs, env=env, noise=False), dtype=torch.float32)
                else:
                    actions, _f, _ = apply_force_charging(env.soc.new_zeros((env.num_stations, env.max_ev_per_station)),
                                                          env, slack_kwh=0.1)
                _, _, _, _, info = env.apply_action(actions)
                raw = torch.as_tensor(info['actual_ev_power_kw'], dtype=torch.float32)
                r = float(info['pre_bess_residual_kw'])
                if on:
                    tracked += 1
                    passed += int(abs(r) <= b)
                fleet_short = r > 0          # EV power below the command: more charging would also track better
                for st, (ids, soc, tgt, rated, spk, dep) in enumerate(before):
                    for k, i in enumerate(ids):
                        p = float(raw[st, k])
                        rec = live[i]
                        if soc[k] < tgt[k] and p < 0.99 * rated[k]:
                            rec['held'] += 1
                            rec['held_on' if on else 'held_off'] += 1
                            if fleet_short:
                                rec['held_fleet_short'] += 1
                                rec['held_fleet_short_on'] += int(on)
                        if p < 0:
                            rec['discharge'] += 1
                            rec['discharge_fleet_short'] += int(fleet_short)
                        if int(dep[k]) == int(env.step_count):
                            final = min(max(soc[k] + p * dt * spk[k], 0.0), 100.0)
                            rec['final'] = final
                            rec['hit'] = final >= tgt[k] - 1e-6
                            rec['deficit_pct'] = max(tgt[k] - final, 0.0)
                            rec['deficit_full_steps'] = rec['deficit_pct'] / max(rated[k] * dt * spk[k], 1e-9)
                            rec['day'] = day
                            evs.append(live.pop(i))
            env_hit += env.metrics['central_departing_evs_soc_met' if run_mode == 'rule' else 'departing_evs_soc_met']
            env_dep += env.metrics['departing_evs']

hit = np.array([v['hit'] for v in evs])
duty = np.array([v['duty'] for v in evs])
defs = np.array([v['deficit_full_steps'] for v in evs])
miss = ~hit


def pct(m):
    return float(100 * m.mean()) if m.size else float('nan')


bins = [(0, 1e-9), (1e-9, 0.25), (0.25, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 10)]
by_duty = {}
for lo, hi in bins:
    m = (duty >= lo) & (duty < hi) if hi > 1e-9 else (duty <= 1e-9)
    by_duty[f'{lo:.2f}-{hi:.2f}'] = {'evs_pct': pct(m), 'hit_pct': pct(hit[m]) if m.any() else None,
                                     'share_of_misses_pct': float(100 * (m & miss).sum() / max(miss.sum(), 1))}
mv = [v for v in evs if not v['hit']]
held = np.array([v['held'] for v in mv], float)
out = {
    'task': task, 'mode': run_mode, 'model': model_dir.name if run_mode == 'actor' else None, 'episode': episode if run_mode == 'actor' else None,
    'departed_evs': len(evs), 'step_pass_pct': 100.0 * passed / max(tracked, 1), 'tracked_steps': tracked,
    'hit_pct': pct(hit), 'env_hit_pct': 100.0 * env_hit / max(env_dep, 1), 'env_departed': int(env_dep),
    'by_needed_charging_time_share': by_duty,
    'misses': {
        'count': int(miss.sum()),
        'deficit_in_full_power_steps_quantiles': {q: float(np.quantile(defs[miss], q / 100)) for q in (25, 50, 75, 90)} if miss.any() else None,
        'deficit_le_1_step_pct': pct(defs[miss] <= 1.0) if miss.any() else None,
        'deficit_le_3_steps_pct': pct(defs[miss] <= 3.0) if miss.any() else None,
        'became_unreachable_pct': pct(np.array([v['min_slack'] < 0 for v in mv])) if mv else None,
        'held_steps_mean': float(held.mean()) if mv else None,
        'held_steps_while_fleet_below_command_share_pct': float(100 * sum(v['held_fleet_short'] for v in mv) / max(held.sum(), 1)),
        'held_steps_tracking_off_share_pct': float(100 * sum(v['held_off'] for v in mv) / max(held.sum(), 1)),
        'held_steps_fleet_below_command_and_tracking_on_share_pct': float(100 * sum(v['held_fleet_short_on'] for v in mv) / max(held.sum(), 1)),
        'misses_with_any_discharge_pct': pct(np.array([v['discharge'] > 0 for v in mv])) if mv else None,
        'deficit_le_held_fleet_short_steps_pct': pct(np.array([v['deficit_full_steps'] <= v['held_fleet_short'] for v in mv])) if mv else None,
        'deficit_le_held_tracking_off_steps_pct': pct(np.array([v['deficit_full_steps'] <= v['held_off'] for v in mv])) if mv else None,
    },
    'hits': {
        'held_steps_mean': float(np.mean([v['held'] for v in evs if v['hit']])) if hit.any() else None,
    },
    'miss_rate_by_arrival_hour': {h: pct(miss[np.array([v['arrive_step'] // 12 == h for v in evs])])
                                  for h in range(24) if any(v['arrive_step'] // 12 == h for v in evs)},
}
path = HERE / (f'soc_miss_analysis_{task}_ep{episode}.json' if run_mode == 'actor' else f'soc_miss_analysis_{task}_rule.json')
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))

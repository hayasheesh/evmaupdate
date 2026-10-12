"""保存した行動器を検証用の入札で動かし、SoC の外れと帯の外れが特定の日・時刻・station・指令に集まっていないかを調べる（学習はしない）。

    python failure_concentration.py <aemo|ercot> [commands_per_day] [ev_seeds]

EV ごとと時刻ごとの記録を CSV に書き、日・時刻・station・指令ごとの集計を JSON に書く。
帯の判定は追従するコマの5分ごと（BESS の前）、SoC は出発時の SoC が目標以上なら達成（環境の集計と同じ）。
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_worker  # noqa: E402

task = sys.argv[1]
n_cmd = int(sys.argv[2]) if len(sys.argv) > 2 else 4
n_ev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
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
ev_rows, step_rows = [], []
for day, entry in enumerate(bank.entries):
    base_bid = dict(bank.load_entry(entry))
    date = str(base_bid.get('service_date'))
    payloads, act_mode = _activation_scenarios_for_day(date, int(base_bid.get('forecast_seed', 0)) + 512_209,
                                                       n_scenarios=n_cmd, scenario_partition='holdout')
    fixed_bid = dict(base_bid, activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=act_mode)
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
            live = {}
            for t in range(EPISODE_STEPS):
                obs = normalize_observation(env.begin_step())
                on = bool(env.current_tracking_enabled)
                b = float(env.tol_narrow_metrics)
                req = float(env.current_net_demand)
                before = []
                for st in range(env.num_stations):
                    idx = env._get_sorted_active_evs(st).clone()
                    ids = env.ev_ids[st, idx].tolist()
                    soc = env.soc[st, idx].tolist()
                    tgt = env.target[st, idx].tolist()
                    dep = env.depart[st, idx].tolist()
                    rated = env.ev_max_power_kw[st, idx].tolist()
                    spk = env.ev_soc_pct_per_kwh[st, idx].tolist()
                    for k, i in enumerate(ids):
                        if i not in live:
                            stay = int(dep[k]) - int(env.step_count) + 1
                            full = stay * rated[k] * dt * spk[k]
                            live[i] = dict(task=task, date=date, day=day, cmd=c, station=st, arrive_step=int(env.step_count),
                                           depart_step=int(dep[k]), init=round(soc[k], 2), target=round(tgt[k], 2),
                                           duty=round(max(tgt[k] - soc[k], 0.0) / full, 4) if full > 0 else 0.0)
                    before.append((ids, soc, tgt, spk, dep))
                n_present = sum(len(x[0]) for x in before)
                agent.update_active_evs(env)
                with torch.no_grad():
                    actions = torch.as_tensor(agent.act(obs, env=env, noise=False), dtype=torch.float32)
                _, _, _, _, info = env.apply_action(actions)
                power = torch.as_tensor(info['actual_ev_power_kw'], dtype=torch.float32)
                r = float(info['pre_bess_residual_kw'])
                step_rows.append(dict(task=task, date=date, day=day, cmd=c, step=t, tracked=int(on), band_kw=round(b, 3),
                                      command_kw=round(req, 3), residual_kw=round(r, 3), passed=int(abs(r) <= b) if on else '',
                                      evs_present=n_present))
                for st, (ids, soc, tgt, spk, dep) in enumerate(before):
                    for k, i in enumerate(ids):
                        if int(dep[k]) == int(env.step_count):
                            p = float(power[st, k])
                            final = min(max(soc[k] + p * dt * spk[k], 0.0), 100.0)
                            rec = live.pop(i)
                            rec['final'] = round(final, 2)
                            rec['hit'] = int(final >= tgt[k] - 1e-6)
                            ev_rows.append(rec)

tag = f'{task}_ep{episode}'
for name, rows in (('evs', ev_rows), ('steps', step_rows)):
    with open(HERE / f'failure_concentration_{name}_{tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def rate(rows, key, ok):
    groups = defaultdict(list)
    for r in rows:
        if r.get(ok) == '':
            continue
        groups[r[key]].append(int(r[ok]))
    return {str(k): {'n': len(v), 'fail_pct': round(100 * (1 - np.mean(v)), 1)} for k, v in sorted(groups.items())}


tracked = [r for r in step_rows if r['tracked']]
for r in tracked:
    r['hour'] = r['step'] // 12
    r['date_hour'] = f"{r['date']} {r['step'] // 12:02d}"
for r in ev_rows:
    r['arrive_hour'] = (r['arrive_step'] - 1) // 12
    r['depart_hour'] = min((r['depart_step'] - 1) // 12, 23) if r['depart_step'] <= EPISODE_STEPS else 24
    r['station_date'] = f"{r['station']} {r['date']}"


def concentration(rows, key, ok):
    """Share of all failures that falls in the worst 10% / 25% of groups (by failure count)."""
    fails = defaultdict(int)
    for r in rows:
        if r.get(ok) == '':
            continue
        fails[r[key]] += 1 - int(r[ok])
    counts = sorted(fails.values(), reverse=True)
    tot = max(sum(counts), 1)
    k10 = max(1, int(round(0.1 * len(counts))))
    k25 = max(1, int(round(0.25 * len(counts))))
    return {'groups': len(counts), 'top10pct_groups_share_of_failures_pct': round(100 * sum(counts[:k10]) / tot, 1),
            'top25pct_groups_share_of_failures_pct': round(100 * sum(counts[:k25]) / tot, 1)}


out = {
    'task': task, 'model': model_dir.name, 'episode': episode,
    'tracking': {
        'tracked_steps': len(tracked), 'fail_pct': round(100 * (1 - np.mean([r['passed'] for r in tracked])), 2),
        'by_date': rate(tracked, 'date', 'passed'), 'by_cmd': rate(tracked, 'cmd', 'passed'),
        'by_hour': rate(tracked, 'hour', 'passed'),
        'concentration_date_hour': concentration(tracked, 'date_hour', 'passed'),
    },
    'soc': {
        'evs': len(ev_rows), 'miss_pct': round(100 * (1 - np.mean([r['hit'] for r in ev_rows])), 2),
        'by_station': rate(ev_rows, 'station', 'hit'), 'by_date': rate(ev_rows, 'date', 'hit'),
        'by_cmd': rate(ev_rows, 'cmd', 'hit'), 'by_arrive_hour': rate(ev_rows, 'arrive_hour', 'hit'),
        'by_depart_hour': rate(ev_rows, 'depart_hour', 'hit'),
        'concentration_station_date': concentration(ev_rows, 'station_date', 'hit'),
    },
}
path = HERE / f'failure_concentration_{tag}.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))

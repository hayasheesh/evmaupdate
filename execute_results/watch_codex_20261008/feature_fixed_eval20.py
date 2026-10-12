"""同じ保存行動器で休日・帯幅の観測を1つずつ固定して比較する。"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = Path(__file__).resolve().parent
task, episode = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, str(HERE.parent / f'allin_20station_{task}_20261007'))
import stage_worker
config = stage_worker.configure()

import numpy as np
import torch
from Config import EPISODE_STEPS
from environment.EVEnv import EVEnv
from environment.normalize import load_observation_normalization_for_archive, normalize_observation, use_instruction_scale
from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES
from tools.evaluator import set_env_seed
from training.bid_bank import BidBank
from training.lower_bid_training import _activation_scenarios_for_day, build_fixed_upper_bid_training_episode
from training.system_controller import build_agent, find_model_path_and_episode

torch.set_num_threads(1)
assert not torch.cuda.is_available()
ROOT = HERE.parents[1]
names = {'aemo': 'allin_aemoplan_AB_20station_20261007_042318', 'ercot': 'allin_ercotplan_AB_20station_20261007_042436'}
run = ROOT / 'archive' / names[task]
load_observation_normalization_for_archive(run)
bank = BidBank(stage_worker.TEST_BANK)
prepared = []
for day, entry in enumerate(bank.entries):
    fixed = dict(bank.load_entry(entry))
    payloads, mode = _activation_scenarios_for_day(fixed.get('service_date'), int(fixed['forecast_seed']) + 512_209,
                                                 n_scenarios=1, scenario_partition='holdout')
    fixed.update(activation_scenario_payload=payloads, activation_scenarios=1, activation_mode=mode)
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed, 0)
    tracking = np.asarray(arr['tracking_enabled_series'], dtype=bool)[:EPISODE_STEPS]
    prepared.append(dict(day=day, date=fixed['service_date'], source=payloads[0]['source'], seed=3_000_000+day,
                         target=target, tol=np.asarray(tol), arr=arr, tracking=tracking))
fixed_band = float(np.concatenate([p['tol'][p['tracking']] for p in prepared]).mean())

def reset(p):
    set_env_seed(p['seed'])
    use_instruction_scale(p['arr'].get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.use_central_ev_residual_allocator = False
    env.use_residual_bess = False
    env.reset(net_demand_series=p['target'], tol_narrow_series=p['tol'], tracking_enabled_series=p['tracking'],
              market_context_series=p['arr'].get('market_context_series'),
              arrival_probabilities_by_station=p['arr'].get('arrival_probabilities_by_station'),
              service_date=p['arr'].get('service_date'), baseline_series=p['arr'].get('baseline_series'))
    return env

# Build and load once, before resetting the identical RNG stream for every rollout.
probe = reset(prepared[0])
agent = build_agent(probe)
ckpt, ep = find_model_path_and_episode(str(run), episode)
agent.load_actors(ckpt, ep, map_location='cpu')
agent.set_test_mode(True)
rows = []
reference_hashes = {}
for mode in ('normal', 'holiday_weekday', 'band_constant'):
    for p in prepared:
        env = reset(p)
        ev_dim = env._get_obs().shape[1] - len(LOCAL_TAIL_FEATURE_NAMES)
        holiday_i = ev_dim + LOCAL_TAIL_FEATURE_NAMES.index('service_day_holiday')
        band_i = ev_dim + LOCAL_TAIL_FEATURE_NAMES.index('tracking_band_kw')
        frozen = {'events': env._arrival_events, 'initial_soc': env.soc.cpu().tolist(),
                  'capacity': env.ev_capacity_kwh.cpu().tolist(), 'power': env.ev_max_power_kw.cpu().tolist(),
                  'depart': env.depart.cpu().tolist(), 'target': env.target.cpu().tolist()}
        signature = hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()
        if mode == 'normal': reference_hashes[p['day']] = signature
        else: assert signature == reference_hashes[p['day']], 'EV realization changed'
        agent.episode_start()
        agent.update_active_evs(env)
        passed = []
        enabled = []
        with torch.no_grad():
            for _ in range(EPISODE_STEPS):
                raw = torch.as_tensor(env.begin_step()).clone()
                if mode == 'holiday_weekday': raw[:, holiday_i] = 0.0
                if mode == 'band_constant': raw[:, band_i] = fixed_band
                obs = normalize_observation(raw)
                agent.update_active_evs(env)
                actions = agent.act(obs, env=env, noise=False)
                env.apply_action(actions)
                enabled.append(bool(env.current_tracking_enabled))
                passed.append(abs(float(env.last_raw_actor_residual_kw)) <= float(env.tol_narrow_metrics))
        m = env.get_metrics()
        en = np.asarray(enabled).reshape(48, 6)
        ok = np.asarray(passed).reshape(48, 6)
        blocks = en.any(1)
        block_ok = np.all(ok | ~en, axis=1)
        track = float(m['tracking_success_rate'])
        soc = 100.0 - float(m['soc_miss_rate'])
        row = dict(mode=mode, day=p['day'], date=p['date'], command_source=p['source'], ev_seed=p['seed'], ev_hash=signature,
                   holiday=bool(p['arr'].get('service_date') and float(env._get_obs()[0, holiday_i]) > 0.5),
                   tracking_pct=track, soc_pct=soc, sum=track+soc, mae_kw=float(m['raw_actor_mae_kw']),
                   assessed_steps=int(m['tracking_steps']), passed_steps=int(m['tracking_within_narrow']),
                   departing_evs=int(m['departing_evs']), soc_met_evs=int(m['departing_evs_soc_met']),
                   participating_blocks=int(blocks.sum()), passed_blocks=int((block_ok & blocks).sum()),
                   block_pass_pct=float(100 * (block_ok & blocks).sum() / max(blocks.sum(), 1)))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

summary = {}
for mode in ('normal', 'holiday_weekday', 'band_constant'):
    rr = [r for r in rows if r['mode'] == mode]
    summary[mode] = {k: float(np.mean([r[k] for r in rr])) for k in ('tracking_pct', 'soc_pct', 'sum', 'mae_kw', 'block_pass_pct')}
result = dict(task=task, episode=episode, run_dir=str(run), device='cpu',
              fixed_band_kw=fixed_band, band_rule='mean actual band over assessed steps of these five validation days',
              comparison='only actor observation altered; physical EVs, targets, assessment bands and RNG streams unchanged',
              block_rule='all six assessed five-minute steps within their own bands; simulation approximation',
              summary=summary, rows=rows)
out = HERE / f'feature_fixed_eval20_{task}_ep{episode}.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved', out, json.dumps(summary, ensure_ascii=False), flush=True)

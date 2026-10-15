"""Complete forecast -> learned bid -> EV control -> hierarchical rewards -> DDPG loop."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from settings import configure, banks


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode(learner, payload, command, *, seed, train, common_bid=False):
    import numpy as np
    import torch
    from environment.EVEnv import EVEnv
    from environment.normalize import normalize_observation, use_instruction_scale
    from tools.evaluator import set_env_seed
    from data import forecast_features, decode_bid, committed_action, make_episode_inputs
    from data import identities, correspondence, individual_rewards

    sp = learner.spec
    forecast = forecast_features(payload, sp)
    raw_bid = learner.choose_bid(forecast, noise=.2 if train else 0)
    bid = copy.deepcopy(payload) if common_bid else decode_bid(payload, raw_bid, sp)
    if common_bid:
        bid['learned_price_fields'] = (raw_bid[:, [1, 4, 5]] + 1) / 2
    target, tol, arr, _ = make_episode_inputs(bid, command)
    market_actions = committed_action(bid, sp)
    set_env_seed(seed)
    use_instruction_scale(arr['instruction_scale_kw'])
    env = EVEnv()
    env.use_central_ev_residual_allocator = False
    env.use_residual_bess = False
    env.reset(net_demand_series=target, tol_narrow_series=tol,
              tracking_enabled_series=arr['tracking_enabled_series'],
              market_context_series=arr['market_context_series'],
              arrival_probabilities_by_station=arr['arrival_probabilities_by_station'],
              service_date=arr['service_date'], baseline_series=arr['baseline_series'])
    state = normalize_observation(env.begin_step())
    day_return = 0.0
    residuals = []
    actual_targets = []
    start = time.perf_counter()
    for step in range(sp.episode_steps):
        block = min(step // (sp.episode_steps // sp.blocks), sp.blocks - 1)
        action, reference, ev_reference = learner.act(state, market_actions[block], noise=.2 if train else 0)
        before = identities(env)
        _, r_local, r_global, dones, info = env.apply_action(action, build_info=False, return_observation=False)
        current_target = float(env.current_net_demand)
        done = bool(np.all(dones))
        powers = torch.as_tensor(info['raw_actor_ev_power_kw'], device=learner.device).reshape(-1) / sp.charger_kw
        ev_reward = individual_rewards(state.to(learner.device), powers, ev_reference, sp)
        next_state = normalize_observation(env._get_obs() if done else env.begin_step())
        after = identities(env)
        next_block = min((step + 1) // (sp.episode_steps // sp.blocks), sp.blocks - 1)
        capacity = float(bid['up_plan'][block] + bid['down_plan'][block]) / (2 * sp.stations * sp.slots * sp.charger_kw)
        upper_reward = float(r_global) + sp.capacity_scale_reward * capacity
        upper_reward += sp.leader_local_scale * sp.local_scale * float(torch.as_tensor(r_local).mean())
        day_return += sp.gamma ** step * upper_reward
        residuals.append(float(info['pre_bess_residual_kw']))
        actual_targets.append(current_target)
        if train:
            learner.replay.add(s=state, s2=next_state, bid=market_actions[block],
                               bid2=market_actions[next_block], action=powers,
                               reference=reference, local_reward=ev_reward,
                               global_reward=torch.tensor([float(r_global)], dtype=torch.float32),
                               done=torch.tensor([float(done)]), correspondence=correspondence(before, after))
            learner.update_control()
        state = next_state
        if done:
            break
    if train:
        normalizer = (1 - sp.gamma) / (1 - sp.gamma ** (step + 1))
        learner.bid_replay.add(forecast=forecast, action=raw_bid.astype(np.float32),
                               reward=np.array([day_return * normalizer], np.float32))
        learner.update_bid()
    metrics = env.get_metrics()
    global_rate = metrics['tracking_success_rate'] if metrics['tracking_steps'] else None
    local_rate = 100 - metrics['soc_miss_rate']
    traces = dict(residual_kw=residuals, target_kw=actual_targets,
                  tol_kw=np.asarray(tol).tolist(), tracking=np.asarray(arr['tracking_enabled_series'], bool).tolist(),
                  baseline_kw=np.asarray(bid['baseline_plan']).tolist(),
                  up_kw=np.asarray(bid['up_plan']).tolist(), down_kw=np.asarray(bid['down_plan']).tolist())
    return dict(global_rate=global_rate, local_rate=local_rate,
                score=None if global_rate is None else global_rate + local_rate,
                mae_kw=metrics['raw_actor_mae_kw'], tracking_steps=metrics['tracking_steps'],
                capacity_kw_block=float(np.sum(np.asarray(bid['up_plan']) + np.asarray(bid['down_plan']))),
                day_return=day_return, seconds=time.perf_counter() - start,
                departed_evs=int(env.metrics['departing_evs']), traces=traces,
                service_date=bid['service_date'], ev_seed=seed,
                command_source=command.get('source'), bid_mode='common' if common_bid else 'learned',
                bid_action=raw_bid.tolist(), baseline_kw=np.asarray(bid['baseline_plan']).tolist(),
                up_kw=np.asarray(bid['up_plan']).tolist(), down_kw=np.asarray(bid['down_plan']).tolist())


def aggregate(rows):
    import numpy as np
    globals_ = [r['global_rate'] for r in rows if r['global_rate'] is not None]
    local = float(np.mean([r['local_rate'] for r in rows]))
    global_ = float(np.mean(globals_)) if globals_ else None
    return dict(sets=len(rows), global_rate=global_, local_rate=local,
                score=global_ + local if global_ is not None else None,
                capacity_kw_block=float(np.mean([r['capacity_kw_block'] for r in rows])),
                mae_kw=float(np.mean([r['mae_kw'] for r in rows])),
                zero_tracking_sets=sum(r['tracking_steps'] == 0 for r in rows))


def validation(learner, payloads, commands):
    """Validation cannot alter training RNGs or either replay buffer."""
    import numpy as np
    import torch
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        own, common = [], []
        for day, (payload, command) in enumerate(zip(payloads, commands)):
            own.append(episode(learner, payload, command, seed=910000 + day, train=False))
            common.append(episode(learner, payload, command, seed=910000 + day, train=False, common_bid=True))
        return dict(learned_bid=aggregate(own), common_bid=aggregate(common))
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def inputs(task):
    from training.bid_bank import BidBank
    from training.lower_bid_training import sample_random_historical_activation
    paths = banks(task)
    train_bank, validation_bank = map(BidBank, paths)
    training = [train_bank.load_entry(e) for e in train_bank.entries]
    validating = [validation_bank.load_entry(e) for e in validation_bank.entries]
    commands = [sample_random_historical_activation(p, day, stream='validation') for day, p in enumerate(validating)]
    excluded = {c['source'] for c in commands}
    # Exclude the exact final holdout commands as well; final EV seeds remain unseen.
    from training.lower_bid_training import _activation_scenarios_for_day
    for p in validating:
        final, _ = _activation_scenarios_for_day(p['service_date'], p['forecast_seed'] + 512209,
                                                n_scenarios=4, scenario_partition='holdout')
        excluded.update(c['source'] for c in final)
    return training, validating, commands, excluded


def fingerprint(task):
    sources = list(HERE.glob('*.py')) + [HERE / 'method_spec.md']
    sources += [ROOT / name for name in
                ('Config.py', 'EnvConfig.py', 'environment/EVEnv.py', 'environment/normalize.py',
                 'environment/observation_config.py', 'environment/central_residual_allocator.py',
                 'training/lower_bid_training.py', 'training/blockwise_bid.py', 'market/activation_scenarios.py',
                 'execute_results/resbid_6station_20261012/stage_worker.py')]
    sources += list((ROOT / 'environment').rglob('*.py')) + list((ROOT / 'market').rglob('*.py'))
    sources += [ROOT / 'tools/evaluator.py']
    sources += list((ROOT / 'data/input_EVinfo').rglob('*.csv'))
    from Config import ACTIVATION_SCENARIO_DIR
    sources += list(Path(ACTIVATION_SCENARIO_DIR).glob('*.csv'))
    for bank in banks(task):
        sources += list(bank.rglob('*.pkl')) + list(bank.rglob('*.json'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources)) if p.exists()}


def save_checkpoint(learner, run, completed, context, learned):
    import numpy as np
    import torch
    payload = dict(completed=completed, learned=learned, context=context, learner=learner.state_dict(),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state())
    tmp = run / 'checkpoint.tmp.pth'
    torch.save(payload, tmp)
    tmp.replace(run / 'checkpoint.pth')
    p = run / 'latest.json'
    p.write_text(json.dumps(dict(completed_environment_episodes=completed, completed_training_episode=learned, updates=learner.updates,
                                bid_updates=learner.bid_updates, saved_at=datetime.now().astimezone().isoformat()), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='aemo')
    parser.add_argument('--run', required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--local-factor', type=float, default=1)
    args = parser.parse_args()
    config = configure(args.task)
    import numpy as np
    import torch
    from environment.normalize import configure_observation_normalization
    from training.lower_bid_training import sample_random_historical_activation
    from networks import AFMLearner, Specification
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    # Preserve the accepted train-bank normalization, not validation statistics.
    main_run = sorted((ROOT / 'archive').glob(base_model(args.task) + '_2*'))[-1]
    profile_path = main_run / 'input/observation_normalization.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    configure_observation_normalization(profile)
    (run / 'observation_normalization.json').write_text(json.dumps(profile, indent=2))
    spec = replace(Specification(), local_scale=config.LOCAL_REWARD_SCALE * args.local_factor)
    seed_all(spec.seed)
    learner = AFMLearner(spec, device='cuda' if torch.cuda.is_available() else 'cpu')
    context = dict(task=args.task, specification=asdict(spec), files=fingerprint(args.task), normalization=profile,
                   runtime=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, device=str(learner.device)))
    start = learned = 0
    if (run / 'checkpoint.pth').exists():
        saved = torch.load(run / 'checkpoint.pth', map_location='cpu', weights_only=False)
        if saved['context'] != context:
            raise ValueError('resume source, data, normalization or settings mismatch')
        learner.load_state_dict(saved['learner'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        start = saved['completed']
        learned = saved['learned']
        # A crash may leave metrics beyond the last durable checkpoint. Preserve
        # those records separately, then replay only the completed prefix.
        for name in ('training.jsonl', 'validation.jsonl'):
            metric_file = run / name
            if metric_file.exists():
                text = metric_file.read_text()
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
                if any(row['episode'] > start for row in rows):
                    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    (run / (name + '.before_resume_' + stamp)).write_text(text)
                    metric_file.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['episode'] <= start))
    else:
        source_dir = run / 'method_source'
        source_dir.mkdir(exist_ok=True)
        import shutil
        for source in HERE.glob('*.py'):
            shutil.copy2(source, source_dir / source.name)
        shutil.copy2(HERE / 'method_spec.md', source_dir / 'method_spec.md')
        (run / 'context.json').write_text(json.dumps(context, indent=2))
        save_checkpoint(learner, run, 0, context, 0)
    training, validating, commands, excluded = inputs(args.task)
    print(f'AFM full component implementation; device={learner.device}; resume={start}; local_factor={args.local_factor}', flush=True)
    for ep in range(start + 1, args.episodes + 1):
        if (run / 'STOP_REQUESTED').exists():
            save_checkpoint(learner, run, ep - 1, context, learned)
            return
        payload = training[(ep - 1) % len(training)]
        command = sample_random_historical_activation(payload, ep, stream='train', exclude_sources=excluded)
        before_updates = learner.updates
        row = episode(learner, payload, command, seed=spec.seed + 100000 + ep, train=True)
        learned += int(learner.updates > before_updates)
        row['episode'] = ep
        row['learned_episodes'] = learned
        with (run / 'training.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(f"[TRAIN{ep}] GLOBAL={row['global_rate']} LOCAL={row['local_rate']:.2f} capacity={row['capacity_kw_block']:.1f} seconds={row['seconds']:.1f} updates={learner.updates}/{learner.bid_updates}", flush=True)
        if ep % 20 == 0 or ep == args.episodes:
            point = dict(episode=ep, learned_episodes=learned, **validation(learner, validating, commands))
            with (run / 'validation.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(point) + '\n')
            p = point['common_bid']
            print(f"[TEST{ep} summary] SoC actor/physical={p['local_rate']:.2f}/{p['local_rate']:.2f}% MARL/Central={p['global_rate']:.2f}/{p['global_rate']:.2f}% SUM={p['score']:.2f}", flush=True)
            policy = dict(specification=asdict(learner.spec),
                          networks={n: getattr(learner, n).state_dict() for n in learner.targets},
                          episode=ep, learned_episodes=learned, validation=point)
            torch.save(policy, run / 'policy_latest.pth')
            if learned >= 1000:
                existing = run / 'policy_best.pth'
                previous_score = torch.load(existing, map_location='cpu', weights_only=False)['validation']['common_bid']['score'] if existing.exists() else -float('inf')
                if p['score'] > previous_score:
                    torch.save(policy, existing)
            save_checkpoint(learner, run, ep, context, learned)
    (run / 'DONE').write_text(f'completed {args.episodes} episodes\n')


def base_model(task):
    from settings import base
    return base.TASKS[task][1]


if __name__ == '__main__':
    main()

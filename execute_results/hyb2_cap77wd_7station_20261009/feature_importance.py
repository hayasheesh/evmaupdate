"""保存した行動器が、観測のどの特徴に強く反応しているかを測る（学習はしない）。

    python feature_importance.py <aemo|ercot> [days] [commands_per_day]

検証用の入札で行動器を動かして各 station の観測（正規化後）を集め、特徴を1つずつ（まとめて扱う特徴は
まとめて）サンプルのあいだで入れ替えたときに、在車の EV の行動（−1〜1）が平均でどれだけ動くかを出す。
EV の特徴は、入れ替えた枠の EV 自身の行動への影響（自分）と、同じ station のほかの EV の行動への影響（他）に分ける。
入れ替えは3回の平均。値が大きいほど、その特徴に強く反応している。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_worker  # noqa: E402

task = sys.argv[1]
n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 5
n_cmd = int(sys.argv[3]) if len(sys.argv) > 3 else 1
config = stage_worker.configure(task)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Config import EPISODE_STEPS  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.normalize import load_observation_normalization_for_archive, normalize_observation, use_instruction_scale  # noqa: E402
from environment.observation_config import EV_FEAT_DIM, EV_FEATURE_NAMES, LOCAL_TAIL_FEATURE_NAMES  # noqa: E402
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
obs_by_station = None
for day, entry in enumerate(bank.entries[:n_days]):
    base_bid = dict(bank.load_entry(entry))
    payloads, act_mode = _activation_scenarios_for_day(base_bid.get('service_date'), int(base_bid.get('forecast_seed', 0)) + 512_209,
                                                       n_scenarios=n_cmd, scenario_partition='holdout')
    fixed_bid = dict(base_bid, activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=act_mode)
    for c in range(len(payloads)):
        target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, c)
        tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
        set_env_seed(4_000_000 + 1000 * day + 10 * c)
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
            obs_by_station = [[] for _ in range(env.num_stations)]
        for t in range(EPISODE_STEPS):
            obs = normalize_observation(env.begin_step())
            for st in range(env.num_stations):
                obs_by_station[st].append(obs[st].detach().cpu().numpy().copy())
            agent.update_active_evs(env)
            with torch.no_grad():
                actions = agent.act(obs, env=env, noise=False)
            env.apply_action(torch.as_tensor(actions, dtype=torch.float32))

n_slots = agent.max_evs
ev_dim = n_slots * EV_FEAT_DIM
tail = list(LOCAL_TAIL_FEATURE_NAMES)
groups = {}
for name in tail:
    if name.startswith('bid_'):
        field, off = name[len('bid_'):].split('_lookahead_')
        off = int(off)
        span = '0' if off == 0 else '1-3' if off <= 3 else '4-11' if off <= 11 else '12-23'
        groups.setdefault(f'入札の先読み・{field}（全24コマ）', []).append(ev_dim + tail.index(name))
        groups.setdefault(f'入札の先読み・3種まとめて・{span}コマ先', []).append(ev_dim + tail.index(name))
    else:
        groups[name] = [ev_dim + tail.index(name)]
rng = np.random.default_rng(0)
REPEATS = 3
result = {'task': task, 'model': model_dir.name, 'episode': episode, 'samples_per_station': len(obs_by_station[0]),
          'ev_features': {}, 'tail_features': {}}
ev_own = {f: [] for f in EV_FEATURE_NAMES}
ev_other = {f: [] for f in EV_FEATURE_NAMES}
tail_eff = {g: [] for g in groups}
base_abs = []
for st, rows in enumerate(obs_by_station):
    X = torch.as_tensor(np.stack(rows), dtype=torch.float32)
    present = X[:, :ev_dim].reshape(len(X), n_slots, EV_FEAT_DIM)[..., 0] > 0.5
    keep = present.any(1)
    X, present = X[keep], present[keep]
    actor = agent.actors[st]
    dev = next(actor.parameters()).device
    X, present = X.to(dev), present.to(dev)
    with torch.no_grad():
        base = actor(X)
    base_abs.append(float(base[present].abs().mean()))

    def delta(Xp):
        with torch.no_grad():
            return (actor(Xp) - base).abs()

    for name, cols in groups.items():
        vals = []
        for _ in range(REPEATS):
            Xp = X.clone()
            perm = torch.as_tensor(rng.permutation(len(X)), device=dev)
            Xp[:, cols] = X[perm][:, cols]
            vals.append(float(delta(Xp)[present].mean()))
        tail_eff[name].append(np.mean(vals))
    for f_idx, f in enumerate(EV_FEATURE_NAMES):
        own_v, oth_v = [], []
        for j in range(n_slots):
            col = j * EV_FEAT_DIM + f_idx
            rows_j = present[:, j]
            if rows_j.sum() < 20:
                continue
            for _ in range(REPEATS):
                Xp = X.clone()
                # Shuffle among samples where slot j holds an EV, so an empty slot never receives an EV's value.
                idx = torch.nonzero(rows_j).squeeze(1)
                Xp[idx, col] = X[idx[torch.as_tensor(rng.permutation(len(idx)), device=dev)], col]
                d = delta(Xp)
                own_v.append((float(d[idx, j].mean()), int(len(idx))))
                other_mask = present.clone(); other_mask[:, j] = False; other_mask[~rows_j] = False
                if other_mask.any():
                    oth_v.append((float(d[other_mask].mean()), int(other_mask.sum())))
        w = lambda v: sum(a * n for a, n in v) / max(sum(n for _, n in v), 1)
        ev_own[f].append(w(own_v))
        ev_other[f].append(w(oth_v))
result['mean_abs_action'] = float(np.mean(base_abs))
result['ev_features'] = {f: {'self': float(np.mean(ev_own[f])), 'other_evs': float(np.mean(ev_other[f]))} for f in EV_FEATURE_NAMES}
result['tail_features'] = {g: float(np.mean(v)) for g, v in sorted(tail_eff.items(), key=lambda kv: -np.mean(kv[1]))}
path = HERE / f'feature_importance_{task}_ep{episode}.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))

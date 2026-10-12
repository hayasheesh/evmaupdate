"""（研究室の20局用）保存した行動器が、観測の末尾の各入力にどれだけ反応しているかを測る（学習はしない）。

    python feature_use.py <aemo|ercot> <行動器の回> [日の番号,...]

検証用入札の日を、保存済みの設計外指令の1本と独立な EV seed で行動器のとおりに動かし、各 step の
正規化した観測を集める。各入力について「出力の入力に対する微分の絶対値の平均 × その入力の標準偏差」
（入力が普段の幅だけ動いたときの出力の動き）を、EV が居る枠について station ごとに出す。
休日の区分は1日の中では変わらないので、平日と休日の日を両方入れる（既定は検証用の0〜4日目すべて）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
task, episode = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, str(HERE.parent / f'allin_20station_{task}_20261007'))
import stage_worker  # noqa: E402

days = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
config = stage_worker.configure()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Config import EPISODE_STEPS  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.normalize import load_observation_normalization_for_archive, normalize_observation, use_instruction_scale  # noqa: E402
from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES, EV_FEATURE_NAMES  # noqa: E402
from tools.evaluator import set_env_seed  # noqa: E402
from training.bid_bank import BidBank  # noqa: E402
from training.lower_bid_training import _activation_scenarios_for_day, build_fixed_upper_bid_training_episode  # noqa: E402
from training.system_controller import build_agent, find_model_path_and_episode  # noqa: E402

ROOT = HERE.parents[1]
model_dir = sorted((ROOT / 'archive').glob(f'allin_{task}plan_AB_20station_2*'))[-1]
test_bank = stage_worker.TEST_BANK
bank = BidBank(test_bank)
load_observation_normalization_for_archive(model_dir)
agent = None
obs_all = []
for day in days:
    fixed_bid = dict(bank.load_entry(list(bank.entries)[day]))
    payloads, mode = _activation_scenarios_for_day(fixed_bid.get('service_date'), int(fixed_bid.get('forecast_seed', 0)) + 512_209,
                                                   n_scenarios=1, scenario_partition='holdout')
    fixed_bid.update(activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=mode)
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, 0)
    tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
    set_env_seed(3_000_000 + day)
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
        ckpt, ep = find_model_path_and_episode(str(model_dir), episode)
        agent.load_actors(ckpt, ep, map_location='cpu')
        agent.set_test_mode(True)
    agent.update_active_evs(env)
    for t in range(EPISODE_STEPS):
        obs = normalize_observation(env.begin_step())
        obs_all.append(obs.detach().cpu())
        agent.update_active_evs(env)
        with torch.no_grad():
            actions = agent.act(obs, env=env, noise=False)
        env.apply_action(actions)

X = torch.stack(obs_all)                       # step × station × dim
n_st, dim = X.shape[1], X.shape[2]
ev_dim = dim - len(LOCAL_TAIL_FEATURE_NAMES)
names = [f'EV:{n}' for n in EV_FEATURE_NAMES] + list(LOCAL_TAIL_FEATURE_NAMES)
rows = {}
for st in range(n_st):
    actor = agent.actors[st].to('cpu').eval()
    x = X[:, st, :].clone().requires_grad_(True)
    a = actor(x)                                # step × slot
    present = (x[:, :ev_dim].view(len(x), -1, len(EV_FEATURE_NAMES))[..., 0] > 0.5).float()
    (a * present).sum().backward()
    g = x.grad.abs()
    active = present.sum(1) > 0
    sd = x.detach()[active].std(0)
    eff = (g[active].mean(0) * sd).numpy()
    # EV 特徴は全枠を足す
    ev = eff[:ev_dim].reshape(-1, len(EV_FEATURE_NAMES)).sum(0)
    for k, n in enumerate(EV_FEATURE_NAMES):
        rows.setdefault(f'EV:{n}', []).append(float(ev[k]))
    for k, n in enumerate(LOCAL_TAIL_FEATURE_NAMES):
        rows.setdefault(n, []).append(float(eff[ev_dim + k]))
# 先読みの入札は24コマ×3をまとめる
merged = {}
for n, v in rows.items():
    key = 'bid_lookahead(72)' if n.startswith('bid_') and 'lookahead' in n else n
    merged.setdefault(key, np.zeros(n_st))
    merged[key] += np.asarray(v)
top = max(float(np.mean(v)) for v in merged.values())
print(f'{task} ep{episode}  日 {days}  station {n_st}  step {len(X)}')
print('  入力 / 反応の大きさ（station平均） / 最大の入力に対する比 / station ごと')
for n, v in sorted(merged.items(), key=lambda kv: -float(np.mean(kv[1]))):
    print(f'    {n:28s} {np.mean(v):.4f}  {np.mean(v)/top:5.2f}  ' + ' '.join(f'{x:.3f}' for x in v))

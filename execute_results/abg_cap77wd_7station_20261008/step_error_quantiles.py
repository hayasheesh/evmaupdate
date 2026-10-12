"""保存した行動器を検証用の入札で動かし、5分ごとの誤差（BESSの前、入札したコマの時刻）の分位点を出す（学習はしない）。

    python step_error_quantiles.py <aemo|ercot> [commands_per_day] [ev_seeds]

指令は holdout（学習にも入札にも使っていない）から日ごとに引き、EV の乱数は日・指令ごとに変える。
最低入札量の候補は「q% 点 ÷ 帯の比率 0.1」。
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
res, band, up_dir = [], [], []
for day, entry in enumerate(bank.entries):
    base_bid = dict(bank.load_entry(entry))
    payloads, mode = _activation_scenarios_for_day(base_bid.get('service_date'), int(base_bid.get('forecast_seed', 0)) + 512_209,
                                                   n_scenarios=n_cmd, scenario_partition='holdout')
    fixed_bid = dict(base_bid, activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=mode)
    for c in range(len(payloads)):
        target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, c)
        tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
        baseline = np.asarray(arr.get('baseline_series'), float)[:EPISODE_STEPS]
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
                req = float(env.current_net_demand)
                agent.update_active_evs(env)
                with torch.no_grad():
                    actions = agent.act(obs, env=env, noise=False)
                env.apply_action(actions)
                if on:
                    res.append(float(env.last_raw_actor_residual_kw)); band.append(b)
                    up_dir.append(req < baseline[t] - 1e-6)
res = np.abs(np.asarray(res)); band = np.asarray(band); up_dir = np.asarray(up_dir)
qs = [50, 80, 90, 95, 98, 99]
out = {
    'task': task, 'model': model_dir.name, 'episode': episode, 'days': len(bank.entries),
    'commands_per_day': n_cmd, 'ev_seeds': n_ev, 'tracked_steps': int(res.size),
    'mae_kw': float(res.mean()), 'step_pass_pct_at_actual_band': float(100 * (res <= band).mean()),
    'quantiles_kw': {str(q): float(np.percentile(res, q)) for q in qs},
    'min_bid_kw_if_band_is_quantile': {str(q): float(np.percentile(res, q) / 0.1) for q in qs},
    'up_steps': int(up_dir.sum()), 'down_or_idle_steps': int((~up_dir).sum()),
    'quantiles_kw_up_steps': {str(q): float(np.percentile(res[up_dir], q)) for q in qs} if up_dir.any() else {},
    'quantiles_kw_other_steps': {str(q): float(np.percentile(res[~up_dir], q)) for q in qs} if (~up_dir).any() else {},
    'actual_band_kw_median': float(np.median(band)),
}
path = HERE / f'step_error_quantiles_{task}_ep{episode}.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))

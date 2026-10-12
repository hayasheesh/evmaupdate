"""学習を止めた7局の行動器を検証用入札の1日で回し、step ごとの記録を残す（学習はしない）。
外した step の原因を、その時点の EV が出せる範囲（step_dispatch_envelope）と比べて分けるために使う。

    python marl_fail_probe.py <aemo|ercot> <行動器の回> <日の番号> [指令の本数] [EV の seed 数]

以下は pjm_band_clip_probe.py と同じ回し方（normal のみ）。

検証用入札（low2max_validation_5_7station_..._pjm_regd_phase_shift）の1日について、holdout 区分の指令と
独立な EV の seed で、同じ行動器を次の2通りで回す。BESS と中央の残差分配は使わない（学習中のテストと同じ marl_raw）。
- normal: 指令はそのまま。判定はアセスメントIIの帯（入札の10%）と、絶対値150 kW（論文の実行の幅）の両方で数える
- bess  : 指令はそのまま。連系点の蓄電池（EnvConfig の BESS、7局で28 kW・36.8 kWh）を動かし、蓄電池を足した連系点の電力で判定する
- clip  : 各 step で、その時点の EV が物理的に出せる範囲（step_dispatch_envelope）に指令を切ってから行動器に見せる。
          判定は切った指令に対して、アセスメントIIの帯で数える
追従率は指令の無い step も含めて数える（学習中のテストの MARL/Central と同じ）。SoC達成率は出発した EV のうち目標に届いた割合。
学習で使った指令を除いていないので、3通りの比較にだけ使う。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_worker  # noqa: E402

TASK = sys.argv[1]
sys.argv = sys.argv[:1] + sys.argv[2:]
config = stage_worker.configure(TASK)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Config import EPISODE_STEPS  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.normalize import (  # noqa: E402
    load_observation_normalization_for_archive, normalize_observation, use_instruction_scale)
from environment.physical_capability import step_dispatch_envelope  # noqa: E402
from tools.evaluator import set_env_seed  # noqa: E402
from training.bid_bank import BidBank  # noqa: E402
from training.lower_bid_training import (  # noqa: E402
    _activation_scenarios_for_day, build_fixed_upper_bid_training_episode)
from training.system_controller import build_agent, find_model_path_and_episode  # noqa: E402

episode = int(sys.argv[1])
day_index = int(sys.argv[2])
n_commands = int(sys.argv[3]) if len(sys.argv) > 3 else 4
n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
VARIANTS = ('normal',)
ROOT = HERE.parents[1]
SIGNAL = {'aemo': 'aemo_plan_deviation', 'ercot': 'ercot_plan_deviation'}[TASK]
model_dir = sorted((ROOT / 'archive').glob(f'low2max_{TASK}plan_AB_7station_2*'))[-1]
_train_bank, test_bank = stage_worker.banks(SIGNAL)
bank = BidBank(test_bank)
entry = list(bank.entries)[day_index]
fixed_bid = dict(bank.load_entry(entry))
payloads, mode = _activation_scenarios_for_day(
    fixed_bid.get('service_date'), int(fixed_bid.get('forecast_seed', 0)) + 512_209,
    n_scenarios=n_commands, scenario_partition='holdout')
fixed_bid['activation_scenario_payload'] = list(payloads)
fixed_bid['activation_scenarios'] = len(payloads)
fixed_bid['activation_mode'] = mode

load_observation_normalization_for_archive(model_dir)
agent = None
rows = []
for s in range(len(payloads)):
    target, tol, arr, _info = build_fixed_upper_bid_training_episode(fixed_bid, s)
    tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
    for k in range(n_seeds):
        for variant in VARIANTS:
            seed = 2_000_000 + 100_000 * day_index + 1000 * s + k
            set_env_seed(seed)
            use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
            env = EVEnv()
            env.use_central_ev_residual_allocator = False
            env.use_residual_bess = variant == 'bess'
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
            disp = np.zeros(EPISODE_STEPS)
            evs = np.zeros((EPISODE_STEPS, env.num_stations))
            spow = np.zeros((EPISODE_STEPS, env.num_stations))
            env_lo = np.zeros(EPISODE_STEPS)
            env_hi = np.zeros(EPISODE_STEPS)
            resp = np.zeros(EPISODE_STEPS)
            clipped_steps = 0
            for t in range(EPISODE_STEPS):
                obs = env.begin_step()
                if variant == 'clip':
                    lo, hi, _obligated = step_dispatch_envelope(env, force_slack_kwh=0.0)
                    want = float(env.current_net_demand)
                    got = min(max(want, lo), hi)
                    if got != want:
                        clipped_steps += 1
                        env.current_net_demand = got
                        env.current_observed_net_demand = got
                        env.current_regulation_kw += got - want
                        obs = env._get_obs()
                evs[t] = env.ev_mask.sum(dim=1).cpu().numpy()
                env_lo[t], env_hi[t], _ob = step_dispatch_envelope(env, force_slack_kwh=0.0)
                agent.update_active_evs(env)
                with torch.no_grad():
                    actions = agent.act(normalize_observation(obs), env=env, noise=False)
                _o, _l, _g, _d, info = env.apply_action(actions)
                disp[t] = float(env.current_net_demand)
                if 'station_powers' in info:
                    spow[t] = np.asarray(info['station_powers'], dtype=float).reshape(-1)[: env.num_stations]
                resp[t] = float(info.get('pcc_power_kw') if variant == 'bess' else info.get('raw_actor_total_power_kw', info.get('total_ev_transport', 0.0)))
            metrics = env.get_metrics()
            err = np.abs(disp - resp)
            tol_arr = np.asarray(tol, float)[:EPISODE_STEPS]
            row = {
                'command': s, 'seed': k, 'variant': variant, 'source': str(payloads[s].get('source', '')),
                'tracking_band_pct': float(100 * np.mean(err[tracking] <= tol_arr[tracking] + 1e-6)),
                'tracking_150kw_pct': float(100 * np.mean(err[tracking] <= 150.0 + 1e-6)),
                'mae_kw': float(err[tracking].mean()),
                'band_mean_kw': float(tol_arr[tracking].mean()),
                'soc_hit_pct': float(100.0 - metrics.get('soc_miss_rate', float('nan'))),
                'clipped_steps': int(clipped_steps), 'assessed_steps': int(tracking.sum()),
                'steps': {'disp': disp.tolist(), 'resp': resp.tolist(), 'tol': tol_arr.tolist(), 'tracking': tracking.astype(int).tolist(),
                          'evs': evs.tolist(), 'station_power': spow.tolist(), 'env_lo': env_lo.tolist(), 'env_hi': env_hi.tolist()},
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'steps'}, ensure_ascii=False), flush=True)
out = HERE / 'marl_fail_probe'
out.mkdir(exist_ok=True)
(out / f'{TASK}_day{day_index}_ep{episode}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

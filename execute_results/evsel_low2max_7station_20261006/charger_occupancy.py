"""7局の充電器の稼働率を、EV の到着の倍率ごとに出す（入札も学習も使わない）。

    python charger_occupancy.py <倍率> [日ごとの実現数]

入札の EV 候補と同じ作り方（seed を固定した EVEnv を出力0で1日回す）で、7局AEMO の学習用入札の
25日について実現を作り、5分ごとに次を数える。
- つながっている EV の数（1局10台の充電器に対する割合が稼働率）
- 着こうとした EV の数と、空きがなくて着けなかった数
結果は charger_occupancy_x<倍率>.json に保存する。
"""
import json
import os
import sys
from pathlib import Path

growth = sys.argv[1]
realizations = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for key in list(os.environ):
    if key.startswith('EVMA_'):
        del os.environ[key]
os.environ.update({'EVMA_NUM_STATIONS': '7', 'EVMA_MARL_ALGORITHM': 'hybrid',
                   'EVMA_ACTIVATION_SIGNAL_SET': 'aemo_plan_deviation',
                   'EVMA_FUTURE_EV_ARRIVAL_GROWTH': growth})
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import EnvConfig  # noqa: E402
from Config import EPISODE_STEPS  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.arrival_context import ArrivalScenarioSampler  # noqa: E402
from tools.evaluator import set_env_seed  # noqa: E402

assert EnvConfig.FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER == float(growth)
bank = ROOT / 'execute_results' / 'bid_banks' / 'low2max_train_25_7station_128cmd_3ev_aemo_plan_deviation'
manifest = json.loads((bank / 'manifest.json').read_text(encoding='utf-8'))
entries = manifest['entries']
sampler = ArrivalScenarioSampler()
chargers = int(EnvConfig.MAX_EV_PER_STATION)
stations = int(EnvConfig.NUM_STATIONS)

connected = np.zeros((len(entries), realizations, stations, EPISODE_STEPS))
attempts = np.zeros((len(entries), realizations, stations, EPISODE_STEPS))
accepted = np.zeros((len(entries), realizations, stations, EPISODE_STEPS))
actions = torch.zeros((stations, chargers), dtype=torch.float32)
for d, entry in enumerate(entries):
    probs = sampler.scenario_for_day(entry['service_date']).arrival_probabilities_by_station
    for r in range(realizations):
        # The same seeds the bid's first candidates use (forecast seed + 10007 * index).
        set_env_seed(int(entry['forecast_seed']) + 10007 * r)
        env = EVEnv()
        env.reset(net_demand_series=np.zeros(EPISODE_STEPS, dtype=np.float32),
                  arrival_probabilities_by_station=probs, service_date=entry['service_date'])
        for t in range(EPISODE_STEPS):
            for s in range(stations):
                attempts[d, r, s, t] = sum(e is not None for e in env._arrival_events[t][s])
        with torch.inference_mode():
            for t in range(EPISODE_STEPS):
                env.begin_step()
                connected[d, r, :, t] = env.ev_mask.sum(dim=1).cpu().numpy()
                accepted[d, r, :, t] = np.asarray(env.arrivals_by_station, dtype=float)
                _o, _l, _g, done, _i = env.apply_action(actions, build_info=False, return_observation=False)
                if bool(np.all(np.asarray(done, dtype=bool))):
                    break
    print(f'x{growth} {entry["service_date"]} done', flush=True)

occupancy = connected / chargers
hourly = occupancy.reshape(len(entries), realizations, stations, 24, EPISODE_STEPS // 24).mean(axis=(0, 1, 4))
lost = 1.0 - accepted.sum() / max(attempts.sum(), 1.0)
lost_hourly = 1.0 - accepted.reshape(*accepted.shape[:3], 24, -1).sum(axis=(0, 1, 2, 4)) / np.maximum(
    attempts.reshape(*attempts.shape[:3], 24, -1).sum(axis=(0, 1, 2, 4)), 1.0)
full = (connected >= chargers).mean(axis=(0, 1, 3))
result = {
    'growth': float(growth), 'days': len(entries), 'realizations_per_day': realizations,
    'stations': list(EnvConfig.PER_STATION_SESSION_IDS), 'chargers_per_station': chargers,
    'mean_occupancy': float(occupancy.mean()),
    'mean_occupancy_by_station': occupancy.mean(axis=(0, 1, 3)).tolist(),
    'hourly_occupancy_all': hourly.mean(axis=0).tolist(),
    'hourly_occupancy_by_station': hourly.tolist(),
    'share_of_time_full_by_station': full.tolist(),
    'lost_arrival_share': float(lost),
    'lost_arrival_share_hourly': lost_hourly.tolist(),
    'connected_evs_mean_per_day': float(connected.mean(axis=3).sum(axis=2).mean()),
    'accepted_arrivals_per_day': float(accepted.sum(axis=(2, 3)).mean()),
    'attempted_arrivals_per_day': float(attempts.sum(axis=(2, 3)).mean()),
}
(HERE / f'charger_occupancy_x{growth}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, ensure_ascii=False))

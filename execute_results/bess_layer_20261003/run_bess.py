"""MARL + force 充電に連系点の BESS を足した層（pipeline=marl_force_bess）を、未見指令の360組で測る。

  python run_bess.py --market aemo|ercot|gb

組・モデル・チェックポイントは、制御の比較の表（difficulty_analysis_20261003/table3.json）の MARL の行と同じ。
  aemo  unseen_decomposition_aemo_20261002 の組、AEMO で学習した AB の ep2200
  ercot unseen_decomposition_ercot_20261002 の組、ERCOT で学習した AB の ep1760
  gb    unseen_decomposition_gb_20261003 の組、GB で学習した AB の ep1240
BESS の大きさは学習時の設定のまま（7 station で 27.99 kW / 36.83 kWh。500 station 換算で 1999.4 kW）。

行動器が観測する「1つ前の時刻の全体のずれ」は BESS を通す前の値なので、EV 側の動きは BESS があっても
marl_force と同じになる。これを確かめるため、BESS を通す前の追従も出す。
要る BESS の大きさを後で計算できるように、時刻ごとの目標・EV の合計電力・許容幅・BESS の電力と蓄電量を
steps.csv.gz に残す（環境のメソッドをこのプロセスの中だけで包んで記録する。コードのファイルは変えない）。
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE.parent
MARKETS = {
    'aemo': (E / 'unseen_decomposition_aemo_20261002', 'archive/prod_aemoplan_AB_7station_20260926_210314', 2200),
    'ercot': (E / 'unseen_decomposition_ercot_20261002', 'archive/prod_ercotplan_AB_7station_20260930_221551', 1760),
    'gb': (E / 'unseen_decomposition_gb_20261003', 'archive/prod_elexonplan_AB_7station_20261002_215844', 1240),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=tuple(MARKETS), required=True)
    args = parser.parse_args()
    common_dir, model_rel, episode = MARKETS[args.market]
    sys.path.insert(0, str(common_dir))
    common = importlib.import_module('common')
    info = common.configure()
    model_dir = common.ROOT / model_rel
    out = HERE / f'results_{args.market}'
    out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    import tools.evaluator as evaluator_tools
    import environment.EVEnv as env_module
    from environment.EVEnv import EVEnv
    from environment.normalize import (
        load_observation_normalization_for_archive,
        normalize_observation,
        use_instruction_scale,
    )
    from training.evaluate_controller_precision import evaluate_controller_precision
    from training.lower_bid_training import build_fixed_upper_bid_training_episode
    from training.system_controller import (
        build_agent,
        find_model_path_and_episode,
        validate_actor_checkpoint_compatibility,
    )
    import Config
    import EnvConfig

    # 時刻ごとの記録（このプロセスの中だけ）
    current = {'day': None, 'seed': None}
    rows: list[tuple] = []
    original_seed = evaluator_tools.set_env_seed

    def set_seed(seed):
        current['seed'] = int(seed)
        return original_seed(seed)

    evaluator_tools.set_env_seed = set_seed
    original_reset = EVEnv.reset

    def reset(self, *a, **k):
        result = original_reset(self, *a, **k)
        self._log_seed = current['seed']
        self._log_step = 0
        tol = k.get('tol_narrow_series')
        self._log_tol = None if tol is None else np.asarray(tol, dtype=float).reshape(-1)
        return result

    original_dispatch = EVEnv._dispatch_residual_bess

    def dispatch(self, request_kw, ev_total_power_kw, tracking_enabled):
        result = original_dispatch(self, request_kw, ev_total_power_kw, tracking_enabled)
        t = int(getattr(self, '_log_step', 0))
        self._log_step = t + 1
        tol = getattr(self, '_log_tol', None)
        rows.append((current['day'], getattr(self, '_log_seed', None), t, float(request_kw), float(ev_total_power_kw),
                     int(bool(tracking_enabled)), float(tol[t]) if tol is not None and t < tol.size else float('nan'),
                     float(self.last_bess_requested_power_kw), float(self.last_bess_power_kw), float(self.bess_energy_kwh)))
        return result

    EVEnv.reset = reset
    EVEnv._dispatch_residual_bess = dispatch

    load_observation_normalization_for_archive(model_dir)
    cases = common.day_cases()
    first_bid = dict(cases[0][2])
    first_bid['activation_scenario_payload'] = first_bid['activation_scenario_payload'][:1]
    first_bid['activation_scenarios'] = 1
    target, tol, arrival, _ = build_fixed_upper_bid_training_episode(first_bid, 0)
    original_seed(common.BASE_SEED)
    use_instruction_scale(arrival.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.reset(net_demand_series=target, tol_narrow_series=tol,
              tracking_enabled_series=arrival.get('tracking_enabled_series'),
              market_context_series=arrival.get('market_context_series'),
              arrival_probabilities_by_station=arrival.get('arrival_probabilities_by_station'),
              day_context=arrival.get('day_context'))
    normalized = normalize_observation(env.begin_step())
    agent = build_agent(env)
    assert int(normalized.shape[1]) == int(agent.s_dim)
    checkpoint_dir, episode = find_model_path_and_episode(str(model_dir), episode)
    validate_actor_checkpoint_compatibility(agent, checkpoint_dir, episode, expected_stations=env.num_stations)
    agent.load_actors(checkpoint_dir, episode, map_location='cpu')
    rows.clear()

    frames = []
    for day_index, entry, fixed_bid in cases:
        current['day'] = day_index
        day_dir = out / f"day_{day_index:03d}_{entry['service_date']}"
        evaluate_controller_precision(
            agent, fixed_bid,
            n_seeds=common.EV_SEEDS,
            base_seed=common.day_base_seed(day_index),
            force_slack_kwh=0.1,
            evaluation_pipeline='marl_force_bess',
            out_dir=day_dir,
            visualize=False,
        )
        frame = pd.read_csv(day_dir / 'results' / 'controller_precision_by_scenario.csv')
        frame.insert(0, 'bid_day_index', day_index)
        frames.append(frame)
        print(f"[{args.market}/marl_force_bess] day {day_index} {entry['service_date']} rollouts {len(frame)} steps logged {len(rows)}", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / 'all_rollouts.csv', index=False)
    steps = pd.DataFrame(rows, columns=['bid_day_index', 'realized_seed', 'step', 'target_kw', 'ev_kw', 'tracking_enabled',
                                        'tol_kw', 'bess_requested_kw', 'bess_kw', 'bess_energy_kwh'])
    steps.to_csv(out / 'steps.csv.gz', index=False, compression='gzip')
    (out / 'run_info.json').write_text(json.dumps({
        'market': args.market, 'pipeline': 'marl_force_bess',
        'model_dir': str(model_dir), 'checkpoint_dir': str(checkpoint_dir), 'episode': int(episode),
        'bess_power_kw': float(EnvConfig.BESS_POWER_KW), 'bess_energy_kwh': float(EnvConfig.BESS_ENERGY_KWH),
        'bess_env_power_kw': float(env_module.BESS_POWER_KW), 'bess_env_energy_kwh': float(env_module.BESS_ENERGY_KWH),
        'num_stations': int(Config.NUM_STATIONS),
        'pinned_source_sha256': (info or {}).get('pinned_source_sha256') if isinstance(info, dict) else None,
        'excluded_commands': len(common.seen_commands()),
    }, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

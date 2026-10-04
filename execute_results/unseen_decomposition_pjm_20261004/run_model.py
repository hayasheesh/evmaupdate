"""PJM（実指令）の360組で、制御を1つ動かす。

  --model pjm         PJM（疑似指令）で学習した AB（チェックポイントは環境変数 PJM_MODEL_EPISODE、既定は200）
  --model central_lp  各時刻に全EVを見て解く中央LP（central_lp_agent.py、AEMO・ERCOT の比較と同じもの）
  --model rule        中央観測のルールベース（pipeline は rule_based_central に固定。行動器は使わない）
  --pipeline          marl_raw（行動器だけ）か marl_force（force 充電を足す）。既定は marl_raw
行動器の作り方と読み込みは tools/evaluate_final_system_on_bid_bank.py の main() と同じ。
観測の正規化は GB のモデルが学習時に使ったもの（中央LPとルールベースは使わない）。
結果のフォルダ名には、GB のモデルではチェックポイントの回数を付ける（results_gb_ep1240 など）。
"""
from __future__ import annotations

import argparse
import json

import common

INFO = common.configure()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=(*common.MODELS, 'central_lp', 'rule'), required=True)
    parser.add_argument('--pipeline', choices=('marl_raw', 'marl_force'), default='marl_raw')
    args = parser.parse_args()
    pipeline = 'rule_based_central' if args.model == 'rule' else args.pipeline
    model_dir, episode = common.MODELS[args.model if args.model in common.MODELS else 'pjm']
    suffix = '' if pipeline in ('marl_raw', 'rule_based_central') else f'_{pipeline}'
    tag = f'{args.model}_ep{episode}' if args.model in common.MODELS else args.model
    out = common.HERE / f'results_{tag}{suffix}'
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd
    import torch
    from environment.EVEnv import EVEnv
    from environment.normalize import (
        load_observation_normalization_for_archive,
        normalize_observation,
        use_instruction_scale,
    )
    from tools.evaluator import set_env_seed
    from training.evaluate_controller_precision import evaluate_controller_precision
    from training.lower_bid_training import build_fixed_upper_bid_training_episode
    from training.system_controller import (
        build_agent,
        find_model_path_and_episode,
        validate_actor_checkpoint_compatibility,
    )

    profile = load_observation_normalization_for_archive(model_dir)
    cases = common.day_cases()
    first_bid = dict(cases[0][2])
    first_bid['activation_scenario_payload'] = first_bid['activation_scenario_payload'][:1]
    first_bid['activation_scenarios'] = 1
    target, tol, arrival, _ = build_fixed_upper_bid_training_episode(first_bid, 0)
    set_env_seed(common.BASE_SEED)
    use_instruction_scale(arrival.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.reset(
        net_demand_series=target, tol_narrow_series=tol,
        tracking_enabled_series=arrival.get('tracking_enabled_series'),
        market_context_series=arrival.get('market_context_series'),
        arrival_probabilities_by_station=arrival.get('arrival_probabilities_by_station'),
        day_context=arrival.get('day_context'),
    )
    normalized = normalize_observation(env.begin_step())
    agent = build_agent(env)
    assert int(normalized.shape[1]) == int(agent.s_dim)
    checkpoint_dir, episode = find_model_path_and_episode(str(model_dir), episode)
    validate_actor_checkpoint_compatibility(agent, checkpoint_dir, episode, expected_stations=env.num_stations)
    agent.load_actors(checkpoint_dir, episode, map_location='cpu')
    if args.model == 'central_lp':
        from central_lp_agent import CentralLPAgent
        agent = CentralLPAgent()

    frames = []
    for day_index, entry, fixed_bid in cases:
        day_dir = out / f"day_{day_index:03d}_{entry['service_date']}"
        evaluate_controller_precision(
            agent, fixed_bid,
            n_seeds=common.EV_SEEDS,
            base_seed=common.day_base_seed(day_index),
            force_slack_kwh=0.1,
            evaluation_pipeline=pipeline,
            out_dir=day_dir,
            visualize=False,
        )
        frame = pd.read_csv(day_dir / 'results' / 'controller_precision_by_scenario.csv')
        frame.insert(0, 'bid_day_index', day_index)
        frames.append(frame)
        print(f"[{args.model}/{pipeline}] day {day_index} {entry['service_date']} rollouts {len(frame)}", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(out / 'all_rollouts.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps({
        'controller': args.model, 'pipeline': pipeline,
        'model_dir': str(model_dir), 'checkpoint_dir': str(checkpoint_dir), 'episode': int(episode),
        'observation_normalization': profile, 'pinned_source_sha256': INFO['pinned_source_sha256'],
        'excluded_commands': len(common.seen_commands()),
    }, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

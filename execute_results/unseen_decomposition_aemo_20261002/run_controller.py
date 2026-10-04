"""MARLと同じ360組で、比較用の制御を走らせる。

  --method rule        中央観測のルールベース（evaluation_pipeline=rule_based_central）。
                       tools/evaluate_final_system_on_bid_bank.py の main() をそのまま呼ぶ
  --method central_lp  各時刻に全EVを見て解く中央LP（central_lp_agent.py）。
                       force充電・中央残差配分・BESSは使わない（pipeline=marl_raw の設定で行動器だけ差し替え）
  --method marl_force  MARL（AEMO で学習した AB、ep2200）に force 充電を足したもの（pipeline=marl_force）
  --method central_lp_force  中央LPに force 充電を足したもの（pipeline=marl_force で行動器だけ差し替え）
  --max-rollouts N     動作確認用。最初の日の最初の指令だけを N 本にする
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import common

INFO = common.configure()


def run_rule(out: Path, pipeline: str = 'rule_based_central') -> int:
    import importlib
    evaluator = importlib.import_module('tools.evaluate_final_system_on_bid_bank')
    return evaluator.main([
        '--model-dir', str(common.MODEL), '--episode', str(common.EPISODE),
        '--bid-bank-dir', str(common.BANK), '--command-scenarios', str(common.COMMANDS_PER_DAY),
        '--ev-seeds', str(common.EV_SEEDS), '--base-seed', str(common.BASE_SEED),
        '--pipeline', pipeline, '--max-days', '5', '--output-dir', str(out),
    ])


def run_central_lp(out: Path, max_rollouts: int, pipeline: str = 'marl_raw') -> int:
    import pandas as pd
    from environment.normalize import load_observation_normalization_for_archive
    from training.evaluate_controller_precision import evaluate_controller_precision
    from central_lp_agent import CentralLPAgent

    load_observation_normalization_for_archive(common.MODEL)
    agent = CentralLPAgent()
    frames = []
    for day_index, entry, fixed_bid in common.day_cases():
        if max_rollouts:
            fixed_bid['activation_scenario_payload'] = fixed_bid['activation_scenario_payload'][:1]
            fixed_bid['activation_scenarios'] = 1
        day_dir = out / f"day_{day_index:03d}_{entry['service_date']}"
        evaluate_controller_precision(
            agent, fixed_bid,
            n_seeds=max_rollouts or common.EV_SEEDS,
            base_seed=common.day_base_seed(day_index),
            force_slack_kwh=0.1,
            evaluation_pipeline=pipeline,
            out_dir=day_dir,
            visualize=False,
        )
        frame = pd.read_csv(day_dir / 'results' / 'controller_precision_by_scenario.csv')
        frame.insert(0, 'bid_day_index', day_index)
        frames.append(frame)
        print(f"[central_lp] day {day_index} {entry['service_date']} rollouts {len(frame)} "
              f"solves {agent.solves} failures {agent.solve_failures}", flush=True)
        if max_rollouts:
            break
    rows = pd.concat(frames, ignore_index=True)
    rows['evaluation_pipeline'] = 'central_lp' if pipeline == 'marl_raw' else f'central_lp+{pipeline}'
    rows.to_csv(out / 'all_rollouts.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps({
        'solves': agent.solves, 'solve_failures': agent.solve_failures,
        'pinned_source_sha256': INFO['pinned_source_sha256'],
    }, indent=2) + '\n', encoding='utf-8')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('rule', 'central_lp', 'marl_force', 'central_lp_force'), required=True)
    parser.add_argument('--max-rollouts', type=int, default=0)
    args = parser.parse_args()
    out = common.HERE / (f'results_{args.method}' + ('_smoke' if args.max_rollouts else ''))
    out.mkdir(parents=True, exist_ok=True)
    if args.method == 'rule':
        return run_rule(out)
    if args.method == 'marl_force':
        return run_rule(out, pipeline='marl_force')
    if args.method == 'central_lp_force':
        return run_central_lp(out, args.max_rollouts, pipeline='marl_force')
    return run_central_lp(out, args.max_rollouts)


if __name__ == '__main__':
    raise SystemExit(main())

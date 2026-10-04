"""保存時のコードと設定で定期検証を再現し、追従誤差の時系列を保存する。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--episode', required=True, type=int)
    parser.add_argument('--reference-test-episode', required=True, type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--controller', choices=('marl', 'rule_based_central'), default='marl')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / 'resume/latest.json').read_text(encoding='utf-8'))
    runtime = manifest['context']['runtime']
    snapshot = run / 'code_snapshot'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(snapshot))
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ['PYTHONUTF8'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # Restore configuration before importing modules that bind its constants.
    for name in ('EnvConfig', 'Config', 'environment.observation_config'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(snapshot)
        for key, value in runtime[name.rsplit('.', 1)[-1]].items():
            if isinstance(value, list) and key.endswith(('FEATURES', 'FEATURE_NAMES')):
                value = tuple(value)
            setattr(module, key, value)
    import Config
    Config.CREATE_AGENT_RUNS_WRITER = False
    if args.controller == 'rule_based_central':
        # The zero-actor adapter applies the existing external force layer;
        # EVEnv then applies central allocation exactly once, with BESS off.
        import EnvConfig
        Config.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR = True
        EnvConfig.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR = True
        Config.TRAIN_USE_RESIDUAL_BESS = False
        EnvConfig.TRAIN_USE_RESIDUAL_BESS = False
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from environment.EVEnv import EVEnv
    from environment.normalize import load_observation_normalization_for_archive, use_instruction_scale
    from training.system_controller import build_agent, apply_force_charging
    from training.bid_bank import BidBank
    from training.lower_bid_training import build_fixed_upper_bid_training_episode
    import tools.evaluator as evaluator
    for name in ('plot_daily_rewards', 'plot_station_cooperation_full', 'plot_ev_detailed_soc',
                 'plot_performance_metrics', 'plot_arrival_counts', 'plot_reward_breakdown',
                 'plot_power_mismatch_analysis'):
        setattr(evaluator, name, lambda *a, **k: None)
    (output / 'input').mkdir(exist_ok=True)
    (output / 'resume').mkdir(exist_ok=True)
    shutil.copy2(run / 'input/observation_normalization.json', output / 'input/observation_normalization.json')
    shutil.copy2(run / 'resume/latest.json', output / 'resume/latest.json')
    load_observation_normalization_for_archive(output)
    bank = BidBank(manifest['context']['test_bid_bank']['path'])
    entries = list(bank.entries)
    info_by_episode = []
    references = []
    for j in range(1, 6):
        reference_path = run / 'results' / f'TEST{args.reference_test_episode}' / f'test_results_zz_station_cooperation_full_episode_{j}.csv'
        with reference_path.open(encoding='utf-8-sig', newline='') as handle:
            references.append((reference_path, list(csv.DictReader(handle))))

    def prepare(env, _demand, date, _sampler, episode_idx):
        entry = bank.entry_for_date(str(date))
        bid = bank.load_entry(entry)
        _, _, kwargs, info = build_fixed_upper_bid_training_episode(bid, int(episode_idx))
        reference_path, reference = references[int(episode_idx)]
        target = np.asarray([float(row['Grid_Request']) for row in reference], dtype=np.float32)
        tol = np.asarray([float(row['Tolerance_kW']) for row in reference], dtype=np.float32)
        tracking = np.asarray([bool(int(float(row['Tracking_Enabled']))) for row in reference])
        np.testing.assert_allclose(np.repeat(bid['baseline_plan'], 6), [float(row['Baseline_kW']) for row in reference], atol=1e-3)
        use_instruction_scale(kwargs.get('instruction_scale_kw', 1.0))
        env.reset(net_demand_series=target, tol_narrow_series=tol,
                  tracking_enabled_series=tracking,
                  market_context_series=kwargs.get('market_context_series'),
                  arrival_probabilities_by_station=kwargs.get('arrival_probabilities_by_station'),
                  day_context=kwargs.get('day_context'),
                  service_date=kwargs.get('service_date'),
                  baseline_series=kwargs.get('baseline_series'))
        info_by_episode.append({'service_date': str(date), 'instruction_reference': str(reference_path),
                                'bid_path': str(bank.root / entry['bid_path']),
                                'baseline': np.asarray(bid['baseline_plan']).tolist(),
                                'up': np.asarray(bid['up_plan']).tolist(),
                                'down': np.asarray(bid['down_plan']).tolist()})
        return info

    env = EVEnv()
    checkpoint = run / 'results' / f'TEST{args.episode}'
    if args.controller == 'marl':
        agent = build_agent(env)
        agent.load_actors(str(checkpoint), args.episode, map_location='cpu')
    else:
        class ZeroActorForceController:
            use_tensorboard = False

            def set_test_mode(self, _enabled):
                pass

            def episode_start(self):
                pass

            def episode_end(self):
                pass

            def act(self, _obs, *, env, noise=False):
                actions = env.soc.new_zeros((env.num_stations, env.max_ev_per_station))
                assert torch.count_nonzero(actions).item() == 0
                forced_actions, _, _ = apply_force_charging(
                    actions, env, slack_kwh=float(Config.TRAIN_FORCE_SLACK_KWH))
                return forced_actions

        agent = ZeroActorForceController()
    payloads = [{'date': row['service_date'], 'series': np.zeros(288)} for row in entries]
    result = evaluator.test(agent, random_window=False, working_dir=str(output),
                            test_episode_num=args.episode, demand_data_override=payloads,
                            episode_preparer=prepare, num_episodes=5,
                            eval_seed=int(runtime['Config']['INTERIM_TEST_SEED']),
                            enable_png=True, enable_history_png=False, save_test_detail_files=True,
                            update_history=False, verbose=True, print_summary=True)
    summary = evaluator._summarize_test_results(result)
    history = json.loads((run / 'results/test_history.json').read_text(encoding='utf-8'))
    i = history['episodes'].index(args.episode)
    reference_tracking = 100 * (history['surplus_within_narrow'][i] + history['shortage_within_narrow'][i]) / (history['surplus_steps'][i] + history['shortage_steps'][i])
    reference_soc = 100 - history['soc_miss_count'][i]
    summary['reference_tracking_rate'] = reference_tracking
    summary['reference_soc_hit_rate'] = reference_soc
    summary['reference_matches'] = bool(abs(summary['dispatch_rate'] - reference_tracking) < 1e-5 and abs(summary['soc_hit'] - reference_soc) < 1e-5)
    summary['controller'] = args.controller
    summary['evaluation_seed'] = int(runtime['Config']['INTERIM_TEST_SEED'])
    summary['score_tracking_rate'] = summary['central_dispatch_rate']
    summary['score_soc_hit_rate'] = summary['central_soc_hit']
    summary['score_sum'] = summary['score_tracking_rate'] + summary['score_soc_hit_rate']
    if args.controller == 'rule_based_central':
        summary['reference_matches'] = None
        summary['actor_input_is_zero'] = True
        summary['force_slack_kwh'] = float(Config.TRAIN_FORCE_SLACK_KWH)
        summary['central_allocator'] = True
        summary['residual_bess'] = False
        reference_steps = history['surplus_steps'][i] + history['shortage_steps'][i]
        if summary['central_dispatch_steps'] != reference_steps:
            raise RuntimeError('中央ルール評価の対象時点数が元の定期検証と一致しない')
    summary['run_dir'] = str(run)
    summary['checkpoint_episode'] = args.episode
    summary['code_snapshot'] = str(snapshot)
    summary['actors_sha256'] = [hashlib.sha256((checkpoint / f'actor_{j}_ep{args.episode}.pth').read_bytes()).hexdigest() for j in range(7)] if args.controller == 'marl' else []
    for episode, data in result['all_episode_data'].items():
        info = info_by_episode[int(episode) - 1]
        with (output / f'episode_{episode}.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            fields = ['episode', 'service_date', 'step', 'block', 'tracking_enabled', 'target_kw', 'actual_kw', 'pre_central_kw', 'tolerance_kw', 'baseline_kw', 'up_kw', 'down_kw'] + [f'station_{j}_kw' for j in range(1, 8)]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for t in range(len(data['ag_requests'])):
                b = t // 6
                row = {'episode': episode, 'service_date': info['service_date'], 'step': t + 1, 'block': b,
                       'tracking_enabled': int(data['tracking_enabled'][t]), 'target_kw': data['ag_requests'][t],
                       'actual_kw': data['total_ev_transport'][t], 'pre_central_kw': data['raw_actor_total_power_kw'][t], 'tolerance_kw': data['tol_narrow'][t],
                       'baseline_kw': info['baseline'][b], 'up_kw': info['up'][b], 'down_kw': info['down'][b]}
                row.update({f'station_{j}_kw': data[f'actual_ev{j}'][t] for j in range(1, 8)})
                writer.writerow(row)
    (output / 'episodes.json').write_text(json.dumps(info_by_episode, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'diagnostic_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.controller == 'marl' and not summary['reference_matches']:
        raise RuntimeError('再現結果が保存済み検証の集計値と一致しない')


if __name__ == '__main__':
    main()

"""実験B0：やり方1（最初のやり方）の入札で、本線と同じ設定の MARL を学習する（6 station、AEMO・ERCOT、2026-10-15〜）。

    python stage_worker.py <task> <stage>      task は aemo、ercot。stage は preflight / train_bank / test_bank / pretrain

入札器は、改善の表（docs/論文の結果と実験計画.md 5.1節）の入札器の項目をすべて外す。
  - EVの実現は、128通りの候補から無作為に3通り（EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION=random_3、日の seed で決まる）
  - 最低入札量なし（1 kW。解く単位の下限）
  - アセスメントIの余裕なし（0 kW）
  - 基準値の段差の罰なし（重み0）
設計用の指令128本、日本の規則（アセスメントI・II）、平日だけは本線と同じ。
学習の設定（観測、報酬の形と倍率、Q_MIX 0.5、2000回）は resbid_6station_20261012 の AEMO・ERCOT と同じ。
入札は Windows で作り、学習は研究室の GPU で行う。
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BANKS = ROOT / 'execute_results' / 'bid_banks'
SELECTION = 'random_3'
BANK_PREFIX = 'wr_cap77wd_b0_min1_margin0_step0_rand3'
TASKS = {
    'aemo': ('aemo_plan_deviation', 'wr_cap77wd_aemoplan_6station_b0_tanh_ls717'),
    'ercot': ('ercot_plan_deviation', 'wr_cap77wd_ercotplan_6station_b0_tanh_ls717'),
}
FLAGS = {'EVMA_EV_CANDIDATE_CACHE': '1', 'EVMA_GLOBAL_BALANCE_REWARD_SHAPE': 'tanh_linear',
         'EVMA_LOCAL_REWARD_SCALE': repr(25.085 / 7.7 * 2.2),
         'EVMA_STATION_ROSTER': 'workplace_residential',
         'EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW': '1',
         'EVMA_ASSESSMENT_I_CONTROL_MARGIN_KW': '0',
         'EVMA_LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT': '0',
         'EVMA_OBSERVE_FLEET_RESIDUAL': '1', 'EVMA_OBSERVE_BID_CONTEXT': '1', 'EVMA_OBSERVE_TRACKING_BAND': '0'}


def banks(task: str) -> tuple[Path, Path]:
    signal_set = TASKS[task][0]
    return (
        BANKS / f'{BANK_PREFIX}_train_25_6station_128cmd_3ev_{signal_set}',
        BANKS / f'{BANK_PREFIX}_validation_5_6station_128cmd_3ev_{signal_set}',
    )


def configure(task: str):
    signal_set, _model = TASKS[task]
    train_bank, test_bank = banks(task)
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ.update({
        'EVMA_NUM_STATIONS': '6',
        'EVMA_MARL_ALGORITHM': 'hybrid',
        'EVMA_ACTIVATION_SIGNAL_SET': signal_set,
        'EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION': SELECTION,
        'EVMA_LOWER_TRAIN_BID_BANK_DIR': str(train_bank),
        'EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR': str(test_bank),
        **FLAGS,
    })
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONUTF8'] = '1'
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import Config
    return Config


def check(config, task: str) -> dict:
    import EnvConfig
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES
    from training.blockwise_bid import minimum_bid_quantity_kw

    signal_set = TASKS[task][0]
    assert EnvConfig.MARKET_TRACKING_RULE == 'japan'
    assert minimum_bid_quantity_kw() == 1.0
    assert EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW == 0.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT == 0.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION == SELECTION
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIOS == 3
    assert config.NUM_STATIONS == 6
    assert EnvConfig.STATION_ROSTER_NAME == 'workplace_residential'
    assert config.MARL_ALGORITHM == 'hybrid'
    assert config.ACTIVATION_SIGNAL_SET == signal_set
    assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert abs(config.LOCAL_REWARD_SCALE - 2.2 / EnvConfig.RATING_KW_SCALE) < 1e-9
    assert config.GLOBAL_REWARD_SCALE == 1.0
    assert config.ACTOR_BOUND_GRADIENT == 'inward'
    assert config.MEMORY_SIZE == 500000
    assert EnvConfig.EV_CHARGER_MAX_POWER_OPTIONS_KW == (7.7, 7.7, 7.7, 7.7)
    assert abs(EnvConfig.RATING_KW_SCALE - 7.7 / 25.085) < 1e-9
    assert abs(EnvConfig.GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW - 150.0 * 6 / 7 * EnvConfig.RATING_KW_SCALE) < 1e-9
    assert abs(EnvConfig.GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW - 60.0 * 6 / 7 * EnvConfig.RATING_KW_SCALE) < 1e-9
    assert EnvConfig.OBS_COMMAND_SCALE == 'bank'
    assert EnvConfig.BANK_DAY_CLASSES == ('weekday',)
    assert EnvConfig.EV_TARGET_SOC_MAX_PCT == 80.0
    assert EnvConfig.LOWER_BID_LOOKAHEAD_BLOCKS == 24
    assert EnvConfig.GLOBAL_BALANCE_REWARD_SHAPE == 'tanh_linear'
    assert EnvConfig.TRACKING_DIFFERENCE_REWARD is False
    assert EnvConfig.FLEET_RESIDUAL_OBS_PER_STATION is False
    assert LOCAL_TAIL_FEATURE_NAMES[:4] == ('demand_0', 'market_tracking_enabled', 'fleet_residual_prev', 'instruction_scale_kw')
    assert 'tracking_band_kw' not in LOCAL_TAIL_FEATURE_NAMES and len(LOCAL_TAIL_FEATURE_NAMES) == 4 + 72 + 1
    assert EnvConfig.PER_STATION_SESSION_IDS == ('ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S', 'ACN_CALTECH_GARAGE1',
                                                'RESIDENTIAL NORWAY _ TRO_R', 'ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S')
    return {
        'task': task, 'model': TASKS[task][1], 'minimum_bid_kw': minimum_bid_quantity_kw(),
        'assessment_i_margin_kw': EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW,
        'baseline_step_weight': EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT,
        'ev_selection': SELECTION,
        'q_mix_global_weight': config.Q_MIX_GLOBAL_WEIGHT, 'local_reward_scale': config.LOCAL_REWARD_SCALE,
        'global_reward_scale': config.GLOBAL_REWARD_SCALE,
    }


def main() -> int:
    task, stage, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    config = configure(task)
    model = TASKS[task][1]
    train_bank, test_bank = banks(task)
    import torch
    torch.set_num_threads(1)
    summary = check(config, task)
    if stage == 'preflight':
        import numpy as np
        from environment.EVEnv import EVEnv
        from environment.normalize import normalize_observation
        from training.system_controller import build_agent

        env = EVEnv()
        env.reset(net_demand_series=np.zeros(config.EPISODE_STEPS, dtype=np.float32), service_date='2024-08-05')
        agent = build_agent(env)
        agent.set_test_mode(True)
        with torch.no_grad():
            action = agent.act(normalize_observation(env.begin_step()), env=env, noise=False)
        assert tuple(action.shape) == (6, env.max_ev_per_station)
        assert bool(torch.isfinite(action).all())
        summary.update({'cuda': torch.cuda.is_available()})
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    if stage in ('train_bank', 'test_bank'):
        module = importlib.import_module('tools.build_training_bid_bank')
        split = 'train' if stage == 'train_bank' else 'test'
        days = '25' if split == 'train' else '5'
        workers = extra[0] if extra else ('5' if split == 'train' else '2')
        scenario_workers = extra[1] if len(extra) > 1 else '3'
        sys.argv = ['build_training_bid_bank.py', '--split', split, '--days', days,
                    '--train-split-count', '25', '--paired-train-days', '25', '--paired-test-days', '5',
                    '--workers', workers, '--scenario-workers', scenario_workers,
                    '--output-dir', str(train_bank if split == 'train' else test_bank)]
        return int(module.main() or 0)
    if stage == 'pretrain':
        assert torch.cuda.is_available(), 'training must run on the GPU'
        module = importlib.import_module('pre_train')
        sys.argv = ['pre_train.py', '--episodes', '2000', '--model-name', model,
                    '--resume-checkpoint-interval', '100',
                    '--bank-dir', str(train_bank), '--test-bank-dir', str(test_bank)] + extra
        return int(module.main() or 0)
    raise ValueError(stage)


if __name__ == '__main__':
    raise SystemExit(main())

"""20 station の本線（resbid の設定、大域＋局所の MARL、行動器の global の勾配の割合0.5）。研究室で回す（2026-10-14〜）。
abg_resbid_20station_20261013/stage_worker.py（20 station の ABG）の写しで、違いは Q_MIX（0.5、Config の既定）とモデル名だけ。
入札は 20 station の本線の入札（最低入札量250 kW、余裕25 kW）、観測・報酬の形と倍率（大域1、局所7.17）は 6 station の本線と同じ。
倍率は 6 station で hobo100100 に合わせた値なので、20 station で勾配の大きさが外れたら報酬で直す（見張りの grad_check）。

    python stage_worker.py <task> <stage>      task は aemo、ercot、pjm（疑似指令 pjm_regd_phase_shift）。stage は preflight / pretrain
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BANKS = ROOT / 'execute_results' / 'bid_banks'
SELECTION = 'low_connection_2_max_count'
BANK_PREFIX = 'wr_cap77wd_min250_margin25'
TASKS = {
    'aemo': ('aemo_plan_deviation', 'wr_cap77wd_aemoplan_20station_resbid_tanh_ls717'),
    'ercot': ('ercot_plan_deviation', 'wr_cap77wd_ercotplan_20station_resbid_tanh_ls717'),
    'pjm': ('pjm_regd_phase_shift', 'wr_cap77wd_pjmplan_20station_resbid_tanh_g04l15'),
}
# PJM だけ報酬の倍率を 6 station の PJM（resbid_6station_20261012）と同じにする。大域0.4倍、局所をさらに1.5倍。
# 6 station で倍率を AEMO と同じにすると、大域の勾配が hobo100100 の2倍を超え、追従率が36〜41%から上がらなかった。
TASK_FLAGS = {'pjm': {'EVMA_GLOBAL_REWARD_SCALE': '0.4', 'EVMA_LOCAL_REWARD_SCALE': repr(25.085 / 7.7 * 2.2 * 1.5)}}
LOCAL_SCALE_FACTOR = {'pjm': 1.5}
GLOBAL_SCALE = {'pjm': 0.4}
# EV realizations are shared between the AEMO and ERCOT banks of the same day.
# 局所（SoC）の報酬は 25.085/7.7 = 3.26 倍（定格7.7 kW で局所の勾配が0.307倍になる分を戻す）。倍率1の Windows の版（wr_cap77wd_aemoplan_6station_resbid_tanh）は
# ep59 で行動器の勾配が 局所0.24・大域0.52（hobo100100 の約1/10）、大域の割合0.694（hobo100100 0.623）だった。研究室で一から回す。
FLAGS = {'EVMA_EV_CANDIDATE_CACHE': '1', 'EVMA_GLOBAL_BALANCE_REWARD_SHAPE': 'tanh_linear',
         # 局所3.26倍の版（resbid_tanh_ls326）は ep329 で 局所1.32・大域5.05（hobo100100 2.93・5.33）、大域の割合0.741 と上がり続けた。
         # 大きさを hobo100100 に寄せるため、局所をさらに 2.93/1.32 ≒ 2.2 倍にする（大域は倍率1のまま）。
         'EVMA_LOCAL_REWARD_SCALE': repr(25.085 / 7.7 * 2.2),
         'EVMA_STATION_ROSTER': 'workplace_residential',
         'EVMA_LOWER_CONTROLLER_MIN_TRACKING_BAND_KW': '25', 'EVMA_ASSESSMENT_I_CONTROL_MARGIN_KW': '25',
         'EVMA_OBSERVE_FLEET_RESIDUAL': '1', 'EVMA_OBSERVE_BID_CONTEXT': '1', 'EVMA_OBSERVE_TRACKING_BAND': '0'}


def banks(task: str) -> tuple[Path, Path]:
    signal_set = TASKS[task][0]
    return (
        BANKS / f'{BANK_PREFIX}_train_25_20station_128cmd_3ev_{signal_set}',
        BANKS / f'{BANK_PREFIX}_validation_5_20station_128cmd_3ev_{signal_set}',
    )


def configure(task: str):
    signal_set, _model = TASKS[task]
    train_bank, test_bank = banks(task)
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ.update({
        'EVMA_NUM_STATIONS': '20',
        'EVMA_MARL_ALGORITHM': 'hybrid',
        'EVMA_ACTIVATION_SIGNAL_SET': signal_set,
        'EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION': SELECTION,
        'EVMA_LOWER_TRAIN_BID_BANK_DIR': str(train_bank),
        'EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR': str(test_bank),
        **FLAGS,
        **TASK_FLAGS.get(task, {}),
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
    assert config.NUM_STATIONS == 20
    assert EnvConfig.STATION_ROSTER_NAME == 'workplace_residential'
    assert config.MARL_ALGORITHM == 'hybrid'
    assert config.ACTIVATION_SIGNAL_SET == signal_set
    assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert abs(config.LOCAL_REWARD_SCALE - 2.2 * LOCAL_SCALE_FACTOR.get(task, 1.0) / EnvConfig.RATING_KW_SCALE) < 1e-9
    assert config.GLOBAL_REWARD_SCALE == GLOBAL_SCALE.get(task, 1.0)
    assert config.ACTOR_BOUND_GRADIENT == 'inward'
    assert EnvConfig.EV_CHARGER_MAX_POWER_OPTIONS_KW == (7.7, 7.7, 7.7, 7.7)
    assert abs(EnvConfig.RATING_KW_SCALE - 7.7 / 25.085) < 1e-9
    assert abs(EnvConfig.GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW - 150.0 * 20 / 7 * EnvConfig.RATING_KW_SCALE) < 1e-9
    assert abs(EnvConfig.GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW - 60.0 * 20 / 7 * EnvConfig.RATING_KW_SCALE) < 1e-9
    assert EnvConfig.OBS_COMMAND_SCALE == 'bank'
    assert EnvConfig.BANK_DAY_CLASSES == ('weekday',)
    assert EnvConfig.EV_TARGET_SOC_MAX_PCT == 80.0
    assert EnvConfig.LOWER_BID_LOOKAHEAD_BLOCKS == 24
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION == SELECTION
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIOS == 3
    assert minimum_bid_quantity_kw() == 250.0
    assert EnvConfig.GLOBAL_BALANCE_REWARD_SHAPE == 'tanh_linear'
    assert EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW == 25.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT == 0.5
    assert EnvConfig.TRACKING_DIFFERENCE_REWARD is False
    assert EnvConfig.FLEET_RESIDUAL_OBS_PER_STATION is False
    assert LOCAL_TAIL_FEATURE_NAMES[:4] == ('demand_0', 'market_tracking_enabled', 'fleet_residual_prev', 'instruction_scale_kw')
    assert 'tracking_band_kw' not in LOCAL_TAIL_FEATURE_NAMES and len(LOCAL_TAIL_FEATURE_NAMES) == 4 + 72 + 1
    assert EnvConfig.PER_STATION_SESSION_IDS == ('ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S',
                                                'ACN_CALTECH_GARAGE1', 'RESIDENTIAL NORWAY _ TRO_R') * 5
    return {
        'task': task, 'model': TASKS[task][1], 'minimum_bid_kw': minimum_bid_quantity_kw(),
        'assessment_i_margin_kw': EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW,
        'q_mix_global_weight': config.Q_MIX_GLOBAL_WEIGHT, 'local_reward_scale': config.LOCAL_REWARD_SCALE,
        'global_reward_scale': config.GLOBAL_REWARD_SCALE,
        'reward_error_scale_kw': EnvConfig.GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW,
        'memory_size': config.MEMORY_SIZE, 'stations': config.NUM_STATIONS,
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
        assert tuple(action.shape) == (20, env.max_ev_per_station)
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

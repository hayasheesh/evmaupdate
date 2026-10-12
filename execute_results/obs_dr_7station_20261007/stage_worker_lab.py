"""研究室で回す。7 station AEMO：観測に休日の区分と今のコマの帯の幅を足した行動器を、2通りの学習で並べる。

    python stage_worker.py <task> <stage>

task
  base : 観測を2つ足しただけ
  dr   : base に、追従の差分報酬（EVMA_TRACKING_DIFFERENCE_REWARD）と、
         全体のずれを station 数で割った観測（EVMA_FLEET_RESIDUAL_OBS_PER_STATION）を加える
入札は2本で共通。EV 3通りは「下側2＋最多」、最低入札量は500 kW（帯50 kW）、
アセスメントIの制御の余裕は25 kW（どちらも EnvConfig の7 station の既定値）。
stage は preflight / train_bank / test_bank / pretrain。既定値が想定どおりかは check() で確かめる。
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

# 研究室では、20 station の学習が使うコードとは別の作業フォルダ（data・archive・execute_results は元の
# 場所への相対リンク）でこの変更を動かす。
ROOT = Path('/workspace/EVMALOCALUPDATE_obsdr')
BANKS = ROOT / 'execute_results' / 'bid_banks'
SIGNAL_SET = 'aemo_plan_deviation'
SELECTION = 'low_connection_2_max_count'
BANK_PREFIX = 'low2max_min500_margin25'
TASKS = {
    'base': ('obs2_aemoplan_AB_7station', {}),
    'dr': ('obs2_dr_share_aemoplan_AB_7station', {
        'EVMA_TRACKING_DIFFERENCE_REWARD': '1', 'EVMA_FLEET_RESIDUAL_OBS_PER_STATION': '1'}),
}


def banks() -> tuple[Path, Path]:
    return (
        BANKS / f'{BANK_PREFIX}_train_25_7station_128cmd_3ev_{SIGNAL_SET}',
        BANKS / f'{BANK_PREFIX}_validation_5_7station_128cmd_3ev_{SIGNAL_SET}',
    )


def configure(task: str):
    _model, flags = TASKS[task]
    train_bank, test_bank = banks()
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ.update({
        'EVMA_NUM_STATIONS': '7',
        'EVMA_MARL_ALGORITHM': 'hybrid',
        'EVMA_ACTIVATION_SIGNAL_SET': SIGNAL_SET,
        'EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION': SELECTION,
        'EVMA_LOWER_TRAIN_BID_BANK_DIR': str(train_bank),
        'EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR': str(test_bank),
        **flags,
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

    dr = task == 'dr'
    assert config.NUM_STATIONS == 7
    assert config.MARL_ALGORITHM == 'hybrid'
    assert config.ACTIVATION_SIGNAL_SET == SIGNAL_SET
    assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert config.MEMORY_SIZE == 500000
    assert EnvConfig.GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW == 150.0
    assert EnvConfig.GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW == 60.0
    assert EnvConfig.LOWER_BID_LOOKAHEAD_BLOCKS == 24
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION == SELECTION
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIOS == 3
    assert minimum_bid_quantity_kw() == 500.0
    assert EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW == 25.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT == 0.5
    assert EnvConfig.TRACKING_DIFFERENCE_REWARD is dr
    assert EnvConfig.FLEET_RESIDUAL_OBS_PER_STATION is dr
    assert LOCAL_TAIL_FEATURE_NAMES[-2:] == ('service_day_holiday', 'tracking_band_kw')
    assert len(EnvConfig.PER_STATION_SESSION_IDS) == 7
    return {
        'task': task, 'model': TASKS[task][0], 'minimum_bid_kw': minimum_bid_quantity_kw(),
        'assessment_i_margin_kw': EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW,
        'difference_reward': dr, 'residual_per_station': dr, 'memory_size': config.MEMORY_SIZE,
    }


def main() -> int:
    task, stage, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    config = configure(task)
    model = TASKS[task][0]
    train_bank, test_bank = banks()
    import torch
    torch.set_num_threads(1)
    summary = check(config, task)
    if stage == 'preflight':
        import numpy as np
        from environment.EVEnv import EVEnv
        from environment.normalize import normalize_observation
        from training.system_controller import build_agent

        env = EVEnv()
        env.reset(net_demand_series=np.zeros(config.EPISODE_STEPS, dtype=np.float32), service_date='2024-08-03')
        agent = build_agent(env)
        agent.set_test_mode(True)
        with torch.no_grad():
            action = agent.act(normalize_observation(env.begin_step()), env=env, noise=False)
        assert tuple(action.shape) == (7, env.max_ev_per_station)
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

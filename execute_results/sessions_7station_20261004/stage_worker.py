"""7 station の作り直し：EV は各局の実セッションから作る（docs/EVデータ.md）。設計指令128本 × EV 3本。

  python stage_worker.py <task> <stage>

task は aemo / ercot / gb / pjm（AB）と ercot_maddpg（標準 MADDPG、ERCOT の入札を使う）。
stage は preflight / train_bank / test_bank / pretrain。設定は EnvConfig と Config の既定値をそのまま使い、
市場、入札バンクの置き場所、学習器の種類だけを指定する。既定値が本線の値であることは check() で確かめる。
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BANKS = ROOT / 'execute_results' / 'bid_banks'
TASKS = {
    'aemo': ('aemo_plan_deviation', 'hybrid', 'sessions_v2_aemoplan_AB_7station'),
    'ercot': ('ercot_plan_deviation', 'hybrid', 'sessions_v2_ercotplan_AB_7station'),
    'gb': ('elexon_plan_deviation', 'hybrid', 'sessions_v2_elexonplan_AB_7station'),
    # PJM は入札と学習（学習中のテストを含む）を疑似指令で行う。実指令は最終評価にだけ使う。
    'pjm': ('pjm_regd_phase_shift', 'hybrid', 'sessions_v2_pjmregd_AB_scale2_7station'),
    'ercot_maddpg': ('ercot_plan_deviation', 'maddpg', 'sessions_v2_ercotplan_MADDPGstd_7station'),
}


# 大域の追従報酬の尺度と直線部 [kW]。既定は 150 / 60。
# PJM は学習中の追従誤差が約280 kW（AEMO・ERCOT は65〜135 kW）で、既定のままだと
# 1ステップの報酬が約-2.3 になり、大域批評家の勾配がほぼ毎回切られた。利用者の判断で、
# PJM だけ尺度と直線部を2倍にする（誤差280 kW の報酬は約-0.7）。
REWARD_KW = {'pjm': (300.0, 120.0)}
DEFAULT_REWARD_KW = (150.0, 60.0)


def banks(signal_set: str) -> tuple[Path, Path]:
    return (
        BANKS / f'sessions_v2_train_25_7station_128cmd_3ev_{signal_set}',
        BANKS / f'sessions_v2_validation_5_7station_128cmd_3ev_{signal_set}',
    )


def configure(task: str):
    signal_set, algorithm, _model = TASKS[task]
    train_bank, test_bank = banks(signal_set)
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ.update({
        'EVMA_NUM_STATIONS': '7',
        'EVMA_MARL_ALGORITHM': algorithm,
        'EVMA_ACTIVATION_SIGNAL_SET': signal_set,
        'EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES': '128',
        'EVMA_LOWER_TRAIN_BID_BANK_DIR': str(train_bank),
        'EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR': str(test_bank),
    })
    if task in REWARD_KW:
        scale, tail = REWARD_KW[task]
        os.environ['EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW'] = str(scale)
        os.environ['EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW'] = str(tail)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONUTF8'] = '1'
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import Config
    return Config


def check(config, task: str) -> dict:
    """The defaults this rebuild relies on."""
    import EnvConfig
    from training.blockwise_bid import minimum_bid_quantity_kw

    signal_set, algorithm, _model = TASKS[task]
    assert config.NUM_STATIONS == 7
    assert config.MARL_ALGORITHM == algorithm
    assert config.ACTIVATION_SIGNAL_SET == signal_set
    assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert config.MEMORY_SIZE == 500000
    scale, tail = REWARD_KW.get(task, DEFAULT_REWARD_KW)
    assert EnvConfig.GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW == scale
    assert EnvConfig.GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW == tail
    assert EnvConfig.LOWER_BID_LOOKAHEAD_BLOCKS == 24
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS == 128
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES == 128
    assert minimum_bid_quantity_kw() == 250.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT == 0.5
    assert len(EnvConfig.PER_STATION_SESSION_IDS) == 7
    for station in EnvConfig.PER_STATION_SESSION_IDS:
        assert (Path(EnvConfig.STATION_SESSION_DIR) / f'{station}.csv').exists(), station
    assert Path(EnvConfig.LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR).exists()
    return {
        'task': task, 'signal_set': signal_set, 'algorithm': algorithm, 'stations': 7,
        'minimum_bid_kw': minimum_bid_quantity_kw(),
        'reward_error_scale_kw': EnvConfig.GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW,
        'reward_linear_tail_kw': EnvConfig.GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW,
        'memory_size': config.MEMORY_SIZE,
    }


def main() -> int:
    task, stage, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    config = configure(task)
    signal_set, _algorithm, model = TASKS[task]
    train_bank, test_bank = banks(signal_set)
    import torch
    torch.set_num_threads(1)
    summary = check(config, task)
    if stage == 'preflight':
        import numpy as np
        from environment.EVEnv import EVEnv
        from environment.normalize import normalize_observation
        from training.system_controller import build_agent

        env = EVEnv()
        env.reset(net_demand_series=np.zeros(config.EPISODE_STEPS, dtype=np.float32), service_date='2024-04-02')
        agent = build_agent(env)
        agent.set_test_mode(True)
        with torch.no_grad():
            action = agent.act(normalize_observation(env.begin_step()), env=env, noise=False)
        assert tuple(action.shape) == (7, env.max_ev_per_station)
        assert bool(torch.isfinite(action).all())
        summary.update({'cuda': torch.cuda.is_available(), 'evs_at_midnight': env.initial_evs_by_station})
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

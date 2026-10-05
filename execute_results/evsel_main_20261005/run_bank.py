"""7局AEMOの入札を、前提のEV実現「下側3＋最多」（4通り）で作る（研究室の本線で動かす）。

    python execute_results/evsel_main_20261005/run_bank.py <設計の指令数> [workers] [scenario_workers]

設計の指令数は 128 または 256。入札日は train の25日。
ほかの設定は execute_results/sessions_7station_20261004/stage_worker.py の aemo と同じで、
前回の選び方の実験（EVMA_evsel_20261005 の run_evsel.py）とも、選び方と指令数のほかは同じ。
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BANKS = ROOT / 'execute_results' / 'bid_banks'
SIGNAL_SET = 'aemo_plan_deviation'
SELECTION = 'low_connection_3_max_count'


def bank_dir(commands: int) -> Path:
    return BANKS / f'evsel_main_train_25_7station_{commands}cmd_low3max_4ev_{SIGNAL_SET}'


def main() -> int:
    commands = int(sys.argv[1])
    if commands not in (128, 256):
        raise SystemExit('設計の指令数は 128 か 256')
    workers = sys.argv[2] if len(sys.argv) > 2 else '4'
    scenario_workers = sys.argv[3] if len(sys.argv) > 3 else '2'
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ.update({
        'EVMA_NUM_STATIONS': '7',
        'EVMA_MARL_ALGORITHM': 'hybrid',
        'EVMA_ACTIVATION_SIGNAL_SET': SIGNAL_SET,
        'EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS': str(commands),
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES': '128',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION': SELECTION,
    })
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONUTF8'] = '1'
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    import EnvConfig
    import training.lower_bid_training as lbt
    from training.blockwise_bid import minimum_bid_quantity_kw

    assert Path(lbt.__file__).resolve().parents[1] == ROOT, lbt.__file__
    assert EnvConfig.NUM_STATIONS == 7
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS == commands
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION == SELECTION
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIOS == 4
    assert minimum_bid_quantity_kw() == 250.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT == 0.5
    assert len(EnvConfig.PER_STATION_SESSION_IDS) == 7
    module = importlib.import_module('tools.build_training_bid_bank')
    sys.argv = ['build_training_bid_bank.py', '--split', 'train', '--days', '25',
                '--train-split-count', '25', '--paired-train-days', '25', '--paired-test-days', '5',
                '--workers', workers, '--scenario-workers', scenario_workers,
                '--output-dir', str(bank_dir(commands))]
    return int(module.main() or 0)


if __name__ == '__main__':
    raise SystemExit(main())

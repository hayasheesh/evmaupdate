"""6 station の大域批評家だけの学習（ABG、EVMA_Q_MIX_GLOBAL_WEIGHT=1.0、局所批評家は学習しない）。最低入札量を行動器の誤差の分位点から
決めるための学習で、本線（wr_gs_6station_20261011）との違いは Q_MIX だけ。入札・観測・局の並び・報酬の倍率は本線と同じ。

6 station（Windows）、大域の報酬を1/16にした版：職場3局と住宅3局（JPL、OSL、Caltech、TRO、JPL、OSL）で、EVの定格7.7 kWの大域＋局所の MARL（本線、global の勾配の割合0.5）を入札から学習まで行う。入札日は平日だけ（EnvConfig.BANK_DAY_CLASSES の既定）。

    python stage_worker.py <task> <stage>

task は aemo。stage は preflight / train_bank / test_bank / pretrain。
  - EVの定格はすべて7.7 kW（EnvConfig の既定）。報酬の kW 定数は平均定格の比0.307で縮める（150→46.0 kW、60→18.4 kW）
  - 最低入札量70 kW（帯7 kW）、アセスメントIの余裕7 kW（7 station の仮の値8 kW を 6/7 倍して丸めた仮の値）
  - station は EVMA_STATION_ROSTER=workplace_residential（職場と住宅を交互に並べる）
  - 観測から、1つ前の時刻の全体のずれ、入札の先読みとその日の指令の大きさ、帯の幅を外す（行動器・局所批評家・大域批評家とも）。
    station 共通の観測は今の指令、今のコマで追従するか、時刻の3つ。入札は観測によらないので、外す前に作り始めた入札をそのまま使う
  - 行動器の global の勾配の割合は0.5（Config の既定、局所批評家も学習する）
  - 局所（SoC）の報酬を 25.085/7.7 = 3.26 倍にする（EVMA_LOCAL_REWARD_SCALE、理由は FLAGS の注記）
  - 観測の指令は学習用入札の95%点で割る、上下限の勾配は内側だけ通す、24時以降のEVは判定しない、目標SoCの上限80%（コードの既定）
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
BANK_PREFIX = 'wr_cap77wd_min70_margin7'
TASKS = {
    'aemo': ('aemo_plan_deviation', 'wr_obsdrop_cap77wd_aemoplan_6station_abg'),
    'ercot': ('ercot_plan_deviation', 'wr_obsdrop_cap77wd_ercotplan_6station_abg'),
}
# EV realizations are shared between the AEMO and ERCOT banks of the same day.
# The local (SoC) reward is scaled by the inverse of the rating ratio. An actor's
# action moves an EV's SoC in proportion to its rating, so capping every EV at
# 7.7 kW (mean 25.085 kW before) cut the local Q gradient reaching the actor to
# about 0.307 of what it was, while the tracking reward's kW constants were
# rescaled with the fleet and its gradient was not. In hyb_cap77wd_* (scale 1)
# the global share of the actor gradient rose to 0.78-0.89 against 0.56-0.68 in
# the runs before the cap, and the SoC hit rate kept falling.
LOCAL_REWARD_SCALE = 25.085 / 7.7
# The tracking reward in the global critics' TD target is scaled by 1/16. With the convex tracking reward
# and scale 1, the global Q gradient reaching the actor was 6-13 times that of docs/hobo100100 at the same
# episode while the local one was about the same, so the global share of the actor gradient sat at 0.89-0.92
# against 0.52-0.62 there (wr_obsdrop_cap77wd_*_6station_2026101*, ep 200-1800). The global/local ratio was
# 8-10 at ep 200-400 and 13-15 after ep 1000. 1/8 brought it below that of hobo100100 in the first episodes
# (global share 0.46, ep 2). With 1/4 the global share rose from 0.45-0.58 (ep 83-140) to 0.76-0.77
# (ep 339-348) against 0.55-0.61 there, while the global magnitude grew to 1.7-2.0 times that of
# hobo100100 and the local one stayed at 0.7 times (wr_obsdrop_cap77wd_*_6station_gs025_20261011_*).
# 1/16 is used, a quarter of the global part of 1/4.
GLOBAL_REWARD_SCALE = 0.0625
FLAGS = {'EVMA_EV_CANDIDATE_CACHE': '1', 'EVMA_Q_MIX_GLOBAL_WEIGHT': '1.0', 'EVMA_LOCAL_REWARD_SCALE': repr(LOCAL_REWARD_SCALE),
         'EVMA_GLOBAL_REWARD_SCALE': repr(GLOBAL_REWARD_SCALE),
         'EVMA_STATION_ROSTER': 'workplace_residential',
         'EVMA_LOWER_CONTROLLER_MIN_TRACKING_BAND_KW': '7', 'EVMA_ASSESSMENT_I_CONTROL_MARGIN_KW': '7',
         'EVMA_OBSERVE_FLEET_RESIDUAL': '0', 'EVMA_OBSERVE_BID_CONTEXT': '0', 'EVMA_OBSERVE_TRACKING_BAND': '0'}


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
    assert config.NUM_STATIONS == 6
    assert EnvConfig.STATION_ROSTER_NAME == 'workplace_residential'
    assert config.MARL_ALGORITHM == 'hybrid'
    assert config.ACTIVATION_SIGNAL_SET == signal_set
    assert config.Q_MIX_GLOBAL_WEIGHT == 1.0
    assert abs(config.LOCAL_REWARD_SCALE - 1.0 / EnvConfig.RATING_KW_SCALE) < 1e-9
    assert config.GLOBAL_REWARD_SCALE == GLOBAL_REWARD_SCALE
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
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION == SELECTION
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIOS == 3
    assert minimum_bid_quantity_kw() == 70.0
    assert EnvConfig.GLOBAL_BALANCE_REWARD_SHAPE == 'convex'
    assert EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW == 7.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT == 0.5
    assert EnvConfig.TRACKING_DIFFERENCE_REWARD is False
    assert EnvConfig.FLEET_RESIDUAL_OBS_PER_STATION is False
    assert LOCAL_TAIL_FEATURE_NAMES == ('demand_0', 'market_tracking_enabled', 'step')
    assert EnvConfig.PER_STATION_SESSION_IDS == ('ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S', 'ACN_CALTECH_GARAGE1',
                                                'RESIDENTIAL NORWAY _ TRO_R', 'ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S')
    return {
        'task': task, 'model': TASKS[task][1], 'minimum_bid_kw': minimum_bid_quantity_kw(),
        'assessment_i_margin_kw': EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW,
        'q_mix_global_weight': config.Q_MIX_GLOBAL_WEIGHT, 'local_reward_scale': config.LOCAL_REWARD_SCALE,
        'global_reward_scale': config.GLOBAL_REWARD_SCALE,
        'reward_error_scale_kw': EnvConfig.GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW,
        'memory_size': config.MEMORY_SIZE,
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

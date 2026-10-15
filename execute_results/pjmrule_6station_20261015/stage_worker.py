"""6 station PJM を、PJM の規則（Performance Score）で入札から学習まで行う（2026-10-15〜、Windows）。

    python stage_worker.py pjm <stage>      stage は preflight / train_bank / test_bank / pretrain

日本の規則の本線（resbid_6station_20261012 の pjm）との違いは、入札が守る規則だけ（EnvConfig.MARKET_TRACKING_RULE=pjm）。
  - 5分ごとの点数 1 − |誤差| / (0.5 × |指令| + 0.5 × AReg) を、設計に使うすべての指令とEVの実現で75%以上に保つ。
    帯は (0.125 + 0.125 × 指令の割合) × AReg、指令がないときは 0.125 × AReg（Manual 12 Rev.57 §4.4.5、OA Schedule 1 §3.2.2(j)）
  - 上げと下げは同じ量（AReg は1つ、Manual 12 §4.3）
  - 能力の確認は5分の出力の上下限（Manual 11 Rev.137 §3.5。30分続ける条件はない）。余裕7 kW は本線と同じ
  - 最低入札量は 帯7 kW ÷ 0.125 = 56 kW（PJM の最低量 0.1 MW を 6/500 に縮めた 1.2 kW より大きい）
指令（疑似指令 pjm_regd_phase_shift）、EV、観測、報酬（大域0.4倍、局所 7.17 × 1.5 倍）、Q_MIX 0.5 は本線の PJM と同じ。
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
BANK_PREFIX = 'wr_cap77wd_pjmrule75_min56_margin7'
SIGNAL_SET = 'pjm_regd_phase_shift'
MODEL = 'wr_cap77wd_pjmplan_6station_pjmrule75_g04l15'
SCORE_FLOOR = 0.75
# paper_eval_20261014/eval_episode.py は configure(task)・banks(task)・TASKS を使う
TASKS = {'pjm': (SIGNAL_SET, MODEL)}
FLAGS = {'EVMA_EV_CANDIDATE_CACHE': '1', 'EVMA_GLOBAL_BALANCE_REWARD_SHAPE': 'tanh_linear',
         'EVMA_MARKET_TRACKING_RULE': 'pjm', 'EVMA_PJM_INTERVAL_SCORE_FLOOR': repr(SCORE_FLOOR),
         'EVMA_GLOBAL_REWARD_SCALE': '0.4', 'EVMA_LOCAL_REWARD_SCALE': repr(25.085 / 7.7 * 2.2 * 1.5),
         'EVMA_STATION_ROSTER': 'workplace_residential',
         'EVMA_LOWER_CONTROLLER_MIN_TRACKING_BAND_KW': '7', 'EVMA_ASSESSMENT_I_CONTROL_MARGIN_KW': '7',
         'EVMA_OBSERVE_FLEET_RESIDUAL': '1', 'EVMA_OBSERVE_BID_CONTEXT': '1', 'EVMA_OBSERVE_TRACKING_BAND': '0'}


def banks(task: str = 'pjm') -> tuple[Path, Path]:
    return (
        BANKS / f'{BANK_PREFIX}_train_25_6station_128cmd_3ev_{SIGNAL_SET}',
        BANKS / f'{BANK_PREFIX}_validation_5_6station_128cmd_3ev_{SIGNAL_SET}',
    )


def configure(task: str = 'pjm'):
    train_bank, test_bank = banks()
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ.update({
        'EVMA_NUM_STATIONS': '6',
        'EVMA_MARL_ALGORITHM': 'hybrid',
        'EVMA_ACTIVATION_SIGNAL_SET': SIGNAL_SET,
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


def check(config) -> dict:
    import EnvConfig
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES
    from training.blockwise_bid import minimum_bid_quantity_kw

    half = 0.5 * (1.0 - SCORE_FLOOR)
    assert EnvConfig.MARKET_TRACKING_RULE == 'pjm'
    assert EnvConfig.PJM_INTERVAL_SCORE_FLOOR == SCORE_FLOOR
    assert abs(EnvConfig.LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC - half) < 1e-12
    assert abs(EnvConfig.TRACKING_BAND_SIGNAL_FRACTION - half) < 1e-12
    assert abs(EnvConfig.TRACKING_IDLE_BAND_FRACTION - 0.5 * half) < 1e-12
    assert EnvConfig.SYMMETRIC_AWARD is True
    assert abs(EnvConfig.CAPABILITY_DURATION_HOURS - 5.0 / 60.0) < 1e-12
    assert EnvConfig.MARKET_MINIMUM_BID_QUANTITY_KW == 100.0
    assert minimum_bid_quantity_kw() == 56.0
    assert config.NUM_STATIONS == 6
    assert EnvConfig.STATION_ROSTER_NAME == 'workplace_residential'
    assert config.MARL_ALGORITHM == 'hybrid'
    assert config.ACTIVATION_SIGNAL_SET == SIGNAL_SET
    assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert abs(config.LOCAL_REWARD_SCALE - 2.2 * 1.5 / EnvConfig.RATING_KW_SCALE) < 1e-9
    assert config.GLOBAL_REWARD_SCALE == 0.4
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
    assert EnvConfig.GLOBAL_BALANCE_REWARD_SHAPE == 'tanh_linear'
    assert EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW == 7.0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT == 0.5
    assert EnvConfig.TRACKING_DIFFERENCE_REWARD is False
    assert EnvConfig.FLEET_RESIDUAL_OBS_PER_STATION is False
    assert LOCAL_TAIL_FEATURE_NAMES[:4] == ('demand_0', 'market_tracking_enabled', 'fleet_residual_prev', 'instruction_scale_kw')
    assert 'tracking_band_kw' not in LOCAL_TAIL_FEATURE_NAMES and len(LOCAL_TAIL_FEATURE_NAMES) == 4 + 72 + 1
    assert EnvConfig.PER_STATION_SESSION_IDS == ('ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S', 'ACN_CALTECH_GARAGE1',
                                                'RESIDENTIAL NORWAY _ TRO_R', 'ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S')
    return {
        'model': MODEL, 'rule': EnvConfig.MARKET_TRACKING_RULE, 'score_floor': SCORE_FLOOR,
        'band_fraction': EnvConfig.LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC,
        'minimum_bid_kw': minimum_bid_quantity_kw(),
        'assessment_i_margin_kw': EnvConfig.ASSESSMENT_I_CONTROL_MARGIN_KW,
        'q_mix_global_weight': config.Q_MIX_GLOBAL_WEIGHT, 'local_reward_scale': config.LOCAL_REWARD_SCALE,
        'global_reward_scale': config.GLOBAL_REWARD_SCALE,
    }


def main() -> int:
    task, stage, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    if task != 'pjm':
        raise ValueError('この実験は pjm だけ')
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    config = configure()
    train_bank, test_bank = banks()
    import torch
    torch.set_num_threads(1)
    summary = check(config)
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
        sys.argv = ['pre_train.py', '--episodes', '2000', '--model-name', MODEL,
                    '--resume-checkpoint-interval', '100',
                    '--bank-dir', str(train_bank), '--test-bank-dir', str(test_bank)] + extra
        return int(module.main() or 0)
    raise ValueError(stage)


if __name__ == '__main__':
    raise SystemExit(main())

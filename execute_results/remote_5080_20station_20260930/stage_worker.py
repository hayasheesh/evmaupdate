"""20 station ABの各段階を、同じ設定で実行する。"""
from __future__ import annotations
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TRAIN_BANK = ROOT / 'execute_results/bid_banks/train_25_20station_256cmd_3ev_aemo_plan_deviation'
TEST_BANK = ROOT / 'execute_results/bid_banks/validation_5_20station_256cmd_3ev_aemo_plan_deviation'


def configure():
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    values = {
        'EVMA_NUM_STATIONS': '20', 'EVMA_MARL_ALGORITHM': 'hybrid',
        'EVMA_ACTIVATION_SIGNAL_SET': 'aemo_plan_deviation',
        'EVMA_ACTOR_EV_COUNT': '1', 'EVMA_LOWER_BID_LOOKAHEAD_BLOCKS': '24',
        'EVMA_GLOBAL_BALANCE_REWARD_MODE': 'bounded_absolute_error',
        'EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW': '150',
        'EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW': '60',
        'EVMA_LOCAL_FLEET_RESIDUAL_OBS': '1',
        'EVMA_GRAD_CLIP_MAX': '5.0', 'EVMA_GRAD_CLIP_MAX_GLOBAL': '5.0',
        'EVMA_Q_MIX_GLOBAL_WEIGHT': '0.5',
        'EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR': '0',
        'EVMA_TRAIN_FORCE_CHARGING': '0', 'EVMA_TRAIN_USE_RESIDUAL_BESS': '0',
        'EVMA_LOWER_TRAIN_BUILD_BID_BANK': '0',
        'EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS': '0',
        'EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS': '256',
        'EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES': '128',
        'EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW': '250',
        'EVMA_LOWER_TRAIN_BID_BANK_DIR': str(TRAIN_BANK),
        'EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR': str(TEST_BANK),
    }
    os.environ.update(values)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import Config
    # Preserve AB's uncorrelated exploration instead of the automatic >=20
    # station default. The override is also recorded in runtime fingerprints.
    Config.GLOBAL_CORRELATED_NOISE_GAIN = 0.0
    return Config


def main():
    stage, extra = sys.argv[1], sys.argv[2:]
    multiprocessing.set_start_method('spawn', force=True)
    config = configure()
    import torch
    torch.set_num_threads(1)
    if stage == 'preflight':
        import EnvConfig
        from environment.EVEnv import EVEnv
        from environment.normalize import normalize_observation
        from training.system_controller import build_agent
        assert config.NUM_STATIONS == 20
        assert config.MARL_ALGORITHM == 'hybrid'
        assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
        assert not config.TRAIN_FORCE_CHARGING
        assert not config.TRAIN_USE_RESIDUAL_BESS
        assert not config.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR
        assert torch.cuda.is_available()
        for key in ('EV_PROFILE_DATA_PATH', 'EV_SOC_ARRIVAL_DISTRIBUTION_PATH', 'DAY_CONTEXT_WEATHER_CSV', 'LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR'):
            assert Path(getattr(EnvConfig, key)).exists(), key
        assert len(EnvConfig.PER_STATION_ARRIVAL_PROFILE_PATHS) == 20
        assert all(Path(p).exists() for p in EnvConfig.PER_STATION_ARRIVAL_PROFILE_PATHS)
        config.CREATE_AGENT_RUNS_WRITER = False
        env = EVEnv()
        import numpy as np
        env.reset(net_demand_series=np.zeros(config.EPISODE_STEPS, dtype=np.float32))
        agent = build_agent(env)
        agent.set_test_mode(True)
        with torch.no_grad():
            action = agent.act(normalize_observation(env.begin_step()), env=env, noise=False)
        assert len(agent.actors) == 20
        assert tuple(action.shape) == (20, env.max_ev_per_station)
        assert bool(torch.isfinite(action).all())
        print(json.dumps({'stations':20,'algorithm':config.MARL_ALGORITHM,
                          'actors':len(agent.actors),'action_shape':list(action.shape),
                          'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                          'lr_actor':config.LR_ACTOR,'lr_local':config.LR_CRITIC_LOCAL,
                          'lr_global':config.LR_GLOBAL_CRITIC,'q_mix_global':config.Q_MIX_GLOBAL_WEIGHT,
                          'global_correlated_noise_gain':config.GLOBAL_CORRELATED_NOISE_GAIN,
                          'design_commands':256,'ev_cases':3,'ev_candidates':128},indent=2),flush=True)
        return
    if stage in ('train_bank', 'test_bank'):
        module = importlib.import_module('tools.build_training_bid_bank')
        split = 'train' if stage == 'train_bank' else 'test'
        days = '25' if split == 'train' else '5'
        bank = TRAIN_BANK if split == 'train' else TEST_BANK
        sys.argv = ['build_training_bid_bank.py','--split',split,'--days',days,
                    '--train-split-count','25','--paired-train-days','25','--paired-test-days','5',
                    '--workers','3','--scenario-workers','5','--output-dir',str(bank)]
        raise SystemExit(module.main())
    if stage == 'pretrain':
        module = importlib.import_module('pre_train')
        sys.argv = ['pre_train.py','--episodes','2000','--model-name','prod_aemoplan_AB_20station_5080',
                    '--resume-checkpoint-interval','100','--bank-dir',str(TRAIN_BANK),'--test-bank-dir',str(TEST_BANK)] + extra
        raise SystemExit(module.main())
    raise ValueError(stage)


if __name__ == '__main__':
    main()

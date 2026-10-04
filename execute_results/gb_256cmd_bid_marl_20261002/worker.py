"""保存したABのコードと設定のまま、GB（BOA − FPN）の256指令入札でAB学習を行う。

ercot_ab_2000_20260930/worker.py と同じ構成。違いは指令ライブラリ、入札バンク
（設計指令256本）、モデル名だけ。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

RUN_ROOT = Path(__file__).resolve().parent
ROOT = RUN_ROOT.parents[1]
SOURCE = RUN_ROOT / 'runtime_source'
TRAIN_BANK = ROOT / 'execute_results/bid_banks/train_25_minmedmax_3of128ev_256cmd_all_commands_elexon_plan_deviation'
TEST_BANK = ROOT / 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_256cmd_all_commands_elexon_plan_deviation'
LIBRARY = ROOT / 'data/elexon/command_libraries/battery_plan_deviation_calendar_day'
MODEL = 'prod_elexonplan256_AB_7station'
DESIGN_COMMANDS = 256


def configure(check_banks: bool = True):
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ['EVMA_ACTIVATION_SIGNAL_SET'] = 'elexon_plan_deviation'
    os.environ['EVMA_LOWER_TRAIN_BUILD_BID_BANK'] = '0'
    os.environ['EVMA_FINETUNE_ENABLE'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(SOURCE))
    os.chdir(ROOT)
    context = json.loads((RUN_ROOT / 'reference_context.json').read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256()
    for relative in sorted(context['source']['files']):
        fingerprint.update(relative.encode())
        fingerprint.update(b'\0')
        fingerprint.update((SOURCE / relative).read_bytes())
        fingerprint.update(b'\0')
    assert fingerprint.hexdigest() == context['source']['sha256']

    overrides = {
        'PROJECT_ROOT': str(SOURCE),
        'ACTIVATION_SIGNAL_SET': 'elexon_plan_deviation',
        'ACTIVATION_SCENARIO_DIR': str(LIBRARY),
        'LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR': str(LIBRARY),
        'LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS': DESIGN_COMMANDS,
        'LOWER_TRAIN_UPPER_BID_BANK_DIR': str(TRAIN_BANK),
        'LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR': str(TEST_BANK),
        'LOWER_TRAIN_UPPER_BID_MODEL_NAME': MODEL,
        'LOWER_TRAIN_UPPER_BID_BANK_BUILD_MISSING': False,
        'LOWER_TRAIN_ACCEPT_BANK_AS_IS': False,
        'FINETUNE_ENABLE': False,
    }
    modules = {}
    differences = {}
    for name in ('EnvConfig', 'Config', 'environment.observation_config'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(SOURCE), module.__file__
        saved = context['runtime'][name.rsplit('.', 1)[-1]]
        for key, value in saved.items():
            if isinstance(value, list) and key.endswith(('FEATURES', 'FEATURE_NAMES')):
                value = tuple(value)
            setattr(module, key, value)
        if name in ('EnvConfig', 'Config'):
            differences[name] = {}
            for key, value in overrides.items():
                old = getattr(module, key, None)
                setattr(module, key, value)
                if old != value:
                    differences[name][key] = {'reference': old, 'current': value}
        modules[name] = module

    config = modules['Config']
    assert config.NUM_STATIONS == 7 and config.MARL_ALGORITHM == 'hybrid'
    assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert config.ACTOR_USE_ACTIVE_EV_COUNT
    assert not config.TRAIN_FORCE_CHARGING and not config.TRAIN_USE_RESIDUAL_BESS
    assert not config.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR
    assert not getattr(config, 'STATION_RULE_ALLOCATION', False)
    assert config.LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS == DESIGN_COMMANDS
    assert config.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES == 128
    if check_banks:
        for bank, days in ((TRAIN_BANK, 25), (TEST_BANK, 5)):
            manifest = json.loads((bank / 'manifest.json').read_text(encoding='utf-8'))
            settings = manifest['settings']
            assert manifest['complete'] and manifest['completed_days'] == days
            assert settings['activation_scenarios'] == DESIGN_COMMANDS and settings['fixed_ev_scenarios'] == 3
            assert Path(settings['activation_source_dir']).resolve() == LIBRARY.resolve()
            entries = list((bank / 'days').glob('*/entry.json'))
            assert len(entries) == days
            for path in entries:
                entry = json.loads(path.read_text(encoding='utf-8'))
                assert entry['summary']['bid_feasible'] and (path.parent / 'fixed_bid.pkl').is_file()

    report = {
        'source_sha256': fingerprint.hexdigest(), 'model': MODEL,
        'stations': 7, 'target_episodes': 2000,
        'algorithm': config.MARL_ALGORITHM, 'q_mix_global': config.Q_MIX_GLOBAL_WEIGHT,
        'force': config.TRAIN_FORCE_CHARGING, 'central': config.USE_CENTRAL_EV_RESIDUAL_ALLOCATOR,
        'bess': config.TRAIN_USE_RESIDUAL_BESS, 'global_correlated_noise_gain': config.GLOBAL_CORRELATED_NOISE_GAIN,
        'train_bank': str(TRAIN_BANK), 'test_bank': str(TEST_BANK),
        'design_commands': DESIGN_COMMANDS, 'ev_cases': 3, 'banks_checked': check_banks,
        'differences_from_reference': differences,
    }
    (RUN_ROOT / 'configuration_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[GB AB configuration] ' + json.dumps({k: v for k, v in report.items() if k != 'differences_from_reference'}), flush=True)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--config-only', action='store_true', help='check the configuration without the banks')
    args = parser.parse_args()
    assert args.episodes == 2000
    configure(check_banks=not args.config_only)
    if args.preflight_only or args.config_only:
        return 0

    train = importlib.import_module('training.train')
    original_snapshot = train.snapshot_code_to_archive
    original_directory = train.create_model_directory

    def snapshot(model_dir, project_root=None):
        result = original_snapshot(model_dir, project_root=str(SOURCE))
        shutil.copytree(SOURCE / 'market', Path(result) / 'market', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        return result

    def create_directory(model_name):
        result = original_directory(model_name)
        marker = RUN_ROOT / 'run_dir.json'
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps({'runDir': str(Path(result).resolve())}) + '\n', encoding='utf-8')
        os.replace(temporary, marker)
        return result

    train.snapshot_code_to_archive = snapshot
    train.create_model_directory = create_directory
    pretrain = importlib.import_module('pre_train')
    sys.argv = ['pre_train.py', '--episodes', str(args.episodes), '--model-name', MODEL,
                '--resume-checkpoint-interval', '20', '--bank-dir', str(TRAIN_BANK),
                '--test-bank-dir', str(TEST_BANK)]
    return pretrain.main()


if __name__ == '__main__':
    raise SystemExit(main())

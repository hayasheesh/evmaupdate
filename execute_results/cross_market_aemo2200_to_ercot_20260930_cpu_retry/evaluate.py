"""AEMO AB checkpoint 2200をERCOTの検証用入札へ変更なしで適用する。"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCE = ROOT / 'execute_results/ercot_ab_2000_20260930/runtime_source'
MODEL = ROOT / 'archive/prod_aemoplan_AB_7station_20260926_210314'
BANK = ROOT / 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_128cmd_all_commands_ercot_plan_deviation'
LIBRARY = ROOT / 'data/ercot/sced/command_libraries/esr_plan_deviation_calendar_day'
EPISODE = 2200


def main() -> int:
    # On this Windows/PyTorch build an empty value still makes is_available()
    # true while device_count() is zero. '-1' reliably selects CPU loading.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ['EVMA_ACTIVATION_SIGNAL_SET'] = 'ercot_plan_deviation'

    context = json.loads((MODEL / 'resume/latest.json').read_text(encoding='utf-8'))['context']
    digest = hashlib.sha256()
    for relative in sorted(context['source']['files']):
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        digest.update((SOURCE / relative).read_bytes())
        digest.update(b'\0')
    assert digest.hexdigest() == context['source']['sha256'], '保存済みABコードとの不一致'
    assert 'aemo_plan_deviation' in context['train_bid_bank']['path']
    manifest = json.loads((BANK / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['completed_days'] == 5
    assert manifest['settings']['activation_scenarios'] == 128
    assert manifest['settings']['fixed_ev_scenarios'] == 3
    assert Path(manifest['settings']['activation_source_dir']).resolve() == LIBRARY.resolve()
    assert all((MODEL / 'results' / f'TEST{EPISODE}' / f'actor_{i}_ep{EPISODE}.pth').is_file() for i in range(7))

    sys.path.insert(0, str(SOURCE))
    os.chdir(ROOT)
    overrides = {
        'PROJECT_ROOT': str(SOURCE),
        'ACTIVATION_SIGNAL_SET': 'ercot_plan_deviation',
        'ACTIVATION_SCENARIO_DIR': str(LIBRARY),
        'LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR': str(LIBRARY),
        'LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR': str(BANK),
    }
    for name in ('EnvConfig', 'Config', 'environment.observation_config'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(SOURCE)
        for key, value in context['runtime'][name.rsplit('.', 1)[-1]].items():
            if isinstance(value, list) and key.endswith(('FEATURES', 'FEATURE_NAMES')):
                value = tuple(value)
            setattr(module, key, value)
        if name in ('EnvConfig', 'Config'):
            for key, value in overrides.items():
                setattr(module, key, value)

    config = importlib.import_module('Config')
    assert config.NUM_STATIONS == 7 and config.MARL_ALGORITHM == 'hybrid'
    assert config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert config.ACTOR_USE_ACTIVE_EV_COUNT

    # Verify the bank against the exact CSV content once. The same immutable
    # signature is reused for each validation date in this one read-only run.
    activation = importlib.import_module('market.activation_scenarios')
    original_signature = activation.activation_library_signature
    signature = original_signature(LIBRARY)
    assert signature['activation_library_sha256'] == manifest['settings']['activation_library_sha256']
    assert signature['activation_signal_mode'] == manifest['settings']['activation_signal_mode']

    def fixed_signature(directory=None):
        path = Path(directory or LIBRARY).resolve()
        if path != LIBRARY.resolve():
            return original_signature(directory)
        return dict(signature)

    activation.activation_library_signature = fixed_signature
    config_info = {
        'model_dir': str(MODEL), 'checkpoint_episode': EPISODE,
        'training_market': 'aemo_plan_deviation',
        'evaluation_market': 'ercot_plan_deviation',
        'evaluation_bank': str(BANK), 'evaluation_command_partition': 'holdout',
        'evaluation_pipeline': 'marl_raw', 'bid_days': 5,
        'commands_per_day': 24, 'independent_ev_seeds_per_command': 3,
        'source_sha256': digest.hexdigest(),
        'ercot_library_sha256': signature['activation_library_sha256'],
        'model_observation_normalization': str(MODEL / 'input/observation_normalization.json'),
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
    }
    (RUN / 'configuration.json').write_text(json.dumps(config_info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[cross-market] verified source, checkpoint and ERCOT library', flush=True)

    evaluator = importlib.import_module('tools.evaluate_final_system_on_bid_bank')
    result = evaluator.main([
        '--model-dir', str(MODEL), '--episode', str(EPISODE),
        '--bid-bank-dir', str(BANK), '--command-scenarios', '24',
        '--ev-seeds', '3', '--base-seed', '1422090',
        '--pipeline', 'marl_raw', '--max-days', '5',
        '--output-dir', str(RUN / 'results'),
    ])
    if result:
        return int(result)

    import pandas as pd
    rows = pd.read_csv(RUN / 'results/all_rollouts.csv')
    assert len(rows) == 5 * 24 * 3
    assert set(rows['evaluation_pipeline']) == {'marl_raw'}
    assessed = int(rows['assessed_steps'].sum())
    missed = int(rows['missed_steps'].sum())
    evs = int(rows['departing_evs'].sum())
    soc_met = int(rows['departing_evs_soc_met'].sum())
    assert assessed > 0 and evs > 0 and 0 <= missed <= assessed and 0 <= soc_met <= evs
    tracking_pct = 100.0 * (assessed - missed) / assessed
    soc_pct = 100.0 * soc_met / evs
    summary = {
        **config_info, 'rollouts': len(rows),
        'assessed_5min_steps': assessed, 'passed_5min_steps': assessed - missed,
        'departing_evs': evs, 'soc_met_evs': soc_met,
        'tracking_pct_weighted': tracking_pct, 'soc_pct_weighted': soc_pct,
        'tracking_plus_soc': tracking_pct + soc_pct,
    }
    (RUN / 'summary_checked.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[cross-market] ' + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

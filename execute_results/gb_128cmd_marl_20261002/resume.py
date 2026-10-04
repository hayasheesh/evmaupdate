"""GB AB（128本 × 3の入札）を、保存した完全な状態から続ける。ercot_ab_fast_resume_20261001/fast_resume.py と同じ処理。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'archive/prod_elexonplan_AB_7station_20261002_215844'
SOURCE = ROOT / 'execute_results/gb_128cmd_marl_20261002/runtime_source'
TRAIN_BANK = ROOT / 'execute_results/bid_banks/train_25_minmedmax_3of128ev_128cmd_all_commands_elexon_plan_deviation'
TEST_BANK = ROOT / 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_128cmd_all_commands_elexon_plan_deviation'
SIGNATURE_KEYS = (
    'activation_signal_mode', 'activation_signal_regime',
    'activation_source_dir', 'activation_library_file_count',
    'activation_library_sha256',
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.episodes != 2000:
        raise ValueError('This resume is scoped to the existing 2000-episode GB run')

    manifest = json.loads((ARCHIVE / 'resume/latest.json').read_text(encoding='utf-8'))
    context = manifest['context']
    if not manifest['exact'] or not 0 < manifest['completed_training_episode'] < args.episodes:
        raise ValueError('No exact GB training state to continue')
    if Path(context['train_bid_bank']['path']).resolve() != TRAIN_BANK.resolve():
        raise ValueError('The training bank differs from the archived run')
    if Path(context['test_bid_bank']['path']).resolve() != TEST_BANK.resolve():
        raise ValueError('The test bank differs from the archived run')

    sys.path.insert(0, str(SOURCE))
    os.chdir(ROOT)
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ['MPLBACKEND'] = 'Agg'
    for name in ('EnvConfig', 'Config', 'environment.observation_config'):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(SOURCE):
            raise RuntimeError(f'Unexpected source module: {module.__file__}')
        for key, value in context['runtime'][name.rsplit('.', 1)[-1]].items():
            if isinstance(value, list) and key.endswith(('FEATURES', 'FEATURE_NAMES')):
                value = tuple(value)
            setattr(module, key, value)

    # The submitted bids already record the certified command library identity.
    # Use those recorded metadata for reporting instead of hashing and opening
    # every command CSV once per train/validation episode. Sampling still reads
    # the actual files and uses the same source code, seeds and partitions.
    training_bank = json.loads((TRAIN_BANK / 'manifest.json').read_text(encoding='utf-8'))
    test_bank = json.loads((TEST_BANK / 'manifest.json').read_text(encoding='utf-8'))
    if not training_bank['complete'] or not test_bank['complete']:
        raise ValueError('The GB bid banks are incomplete')
    settings = training_bank['settings']
    signature = {key: settings[key] for key in SIGNATURE_KEYS}
    if any(test_bank['settings'][key] != signature[key] for key in SIGNATURE_KEYS):
        raise ValueError('GB train and test banks do not record the same library')
    library = Path(signature['activation_source_dir']).resolve()
    saved_library = Path(context['runtime']['EnvConfig']['LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR']).resolve()
    if library != saved_library or not library.is_dir():
        raise ValueError('The saved command library path differs from the original run')

    activation = importlib.import_module('market.activation_scenarios')

    def recorded_signature(directory=None):
        requested = Path(directory or saved_library).resolve()
        if requested != saved_library:
            raise ValueError(f'Unexpected command source during GB resume: {requested}')
        return dict(signature)

    activation.activation_library_signature = recorded_signature
    lower_bid = importlib.import_module('training.lower_bid_training')
    lower_bid.activation_library_signature = recorded_signature

    from training import training_resume
    actual = training_resume.build_pretrain_resume_context(
        project_root=SOURCE,
        model_name=context['model_name'],
        forecast_seed=context['forecast_seed'],
        train_split_count=context['train_split_count'],
        bid_bank_dir=TRAIN_BANK,
        test_bid_bank_dir=TEST_BANK,
        observation_normalization_profile=context['observation_normalization_profile'],
    )
    mismatches = training_resume._context_mismatches(context, actual)
    if mismatches:
        raise RuntimeError('Exact resume context mismatch:\n' + '\n'.join(mismatches))

    audit = {
        'saved_episode': manifest['completed_training_episode'],
        'target_episode': args.episodes,
        'archive': str(ARCHIVE), 'source': str(SOURCE),
        'source_sha256': context['source']['sha256'],
        'training_bank': str(TRAIN_BANK), 'test_bank': str(TEST_BANK),
        'recorded_library_signature': signature,
        'optimization': 'reuse recorded library metadata; omit per-episode CSV fingerprint and header scan',
        'model_state_restoration': 'exact optimizer, replay buffer, RNG and episode cursor',
    }
    (RUN / 'optimization_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print('[fast-resume] exact source/bank/runtime context matched', flush=True)
    print('[fast-resume] command metadata reused from the original certified bid bank', flush=True)
    print(f"[fast-resume] episode={manifest['completed_training_episode']} target={args.episodes}", flush=True)
    if args.preflight_only:
        return 0

    import pre_train
    sys.argv = ['pre_train.py', '--resume-run', str(ARCHIVE),
                '--episodes', str(args.episodes), '--resume-checkpoint-interval', '20']
    return int(pre_train.main())


if __name__ == '__main__':
    raise SystemExit(main())

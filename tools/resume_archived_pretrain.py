"""保存コードを別ディレクトリから読み込み、同一設定で既存学習を再開する。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    source = args.source_dir.resolve()
    manifest = json.loads((run / 'resume/latest.json').read_text(encoding='utf-8'))
    context = manifest['context']
    if args.episodes <= manifest['completed_training_episode']:
        raise ValueError('再開目標は保存済みの学習回数より大きい必要がある')
    sys.path.insert(0, str(source))
    # The archived module values are the experiment identity. Restore paths to
    # original data while code is imported from the pinned source directory.
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    for name in ('EnvConfig', 'Config', 'environment.observation_config'):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f'保存コード以外から設定を読み込んだ: {module.__file__}')
        for key, value in context['runtime'][name.rsplit('.', 1)[-1]].items():
            if isinstance(value, list) and key.endswith(('FEATURES', 'FEATURE_NAMES')):
                value = tuple(value)
            setattr(module, key, value)

    from training import training_resume
    actual = training_resume.build_pretrain_resume_context(
        project_root=source,
        model_name=context['model_name'],
        forecast_seed=context['forecast_seed'],
        train_split_count=context['train_split_count'],
        bid_bank_dir=context['train_bid_bank']['path'],
        test_bid_bank_dir=context['test_bid_bank']['path'],
        observation_normalization_profile=context['observation_normalization_profile'],
    )
    mismatches = training_resume._context_mismatches(context, actual)
    if mismatches:
        raise RuntimeError('保存状態と再開条件が一致しない:\n' + '\n'.join(mismatches))
    print(f"[archive-resume] source/bank/runtime fingerprints matched; "
          f"checkpoint={manifest['completed_training_episode']} target={args.episodes}", flush=True)
    print(f"[archive-resume] code={source} run={run}", flush=True)
    if args.preflight_only:
        return
    import pre_train
    sys.argv = [str(source / 'pre_train.py'), '--resume-run', str(run),
                '--episodes', str(args.episodes), '--resume-checkpoint-interval', '20']
    raise SystemExit(pre_train.main())


if __name__ == '__main__':
    main()

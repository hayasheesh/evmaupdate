"""学習の前の入札の束の照合で、manifest と食い違う設定を書き出す（学習はしない）。

    python diag_manifest.py <task>      task は aemo、ercot
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_worker  # noqa: E402


def main():
    task = sys.argv[1]
    config = stage_worker.configure(task)
    stage_worker.check(config, task)
    train_bank, test_bank = stage_worker.banks(task)
    import training.bid_bank as bank_module

    def report(manifest, required, **_kw):
        actual = manifest.get('settings') or {}
        for key, expected in required.items():
            if key not in actual:
                print('MISSING', key, json.dumps(expected)[:200])
            elif actual[key] != expected:
                print('DIFF', key, 'manifest=', json.dumps(actual[key])[:300], 'now=', json.dumps(expected)[:300])
        raise SystemExit(0)

    bank_module.manifest_settings_match = report
    module = importlib.import_module('pre_train')
    sys.argv = ['pre_train.py', '--episodes', '2000', '--model-name', 'diag_manifest_only',
                '--bank-dir', str(train_bank), '--test-bank-dir', str(test_bank)]
    module.main()


if __name__ == '__main__':
    main()

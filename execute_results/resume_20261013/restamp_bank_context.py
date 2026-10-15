"""再開の記録にある入札データのハッシュを、今の入札データに合わせて付け直す（入札そのものが変わっていない場合だけ）。

    python restamp_bank_context.py <run_dir> --reason "<根拠>"

再開の照合（training.training_resume.load_training_resume）は、入札データのフォルダの .json と .pkl をすべて
ハッシュして、保存したときの値と比べる。入札の段を起動し直すと、入札が同じでも目録（manifest*.json）の
更新時刻（updated_at）が書き換わり、ハッシュが変わって再開できなくなる。

付け直す前に次を確かめ、どれかが成り立たなければ何もしないで止まる。
  - 入札データのファイルの一覧が、保存したときと同じ
  - .pkl（入札そのもの）はすべて、実行のフォルダの名前の時刻（学習を始めた時刻）より前に書かれている
ハッシュは目録（latest.json）と学習状態のファイルの両方にあり、両方が一致している必要がある。学習状態は
別の名前で書き直し、元のファイルと元の latest.json は残す（元に戻せる）。理由と前後のハッシュは
resume/events.jsonl に残す。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from training import training_resume as tr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--reason', required=True)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    resume = run / tr.RESUME_DIRNAME
    manifest_path = resume / tr.RESUME_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    started = datetime.strptime('_'.join(run.name.split('_')[-2:]), '%Y%m%d_%H%M%S').timestamp()

    context = json.loads(json.dumps(manifest['context']))
    changes = {}
    for key in ('train_bid_bank', 'test_bid_bank'):
        saved = context[key]
        bank = Path(saved['path'])
        current = tr.fingerprint_bid_bank(bank)
        if current['files'] != saved['files']:
            raise SystemExit(f'{key}: the file list differs from the saved one; not restamping')
        newer = [p for p in bank.rglob('*.pkl') if p.stat().st_mtime > started]
        if newer:
            raise SystemExit(f'{key}: {len(newer)} bid pickle(s) were written after the run started; not restamping')
        if current['sha256'] != saved['sha256']:
            changes[key] = {'saved': saved['sha256'], 'current': current['sha256']}
            saved['sha256'] = current['sha256']
    if not changes:
        print('nothing to restamp')
        return 0

    state_path = resume / manifest['state_file']
    if tr._sha256_file(state_path) != manifest['state_sha256']:
        raise SystemExit('the state file does not match its manifest; not restamping')
    state = torch.load(state_path, map_location=None, weights_only=False)
    if state.get('context') != manifest['context']:
        raise SystemExit('the state payload context does not match its manifest; not restamping')
    state['context'] = context

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_state = state_path.with_name(f'{state_path.stem}_bankrestamp_{stamp}{state_path.suffix}')
    with open(new_state, 'wb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    shutil.copy2(manifest_path, manifest_path.with_name(f'{manifest_path.name}.before_bankrestamp_{stamp}'))
    manifest.update({'context': context, 'state_file': new_state.name,
                     'state_bytes': new_state.stat().st_size, 'state_sha256': tr._sha256_file(new_state)})
    tmp = manifest_path.with_name(manifest_path.name + '.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, manifest_path)
    tr._append_event(run, {'event': 'bank_fingerprint_restamped', 'at_utc': datetime.now(timezone.utc).isoformat(),
                           'reason': args.reason, 'changes': changes, 'old_state_file': state_path.name,
                           'new_state_file': new_state.name})
    print(json.dumps({'run': run.name, 'changes': changes, 'new_state_file': new_state.name}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

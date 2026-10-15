"""別の機械（研究室）で保存した学習状態を、この機械で再開できるように再開の記録を付け直す。

    python rehost_resume.py <stage_worker のフォルダ> <task> <run_dir> --from-root /workspace/EVMALOCALUPDATE --reason "<根拠>" [--apply]

再開の照合（training.training_resume.load_training_resume）は、保存したときの context（入札データの場所とハッシュ、
Config・EnvConfig・observation_config の大文字の値、主なソースのハッシュ、観測の正規化）を、再開する機械で作り直した
context と1項目ずつ比べる。機械が変わると、場所（/workspace/... と C:\\...）と入札データのハッシュが必ず変わる。

この道具は、学習と同じ設定（stage_worker.configure(task)）で、この機械の context を学習と同じ式で作り直し、
保存した context の from-root で始まる文字列をこの機械の根に置き換えてから比べる。
  - 次の違いだけなら付け直してよい（--apply）。それ以外の違いが1つでもあれば、一覧を出して何もしない。
      入札データの sha256（この機械の .pkl がすべて学習を始めた時刻より前のものに限る）
      入札データのファイルの一覧で、保存した側にだけ manifest.windows_path.json がある（研究室へ写したときの目録の控え）
      ソースの sha256 で、--source-sha（保存した機械での sha256sum）とファイルごとにそのまま、または改行を LF にして一致し、
      その中身で計算し直すと保存した値に一致する（中身が同じで改行だけ違う）
  - Git Bash から呼ぶときは MSYS_NO_PATHCONV=1 を付ける（付けないと --from-root の /workspace/... が書き換えられる）。
  - 付け直すと、学習状態を別の名前で書き直し、元の学習状態と元の latest.json は残す（元に戻せる）。
    理由と違いは resume/events.jsonl に残す。
--apply を付けなければ、違いを出すだけで何も書かない。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def remap(value, src, dst):
    if isinstance(value, dict):
        return {k: remap(v, src, dst) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, src, dst) for v in value]
    if isinstance(value, str) and value.startswith(src):
        return str(Path(dst + value[len(src):].replace('/', os.sep)))
    return value


def source_sha_matching(root, files, other_sha):
    """保存した側のファイルごとの sha256（other_sha）に、この機械のファイルをそのまま、または改行を LF にして合わせ、
    合わせた中身で _fingerprint_files と同じ式のハッシュを出す。どちらでも合わないファイルがあれば None。"""
    import hashlib
    digest = hashlib.sha256()
    for rel in sorted(files):
        raw = (Path(root) / rel).read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        if hashlib.sha256(raw).hexdigest() == other_sha.get(rel):
            data = raw
        elif hashlib.sha256(lf).hexdigest() == other_sha.get(rel):
            data = lf
        else:
            print(f'  source {rel}: differs from the saved machine even after CRLF to LF')
            return None
        digest.update(rel.encode('utf-8'))
        digest.update(b'\0')
        digest.update(data)
        digest.update(b'\0')
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('sw_dir')
    ap.add_argument('task')
    ap.add_argument('run_dir')
    ap.add_argument('--from-root', required=True)
    ap.add_argument('--reason', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--source-sha', help='保存した機械での学習のソースのファイルごとの sha256（sha256sum の出力）')
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.sw_dir).resolve()))
    import stage_worker
    config = stage_worker.configure(args.task)
    train_dir, test_dir = stage_worker.banks(args.task)

    import torch
    from training import training_resume as tr
    from training.bid_bank import BidBank, derive_bid_bank_observation_normalization
    from training.lower_bid_training import build_fixed_upper_bid_training_episode

    run = Path(args.run_dir).resolve()
    resume = run / tr.RESUME_DIRNAME
    manifest_path = resume / tr.RESUME_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    saved = manifest['context']
    profile = derive_bid_bank_observation_normalization(BidBank(train_dir), build_episode=build_fixed_upper_bid_training_episode,
                                                        quantile=0.95)
    expected = tr.build_pretrain_resume_context(
        project_root=config.PROJECT_ROOT, model_name=saved['model_name'], forecast_seed=int(saved['forecast_seed']),
        train_split_count=int(saved['train_split_count']), bid_bank_dir=BidBank(train_dir).root,
        test_bid_bank_dir=BidBank(test_dir).root, observation_normalization_profile=profile)
    expected = json.loads(json.dumps(expected))
    mapped = remap(json.loads(json.dumps(saved)), args.from_root, str(Path(config.PROJECT_ROOT)))
    mismatches = tr._context_mismatches(mapped, expected)
    allowed = {'context.train_bid_bank.sha256', 'context.test_bid_bank.sha256'}
    for key in ('train_bid_bank', 'test_bid_bank'):
        extra = set(mapped[key]['files']) - set(expected[key]['files'])
        if not (set(expected[key]['files']) - set(mapped[key]['files'])) and all(Path(f).name == 'manifest.windows_path.json' for f in extra):
            allowed.add(f'context.{key}.files')
    if args.source_sha and expected['source']['files'] == mapped['source']['files']:
        other_sha = {}
        for line in Path(args.source_sha).read_text(encoding='utf-8').splitlines():
            if line.strip():
                h, rel = line.split(None, 1)
                other_sha[rel.lstrip('*').strip()] = h
        if source_sha_matching(config.PROJECT_ROOT, expected['source']['files'], other_sha) == mapped['source']['sha256']:
            allowed.add('context.source.sha256')
    other = [m for m in mismatches if m.split(':')[0] not in allowed]
    for m in mismatches:
        print(('  allowed ' if m.split(':')[0] in allowed else '  OTHER   ') + m[:300])
    if other:
        print(f'{len(other)} difference(s) other than the bid-bank hash; not rehosting')
        return 1
    started = datetime.strptime('_'.join(run.name.split('_')[-2:]), '%Y%m%d_%H%M%S').timestamp()
    for key, bank in (('train_bid_bank', train_dir), ('test_bid_bank', test_dir)):
        newer = [p for p in Path(bank).rglob('*.pkl') if p.stat().st_mtime > started]
        if newer:
            print(f'{key}: {len(newer)} bid pickle(s) here were written after the run started; not rehosting')
            return 1
    if not args.apply:
        print('dry run: only the bid-bank hash differs; rerun with --apply to rehost')
        return 0

    state_path = resume / manifest['state_file']
    if tr._sha256_file(state_path) != manifest['state_sha256']:
        raise SystemExit('the state file does not match its manifest; not rehosting')
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    if state.get('context') != saved:
        raise SystemExit('the state payload context does not match its manifest; not rehosting')
    state['context'] = expected
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_state = state_path.with_name(f'{state_path.stem}_rehost_{stamp}{state_path.suffix}')
    with open(new_state, 'wb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    shutil.copy2(manifest_path, manifest_path.with_name(f'{manifest_path.name}.before_rehost_{stamp}'))
    manifest.update({'context': expected, 'state_file': new_state.name,
                     'state_bytes': new_state.stat().st_size, 'state_sha256': tr._sha256_file(new_state)})
    tmp = manifest_path.with_name(manifest_path.name + '.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, manifest_path)
    tr._append_event(run, {'event': 'rehosted', 'at_utc': datetime.now(timezone.utc).isoformat(), 'reason': args.reason,
                           'from_root': args.from_root, 'to_root': str(Path(config.PROJECT_ROOT)),
                           'differences': mismatches, 'old_state_file': state_path.name, 'new_state_file': new_state.name})
    print(json.dumps({'run': run.name, 'new_state_file': new_state.name, 'differences': len(mismatches)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

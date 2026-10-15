"""Windows で作った入札の束を研究室で使えるように、manifest.json の指令データの場所とハッシュを研究室の値に付け直す。

    python rehome_bank_manifest.py <bank名> [...]        研究室のコンテナの中で動かす

指令データのハッシュ（market.activation_scenarios._activation_library_signature）は、CSV をパスの順に並べて
名前と中身をつなげたものの sha256 である。Windows のパスは大文字と小文字を区別せずに並ぶので、同じ CSV でも
Linux と順番が変わり、ハッシュが違う値になる（ERCOT の 46,994 本では 14,714 か所で順番が違った）。
保存場所の activation_source_dir も OS ごとに違う。このため、Windows で作った束は研究室の学習の照合で止まる。

付け直す前に、研究室の指令データを Windows の順に並べてハッシュを計算し、manifest の値と一致することを確かめる
（中身が同じことの確認）。ファイルの数も比べる。どちらかが合わなければ何もしないで止まる。
元の manifest.json は manifest.json.windows_original として残し、付け直した記録を manifest の rehomed に書く。
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
# このフォルダの queue.py が標準の queue を隠すので、このフォルダを探す場所から外す
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

from market.activation_scenarios import activation_library_signature  # noqa: E402

KEYS = ('activation_source_dir', 'activation_library_file_count', 'activation_library_sha256')


def windows_order_sha256(root: Path) -> tuple[int, str]:
    paths = sorted(root.glob('*.csv'), key=lambda p: PureWindowsPath(p.name))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode('utf-8'))
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
    return len(paths), digest.hexdigest()


def rehome(bank: Path) -> None:
    manifest_path = bank / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    settings = manifest['settings']
    source = str(settings['activation_source_dir'])
    if '\\' not in source:
        print('already a lab path, nothing to do:', bank.name, source, flush=True)
        return
    parts = PureWindowsPath(source).parts
    relative = Path(*parts[parts.index('data'):])
    root = (ROOT / relative).resolve()
    count, windows_sha = windows_order_sha256(root)
    if count != settings['activation_library_file_count'] or windows_sha != settings['activation_library_sha256']:
        raise SystemExit(f'{bank.name}: lab library differs (files {count} vs {settings["activation_library_file_count"]}, '
                         f'sha {windows_sha} vs {settings["activation_library_sha256"]})')
    lab = activation_library_signature(root)
    for key in ('activation_signal_mode', 'activation_signal_regime'):
        if lab[key] != settings[key]:
            raise SystemExit(f'{bank.name}: {key} differs ({lab[key]} vs {settings[key]})')
    backup = bank / 'manifest.json.windows_original'
    if not backup.exists():
        shutil.copy2(manifest_path, backup)
    before = {key: settings[key] for key in KEYS}
    settings.update({key: lab[key] for key in KEYS})
    manifest['rehomed'] = {
        'at': datetime.now().isoformat(timespec='seconds'),
        'reason': 'same CSV contents; Windows sorts paths case-insensitively, so the library hash differs by OS',
        'windows_order_sha256_on_lab': windows_sha,
        'before': before,
        'after': {key: settings[key] for key in KEYS},
    }
    tmp = manifest_path.with_name('manifest.json.rehome_tmp')
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(manifest_path)
    print('rehomed', bank.name, before['activation_library_sha256'][:12], '->', lab['activation_library_sha256'][:12],
          flush=True)


def main():
    for name in sys.argv[1:]:
        rehome(ROOT / 'execute_results' / 'bid_banks' / name)


if __name__ == '__main__':
    main()

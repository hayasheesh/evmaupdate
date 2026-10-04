"""指令ライブラリの中身を、OS によらない順（ファイル名の文字列順）で1つのハッシュにまとめる（読むだけ）。

usage: library_digest.py LIBRARY_DIR
activation_library_signature は Path の並べ方を使うので、Windows（大文字小文字を区別しない）と
Linux（区別する）で並びが変わる。ここでは名前の文字列で並べてから、名前と中身を順に入れる。
"""
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted((p for p in root.iterdir() if p.suffix == '.csv'), key=lambda p: p.name)
digest = hashlib.sha256()
for path in paths:
    digest.update(path.name.encode('utf-8'))
    digest.update(path.read_bytes())
meta = root / 'metadata.json'
print(len(paths), digest.hexdigest(), hashlib.sha256(meta.read_bytes()).hexdigest() if meta.exists() else '-')

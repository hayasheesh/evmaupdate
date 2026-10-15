"""Windows で作った入札の束を、研究室で読める形に書き直した写しを作る（Windows で動かす）。

    python make_portable_bank.py <bank名> [...]      execute_results/bid_banks_portable/<bank名> に書く

研究室で読めない点が2つある。
  - pickle：Windows の numpy 2 は配列を numpy._core から復元する形で書くので、研究室の numpy 1.26 では読めない。
    resbid_6station_20261012/portable_pickle.py と同じ方法で、両方の版にある名前で書き直す（読み直して元と一致することを確かめる）。
  - 目録（manifest*.json、days/*/entry.json）の bid_path・artifact_dir：Windows の区切り文字（\\）で書いた相対パスなので、
    研究室ではファイル名の一部として読まれる。/ に直す。
元の束は変えない。指令データの場所とハッシュは研究室で rehome_bank_manifest.py が付け直す。
"""
from __future__ import annotations

import io
import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# このフォルダの queue.py が標準の queue を隠すので、このフォルダを探す場所から外す
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'execute_results' / 'resbid_6station_20261012'))

from portable_pickle import PortablePickler, same  # noqa: E402

SRC = ROOT / 'execute_results' / 'bid_banks'
DST = ROOT / 'execute_results' / 'bid_banks_portable'
PATH_KEYS = ('bid_path', 'artifact_dir')
BACK = chr(92)


def fix_paths(value):
    if isinstance(value, dict):
        return {k: (v.replace(BACK, '/') if k in PATH_KEYS and isinstance(v, str) and ':' not in v else fix_paths(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [fix_paths(v) for v in value]
    return value


def convert(name: str) -> None:
    src, dst = SRC / name, DST / name
    if not (src / 'manifest.json').exists():
        raise SystemExit(f'作り終えていない: {name}')
    tmp = DST / f'{name}.making'
    if tmp.exists():
        shutil.rmtree(tmp)
    pickles = jsons = 0
    for p in src.rglob('*'):
        if p.is_dir():
            continue
        out = tmp / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == '.pkl':
            obj = pickle.loads(p.read_bytes())
            buf = io.BytesIO()
            PortablePickler(buf, protocol=4).dump(obj)
            data = buf.getvalue()
            assert b'numpy._core' not in data, p
            assert same(obj, pickle.loads(data)), p
            out.write_bytes(data)
            pickles += 1
        elif p.suffix == '.json' and (p.name.startswith('manifest') or p.name == 'entry.json'):
            payload = fix_paths(json.loads(p.read_text(encoding='utf-8')))
            text = json.dumps(payload, indent=2, ensure_ascii=False)
            out.write_text(text, encoding='utf-8')
            jsons += 1
        else:
            shutil.copy2(p, out)
    manifest = json.loads((tmp / 'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['entries']:
        assert BACK not in entry['bid_path'] and (tmp / entry['bid_path']).is_file(), entry['bid_path']
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    print('portable', name, 'pickles', pickles, 'json', jsons, flush=True)


def main():
    DST.mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        convert(name)


if __name__ == '__main__':
    main()

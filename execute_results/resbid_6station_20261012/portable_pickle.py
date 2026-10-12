"""Windows（numpy 2）で作った入札の pickle を、研究室の numpy でも読める形に書き直す。

    python portable_pickle.py <入札フォルダ> <出力フォルダ>

numpy 2 の pickle は配列と数値を numpy._core から復元するので、numpy 1 では読めない。
配列は numpy.array(リスト, dtype)、数値は numpy.<型>(値) として書き直す。どちらも両方の版にある名前で、
読み直した配列が元と dtype・形・値まで一致することを確かめてから書く。pkl 以外のファイルはそのまま写す。
"""
from __future__ import annotations

import io
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class PortablePickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, np.ndarray) and obj.dtype != object:
            return np.array, (obj.tolist(), obj.dtype.str)
        if isinstance(obj, np.ndarray):
            return np.array, (list(obj.ravel()), object), None, None, None
        if isinstance(obj, np.generic):
            return getattr(np, type(obj).__name__), (obj.item(),)
        return NotImplemented


def same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if hasattr(a, '__dict__') and not isinstance(a, type):
        return type(a) is type(b) and same(vars(a), vars(b))
    try:
        return bool(a == b) or (a != a and b != b)
    except Exception:
        return True


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    n = 0
    for p in src.rglob('*'):
        if p.is_dir():
            continue
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix != '.pkl':
            shutil.copy2(p, out)
            continue
        obj = pickle.loads(p.read_bytes())
        buf = io.BytesIO()
        PortablePickler(buf, protocol=4).dump(obj)
        data = buf.getvalue()
        assert b'numpy._core' not in data, p
        assert same(obj, pickle.loads(data)), p
        out.write_bytes(data)
        n += 1
    print('rewrote', n, 'pickles into', dst)


if __name__ == '__main__':
    main()

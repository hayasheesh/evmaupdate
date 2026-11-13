"""eval_real.py の結果をまとめる（本当の状況、指令20本）。追従率、SoC達成率、その和、コマの合格率を、指令の平均と標準誤差で出す。

    python summarize_real.py

同じ指令どうしの差を、本線の行動器（main_2000、同じ入札）と、同じ回の control を基準に出す。summary_real.json に書く。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEYS = ('step_pass_pct', 'soc_hit_pct', 'block_pass_pct')


def load(p):
    return {r['command']: r for r in map(json.loads, p.open(encoding='utf-8'))}


def stats(x):
    x = np.asarray(x, float)
    return {'mean': float(x.mean()), 'se': float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None, 'n': int(len(x))}


def diff(a, b):
    common = sorted(set(a) & set(b))
    out = {k: stats([a[c][k] - b[c][k] for c in common]) for k in KEYS}
    out['sum'] = stats([a[c]['step_pass_pct'] + a[c]['soc_hit_pct'] - b[c]['step_pass_pct'] - b[c]['soc_hit_pct'] for c in common])
    return out


def main():
    out = {}
    for d in sorted((HERE / 'eval').glob('*_bid_*')):
        res = {p.stem[len('result_'):]: load(p) for p in sorted(d.glob('result_*.jsonl'))}
        block = {}
        for name, rows in sorted(res.items(), key=lambda kv: (re.sub(r'_\d+_', '_', kv[0]), int(re.findall(r'_(\d+)_', kv[0])[0]))):
            mode = 'marl_force' if name.endswith('marl_force') else 'marl'
            ep = re.findall(r'_(\d+)_', name)[0]
            row = {k: stats([r[k] for r in rows.values()]) for k in KEYS}
            row['sum'] = stats([r['step_pass_pct'] + r['soc_hit_pct'] for r in rows.values()])
            row['capacity_kw_block'] = next(iter(rows.values()))['offered_capacity_kw_block']
            base = res.get(f'main_2000_{mode}')
            if base and not name.startswith('main_2000'):
                row['diff_vs_main'] = diff(rows, base)
            ctrl = res.get(f'control_{ep}_{mode}')
            if ctrl and not name.startswith('control') and not name.startswith('main'):
                row['diff_vs_control'] = diff(rows, ctrl)
            block[name] = row
            line = (f"{d.name:20s} {name:24s} n={row['sum']['n']:2d} 追従 {row['step_pass_pct']['mean']:6.2f} "
                    f"SoC {row['soc_hit_pct']['mean']:6.2f} 和 {row['sum']['mean']:6.2f} コマ {row['block_pass_pct']['mean']:6.2f}")
            if 'diff_vs_main' in row:
                line += f"  対本線 {row['diff_vs_main']['sum']['mean']:+.2f}±{row['diff_vs_main']['sum']['se'] or 0:.2f}"
            if 'diff_vs_control' in row:
                line += f"  対control {row['diff_vs_control']['sum']['mean']:+.2f}±{row['diff_vs_control']['sum']['se'] or 0:.2f}"
            print(line)
        out[d.name] = block
    (HERE / 'summary_real.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

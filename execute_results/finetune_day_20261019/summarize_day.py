"""eval_day.py の結果をまとめる。追従率（5分ごとに帯に入った割合）、SoC達成率、その和、コマの合格率を、組の平均と標準誤差で出す。

    python summarize_day.py

組（指令 × EVの乱数）が同じものどうしの差も、本線の行動器（main_2000）を基準に出す。summary_day.json に書く。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(p):
    return {(r['command'], r['ev_seed']): r for r in map(json.loads, p.open(encoding='utf-8'))}


def stats(x):
    x = np.asarray(x, float)
    return {'mean': float(x.mean()), 'se': float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None, 'n': int(len(x))}


def main():
    out = {}
    for d in sorted((HERE / 'eval').glob('world_*')):
        res = {p.stem[len('result_'):]: load(p) for p in sorted(d.glob('result_*.jsonl'))}
        block = {}
        for name, rows in res.items():
            mode = 'marl_force' if name.endswith('marl_force') else 'marl'
            base = res.get(f'main_2000_{mode}', {})
            keys = sorted(rows)
            row = {k: stats([rows[s][k] for s in keys]) for k in ('step_pass_pct', 'soc_hit_pct', 'block_pass_pct')}
            row['sum'] = stats([rows[s]['step_pass_pct'] + rows[s]['soc_hit_pct'] for s in keys])
            common = [s for s in keys if s in base]
            if common and not name.startswith('main_2000'):
                row['diff_vs_main'] = {k: stats([rows[s][k] - base[s][k] for s in common])
                                       for k in ('step_pass_pct', 'soc_hit_pct', 'block_pass_pct')}
                row['diff_vs_main']['sum'] = stats([rows[s]['step_pass_pct'] + rows[s]['soc_hit_pct']
                                                    - base[s]['step_pass_pct'] - base[s]['soc_hit_pct'] for s in common])
            row['capacity_kw_block'] = rows[keys[0]]['offered_capacity_kw_block']
            block[name] = row
            print(f"{d.name:24s} {name:24s} n={len(keys):2d} 追従 {row['step_pass_pct']['mean']:6.2f} SoC {row['soc_hit_pct']['mean']:6.2f} "
                  f"和 {row['sum']['mean']:6.2f}±{row['sum']['se'] or 0:.2f} コマ {row['block_pass_pct']['mean']:6.2f}"
                  + (f"  差(和) {row['diff_vs_main']['sum']['mean']:+.2f}±{row['diff_vs_main']['sum']['se'] or 0:.2f}" if 'diff_vs_main' in row else ''))
        out[d.name] = block
    (HERE / 'summary_day.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

"""eval_seconds.py の結果（results/*.jsonl）をまとめる（2026-10-17）。

    python summarize.py [results/<名前>.jsonl ...]      引数がなければ results/ のすべて

1行が1つの設定（入札・制御・秒単位の層の有無）で、40組の平均を出す。
  - 追従率（5分ごとの目標に対して帯に入った5分の割合、GLOBAL）、SoC達成率（LOCAL）、その和
  - PJM の点数の平均：10秒ごとの採点（遅れの許容あり・なし）と、5分に1点の採点
  - 10秒の採点で25%未満（支払い0）の5分の割合、点数と AReg で重みを付けた支払いの割合
  - 同じ指令の番号と EV の乱数の5日を日付の順につなぎ（8並び）、100時間の移動平均の最小と、40%を保った並びの数
ファイル名の末尾は層の有無と余力の取り方（off：層なし、plain：放電の下限が SoC 0%、reach：出発までに目標へ届く分を残す）。
同じ入札・制御で層あり・なしがそろえば、40組の対の差（あり − なし）と95%区間（組を4000回復元抽出、乱数0）を出す。
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from market.pjm_performance import qualification_summary, score_summary  # noqa: E402


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return sorted(rows, key=lambda r: (r['day'], r['command'], r['ev_seed']))


def summarize(rows: list[dict]) -> dict:
    out = {'sets': len(rows),
           'global_pct': float(np.mean([r['step_pass_pct'] for r in rows])),
           'local_pct': float(np.mean([r['soc_hit_pct'] for r in rows]))}
    out['sum'] = out['global_pct'] + out['local_pct']
    for key in ('score_10s', 'score_10s_nolag', 'score_5min'):
        scores = np.concatenate([np.asarray(r[key], float) for r in rows])
        areg = np.concatenate([np.asarray(r['areg_kw'], float) for r in rows])
        s = score_summary(scores, areg)
        out[key] = s['mean_score']
        if key == 'score_10s':
            out['intervals'] = s['intervals']
            out['share_below_25pct'] = s['share_below_payment_floor']
            out['credit_ratio'] = s['credit_ratio']
            out['min_score'] = s['min_score']
    chains = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r['command'], r['ev_seed'], r['day'])):
        chains[(r['command'], r['ev_seed'])].append(np.asarray(r['score_10s'], float))
    q = [qualification_summary(np.concatenate(parts)) for parts in chains.values()]
    full = [x for x in q if x['window_full']]
    out['chains'] = len(q)
    out['chains_full_window'] = len(full)
    out['chains_qualified'] = int(sum(x['qualified'] is True for x in full))
    out['min_historic_score'] = float(min(x['min_historic_score'] for x in full)) if full else float('nan')
    out['fast_saturated_share'] = float(np.mean([r['fast_saturated_share'] for r in rows]))
    out['fast_abs_energy_kwh'] = float(np.mean([r['fast_abs_energy_kwh'] for r in rows]))
    return out


def paired(on: list[dict], off: list[dict]) -> dict:
    key = lambda r: (r['day'], r['command'], r['ev_seed'])
    a = {key(r): r for r in on}
    b = {key(r): r for r in off}
    common = sorted(set(a) & set(b))
    rng = np.random.default_rng(0)
    res = {'pairs': len(common)}
    for name, f in (('global', lambda r: r['step_pass_pct']), ('local', lambda r: r['soc_hit_pct']),
                    ('sum', lambda r: r['step_pass_pct'] + r['soc_hit_pct'])):
        d = np.array([f(a[k]) - f(b[k]) for k in common])
        boot = [rng.choice(d, d.size).mean() for _ in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res[name] = (float(d.mean()), float(lo), float(hi))
    return res


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted((HERE / 'results').glob('*.jsonl'))
    table = {}
    for p in paths:
        rows = load(p)
        if rows:
            table[p.stem] = (rows, summarize(rows))
    head = (f"{'設定':<24}{'組':>3}{'追従率':>7}{'SoC':>6}{'和':>7}{'10秒':>7}{'許容なし':>7}{'5分1点':>7}"
            f"{'<25%':>7}{'支払い':>7}{'100h最小':>8}{'40%保つ':>7}{'層の飽和':>7}")
    print(head)
    for name, (_rows, s) in table.items():
        print(f"{name:<24}{s['sets']:>3}{s['global_pct']:>7.1f}{s['local_pct']:>6.1f}{s['sum']:>7.1f}"
              f"{s['score_10s']:>7.3f}{s['score_10s_nolag']:>7.3f}{s['score_5min']:>7.3f}"
              f"{100 * s['share_below_25pct']:>6.1f}%{s['credit_ratio']:>7.3f}{s['min_historic_score']:>8.3f}"
              f"{s['chains_qualified']:>4}/{s['chains_full_window']}{s['fast_saturated_share']:>7.2f}")
    print()
    for name in table:
        for layer in ('_reach', '_plain'):
            base = name[:-len(layer)] + '_off'
            if name.endswith(layer) and base in table:
                d = paired(table[name][0], table[base][0])
                text = '  '.join(f"{k} {v[0]:+.1f}[{v[1]:+.1f}, {v[2]:+.1f}]" for k, v in d.items() if k != 'pairs')
                print(f"{name:<24} − off（{d['pairs']}組）  {text}")
    (HERE / 'summary.json').write_text(json.dumps({k: v[1] for k, v in table.items()}, ensure_ascii=False, indent=1),
                                       encoding='utf-8')


if __name__ == '__main__':
    main()

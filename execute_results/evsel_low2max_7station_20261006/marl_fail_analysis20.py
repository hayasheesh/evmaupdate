"""marl_fail_probe の記録から、帯を外した step を原因で分ける。

    python marl_fail_analysis.py <aemo|ercot> <行動器の回>

外した step（|指令 - 出力| > 帯、評価対象の step のみ）を次の2つに分ける。
- 範囲外: その step の初めの EV で出せる電力の範囲（step_dispatch_envelope）に、帯がかからない。
  その時点の状態からは、どう配分しても帯に入らない
- 範囲内: 帯の一部が範囲に入っているのに外した
範囲内の外しは、出力が指令より上か下か、範囲の端からどれだけ離れていたかでも数える。
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
task, episode = sys.argv[1], int(sys.argv[2])
rows = []
for f in sorted((HERE / 'marl_fail_probe20').glob(f'{task}_day*_ep{episode}.json')):
    rows += json.loads(f.read_text(encoding='utf-8'))

tot = dict(steps=0, hit=0, fail=0, out=0, inn=0, inn_above=0, inn_below=0, inn_edge=0)
err_in, err_out, err_hit, signed_in = [], [], [], []
for r in rows:
    s = r['steps']
    disp, resp, tol = (np.asarray(s[k], float) for k in ('disp', 'resp', 'tol'))
    lo, hi = np.asarray(s['env_lo'], float), np.asarray(s['env_hi'], float)
    m = np.asarray(s['tracking'], bool)
    e = resp - disp
    fail = m & (np.abs(e) > tol + 1e-6)
    out = fail & ((disp - tol > hi + 1e-6) | (disp + tol < lo - 1e-6))
    inn = fail & ~out
    # 範囲内の外しで、帯と範囲の重なりが帯の幅の半分未満（範囲の端ぎりぎり）
    overlap = np.minimum(disp + tol, hi) - np.maximum(disp - tol, lo)
    edge = inn & (overlap < tol)
    tot['steps'] += int(m.sum()); tot['hit'] += int((m & ~fail).sum()); tot['fail'] += int(fail.sum())
    tot['out'] += int(out.sum()); tot['inn'] += int(inn.sum()); tot['inn_edge'] += int(edge.sum())
    tot['inn_above'] += int((inn & (e > 0)).sum()); tot['inn_below'] += int((inn & (e < 0)).sum())
    err_in += list(np.abs(e[inn]) - tol[inn]); err_out += list(np.abs(e[out]) - tol[out])
    err_hit += list(np.abs(e[m & ~fail])); signed_in += list(e[inn])
    r['_tr'] = 100 * (m & ~fail).sum() / max(m.sum(), 1)

n = tot['steps']
print(f'{task} ep{episode}: エピソード {len(rows)}, 評価 step {n}')
print(f"  帯の中 {tot['hit']} ({100*tot['hit']/n:.1f}%)  外した {tot['fail']} ({100*tot['fail']/n:.1f}%)")
f = max(tot['fail'], 1)
print(f"  外したうち 範囲外 {tot['out']} ({100*tot['out']/f:.1f}%)  範囲内 {tot['inn']} ({100*tot['inn']/f:.1f}%)")
print(f"  範囲内のうち 出力が指令より上 {tot['inn_above']}  下 {tot['inn_below']}  範囲の端ぎりぎり {tot['inn_edge']}")
for name, v in (('範囲内の外し 帯からのはみ出し', err_in), ('範囲外の外し 帯からのはみ出し', err_out)):
    if v:
        print(f'  {name} kW: 中央値 {np.median(v):.1f}  平均 {np.mean(v):.1f}  90% {np.percentile(v, 90):.1f}')
if signed_in:
    print(f'  範囲内の外し 誤差（出力-指令）の平均 {np.mean(signed_in):+.1f} kW')
tr = [r['_tr'] for r in rows]
print(f"  追従率（エピソードごと） 平均 {np.mean(tr):.1f}  最小 {np.min(tr):.1f}  最大 {np.max(tr):.1f}")
print(f"  SoC達成率 平均 {np.mean([r['soc_hit_pct'] for r in rows]):.1f}  帯の平均 {np.mean([r['band_mean_kw'] for r in rows]):.1f} kW")

# 帯の幅ごと、出せる範囲の中の指令の位置ごとの外した割合
T, F, Z, P = [], [], [], []
for r in rows:
    s = r['steps']
    disp, resp, tol = (np.asarray(s[k], float) for k in ('disp', 'resp', 'tol'))
    lo, hi = np.asarray(s['env_lo'], float), np.asarray(s['env_hi'], float)
    m = np.asarray(s['tracking'], bool)
    T += list(tol[m]); F += list((np.abs(resp - disp) > tol + 1e-6)[m]); Z += list(np.abs(resp - disp)[m])
    P += list(((disp - lo) / np.maximum(hi - lo, 1e-6))[m])
T, F, Z, P = map(np.asarray, (T, F, Z, P))
edges = np.percentile(T, [0, 20, 40, 60, 80, 100])
print('  帯の幅 kW（step数で5等分） / step数 / 外した割合 / 外した step 全体に占める割合 / 誤差の中央値 kW')
for a, b in zip(edges[:-1], edges[1:]):
    k = (T >= a) & (T <= b if b == edges[-1] else T < b)
    print(f'    {a:6.0f}〜{b:<6.0f} {k.sum():5d} {100*F[k].mean():5.1f}% {100*F[k].sum()/max(F.sum(),1):5.1f}% {np.median(Z[k]):6.1f}')
print('  指令の位置（出せる範囲の下端0〜上端1） / step数 / 外した割合')
for a, b in [(-9, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 9)]:
    k = (P >= a) & (P < b)
    if k.sum():
        print(f'    {a:>4}〜{b:<4} {k.sum():5d} {100*F[k].mean():5.1f}%')

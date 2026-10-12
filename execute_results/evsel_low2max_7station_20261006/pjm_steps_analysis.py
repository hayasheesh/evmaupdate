"""pjm_band_clip_probe.py（normal）の5分ごとの記録から、誤差と局ごとの EV 台数・出力の関係を出す。

    python pjm_steps_analysis.py

局の並び（7局）: 0 WALNUT, 1 SPRUCE, 2 PEARL（商業）, 3 JPL, 4 Caltech（職場）, 5 OSL_S, 6 TRO_R（住宅）。
誤差は |指令 − 行動器の出力の合計|、判定の対象の step だけで数える。
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GROUPS = {'商業': [0, 1, 2], '職場': [3, 4], '住宅': [5, 6]}
NAMES = ['WALNUT', 'SPRUCE', 'PEARL', 'JPL', 'Caltech', 'OSL_S', 'TRO_R']

err, over, evs, spow, disp, hour, width = [], [], [], [], [], [], []
for f in sorted(glob.glob(str(HERE / 'pjm_band_clip_probe' / 'day*_ep440_normal.json'))):
    for row in json.load(open(f, encoding='utf-8')):
        s = row['steps']
        m = np.asarray(s['tracking'], bool)
        d, r, t = (np.asarray(s[k], float) for k in ('disp', 'resp', 'tol'))
        e = np.abs(d - r)
        err.append(e[m]); over.append((e - t)[m]); disp.append(d[m])
        evs.append(np.asarray(s['evs'], float)[m]); spow.append(np.asarray(s['station_power'], float)[m])
        hour.append((np.arange(len(d)) // 12)[m])
        width.append((np.asarray(s['env_hi'], float) - np.asarray(s['env_lo'], float))[m])
err, over, disp, hour, width = (np.concatenate(x) for x in (err, over, disp, hour, width))
evs, spow = np.concatenate(evs), np.concatenate(spow)
inside = over <= 1e-6
print(f'step数 {len(err)}  平均誤差 {err.mean():.1f} kW  帯の内側 {100 * inside.mean():.1f}%')

print('\n時 | 平均誤差 帯の内側 | EV台数 商業/職場/住宅 | 1stepで出せる幅')
for h in range(24):
    k = hour == h
    if not k.any():
        continue
    g = [evs[k][:, idx].sum(axis=1).mean() for idx in GROUPS.values()]
    print(f'{h:2d} | {err[k].mean():6.1f} {100 * inside[k].mean():5.1f}% | {g[0]:4.1f} / {g[1]:4.1f} / {g[2]:4.1f} | {width[k].mean():6.0f} kW')

print('\n商業3局のEV台数ごと')
com = evs[:, GROUPS['商業']].sum(axis=1)
for lo, hi, label in ((0, 0, '0台'), (1, 2, '1-2台'), (3, 5, '3-5台'), (6, 99, '6台以上')):
    k = (com >= lo) & (com <= hi)
    if k.any():
        print(f'  {label:6s} step {k.sum():5d} 平均誤差 {err[k].mean():6.1f} 帯の内側 {100 * inside[k].mean():5.1f}% 全体のEV {evs[k].sum(axis=1).mean():5.1f}台')

print('\n全体のEV台数ごと')
tot = evs.sum(axis=1)
for lo, hi in ((0, 15), (16, 25), (26, 35), (36, 70)):
    k = (tot >= lo) & (tot <= hi)
    if k.any():
        print(f'  {lo:2d}-{hi:2d}台 step {k.sum():5d} 平均誤差 {err[k].mean():6.1f} 帯の内側 {100 * inside[k].mean():5.1f}%')

print('\n局ごと: 平均EV台数 | 出力と指令の相関 | 出力の1stepの変化の平均')
for i, n in enumerate(NAMES):
    p = spow[:, i]
    c = np.corrcoef(p, disp)[0, 1] if p.std() > 0 else float('nan')
    print(f'  {n:8s} {evs[:, i].mean():4.1f}台 | {c:5.2f} | {np.abs(np.diff(p)).mean():6.1f} kW')

big = over > 20
print(f'\n帯を20 kW以上外れたstep {big.sum()} の平均EV台数 商業/職場/住宅:',
      [round(float(evs[big][:, idx].sum(axis=1).mean()), 1) for idx in GROUPS.values()],
      ' それ以外:', [round(float(evs[~big][:, idx].sum(axis=1).mean()), 1) for idx in GROUPS.values()])

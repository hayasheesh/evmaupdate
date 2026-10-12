"""古い20 station（10月3日、再現した評価40本）と今の20 station（calendar20 の120組、1380回目）を同じ物差しで比べる。

    python compare_old_new.py

どちらも評価用の日・指令・EVで、学習で見ていない指令を使う。日は同じではない。
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/workspace/EVMALOCALUPDATE')
from EnvConfig import POWER_TO_ENERGY as P2E  # noqa: E402

OLD = sorted(glob.glob('/workspace/EVMALOCALUPDATE/execute_results/repro_old20_20261008/trace_ep1100/trace/episode_*.npz'))
NEW = sorted(glob.glob('/workspace/EVMALOCALUPDATE/execute_results/watch_codex_20261008/calendar20_aemo_20261008/best_*.npz'))


def unify(path, kind):
    z = np.load(path)
    s, e = z['step'], z['ev']
    if kind == 'old':
        step = dict(step=s[:, 0], on=s[:, 1] > 0, req=s[:, 2], band=s[:, 3], resid=s[:, 4])
        ev = dict(step=e[:, 0], st=e[:, 1], soc=e[:, 3], target=e[:, 4], cap=e[:, 5], rated=e[:, 6],
                  remain=e[:, 7], lo=e[:, 8], hi=e[:, 9], floor=e[:, 10], action=e[:, 11], ex=e[:, 12])
    else:
        step = dict(step=s[:, 0], on=s[:, 1] > 0, req=s[:, 2], band=s[:, 3], resid=s[:, 4])
        ev = dict(step=e[:, 0], st=e[:, 1], soc=e[:, 5], target=e[:, 7], cap=e[:, 8], rated=e[:, 9],
                  remain=e[:, 4], lo=e[:, 13], hi=e[:, 14], floor=e[:, 15], action=e[:, 10], ex=e[:, 12])
    return step, ev


def summarize(paths, kind):
    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(float(v))

    for p in paths:
        s, e = unify(p, kind)
        steps = s['step'].astype(int)
        idx = {t: i for i, t in enumerate(steps)}
        n = len(steps)
        es = e['step'].astype(int)
        # Population
        add('EVs per step', len(es) / n)
        occ = np.zeros((n, 20))
        np.add.at(occ, (np.array([idx[t] for t in es]), e['st'].astype(int)), 1)
        add('station-steps full (10 EVs) %', 100 * (occ >= 10).mean())
        add('rated kW (EV-step mean)', e['rated'].mean())
        add('capacity kWh (EV-step mean)', e['cap'].mean())
        add('remaining hours (EV-step mean)', e['remain'].mean() / 12)
        need = np.maximum(e['target'] - e['soc'], 0) * e['cap'] / 100
        reach = np.maximum(e['remain'], 1) * e['rated'] * P2E
        urg = need / reach
        add('energy still needed per EV-step kWh', need.mean())
        add('urgency need/(remaining*rated) mean', urg.mean())
        add('EV-steps with urgency > 0.8 %', 100 * (urg > 0.8).mean())
        add('EV-steps with deadline floor > 0 %', 100 * (e['floor'] > 1e-6).mean())
        fmin = np.zeros(n); fmax = np.zeros(n); fl = np.zeros(n)
        np.add.at(fmin, [idx[t] for t in es], e['lo'])
        np.add.at(fmax, [idx[t] for t in es], e['hi'])
        np.add.at(fl, [idx[t] for t in es], np.maximum(e['floor'], e['lo']))
        on = s['on']
        add('fleet charge range kW (assessed mean)', fmax[on].mean())
        add('fleet discharge range kW (assessed mean)', -fmin[on].mean())
        # Command difficulty
        req, band, resid = s['req'], s['band'], s['resid']
        span = np.maximum(fmax - fmin, 1e-6)
        pos = (req - fmin) / span
        add('request position in fleet range (0=min,1=max)', pos[on].mean())
        add('assessed steps with request in outer 10% of range %', 100 * ((pos < 0.1) | (pos > 0.9))[on].mean())
        add('assessed steps physically infeasible now %', 100 * ((req - band > fmax) | (req + band < fmin))[on].mean())
        add('assessed steps request above deadline-safe minimum? (req+band < floor sum) %',
            100 * (req + band < fl)[on].mean())
        add('band kW (assessed mean)', band[on].mean())
        add('band / fleet range %', 100 * (2 * band / span)[on].mean())
        dreq = np.abs(np.diff(req))
        both = on[1:] & on[:-1]
        add('|request change| per step / band (assessed)', (dreq / np.maximum(band[1:], 1e-6))[both].mean())
        add('request |kW| (assessed mean)', np.abs(req[on]).mean())
        # Outcome
        ok = np.abs(resid) <= band
        add('tracking %', 100 * ok[on].mean())
        add('MAE kW', np.abs(resid[on]).mean())
        add('MAE / band', (np.abs(resid) / np.maximum(band, 1e-6))[on].mean())
        bad = on & ~ok
        feas = ~((req - band > fmax) | (req + band < fmin))
        add('missed steps with instant headroom %', 100 * (bad & feas).sum() / max(bad.sum(), 1))
        nominal = np.clip(e['action'], -1, 1) * e['rated']
        clip = np.zeros(n)
        np.add.at(clip, [idx[t] for t in es], nominal - e['ex'])
        add('SoC-clip shift |nominal-executed| per assessed step kW', np.abs(clip[on]).mean())
        add('assessed steps where clip shift > band %', 100 * (np.abs(clip) > band)[on].mean())
        ch = np.zeros(n); dis = np.zeros(n)
        np.add.at(ch, [idx[t] for t in es], np.maximum(e['ex'], 0))
        np.add.at(dis, [idx[t] for t in es], np.maximum(-e['ex'], 0))
        add('simultaneous charge&discharge kW (assessed mean)', np.minimum(ch, dis)[on].mean())
        # Departures
        dep = e['remain'] == 1
        after = np.clip(e['soc'] + e['ex'] * P2E * 100 / e['cap'], 0, 100)
        add('departures per episode', dep.sum())
        add('SoC hit at departure %', 100 * (after[dep] >= e['target'][dep] - 1e-6).mean() if dep.any() else np.nan)
        add('departures at 24:00 %', 100 * (dep & (e['step'] == 288)).sum() / max(dep.sum(), 1))
    return {k: (float(np.nanmean(v)), len(v)) for k, v in acc.items()}


def main():
    old = summarize(OLD, 'old')
    new = summarize(NEW, 'new')
    print(f"{'metric':78s} {'old 20st (10/3)':>16s} {'new 20st (allin)':>16s}")
    for k in old:
        print(f"{k:78s} {old[k][0]:16.3f} {new.get(k, (float('nan'),))[0]:16.3f}")
    print('episodes', len(OLD), len(NEW))
    out = Path('/workspace/EVMALOCALUPDATE/execute_results/repro_old20_20261008/compare_old_new.json')
    out.write_text(json.dumps({'old': old, 'new': new, 'old_episodes': len(OLD), 'new_cases': len(NEW)}, indent=1))


if __name__ == '__main__':
    main()

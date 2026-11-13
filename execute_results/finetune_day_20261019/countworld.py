"""その日の到着台数を中心にした世界（finetune の試し、Windows、2026-10-19）。

前日に分かるのは、その日の時刻ごとの到着の量までとする。来たEVの中身（滞在時間、電力量、到着時のSoC、電池の容量）と
1時間の中の来る時刻、0時にいるEVは、本線と同じく毎回引き直す。

その日の到着台数 n_D(局, 時)：dayworld.center の実現 D（AEMO の検証用の2日目、2024-02-01）の、局ごと・1時間ごとの到着台数。
with count_world(n_D, k): の中で作る EVEnv の1日は、次のように作る。
  1. 本線と同じ作り方で1日を引く（環境の乱数 random を本線と同じ順で使う）。その局・その1時間に来た台数を n、
     本線の平均の台数（到着の確率の和 × 充電口の数）を μ とする。
  2. 目標の台数を n_D + k(n − μ) とし、確率的に丸める（0未満は0）。k = 1 なら、D からのずれは本線の1日の平均からのずれと同じ大きさ。
  3. 台数が多すぎればその1時間に来たEVから無作為に外し、少なければ空いている（時刻・口）に、本線と同じ作り方
     （EVEnv._session_attributes）の中身のEVを足す。充電口が埋まっているかは環境がそのまま判定する。
足し引きの乱数は、その実現の到着の並びから決める（world.shrink_events と同じ）ので、環境の乱数の流れは本線と変わらない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dayworld as dw  # noqa: E402

world = dw.world
CALIB = HERE / 'calib_count.json'


def day_counts(D):
    """D の局ごと・1時間ごとの到着台数（局 × 24）。"""
    events = D[0]
    steps, n_st = len(events), len(events[0])
    per_hour = steps // 24
    return [[sum(1 for t in range(h * per_hour, (h + 1) * per_hour) for ev in events[t][st] if ev is not None)
             for h in range(24)] for st in range(n_st)]


def recount(env, events, n_day, k: float):
    import random as _r
    import zlib
    steps, n_st = len(events), len(events[0])
    max_evs = int(env.max_ev_per_station)
    per_hour = steps // 24
    prob = getattr(env, '_arrival_probability_override', None)
    if prob is None:
        prob = env.arrival_profiles_by_station
    key = ';'.join(f'{t},{st},{j},{ev.get("dwell_steps")}' for t in range(steps) for st in range(n_st)
                   for j, ev in enumerate(events[t][st]) if ev is not None)
    rng = _r.Random(zlib.crc32(key.encode()))
    attr_seed = rng.randrange(2 ** 31)
    for st in range(n_st):
        prof = prob[st]
        for h in range(24):
            ts = range(h * per_hour, (h + 1) * per_hour)
            filled = [(t, j) for t in ts for j in range(max_evs) if events[t][st][j] is not None]
            empty = [(t, j) for t in ts for j in range(max_evs) if events[t][st][j] is None]
            mu = sum(float(prof[min(t, len(prof) - 1)]) for t in ts) * max_evs
            target = max(0.0, n_day[st][h] + k * (len(filled) - mu))
            n_new = int(target) + (1 if rng.random() < target - int(target) else 0)
            if n_new < len(filled):
                for t, j in rng.sample(filled, len(filled) - n_new):
                    events[t][st][j] = None
            elif n_new > len(filled):
                for t, j in rng.sample(empty, min(n_new - len(filled), len(empty))):
                    events[t][st][j] = env._session_attributes(st, env.day_class, t,
                                                               _r.Random(attr_seed + (t * n_st + st) * max_evs + j))
    return events


class count_world:
    """with count_world(n_day, k): の中で作る EVEnv の到着を、その日の到着台数を中心にしたものにする。"""

    def __init__(self, n_day, k: float = 1.0):
        self.n_day, self.k = n_day, float(k)

    def __enter__(self):
        from environment.EVEnv import EVEnv
        self.original = EVEnv._pregenerate_arrival_events
        original, n_day, k = self.original, self.n_day, self.k

        def patched(env):
            return recount(env, original(env), n_day, k)

        EVEnv._pregenerate_arrival_events = patched
        return self

    def __exit__(self, *exc):
        from environment.EVEnv import EVEnv
        EVEnv._pregenerate_arrival_events = self.original
        return False


def candidates(base, n_day, k, n=128):
    """入札の候補：本線の候補と同じ種で、この世界から n 通り。"""
    fs, probs, date = int(base['forecast_seed']), base['arrival_probabilities_by_station'], base.get('service_date')
    with count_world(n_day, k):
        return [world.capture_seeded(fs + 10007 * i, probs, date) for i in range(n)]


def actuals(base, n_day, k, js=tuple(range(1, 21))):
    """当日：同じ世界から、候補と別の種で（dayworld.actuals と同じ種）。"""
    fs, probs, date = int(base['forecast_seed']), base['arrival_probabilities_by_station'], base.get('service_date')
    with count_world(n_day, k):
        return [world.capture_seeded(fs + 10_000_000 + 10007 * j, probs, date) for j in js]


def _measure(args):
    """k ごとの接続台数（5分ごと）とEVの数。k が None なら本線の世界（その日の中心なし）。
    子から親へは数の配列だけを返す（EVSpec のまま返すと、親での読み込みで止まった）。"""
    k, part, chunk = args
    world.configure(dw.TASK)
    base = dw.target_base()
    n_day = day_counts(dw.center(base))
    fs, probs, date = int(base['forecast_seed']), base['arrival_probabilities_by_station'], base.get('service_date')
    if part == 'cand':
        seeds = [fs + 10007 * i for i in range(8 * chunk, 8 * chunk + 8)]
    else:
        seeds = [fs + 10_000_000 + 10007 * j for j in range(1 + 10 * chunk, 11 + 10 * chunk)]
    if k is None:
        caps = [world.capture_seeded(s, probs, date) for s in seeds]
    else:
        with count_world(n_day, k):
            caps = [world.capture_seeded(s, probs, date) for s in seeds]
    return [(world.connected(c).tolist(), len(c)) for c in caps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', default='1,0.75,0.5,0')
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    import multiprocessing as mp
    import numpy as np
    ks = [float(x) for x in args.k.split(',')]
    mp.set_start_method('spawn', force=True)
    # 候補128通りは8通りずつ、当日20通りは10通りずつに分けて並べる。本線の世界は候補だけ（接続量の平均を比べるため）
    jobs = [(None, 'cand', c) for c in range(16)]
    jobs += [(k, 'act', c) for k in ks for c in range(2)] + [(k, 'cand', c) for k in ks for c in range(16)]
    with mp.Pool(args.workers, maxtasksperchild=1) as pool:
        res = pool.map(_measure, jobs, chunksize=1)
    got = {}
    for (k, part, _c), r in zip(jobs, res):
        got.setdefault((k, part), []).extend(r)
    world.configure(dw.TASK)
    base = dw.target_base()
    n_day = day_counts(dw.center(base))
    main_vol = np.array([sum(c) for c, _ in got[(None, 'cand')]])
    rows = []
    for k in ks:
        cands = np.array([c for c, _ in got[(k, 'cand')]])
        acts = np.array([c for c, _ in got[(k, 'act')]])
        mean = cands.mean(axis=0)
        vol = cands.sum(axis=1)
        lo, hi = np.percentile(vol, [5, 95])
        rows.append({'k': k, 'wape': [float(100 * np.abs(mean - a).sum() / max(a.sum(), 1e-9)) for a in acts],
                     'p5_pct': float(100 * (lo / vol.mean() - 1)), 'p95_pct': float(100 * (hi / vol.mean() - 1)),
                     'mean_volume_vs_main_pct': float(100 * (vol.mean() / main_vol.mean() - 1)),
                     'evs_mean': float(np.mean([n for _, n in got[(k, 'cand')]]))})
    for r in rows:
        r['wape_mean'] = float(np.mean(r['wape']))
        print(f"k {r['k']}: WAPE {r['wape_mean']:.2f}%  接続量の5%点 {r['p5_pct']:+.1f}%  95%点 {r['p95_pct']:+.1f}%  "
              f"接続量の平均は本線の {r['mean_volume_vs_main_pct']:+.1f}%  EVの数の平均 {r['evs_mean']:.1f}", flush=True)
    main_lo, main_hi = np.percentile(main_vol, [5, 95])
    out = {'n_day_total': int(sum(map(sum, n_day))), 'n_day': n_day,
           'main_candidates': {'evs_mean': float(np.mean([n for _, n in got[(None, 'cand')]])),
                               'p5_pct': float(100 * (main_lo / main_vol.mean() - 1)),
                               'p95_pct': float(100 * (main_hi / main_vol.mean() - 1))},
           'measurements': rows}
    print('本線の候補', out['main_candidates'], flush=True)
    CALIB.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

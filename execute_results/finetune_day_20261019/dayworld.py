"""その日を中心にした世界（finetune の試し、Windows、2026-10-19）。

本線の世界は、平日ならどの日も同じ到着の確率から毎回くじ引きで1日を作り、日ごとのばらつきは誤差率9.47%
（5分ごとの接続台数について、128通りの平均と当日との差の合計を当日の合計で割った値）である。
ここでは、対象の1日（AEMO の検証用の2日目、2024-02-01）について、本線の分布から1つの実現 D（その日のEVの状況）を決め、
D を中心に、本線と同じ大きさ（誤差率9.47%）のばらつきを足した世界を作る。

ばらつきの足し方（with day_world(D, kappa, tau):）：
  - 局ごとに、来る台数を D の台数 n の前後でばらつかせる（n' = n + kappa × (ポアソン(n) − n) を丸める）。
  - 来る車は、D のその局の到着から、同じ車を何度でも選べる形で n' 台を引き直す（来る時刻と中身は D のもの）。
    tau > 0 なら、来る時刻を標準偏差 tau ステップでずらす。
  - 0時にいるEVは、局ごとに D の台数のまま、D の0時のEVから同じ形で引き直す。
  - 充電口が埋まっているかは環境がそのまま判定する。
乱数は環境の乱数（random）から1つ取って決めるので、学習ではエピソードごとに違う1日になり、入札の候補は種で決まる。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FCW = ROOT / 'execute_results' / 'fcworld_6station_20261018'
sys.path.insert(0, str(FCW))
import world  # noqa: E402

TASK = 'aemo'
DAY_INDEX = 1                 # 検証用の束の2日目（2024-02-01）
CENTER_J = 0                  # D は未見の評価の当日の0番目の種（world.unseen_actual と同じ）
CALIB = HERE / 'calib_day.json'


def target_base():
    from training.bid_bank import BidBank
    bank = BidBank(world.main_bank(TASK, 'validation'))
    return bank.load_entry(bank.entries[DAY_INDEX])


def center(base):
    """D：その日のEVの状況（到着の試行の結果と0時のEV）。"""
    seed = int(base['forecast_seed']) + 10_000_000 + 10007 * CENTER_J
    return world.arrival_events(seed, 'seeded', base['arrival_probabilities_by_station'], base.get('service_date'))


class day_world:
    """with day_world(D, kappa, tau): の中で作る EVEnv の1日を、D を中心に引き直したものにする。"""

    def __init__(self, D, kappa: float = 1.0, tau: float = 0.0):
        self.events, self.initial = D
        self.kappa, self.tau = float(kappa), float(tau)

    def __enter__(self):
        import numpy as np
        from environment.EVEnv import EVEnv
        self.orig = (EVEnv._pregenerate_arrival_events, EVEnv._preroll_initial_evs)
        orig_events, orig_initial = self.orig
        D_events, D_initial, kappa, tau = self.events, self.initial, self.kappa, self.tau
        steps, n_st, n_slot = len(D_events), len(D_events[0]), len(D_events[0][0])
        by_station = [[(t, ev) for t in range(steps) for ev in D_events[t][st] if ev is not None] for st in range(n_st)]
        state = {}

        def resampled(env):
            orig_events(env)  # 環境の乱数の流れを本線と同じに進める
            rng = np.random.default_rng(random.randrange(2 ** 31))
            state['rng'] = rng
            out = [[[None] * n_slot for _ in range(n_st)] for _ in range(steps)]
            for st in range(n_st):
                pool = by_station[st]
                n = len(pool)
                if n == 0:
                    continue
                m = max(0, int(round(n + kappa * (rng.poisson(n) - n))))
                for i in rng.integers(0, n, size=m):
                    t, ev = pool[int(i)]
                    if tau > 0:
                        t = int(min(max(t + round(rng.normal(0.0, tau)), 0), steps - 1))
                    for t2 in range(t, steps):  # 同じ時刻の口が埋まっていたら次の時刻へ
                        free = [k for k in range(n_slot) if out[t2][st][k] is None]
                        if free:
                            out[t2][st][free[int(rng.integers(0, len(free)))]] = copy.deepcopy(ev)
                            break
            return out

        def resampled_initial(env, seed_):
            orig_initial(env, seed_)
            rng = state.get('rng') or np.random.default_rng(int(seed_))
            out = []
            for st in range(n_st):
                pool = D_initial[st]
                out.append([copy.deepcopy(pool[int(i)]) for i in rng.integers(0, len(pool), size=len(pool))] if pool else [])
            return out

        EVEnv._pregenerate_arrival_events, EVEnv._preroll_initial_evs = resampled, resampled_initial
        return self

    def __exit__(self, *exc):
        from environment.EVEnv import EVEnv
        EVEnv._pregenerate_arrival_events, EVEnv._preroll_initial_evs = self.orig
        return False


def candidates(base, D, kappa, tau, n=128):
    """入札の候補：本線の候補と同じ種で、その日の世界から n 通り。"""
    fs, probs, date = int(base['forecast_seed']), base['arrival_probabilities_by_station'], base.get('service_date')
    with day_world(D, kappa, tau):
        return [world.capture_seeded(fs + 10007 * i, probs, date) for i in range(n)]


def actuals(base, D, kappa, tau, js=tuple(range(1, 21))):
    """当日：同じ世界から、候補と別の種で。"""
    fs, probs, date = int(base['forecast_seed']), base['arrival_probabilities_by_station'], base.get('service_date')
    with day_world(D, kappa, tau):
        return [world.capture_seeded(fs + 10_000_000 + 10007 * j, probs, date) for j in js]


def _measure(args):
    kappa, tau = args
    world.configure(TASK)
    import numpy as np
    base = target_base()
    D = center(base)
    cands = candidates(base, D, kappa, tau)
    acts = actuals(base, D, kappa, tau)
    vol = np.array([world.connected(c).sum() for c in cands])
    lo, hi = np.percentile(vol, [5, 95])
    return {'kappa': kappa, 'tau': tau, 'wape': [world.wape(cands, a) for a in acts],
            'p5_pct': 100 * (lo / vol.mean() - 1), 'p95_pct': 100 * (hi / vol.mean() - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='1:0,1:3,1:6,1:12,0.5:0,1.5:0')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    import multiprocessing as mp
    import numpy as np
    grid = [tuple(float(x) for x in g.split(':')) for g in args.grid.split(',')]
    mp.set_start_method('spawn', force=True)
    with mp.Pool(args.workers, maxtasksperchild=1) as pool:
        rows = pool.map(_measure, grid)
    for r in rows:
        r['wape_mean'] = float(np.mean(r['wape']))
        print(f"kappa {r['kappa']} tau {r['tau']}: WAPE {r['wape_mean']:.2f}%  接続量の5%点 {r['p5_pct']:+.1f}%  95%点 {r['p95_pct']:+.1f}%",
              flush=True)
    out = json.loads(CALIB.read_text(encoding='utf-8')) if CALIB.exists() else {}
    out.setdefault('measurements', []).extend(rows)
    CALIB.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

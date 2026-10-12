"""入札後の調整の案1・2・4を、研究室の20 stationの行動器で測る（学習のやり直しなし）。

    python bidfeedback20.py <aemo|ercot> run --variants base opt2_r075 opt2_r050 [--workers 9]
    python bidfeedback20.py <aemo|ercot> fit
    python bidfeedback20.py <aemo|ercot> run --variants opt1_q075 opt1_q090 opt1_q095 opt4_t060 opt4_t075 opt4_t085
    （名前の末尾に s を付けると、行動器の観測の尺度を元の入札のまま保つ。build_episode を参照）
    python bidfeedback20.py <aemo|ercot> summarize

日・指令・EVの選び方は Codex の prepare_calendar20.py と同じにする。入札日30日（学習用25日と検証用5日）、
学習で使っていない指令2本、EVの実現2通りで、1市場120組になる。行動器は1000回以降で和が最大の回
（AEMO 1380、ERCOT 1340）。

案は、入札を作り直さずに入札後の調整として当てる。
- base：入札そのまま
- opt2_rXXX：基準値−上げ幅と基準値＋下げ幅が、アセスメントIの出力範囲（入札の前提のEV実現3通りで
  30分続けられる放電・充電電力の最小）のρ倍に収まるよう幅を縮める。入札器はこの範囲から一律80 kW内側までを使う
- opt1_qXX：時間帯（3時間）と休日（土日と祝日、環境の区分と同じ）の組ごとの「帯に入るのに要る半幅」の分位点より帯が細い向きを外す
- opt4_tXX：コマの合格確率をロジスティック回帰で推定し、τ未満のコマを外す
幅が最低入札量に届かない向きは外す。案1と案4の値は、日を偶数番目と奇数番目に分け、
片方の日の base の結果で当てはめて、もう片方の日に使う（交差当てはめ）。
"""
import argparse
import datetime
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BEST = {'aemo': 1380, 'ercot': 1340}
SAVED = {'aemo': 1634, 'ercot': 1595}
SCREEN = None
ACTORS = None


def load_screen(market):
    global SCREEN
    spec = importlib.util.spec_from_file_location(
        'screen', ROOT / 'execute_results/watch_codex_20261008/trace_screen_helpers.py')
    SCREEN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SCREEN)
    SCREEN.ROOT = ROOT
    SCREEN.configure(market)
    SCREEN.CONSTANTS = {}
    SCREEN.MODE = 'normal'
    sys.path.insert(0, str(ROOT))


def prepare_days(market):
    """Days, bids and the two held-out commands per day, as in prepare_calendar20.py."""
    import numpy as np
    from training.bid_bank import BidBank
    from training.lower_bid_training import interim_test_command_sources, _lower_command_seed
    from market.activation_scenarios import (load_proxy_shape_library, _row_source, stable_seed_for_context,
                                             LOWER_CONTROL_POOL, build_activation_scenario_set,
                                             scenarios_to_solver_payload)
    from EnvConfig import LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR
    from environment.calendars import day_class, HOLIDAY
    tr, va, _ = SCREEN.roots(market)
    train, valid = BidBank(tr), BidBank(va)
    lib = load_proxy_shape_library(LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR)
    sources = [_row_source(lib.iloc[i]) for i in range(len(lib))]
    interim = interim_test_command_sources(valid.entries, interim_test_episodes=5,
                                           library_dir=LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR)
    pool = [s for s, d in zip(sources, lib['declared_partition']) if d in ('validation', 'test') and s not in interim]
    seen = set(interim)
    for idx in range(SAVED[market] + 25 + 1):
        e = train.entries[max(idx - 1, 0) % len(train.entries)]
        _, seed = _lower_command_seed(e, idx, 'train')
        rng = np.random.default_rng(stable_seed_for_context(e['service_date'], seed, 'explicit_5min_random', LOWER_CONTROL_POOL))
        seen.add(pool[int(rng.permutation(len(pool))[0])])
    days = []
    for split, bank in [('train_bid', train), ('validation_bid', valid)]:
        for i, entry in enumerate(bank.entries):
            b = dict(bank.load_entry(entry))
            day = b['service_date']
            date = datetime.date.fromisoformat(day)
            signals = build_activation_scenario_set(day, 2, int(b['forecast_seed']) + 31_670_009,
                                                    LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR, 'holdout', True, seen)
            assert all(s.source not in seen for s in signals)
            days.append({
                'key': f'{split}_d{i}', 'date': day, 'split': split, 'index': i,
                'holiday': day_class(date, 'JP') == HOLIDAY, 'bid': b,
                'signals': [scenarios_to_solver_payload([s]) for s in signals],
                'sources': [s.source for s in signals],
                'ev_seeds': [39_000_000 + (1_000_000 if split == 'validation_bid' else 0) + 1009 * i + 10007 * ev
                             for ev in range(2)],
            })
    assert len(days) == 30
    return days


def design_capability(bid):
    """Per-block fleet charge / discharge power the bid's EV realizations sustain for 30 minutes (Assessment I)."""
    import numpy as np
    charge = np.maximum(np.asarray(bid['assessment_i_sustained_power_max_kw'], float), 0.0)
    discharge = np.maximum(-np.asarray(bid['assessment_i_sustained_power_min_kw'], float), 0.0)
    return charge, discharge


def block_features(bid, holiday, cap):
    import numpy as np
    base = np.asarray(bid['baseline_plan'], float)
    up = np.asarray(bid['up_plan'], float)
    down = np.asarray(bid['down_plan'], float)
    c, d = cap
    hour = np.arange(48) / 2.0
    return np.column_stack([
        np.log1p(0.1 * up), np.log1p(0.1 * down),
        (base + down) / np.maximum(c, 1.0), (up - base) / np.maximum(d, 1.0), base / np.maximum(c, 1.0),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.full(48, float(holiday)), ((up > 0) & (down <= 0)).astype(float), ((down > 0) & (up <= 0)).astype(float),
    ])


def fold_of(day):
    return int(day['index'] + (100 if day['split'] == 'validation_bid' else 0)) % 2


def split_variant(variant):
    """A trailing 's' keeps the observation scale of the bid as submitted (see build_episode)."""
    return (variant[:-1], True) if variant.startswith('opt') and variant.endswith('s') else (variant, False)


def apply_variant(day, variant, params, floor):
    import numpy as np
    variant, _ = split_variant(variant)
    b = day['bid']
    base = np.asarray(b['baseline_plan'], float)
    up = np.asarray(b['up_plan'], float).copy()
    down = np.asarray(b['down_plan'], float).copy()
    if variant == 'base':
        pass
    elif variant.startswith('opt2_r'):
        rho = int(variant[6:]) / 100.0
        c, d = design_capability(b)
        down = np.minimum(down, np.maximum(0.0, rho * c - base))
        up = np.minimum(up, np.maximum(0.0, base + rho * d))
    elif variant.startswith('opt1_q'):
        table = params['opt1'][str(fold_of(day))][variant[6:]]
        for blk in range(48):
            need = table[f"{(blk // 2) // 3}_{int(day['holiday'])}"]
            if 0.1 * up[blk] < need:
                up[blk] = 0.0
            if 0.1 * down[blk] < need:
                down[blk] = 0.0
    elif variant.startswith('opt4_t'):
        tau = int(variant[6:]) / 100.0
        model = params['opt4'][str(fold_of(day))]
        x = block_features(b, day['holiday'], design_capability(b))
        z = (x - np.asarray(model['mean'])) / np.asarray(model['std'])
        p = 1.0 / (1.0 + np.exp(-(z @ np.asarray(model['coef']) + model['intercept'])))
        drop = (p < tau)
        up[drop] = 0.0
        down[drop] = 0.0
    else:
        raise ValueError(variant)
    up[up < floor - 1e-6] = 0.0
    down[down < floor - 1e-6] = 0.0
    return up, down


def build_episode(day, cmd, up, down, keep_scale=False):
    """One episode of the changed bid.

    The actor sees the instruction divided by the widest instruction the day's bid can produce, and the bid
    lookahead in the same unit. Narrowing the bid shrinks that divisor. With keep_scale the divisor stays at
    the bid as submitted, which is also fixed the day before.
    """
    import numpy as np
    from training.lower_bid_training import (build_fixed_upper_bid_training_episode, bid_instruction_scale_kw,
                                             bid_lookahead_context_series, LOWER_BID_LOOKAHEAD_BLOCKS)
    fixed = dict(day['bid'])
    fixed.pop('_runtime_market_context_cache', None)
    fixed.update(up_plan=up, down_plan=down, submitted_up_plan=up, submitted_down_plan=down,
                 activation_scenario_payload=day['signals'][cmd], activation_scenarios=1)
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed, 0)
    if keep_scale:
        b = day['bid']
        scale = bid_instruction_scale_kw(b['baseline_plan'], b['up_plan'], b['down_plan'])
        arr = dict(arr)
        arr['instruction_scale_kw'] = scale
        arr['market_context_series'] = {
            'instruction_scale_kw': np.full(len(target), scale, dtype=np.float32),
            **bid_lookahead_context_series(b['baseline_plan'], up, down, instruction_scale_kw=scale,
                                           lookahead_blocks=int(LOWER_BID_LOOKAHEAD_BLOCKS)),
        }
    keep = {k: arr.get(k) for k in ('arrival_probabilities_by_station', 'baseline_series',
                                     'instruction_scale_kw', 'market_context_series', 'tracking_enabled_series')}
    return {'target': target, 'tol': tol, 'arr': keep}


def worker_init(market):
    global ACTORS
    load_screen(market)
    import torch
    from environment.normalize import load_observation_normalization_for_archive
    torch.set_num_threads(1)
    load_observation_normalization_for_archive(SCREEN.roots(market)[2])
    ACTORS = SCREEN.load_actors(market, BEST[market])


def rollout(task):
    import numpy as np
    import torch
    from tools.evaluator import set_env_seed
    from environment.EVEnv import EVEnv
    from environment.normalize import use_instruction_scale, normalize_observation
    ep = task['episode']
    arr = ep['arr']
    set_env_seed(task['ev_seed'])
    use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.use_central_ev_residual_allocator = False
    env.use_residual_bess = False
    env.reset(net_demand_series=ep['target'], tol_narrow_series=ep['tol'],
              tracking_enabled_series=np.asarray(arr['tracking_enabled_series'], bool),
              market_context_series=arr.get('market_context_series'),
              arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
              service_date=task['date'], baseline_series=arr.get('baseline_series'))
    req, band, resid, enabled = [], [], [], []
    with torch.no_grad():
        for _ in range(288):
            obs = normalize_observation(env.begin_step())
            action = torch.stack([ACTORS[i](obs[i]) for i in range(20)])
            req.append(float(env.current_net_demand))
            band.append(float(env.tol_narrow_metrics))
            enabled.append(bool(env.current_tracking_enabled))
            _, _, _, _, info = env.apply_action(action, return_observation=False, include_active_evs=False,
                                                include_ev_physics=False, include_reward_breakdown=False)
            resid.append(float(info['raw_actor_residual_kw']))
    m = env.get_metrics()
    out = OUT / task['market'] / task['variant']
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{task['id']}.npz", req=np.asarray(req), band=np.asarray(band),
                        resid=np.asarray(resid), enabled=np.asarray(enabled), up=task['up'], down=task['down'])
    row = {'id': task['id'], 'variant': task['variant'], 'date': task['date'], 'source': task['source'],
           'ev_seed': task['ev_seed'], 'tracking': float(m['tracking_success_rate']),
           'soc': 100.0 - float(m['soc_miss_rate']), 'mae': float(m.get('raw_actor_mae_kw', float('nan')))}
    (out / f"{task['id']}.json").write_text(json.dumps(row), encoding='utf-8')
    return row


def run(market, variants, workers, only=None):
    import numpy as np
    from training.blockwise_bid import minimum_bid_quantity_kw
    floor = float(minimum_bid_quantity_kw())
    days = prepare_days(market)
    params_file = OUT / market / 'params.json'
    params = json.loads(params_file.read_text(encoding='utf-8')) if params_file.exists() else {}
    tasks = []
    for variant in variants:
        for day in days:
            up, down = apply_variant(day, variant, params, floor)
            for cmd in range(2):
                ids = [(ev, f"{day['key']}_c{cmd}_e{ev}") for ev in range(2)]
                ids = [(ev, tid) for ev, tid in ids if (not only or tid in only)
                       and not (OUT / market / variant / f'{tid}.json').exists()]
                if not ids:
                    continue
                ep = build_episode(day, cmd, up, down, split_variant(variant)[1])
                for ev, tid in ids:
                    tasks.append({'market': market, 'variant': variant, 'id': tid, 'date': day['date'],
                                  'source': day['sources'][cmd], 'ev_seed': day['ev_seeds'][ev],
                                  'episode': ep, 'up': up, 'down': down})
    print(market, 'tasks', len(tasks), 'floor', floor, flush=True)
    import multiprocessing as mp
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=worker_init, initargs=(market,)) as pool:
        for fut in as_completed([pool.submit(rollout, t) for t in tasks]):
            fut.result()
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(market, 'done', done, '/', len(tasks), flush=True)


def step_need(req, band, resid, enabled):
    """Half-width each assessed step needs to pass, with the band widened before a command change."""
    import numpy as np
    resp = req - resid
    nxt = np.r_[req[1:], req[-1]]
    nxt_on = np.r_[enabled[1:], False]
    changed = enabled & nxt_on & (np.abs(nxt - req) > 1e-6)
    lo = np.where(changed, np.minimum(req, nxt), req)
    hi = np.where(changed, np.maximum(req, nxt), req)
    return np.maximum.reduce([np.zeros_like(req), lo - resp, resp - hi])


def fit(market):
    import numpy as np
    days = prepare_days(market)
    rows = {f: [] for f in (0, 1)}
    blocks = {f: ([], [], []) for f in (0, 1)}
    held_out = []
    for day in days:
        f = fold_of(day)
        cap = design_capability(day['bid'])
        x = block_features(day['bid'], day['holiday'], cap)
        passes = np.zeros(48)
        count = 0
        for cmd in range(2):
            for ev in range(2):
                z = np.load(OUT / market / 'base' / f"{day['key']}_c{cmd}_e{ev}.npz")
                need = step_need(z['req'], z['band'], z['resid'], z['enabled'].astype(bool))
                for s in np.nonzero(z['enabled'])[0]:
                    rows[f].append(((s // 12) // 3, int(day['holiday']), need[s]))
                ok = (need <= z['band'] + 1e-6) | ~z['enabled'].astype(bool)
                passes += ok.reshape(48, 6).all(1)
                count += 1
        part = (np.asarray(day['bid']['up_plan']) > 0) | (np.asarray(day['bid']['down_plan']) > 0)
        award = np.asarray(day['bid']['up_plan'], float) + np.asarray(day['bid']['down_plan'], float)
        blocks[f][0].append(x[part])
        blocks[f][1].append(passes[part] / count)
        blocks[f][2].append(np.full(part.sum(), float(count)))
        held_out.append((f, x[part], passes[part] / count, award[part]))
    params = {'opt1': {}, 'opt4': {}}
    for f in (0, 1):
        other = 1 - f
        table = {}
        arr = np.asarray(rows[other], float)
        for q in ('075', '090', '095'):
            t = {}
            for h in range(8):
                for hol in (0, 1):
                    sel = (arr[:, 0] == h) & (arr[:, 1] == hol)
                    vals = arr[sel, 2] if sel.sum() >= 30 else arr[arr[:, 1] == hol, 2]
                    t[f'{h}_{hol}'] = float(np.quantile(vals, int(q) / 100.0))
            table[q] = t
        params['opt1'][str(f)] = table
        X = np.vstack(blocks[other][0])
        y = np.concatenate(blocks[other][1])
        w = np.concatenate(blocks[other][2])
        mean, std = X.mean(0), X.std(0) + 1e-9
        Z = (X - mean) / std
        coef, b0 = np.zeros(Z.shape[1]), 0.0
        lam = 1.0
        for _ in range(50):
            p = 1 / (1 + np.exp(-(Z @ coef + b0)))
            W = w * p * (1 - p) + 1e-9
            A = np.column_stack([np.ones(len(Z)), Z])
            H = A.T @ (A * W[:, None]) + lam * np.diag([0.0] + [1.0] * Z.shape[1])
            g = A.T @ (w * (y - p)) - lam * np.r_[0.0, coef]
            step = np.linalg.solve(H, g)
            b0 += step[0]
            coef += step[1:]
            if np.abs(step).max() < 1e-8:
                break
        p = 1 / (1 + np.exp(-(Z @ coef + b0)))
        params['opt4'][str(f)] = {'mean': mean.tolist(), 'std': std.tolist(), 'coef': coef.tolist(),
                                  'intercept': float(b0), 'fit_blocks': int(len(Z)),
                                  'fit_mean_abs_error': float(np.average(np.abs(p - y), weights=w))}
    # Static estimate: base outcomes held fixed, dropped blocks simply removed. The rollouts with the
    # changed bid measure how far the rest of the day moves through the SoC.
    sweep = []
    for tau in np.round(np.arange(0.0, 0.96, 0.05), 2):
        kept = failed = submitted = passing = 0.0
        for f, x, frac, award in held_out:
            model = params['opt4'][str(f)]
            z = (x - np.asarray(model['mean'])) / np.asarray(model['std'])
            p = 1 / (1 + np.exp(-(z @ np.asarray(model['coef']) + model['intercept'])))
            keep = p >= tau
            kept += keep.sum()
            failed += ((1 - frac) * keep).sum()
            submitted += award[keep].sum()
            passing += (frac * award * keep).sum()
        n = len(held_out)
        sweep.append({'tau': float(tau), 'blocks_per_day': kept / n, 'failed_blocks_per_day': failed / n,
                      'submitted_kw_block_per_day': submitted / n, 'passing_kw_block_per_day': passing / n})
    params['opt4_static_sweep'] = sweep
    (OUT / market / 'params.json').write_text(json.dumps(params, indent=1), encoding='utf-8')
    for r in sweep:
        print(f"static tau {r['tau']:.2f} blocks/day {r['blocks_per_day']:5.1f} fail/day {r['failed_blocks_per_day']:5.2f} "
              f"submitted/day {r['submitted_kw_block_per_day']:8.0f} passing/day {r['passing_kw_block_per_day']:8.0f}")
    print(json.dumps({'opt1': params['opt1'],
                      'opt4': {f: {k: v[k] for k in ('fit_blocks', 'fit_mean_abs_error')} for f, v in params['opt4'].items()}},
                     indent=1))


def summarize(market):
    import numpy as np
    result = {}
    for vdir in sorted((OUT / market).iterdir()):
        if not vdir.is_dir():
            continue
        acc = {j: {'submitted': 0.0, 'passing': 0.0, 'blocks': 0, 'pass_blocks': 0, 'in_band': [], 'err_rate': []}
               for j in ('none', 'forward')}
        tracking, soc, n = [], [], 0
        for npz in sorted(vdir.glob('*.npz')):
            z = np.load(npz)
            row = json.loads(npz.with_suffix('.json').read_text(encoding='utf-8'))
            tracking.append(row['tracking'])
            soc.append(row['soc'])
            n += 1
            en = z['enabled'].astype(bool)
            award = (z['up'] + z['down'])
            for j in ('none', 'forward'):
                need = np.abs(z['resid']) if j == 'none' else step_need(z['req'], z['band'], z['resid'], en)
                ok = need <= z['band'] + 1e-6
                part = en.reshape(48, 6).any(1)
                bpass = (ok | ~en).reshape(48, 6).all(1)
                a = acc[j]
                a['submitted'] += float(award[part].sum())
                a['passing'] += float(award[part & bpass].sum())
                a['blocks'] += int(part.sum())
                a['pass_blocks'] += int((part & bpass).sum())
                a['in_band'] += list(ok[en])
                a['err_rate'] += list((np.abs(z['resid']) / np.maximum(10 * z['band'], 1e-9))[en])
        res = {'cases': n, 'tracking_env': float(np.mean(tracking)), 'soc': float(np.mean(soc))}
        for j, a in acc.items():
            per_case = max(n, 1)
            res[j] = {
                'submitted_kw_block_per_day': a['submitted'] / per_case,
                'passing_kw_block_per_day': a['passing'] / per_case,
                'blocks_per_day': a['blocks'] / per_case,
                'block_pass_pct': 100 * a['pass_blocks'] / max(a['blocks'], 1),
                'failed_blocks_per_day': (a['blocks'] - a['pass_blocks']) / per_case,
                'step_in_band_pct': 100 * float(np.mean(a['in_band'])) if a['in_band'] else float('nan'),
                'mean_err_over_award_pct': 100 * float(np.mean(a['err_rate'])) if a['err_rate'] else float('nan'),
            }
        result[vdir.name] = res
    (OUT / market / 'summary.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    for k, v in result.items():
        f = v['forward']
        print(f"{market} {k:10s} n={v['cases']:3d} blocks/day {f['blocks_per_day']:5.1f} submitted/day {f['submitted_kw_block_per_day']:8.0f} "
              f"passing/day {f['passing_kw_block_per_day']:8.0f} pass {f['block_pass_pct']:5.1f}% fail/day {f['failed_blocks_per_day']:5.2f} "
              f"(x30 {30 * f['failed_blocks_per_day']:6.1f}) in-band {f['step_in_band_pct']:5.1f}% "
              f"| none pass {v['none']['block_pass_pct']:5.1f}% | tracking {v['tracking_env']:5.1f} soc {v['soc']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('market', choices=['aemo', 'ercot'])
    ap.add_argument('action', choices=['run', 'fit', 'summarize'])
    ap.add_argument('--variants', nargs='+', default=['base'])
    ap.add_argument('--workers', type=int, default=9)
    ap.add_argument('--only', nargs='*', help='case ids to run, for checking against the Codex calendar20 rows')
    a = ap.parse_args()
    load_screen(a.market)
    if a.action == 'run':
        run(a.market, a.variants, a.workers, set(a.only or []))
    elif a.action == 'fit':
        fit(a.market)
    else:
        summarize(a.market)


if __name__ == '__main__':
    main()

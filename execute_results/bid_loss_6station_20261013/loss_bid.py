"""入札時に発生する損を表にする（6 station、検証用の40組）。前日に知っている情報（当日の指令、EVの予測の精度）と、
入札器が使うEVの実現の選び方を振り、入札量と、その組の当日の指令とEVで入札が成り立つかを測る。

    python loss_bid.py prep [workers]                 # 5日分の128通り（入札器が見るEVの実現の候補）と、40組の当日のEVを保存する
    python loss_bid.py calibrate [95|85]              # 精度95%の k と85%の σ を、WAPE が5%と15%になるように決め、calib.json に書く
    python loss_bid.py run <aemo|ercot> [workers]     # 下の18通りを、精度91% → 85% → 95% の順にすべて回す
    python loss_bid.py summarize

組、判定、入札器の条件は info_bid_6station_20261012/info_bid.py と同じ（検証用の平日5日 × holdout の指令4本 × EVの乱数2通り。
判定は、その組の当日の指令とEVで、全情報のLPが帯（5分すべて）と出発時SoCを満たせるか。1組20秒まで）。
入札量の基準は、同じ40組で当日の指令とEVを両方渡した入札（info_bid の oracle、AEMO 22,963・ERCOT 22,964 kW・コマ／日）。

当日の指令
  known    当日の指令1本を渡し、全コマで上げと下げの両方を許す（info_bid の cmd と同じ）
  unknown  本線と同じ設計用の指令128本と、本線と同じ向きの制限
EVの予測の精度（5分ごとの接続台数（6 station の合計）の WAPE を当日と比べ、精度 = 100% − WAPE。docs/EVデータ.md の「予測の精度」）
  91  今の予測。128通りをそのまま使う。入札は、指令 known なら日と指令ごと（20本）、unknown なら日ごと（5本）
  95  128通りの各実現の各 station を、確率 k で当日のその station のEVに丸ごと入れ替える。組ごとに入札する（40本）。
      station 単位で入れ替えるので充電口の数を超えず、予測の平均は k×当日＋(1−k)×今の予測になる
  85  組ごとに1つの倍率 f = 1+ε（ε ~ N(0, σ)、0.3〜2.0 に収める）を、128通りすべての到着の量に掛ける。
      f < 1 なら各EVを確率 f で残し、f > 1 なら別の実現のEVを確率 f−1 で足す。組ごとに入札する（40本）
  85で足したEVは、1 station の同時接続が充電口の数（EnvConfig.MAX_EV_PER_STATION）を超えたら、後から来たEVを外す。
  95と85の乱数は組から決めるので、指令 known と unknown は同じ予測を使う。
EVの実現の選び方（128通りから3通り）
  low2max    本線。接続の少ない2通り＋接続数が最多の1通り（low_connection_2_max_count）
  minmedmax  前のやり方。接続数の最小・中央・最大（session_count）
  random3    最初のやり方。無作為に3通り（乱数は、日ごとの入札では日から、組ごとの入札では組から決める）
結果は result_<task>_<cmd>_<acc>_<sel>.jsonl（1行1組）。済んだ組は飛ばす。
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import types
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Windows では空の文字列だと変数が消え、GPU が見えてしまう
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'info_bid_6station_20261012'))
import info_bid as ib  # noqa: E402

t3 = ib.t3
N_DAY, N_CMD, N_EV = ib.N_DAY, ib.N_CMD, ib.N_EV
TASKS = ('aemo', 'ercot')
CMDS = ('known', 'unknown')
ACCS = ('85', '91', '95')
RUN_ORDER = ('91', '85', '95')  # 回す順。95% は作り方を確かめてから回す
SELS = ('low2max', 'minmedmax', 'random3')
TARGET_WAPE = {'95': 5.0, '85': 15.0}
CALIB = HERE / 'calib.json'
ORACLE = {t: HERE.parent / 'info_bid_6station_20261012' / f'result_{t}_oracle.jsonl' for t in TASKS}


def sets():
    return [(d, c, e) for d in range(N_DAY) for c in range(N_CMD) for e in range(N_EV)]


def cand_path(day):
    return HERE / 'data' / f'candidates_day{day}.pkl'


def actual_path(task):
    return HERE / 'data' / f'actual_{task}.pkl'


# ---- 予測（128通り）を作る ----

def relabel(evs, tag):
    return [dataclasses.replace(ev, ev_id=f'{tag}:{ev.ev_id}') for ev in evs]


def port_limit(evs):
    """1 station の同時接続を充電口の数以内にする（到着の早い順に入れ、埋まっていれば外す）。"""
    from EnvConfig import MAX_EV_PER_STATION
    kept = []
    for st in sorted({int(x.station_id) for x in evs}):
        active = []
        for ev in sorted((x for x in evs if int(x.station_id) == st), key=lambda x: (x.arrival_t, x.departure_t)):
            active = [d for d in active if d > ev.arrival_t]
            if len(active) < int(MAX_EV_PER_STATION):
                kept.append(ev)
                active.append(ev.departure_t)
    return kept


def mix_95(cands, actual, k, rng):
    # 当日のEVと候補のEVを1台ずつ混ぜると、同じ station で充電口の数を超えたEVを外すことになり、予測が少なめに偏る
    # （k=0.43 で WAPE 9.6%、口の上限を外すと 4.5%）。そのため station ごとに丸ごと入れ替える。
    stations = sorted({int(ev.station_id) for x in [actual, *cands] for ev in x})
    real = relabel(actual, 'a')
    out = []
    for cand in cands:
        own = relabel(cand, 'c')
        evs = []
        for st in stations:
            src = real if rng.random() < k else own
            evs.extend(ev for ev in src if int(ev.station_id) == st)
        out.append(evs)
    return out


def shift_85(cands, sigma, rng):
    f = float(min(max(1 + rng.normal(0.0, sigma), 0.3), 2.0))
    out = []
    n = len(cands)
    for i, cand in enumerate(cands):
        if f <= 1:
            out.append(port_limit([ev for ev in relabel(cand, 'c') if rng.random() < f]))
        else:
            extra = [ev for ev in relabel(cands[(i + 37) % n], 'x') if rng.random() < f - 1]
            out.append(port_limit(relabel(cand, 'c') + extra))
    return out, f


def forecast(acc, day, c, e, cands, actual, calib):
    """精度ごとの128通り。91 はそのまま。95 と 85 の乱数は組から決める（指令 known と unknown で同じ）。"""
    import numpy as np
    if acc == '91':
        return cands, None
    rng = np.random.default_rng(1_300_000 + 1000 * day + 10 * c + e + (0 if acc == '95' else 500_000))
    if acc == '95':
        return mix_95(cands, actual, calib['k95'], rng), None
    return shift_85(cands, calib['sigma85'], rng)


def connected(evs, steps=288):
    import numpy as np
    out = np.zeros(steps)
    for ev in evs:
        out[int(ev.arrival_t):min(int(ev.departure_t), steps)] += 1
    return out


def wape(cands, actual):
    import numpy as np
    mean = np.mean([connected(x) for x in cands], axis=0)
    a = connected(actual)
    return 100.0 * float(np.abs(mean - a).sum() / max(a.sum(), 1e-9))


# ---- 128通りから3通りを選ぶ ----

def select(sel, cands, seed):
    import numpy as np
    from training import lower_bid_training as lbt
    if sel == 'low2max':
        return lbt._select_ev_scenarios_low_connection(cands, low_picks=2, count_picks=('maximum',))
    if sel == 'minmedmax':
        return lbt._select_ev_scenarios_by_count(cands)
    picks = [int(i) for i in np.random.default_rng(seed).choice(len(cands), 3, replace=False)]
    # 入札器の記録が使う接続数の順位（_select_ev_scenarios_by_count と同じ並べ方）も付ける
    rank = {index: r for r, index in enumerate(sorted(range(len(cands)), key=lambda index: (len(cands[index]), index)))}
    return [cands[i] for i in picks], [{'label': f'random_{j + 1}', 'candidate_index': i, 'rank_zero_based': rank[i],
                                        'candidate_count': len(cands), 'ev_count': len(cands[i])} for j, i in enumerate(picks)]


# ---- 準備 ----

def _actual_job(args):
    task, d, c, e = args
    t3._setup(task)
    return (d, c, e), ib.actual_evs(task, d, c, e)


def cmd_prep(workers):
    t3._setup('aemo')  # stage_worker を読み込むと、リポジトリのモジュールを読めるようになる
    from training.bid_bank import BidBank
    from training import lower_bid_training as lbt
    (HERE / 'data').mkdir(exist_ok=True)
    bank = BidBank(t3._CACHE['test_bank'])
    for d in range(N_DAY):
        if cand_path(d).exists():
            continue
        base = dict(bank.load_entry(bank.entries[d]))
        seed = int(base['forecast_seed'])
        count = int(lbt.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES)
        payloads = [(seed + 10007 * i, base['arrival_probabilities_by_station'], base['service_date']) for i in range(count)]
        cands = lbt._sample_ev_scenario_bank_uncached(payloads, label='loss', workers=workers)
        # 本線の選び方で選んだ3通りが、入札データに残っている3通りと一致するか（同じ128通りを作れているか）を確かめる
        _bank, meta = select('low2max', cands, 0)
        stored = [int(r['candidate_index']) for r in base['ev_scenario_selection']]
        picked = [int(r['candidate_index']) for r in meta]
        same = [sorted(map(repr, _bank[i])) == sorted(map(repr, base['bid_ev_scenario_bank'][i])) for i in range(3)]
        print(f'day{d} {base["service_date"]}: picked {picked} stored {stored} same_evs {same}', flush=True)
        if picked != stored or not all(same):
            raise SystemExit(f'day{d}: 128通りが入札データの作成時と一致しない')
        cand_path(d).write_bytes(pickle.dumps(cands))
    for task in TASKS:
        if actual_path(task).exists():
            continue
        with mp.get_context('spawn').Pool(workers) as pool:
            out = dict(pool.map(_actual_job, [(task, d, c, e) for d, c, e in sets()]))
        actual_path(task).write_bytes(pickle.dumps(out))
        print(task, 'actual EVs', sum(len(v) for v in out.values()) / len(out), 'per set', flush=True)
    a, b = (pickle.loads(actual_path(t).read_bytes()) for t in TASKS)
    print('actual EVs equal across markets:', all(sorted(map(repr, a[s])) == sorted(map(repr, b[s])) for s in sets()), flush=True)


def _mean_wape(acc, strength, act):
    import numpy as np
    calib = {'k95': strength, 'sigma85': strength}
    vals = []
    for d, c, e in sets():
        cands = pickle.loads(cand_path(d).read_bytes())
        vals.append(wape(forecast(acc, d, c, e, cands, act[(d, c, e)], calib)[0], act[(d, c, e)]))
    return float(np.mean(vals))


def cmd_calibrate(only=None):
    t3._setup('aemo')
    act = pickle.loads(actual_path('aemo').read_bytes())
    out = json.loads(CALIB.read_text(encoding='utf-8')) if only and CALIB.exists() else {}
    out['wape_91'] = _mean_wape('91', 0.0, act)
    print(f"now: WAPE {out['wape_91']:.2f}%", flush=True)
    # WAPE は k に対して減り、σ に対して増える。二分法で目標に合わせる（乱数は組で固定なので、強さだけが変わる）。
    for acc, key, lo, hi in (('95', 'k95', 0.0, 1.0), ('85', 'sigma85', 0.0, 0.6)):
        if only and acc != only:
            continue
        for _ in range(14):
            mid = (lo + hi) / 2
            w = _mean_wape(acc, mid, act)
            print(f'{acc}: {key}={mid:.4f} WAPE {w:.2f}%', flush=True)
            if (w > TARGET_WAPE[acc]) == (acc == '95'):
                lo = mid
            else:
                hi = mid
        out[key] = round((lo + hi) / 2, 4)
        out[f'wape_{acc}'] = _mean_wape(acc, out[key], act)
    out['mix95'] = 'station_swap'
    tmp = CALIB.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, indent=1), encoding='utf-8')
    os.replace(tmp, CALIB)  # 回っている入札が途中の内容を読まないように置き換える
    print(out, flush=True)


# ---- 入札と判定 ----

def units(cmd, acc):
    """入札1本ごとの (day, c, e)。None は、その入札をその軸のすべての組で判定することを表す。"""
    if acc != '91':
        return list(sets())
    if cmd == 'known':
        return [(d, c, None) for d in range(N_DAY) for c in range(N_CMD)]
    return [(d, None, None) for d in range(N_DAY)]


def judged_sets(d, c, e):
    cs = range(N_CMD) if c is None else [c]
    es = range(N_EV) if e is None else [e]
    return [(d, cc, ee) for cc in cs for ee in es]


def run_job(args):
    task, cmd, acc, sel, d, c, e = args
    t3._setup(task)
    import numpy as np
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank
    from training.blockwise_bid import build_blockwise_bid_for_day
    import training.blockwise_bid as bb
    t0 = time.time()
    calib = json.loads(CALIB.read_text(encoding='utf-8')) if acc != '91' else {}
    if acc == '95' and calib.get('mix95') != 'station_swap':
        raise RuntimeError('calib.json の k95 は前の混ぜ方で決めた値。python loss_bid.py calibrate 95 で決め直す')
    act = pickle.loads(actual_path(task).read_bytes())
    cands = pickle.loads(cand_path(d).read_bytes())
    actual = act[(d, c, e)] if e is not None else None
    fc, f = forecast(acc, d, c if c is not None else 0, e if e is not None else 0, cands, actual, calib)
    seed = 2_100_000 + 1000 * d + (0 if e is None else 10 * c + e)
    ev_bank, selection = select(sel, fc, seed)
    bank = BidBank(t3._CACHE['test_bank'])
    base = dict(bank.load_entry(bank.entries[d]))
    originals = (lbt._activation_scenarios_for_day, lbt._selected_ev_scenario_bank, bb._observed_activation_directions_by_block)
    lbt._selected_ev_scenario_bank = lambda **kw: (ev_bank, selection, None)
    if cmd == 'known':
        _fixed, command, *_ = t3._episode_inputs(task, d, c, N_CMD)
        mode_name = originals[0](base.get('service_date'), int(base.get('forecast_seed', 0)))[1]
        lbt._activation_scenarios_for_day = lambda *a, **k: ([command], mode_name)
        bb._observed_activation_directions_by_block = lambda *a, **k: (np.ones(bb.N_BLOCKS, bool), np.ones(bb.N_BLOCKS, bool))
    try:
        bid = build_blockwise_bid_for_day(
            np.asarray(base['base_series'], float), base.get('service_date'),
            arrival_scenario=types.SimpleNamespace(arrival_probabilities_by_station=base.get('arrival_probabilities_by_station')),
            forecast_seed=int(base.get('forecast_seed', 0)),
            assessment_band_fraction=float(base['assessment_band_fraction']), scenario_workers=1)
    finally:
        lbt._activation_scenarios_for_day, lbt._selected_ev_scenario_bank, bb._observed_activation_directions_by_block = originals
    solve_s = round(time.time() - t0, 1)
    rows = []
    for dd, cc, ee in judged_sets(d, c, e):
        feasible, reason = ib.judge(task, bid, dd, cc, ee, evs=act[(dd, cc, ee)])
        rows.append({'task': task, 'cmd': cmd, 'acc': acc, 'sel': sel, 'day': dd, 'command': cc, 'ev_seed': ee,
                     'service_date': str(base.get('service_date')),
                     'capacity_kw_block': float(np.sum(bid['up_plan']) + np.sum(bid['down_plan'])),
                     'up_kw_block': float(np.sum(bid['up_plan'])), 'down_kw_block': float(np.sum(bid['down_plan'])),
                     'participating_blocks': int(np.sum((np.asarray(bid['up_plan']) > 0) | (np.asarray(bid['down_plan']) > 0))),
                     'forecast_wape_pct': wape(fc, act[(dd, cc, ee)]), 'shift_factor': f,
                     'selected': [(r['label'], int(r['candidate_index']), int(r['ev_count'])) for r in selection],
                     'feasible': feasible, 'reason': reason, 'solve_s': solve_s})
    return rows


def out_path(task, cmd, acc, sel):
    return HERE / f'result_{task}_{cmd}_{acc}_{sel}.jsonl'


def cmd_run(task, workers):
    jobs = []
    for acc in RUN_ORDER:
        for cmd in CMDS:
            for sel in SELS:
                f = out_path(task, cmd, acc, sel)
                done = set()
                if f.exists():
                    done = {(r['day'], r['command'], r['ev_seed']) for r in map(json.loads, f.read_text(encoding='utf-8').splitlines())}
                for d, c, e in units(cmd, acc):
                    if not all(s in done for s in judged_sets(d, c, e)):
                        jobs.append((task, cmd, acc, sel, d, c, e))
    print(f'{task}: {len(jobs)} bids', flush=True)
    with mp.get_context('spawn').Pool(workers, maxtasksperchild=20) as pool:
        for rows in pool.imap_unordered(run_job, jobs):
            r0 = rows[0]
            with out_path(task, r0['cmd'], r0['acc'], r0['sel']).open('a', encoding='utf-8') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(r0['cmd'], r0['acc'], r0['sel'], r0['day'], r0['command'], r0['ev_seed'], round(r0['capacity_kw_block']),
                  [r['feasible'] for r in rows], r0['solve_s'], flush=True)


def cmd_summarize():
    print('| 市場 | 当日の指令 | EVの予測の精度 | EVの選び方 | 入札の数 | 入札量（kW・コマ／日） | 上限との比 | 損 | 当日に成り立つ組 |')
    print('|---|---|---|---|---|---|---|---|---|')
    names = {'known': '知る', 'unknown': '知らない'}
    for task in TASKS:
        oracle = [json.loads(x) for x in ORACLE[task].read_text(encoding='utf-8').splitlines()]
        top = sum(r['capacity_kw_block'] for r in oracle) / len(oracle)
        print(f'| {task.upper()} | 知る | 当日のEVを知る | — | {len(oracle)} | {top:,.0f} | 100.0% | 0.0% | '
              f'{sum(r["feasible"] is True for r in oracle)}/{len(oracle)} |')
        ev = [json.loads(x) for x in (ORACLE[task].parent / f'result_{task}_ev.jsonl').read_text(encoding='utf-8').splitlines()]
        cap = sum(r['capacity_kw_block'] for r in ev) / len(ev)
        print(f'| {task.upper()} | 知らない | 当日のEVを知る | — | {len(ev)} | {cap:,.0f} | {cap / top:.1%} | {1 - cap / top:.1%} | '
              f'{sum(r["feasible"] is True for r in ev)}/{len(ev)} |')
        for cmd in CMDS:
            for acc in ACCS:
                for sel in SELS:
                    f = out_path(task, cmd, acc, sel)
                    if not f.exists():
                        continue
                    rows = [json.loads(x) for x in f.read_text(encoding='utf-8').splitlines()]
                    if not rows:
                        continue
                    cap = sum(r['capacity_kw_block'] for r in rows) / len(rows)
                    ok = sum(r['feasible'] is True for r in rows)
                    unk = sum(r['feasible'] is None for r in rows)
                    nbid = len({(r['day'], r['command'] if acc != '91' or cmd == 'known' else None,
                                 r['ev_seed'] if acc != '91' else None) for r in rows})
                    print(f'| {task.upper()} | {names[cmd]} | {acc}% | {sel} | {nbid} | {cap:,.0f} | {cap / top:.1%} | {1 - cap / top:.1%} | '
                          f'{ok}/{len(rows)}' + (f'（判定できず {unk}）' if unk else '') + ' |')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'prep':
        cmd_prep(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif mode == 'calibrate':
        cmd_calibrate(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == 'run':
        cmd_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif mode == 'summarize':
        cmd_summarize()

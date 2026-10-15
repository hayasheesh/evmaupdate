"""前日に当日の指令やEVを知っていたら、どれだけ入札できたかを測る（20 station の検証用の40組。研究室で回す）。

    python info_bid.py <aemo|ercot|pjm> <main|oracle|oracle_used|cmd|ev> [workers]

info_bid_6station_20261012/info_bid.py の写しで、違いは読み込む設定と判定の時間の上限だけ。
組は table3_20station_20261012/run_table3.py（20 station の表3）と同じで、決め方は 6 station とも同じ
（検証用の平日5日 × holdout の指令4本 × EVの乱数2通り、EV は set_env_seed(4_000_000 + 1000 * day + 10 * c + e)）。
入札器の条件は 20 station の本線の検証用の入札（wr_cap77wd_min250_margin25_validation_5_20station_128cmd_3ev、
wr_gs_20station_<task>_20261011/stage_worker.py の configure）と同じで、渡す指令とEVだけを替える。
判定の時間の上限は INFO_BID_CERTIFY_S（既定120秒）。20 station の LP は 20秒では判定できない組がある（表3の ERCOT で1組）。
PJM は TABLE3_SW_DIR に resbid_20station_20261014 を渡す（wr_gs_20station_* には PJM がない。入札の束は同じ min250_margin25）。
  main         本線の検証用の入札をそのまま判定する（入札し直さない）。4.3節の「全情報のLP」にあたる（40組）
  oracle       当日の指令1本と当日のEV（実現1通り）を渡す。組ごとに入札する（40本）。上限にあたる
               入札器は設計用の指令で一度も出ない向きの入札を0にする（_observed_activation_directions_by_block）。
               当日の指令が分かっていれば、出ない向きの入札も判定では外れないので、ここでは全コマの両方の向きを許す
  oracle_used  oracle と同じで、向きの制限だけ本線と同じにする（当日の指令が使う向きにだけ入札する）
  cmd          当日の指令1本と、本線と同じ選び方のEVの実現3通りを渡す。日と指令ごとに入札する（20本）。向きは oracle と同じく全部許す
  ev           本線と同じ設計用の指令128本と、当日のEVを渡す。組ごとに入札する（40本。EVの乱数は指令の番号でも変わるため）
どの入札も、評価する組の当日の指令とEVで、全情報のLPが帯（5分すべて）と出発時SoCを満たせるかを判定する。
当日のEVは run_table3.py と同じく、同じ seed で環境を reset して出力0の1日を流して取り出す。
結果は result_<task>_<mode>.jsonl（1行1組）。済んだ組は飛ばす。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import types
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Windows では空の文字列だと変数が消え、GPU が見えてしまう
HERE = Path(__file__).resolve().parent
# run_table3 は読み込むときに TABLE3_SW_DIR を見るので、先に 20 station の stage_worker を指す。
os.environ.setdefault('TABLE3_SW_DIR', str(HERE.parent / 'wr_gs_20station_{task}_20261011'))
sys.path.insert(0, str(HERE.parent / 'table3_20station_20261012'))
import run_table3 as t3  # noqa: E402

CERTIFY_S = float(os.environ.get('INFO_BID_CERTIFY_S', '120'))

N_CMD, N_EV, N_DAY = 4, 2, 5


def actual_evs(task, day, c, e):
    _fixed, _command, target, tol, tracking, arr = t3._episode_inputs(task, day, c, N_CMD)
    env = t3._make_env(task, day, c, e, target, tol, tracking, arr, central=False)
    return t3._capture_evs(env)


def judge(task, bid, day, c, e, evs=None):
    import numpy as np
    from market.physical_lp_bidding.colgen_feasibility import certify
    from market.physical_lp_bidding.data_classes import BiddingLPConfig
    from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands

    _fixed, command, *_ = t3._episode_inputs(task, day, c, N_CMD)
    cfg = BiddingLPConfig(assessment_band_fraction=float(bid['assessment_band_fraction']),
                          apply_transition_band=bool(bid['apply_transition_band']))
    _t, _tol, lower, upper = fixed_bid_tracking_bands(
        cfg, np.asarray(bid['baseline_plan'], float), np.asarray(bid['up_plan'], float), np.asarray(bid['down_plan'], float),
        np.asarray(command['up_proxy'], float), np.asarray(command['down_proxy'], float), apply_transition_band=cfg.apply_transition_band)
    evs = evs if evs is not None else actual_evs(task, day, c, e)
    feasible, _, info = certify(evs, lower, upper, steps=cfg.steps, dt=cfg.dt_hours, eta_ch=cfg.eta_ch,
                                return_dispatch=False, time_limit_s=CERTIFY_S)
    return feasible, (info.get('reason') if feasible is None else '')


def solve(task, mode, day, c, e):
    """本線の検証用の入札と同じ日・seed・条件で、渡す指令とEVだけを替えて入札する。"""
    import numpy as np
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank
    from training.blockwise_bid import build_blockwise_bid_for_day

    t3._setup(task)
    base = dict(BidBank(t3._CACHE['test_bank']).load_entry(BidBank(t3._CACHE['test_bank']).entries[day]))
    if mode == 'main':
        return base, base
    # 当日の指令とEVは、入札器の関数を差し替える前に取り出す（取り出しも同じ関数を使うため）。
    _fixed, command, *_ = t3._episode_inputs(task, day, c, N_CMD)
    evs = actual_evs(task, day, c, e) if mode in ('oracle', 'oracle_used', 'ev') else None
    import training.blockwise_bid as bb
    original_commands, original_evs = lbt._activation_scenarios_for_day, lbt._selected_ev_scenario_bank
    original_dirs = bb._observed_activation_directions_by_block
    if mode in ('oracle', 'cmd'):
        bb._observed_activation_directions_by_block = lambda *a, **k: (np.ones(bb.N_BLOCKS, bool), np.ones(bb.N_BLOCKS, bool))
    if mode in ('oracle', 'oracle_used', 'cmd'):
        mode_name = original_commands(base.get('service_date'), int(base.get('forecast_seed', 0)))[1]
        lbt._activation_scenarios_for_day = lambda *a, **k: ([command], mode_name)
    if mode in ('oracle', 'oracle_used', 'ev'):
        row = {'label': 'actual', 'candidate_index': 0, 'rank_zero_based': 0, 'candidate_count': 1, 'ev_count': len(evs)}
        lbt._selected_ev_scenario_bank = lambda **k: ([evs], [row], None)
    try:
        bid = build_blockwise_bid_for_day(
            np.asarray(base['base_series'], float), base.get('service_date'),
            arrival_scenario=types.SimpleNamespace(arrival_probabilities_by_station=base.get('arrival_probabilities_by_station')),
            forecast_seed=int(base.get('forecast_seed', 0)),
            assessment_band_fraction=float(base['assessment_band_fraction']), scenario_workers=1)
    finally:
        lbt._activation_scenarios_for_day, lbt._selected_ev_scenario_bank = original_commands, original_evs
        bb._observed_activation_directions_by_block = original_dirs
    return base, bid


def run_job(args):
    task, mode, day, c, e = args
    t3._setup(task)  # 入札の条件を環境変数に置いてから EnvConfig を読む
    import numpy as np
    t0 = time.time()
    base, bid = solve(task, mode, day, c, e)
    pairs = [(c, k) for k in range(N_EV)] if mode == 'cmd' else [(c, e)]
    rows = []
    for cc, ee in pairs:
        feasible, reason = judge(task, bid, day, cc, ee)
        rows.append({'task': task, 'mode': mode, 'day': day, 'command': cc, 'ev_seed': ee, 'service_date': str(base.get('service_date')),
                     'capacity_kw_block': float(np.sum(bid['up_plan']) + np.sum(bid['down_plan'])),
                     'main_capacity_kw_block': float(np.sum(base['up_plan']) + np.sum(base['down_plan'])),
                     'up_kw_block': float(np.sum(bid['up_plan'])), 'down_kw_block': float(np.sum(bid['down_plan'])),
                     'participating_blocks': int(np.sum((np.asarray(bid['up_plan']) > 0) | (np.asarray(bid['down_plan']) > 0))),
                     'feasible': feasible, 'reason': reason, 'solve_s': round(time.time() - t0, 1)})
    return rows


def main():
    task, mode = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out = HERE / f'result_{task}_{mode}.jsonl'
    done = set()
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            done.add((r['day'], r['command'], r['ev_seed']))
    if mode in ('main', 'oracle', 'oracle_used', 'ev'):
        jobs = [(task, mode, d, c, e) for d in range(N_DAY) for c in range(N_CMD) for e in range(N_EV)]
        jobs = [j for j in jobs if (j[2], j[3], j[4]) not in done]
    elif mode == 'cmd':
        jobs = [(task, mode, d, c, 0) for d in range(N_DAY) for c in range(N_CMD) if (d, c, 0) not in done]
    print(f'{task} {mode}: {len(jobs)} bids', flush=True)
    with mp.get_context('spawn').Pool(workers) as pool, out.open('a', encoding='utf-8') as f:
        for rows in pool.imap_unordered(run_job, jobs):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            print(rows[0]['day'], rows[0]['command'], rows[0]['ev_seed'], round(rows[0]['capacity_kw_block']), [r['feasible'] for r in rows], rows[0]['solve_s'], flush=True)


if __name__ == '__main__':
    main()

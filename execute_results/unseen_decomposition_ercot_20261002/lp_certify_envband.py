"""帯がずれた組だけ、制御の評価と同じ帯（評価する時刻は目標±許容幅、それ以外は制約なし）で判定し直す。

ずれの原因は、指令が0の時刻の許容幅の決め方。入札段階（fixed_bid_tracking_bands）は上げ・下げの約定量の
合計の10%、環境の評価は片方向の約定量の10%を使う。ERCOT の指令は参加コマの中に指令0の時刻を含むことがある。
"""
import csv
import json
import time

import numpy as np
import pandas as pd

import lp_certify as base  # configure() はここで1回だけ走る
import common
from Config import EPISODE_STEPS
from environment.EVEnv import EVEnv
from environment.normalize import use_instruction_scale
from market.physical_lp_bidding.colgen_feasibility import certify
from market.physical_lp_bidding.data_classes import BiddingLPConfig
from tools.evaluator import set_env_seed
from training.lower_bid_training import build_fixed_upper_bid_training_episode

lp = pd.read_csv(common.HERE / 'lp_certify_cases.csv')
diff = lp[['band_lower_max_diff_kw', 'band_upper_max_diff_kw']].max(axis=1)
targets = {(int(r.bid_day_index), int(r.scenario), int(r.seed)) for r in lp[diff > 1e-3].itertuples()}
rows = []
for day_index, entry, fixed_bid in common.day_cases():
    cfg = BiddingLPConfig(assessment_band_fraction=float(fixed_bid['assessment_band_fraction']),
                          apply_transition_band=bool(fixed_bid['apply_transition_band']))
    env = EVEnv()
    for s in range(len(fixed_bid['activation_scenario_payload'])):
        for k in range(common.EV_SEEDS):
            if (day_index, s, k) not in targets:
                continue
            target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, s)
            target = np.asarray(target, float).reshape(-1)[:EPISODE_STEPS]
            tol = np.asarray(tol, float).reshape(-1)[:EPISODE_STEPS]
            enabled = np.asarray(arr.get('tracking_enabled_series'), bool).reshape(-1)[:EPISODE_STEPS]
            lower = np.where(enabled, target - tol, -np.inf)
            upper = np.where(enabled, target + tol, np.inf)
            set_env_seed(common.realized_seed(day_index, s, k))
            use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
            env.reset(net_demand_series=target, tol_narrow_series=tol, tracking_enabled_series=enabled,
                      market_context_series=arr.get('market_context_series'),
                      arrival_probabilities_by_station=fixed_bid.get('arrival_probabilities_by_station'),
                      day_context=fixed_bid.get('day_context'))
            evs = base.capture_evs(env)
            t0 = time.perf_counter()
            feasible, _, info = certify(evs, lower, upper, steps=cfg.steps, dt=cfg.dt_hours, eta_ch=cfg.eta_ch,
                                        return_dispatch=False, time_limit_s=base.TIME_LIMIT_S)
            rows.append({'bid_day_index': day_index, 'scenario': s, 'seed': k, 'feasible_env_band': feasible,
                         'reason': info.get('reason') if feasible is None else '', 'certify_s': time.perf_counter() - t0})
            print(f"day {day_index} s {s} k {k} feasible_env_band={feasible}", flush=True)
with open(common.HERE / 'lp_certify_envband_cases.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps({'cases': len(rows), 'feasible': sum(r['feasible_env_band'] is True for r in rows),
                  'infeasible': sum(r['feasible_env_band'] is False for r in rows),
                  'unknown': sum(r['feasible_env_band'] is None for r in rows)}), flush=True)

"""帯がずれている組で、指令の値と2つの帯を並べる（読むだけ）。"""
import numpy as np
import pandas as pd
import common
common.configure()
from market.physical_lp_bidding.data_classes import BiddingLPConfig
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
from training.lower_bid_training import build_fixed_upper_bid_training_episode

lp = pd.read_csv(common.HERE / 'lp_certify_cases.csv')
d = lp[['band_lower_max_diff_kw', 'band_upper_max_diff_kw']].max(axis=1)
bad = lp[(d > 1e-3) & (lp['seed'] == 0)].head(3)
cases = {c[0]: c for c in common.day_cases()}
for _, row in bad.iterrows():
    day_index, entry, fixed_bid = cases[int(row['bid_day_index'])]
    s = int(row['scenario'])
    cmd = fixed_bid['activation_scenario_payload'][s]
    up = np.asarray(cmd['up_proxy'], float); dn = np.asarray(cmd['down_proxy'], float)
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, s)
    target = np.asarray(target, float)[:288]; tol = np.asarray(tol, float)[:288]
    cfg = BiddingLPConfig(assessment_band_fraction=float(fixed_bid['assessment_band_fraction']), apply_transition_band=bool(fixed_bid['apply_transition_band']))
    _, _, lo, hi = fixed_bid_tracking_bands(cfg, np.asarray(fixed_bid['baseline_plan'], float), np.asarray(fixed_bid['up_plan'], float), np.asarray(fixed_bid['down_plan'], float), up, dn, apply_transition_band=cfg.apply_transition_band)
    en = np.asarray(arr.get('tracking_enabled_series'), bool)[:288]
    diff = np.maximum(np.abs(lo - (target - tol)), np.abs(hi - (target + tol)))
    diff[~en] = 0
    idx = np.where(diff > 1e-3)[0]
    print(f"== {entry['service_date']} scenario {s} {row['command_source']} feasible={row['feasible']} steps differing {len(idx)}")
    print('   up_proxy max', round(up.max(), 3), 'down_proxy max', round(dn.max(), 3), 'both>0 steps', int(((up > 1e-9) & (dn > 1e-9)).sum()))
    for t in idx[:4]:
        b = t // 6
        print(f"   t={t} up={up[t]:.3f} down={dn[t]:.3f} upkw={fixed_bid['up_plan'][b]:.1f} downkw={fixed_bid['down_plan'][b]:.1f} base={fixed_bid['baseline_plan'][b]:.1f} | env target={target[t]:.1f} tol={tol[t]:.1f} | bid lo={lo[t]:.1f} hi={hi[t]:.1f}")

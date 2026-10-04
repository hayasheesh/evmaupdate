"""市場ごとの360組（検証日5日 × 未見指令24本）について、指令と入札の特徴を1組ずつ数える。

使う組は制御の比較（unseen_decomposition_*）と同じで、その市場の common.py の day_cases() から作る。
EV には依存しない量だけを数える（EV 実現3本は3市場で同じ seed なので、市場の差は指令と入札だけから来る）。

  python features.py --market aemo|ercot|gb

指令から目標を作る式は training.lower_bid_training._bid_to_target_tol_from_activation と同じ：
  ずれ（kW、需要の増加が正）= 下げ約定量 × 下げ利用率 − 上げ約定量 × 上げ利用率
  目標 = 基準計画 + ずれ、許容幅 = 帯の割合 × その向きの約定量（指令が0のときは別の幅）
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E = HERE.parent
COMMON = {
    'aemo': E / 'unseen_decomposition_aemo_20261002',
    'ercot': E / 'unseen_decomposition_ercot_20261002',
    'gb': E / 'unseen_decomposition_gb_20261003',
    'pjm': E / 'unseen_decomposition_pjm_20261004',
}
DT_H = 5.0 / 60.0
STEPS_PER_BLOCK = 6
EPS = 1e-6


def runs(sign: np.ndarray) -> list[tuple[int, int]]:
    """同じ向き（+1/−1）が続く区間の (向き, 長さ) の並び。0 は区切りとして数えない。"""
    out: list[tuple[int, int]] = []
    for s in sign:
        if s == 0:
            continue
        if out and out[-1][0] == s:
            out[-1] = (s, out[-1][1] + 1)
        else:
            out.append((int(s), 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=tuple(COMMON), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(COMMON[args.market]))
    common = importlib.import_module('common')
    common.configure()
    from training.lower_bid_training import build_fixed_upper_bid_training_episode

    rows = []
    for day_index, entry, fixed_bid in common.day_cases():
        baseline = np.asarray(fixed_bid['baseline_plan'], dtype=float).reshape(-1)[:48]
        up = np.asarray(fixed_bid['up_plan'], dtype=float).reshape(-1)[:48]
        down = np.asarray(fixed_bid['down_plan'], dtype=float).reshape(-1)[:48]
        live = (up > EPS) | (down > EPS)
        for s, command in enumerate(fixed_bid['activation_scenario_payload']):
            target, tol, arr, _info = build_fixed_upper_bid_training_episode(fixed_bid, s)
            target = np.asarray(target, dtype=float)[:288]
            tol = np.asarray(tol, dtype=float)[:288]
            enabled = np.asarray(arr['tracking_enabled_series'], dtype=bool)[:288]
            up_p = np.asarray(command['up_proxy'], dtype=float)[:288]
            down_p = np.asarray(command['down_proxy'], dtype=float)[:288]
            base_steps = np.repeat(baseline, STEPS_PER_BLOCK)[:288]
            reg = (target - base_steps) * enabled
            sign = np.where(reg > EPS, 1, np.where(reg < -EPS, -1, 0))
            r = runs(sign[enabled])
            run_len = [n for _s, n in r]
            cum = np.cumsum(reg) * DT_H
            net_util = (down_p - up_p)[enabled]
            # 基準計画のままでも帯に入るか（|ずれ| ≤ 許容幅）。入らない分が、実際に動かす必要のある量
            beyond = np.where(enabled, np.clip(np.abs(reg) - tol, 0.0, None), 0.0)
            need_up = np.where(reg < 0, beyond, 0.0)
            need_down = np.where(reg > 0, beyond, 0.0)
            need_cum = np.cumsum(need_down - need_up) * DT_H
            big = enabled & (np.abs(reg) > 1.0)
            rows.append({
                'market': args.market, 'bid_day_index': day_index, 'service_date': entry['service_date'],
                'scenario': s, 'command_source': str(command.get('source', '')),
                # 入札（その日で共通）
                'bid_blocks': int(live.sum()),
                'bid_up_kw_mean': float(up[live].mean()) if live.any() else 0.0,
                'bid_down_kw_mean': float(down[live].mean()) if live.any() else 0.0,
                'bid_baseline_kw_mean': float(baseline.mean()),
                'bid_min_award_share': float(((up[live] > EPS) & (up[live] <= 250.0 + 1e-6)).mean()
                                             + ((down[live] > EPS) & (down[live] <= 250.0 + 1e-6)).mean()) / 2.0
                                       if live.any() else 0.0,
                # 指令の形（利用率、約定量によらない）
                'assessed_steps': int(enabled.sum()),
                'up_active_frac': float((up_p[enabled] > EPS).mean()) if enabled.any() else 0.0,
                'down_active_frac': float((down_p[enabled] > EPS).mean()) if enabled.any() else 0.0,
                'idle_frac': float((sign[enabled] == 0).mean()) if enabled.any() else 0.0,
                'up_util_when_active': float(up_p[enabled][up_p[enabled] > EPS].mean()) if (up_p[enabled] > EPS).any() else 0.0,
                'down_util_when_active': float(down_p[enabled][down_p[enabled] > EPS].mean()) if (down_p[enabled] > EPS).any() else 0.0,
                'direction_switches': max(len(r) - 1, 0),
                'longest_run_steps': int(max(run_len)) if run_len else 0,
                'mean_run_steps': float(np.mean(run_len)) if run_len else 0.0,
                'util_ramp_mean': float(np.mean(np.abs(np.diff(net_util)))) if net_util.size > 1 else 0.0,
                # 指令 × 入札（kW、kWh）
                'reg_abs_mean_kw': float(np.abs(reg[enabled]).mean()) if enabled.any() else 0.0,
                'reg_abs_max_kw': float(np.abs(reg).max()),
                'reg_ramp_mean_kw': float(np.mean(np.abs(np.diff(reg[enabled])))) if enabled.sum() > 1 else 0.0,
                'energy_up_kwh': float(np.clip(-reg, 0, None).sum() * DT_H),
                'energy_down_kwh': float(np.clip(reg, 0, None).sum() * DT_H),
                'energy_net_kwh': float(reg.sum() * DT_H),
                'cum_min_kwh': float(cum.min()),
                'cum_max_kwh': float(cum.max()),
                'cum_range_kwh': float(cum.max() - cum.min()),
                # 帯
                'tol_mean_kw': float(tol[enabled].mean()) if enabled.any() else 0.0,
                'tol_over_reg_active': float(tol[big].mean() / np.abs(reg[big]).mean()) if big.any() else float('nan'),
                'in_band_at_baseline_share': float((beyond[enabled] <= 0.0).mean()) if enabled.any() else 0.0,
                'need_move_kw_mean': float(beyond[enabled].mean()) if enabled.any() else 0.0,
                'need_up_kwh': float(need_up.sum() * DT_H),
                'need_down_kwh': float(need_down.sum() * DT_H),
                'need_cum_min_kwh': float(need_cum.min()),
                'need_cum_max_kwh': float(need_cum.max()),
                'need_cum_range_kwh': float(need_cum.max() - need_cum.min()),
                'target_mean_kw': float(target[enabled].mean()) if enabled.any() else 0.0,
                'target_min_kw': float(target[enabled].min()) if enabled.any() else 0.0,
                'target_max_kw': float(target[enabled].max()) if enabled.any() else 0.0,
            })
    out = HERE / f'features_{args.market}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows), flush=True)


if __name__ == '__main__':
    main()

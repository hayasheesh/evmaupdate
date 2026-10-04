"""各時刻に全EVを見て解く中央LP（execute_results/ft_sweep/milp_agent.py と同じ定式化）。

元の定式化（horizon 1、切替制約なし）
  変数    各EVの電力 x_j、出発時のSoC不足 u_j、指令とのずれ e
  目的    w_ag * e / (27.5 kW * N) + w_soc * sum(u_j) / (容量 * N)、w_ag = 1、w_soc = 100
  制約    e >= |sum(x_j) - その時刻の指令|、SoC は 0〜満充電、
          その日のうちに出発するEVだけ u_j >= 目標SoC - (今のSoC + 1ステップ分の充放電)
今の環境に合わせて直したのは単位だけ。
  - SoC は % で持つ。1ステップの変化は x_j * Δt * (100 / そのEVの容量 kWh)
  - 出力の上限はEVごとの値。行動は x_j / そのEVの上限 にする（環境がそのEVの上限を掛ける）
  - SoC不足の正規化は、元の「kWh / 100 kWh」を「% / 100」にした（100 kWh のEVで元と同じ）
中央が全EVの状態を毎時刻集めるので、分散制御の実行時の条件は満たさない比較用の制御。
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linprog

from Config import MAX_EV_POWER_KW, POWER_TO_ENERGY

W_AG = 1.0
W_SOC = 100.0


class CentralLPAgent:
    def __init__(self) -> None:
        self.test_mode = True
        self.solve_failures = 0
        self.solves = 0

    def set_test_mode(self, mode: bool) -> None:
        self.test_mode = bool(mode)

    def update_active_evs(self, env) -> None:
        return

    def act(self, state, env=None, noise: bool = False):
        step = int(env.step_count)
        idx = step - 1
        command = float(env.net_demand_series[idx].item()) if 0 <= idx < len(env.net_demand_series) else 0.0
        episode_steps = int(getattr(env, 'episode_steps', 288))
        evs = []  # (station, column, soc %, target %, pct per kWh, power limit kW, departs in the day)
        for st in range(env.num_stations):
            ordered = env._get_sorted_active_evs(st)
            for col, slot_t in enumerate(ordered.tolist()):
                slot = int(slot_t)
                remaining = int(env.depart[st, slot].item()) - step + 1
                evs.append((
                    st, col,
                    float(env.soc[st, slot].item()),
                    float(env.target[st, slot].item()),
                    float(env.ev_soc_pct_per_kwh[st, slot].item()),
                    float(env.ev_max_power_kw[st, slot].item()),
                    remaining > 0 and step + remaining <= episode_steps,
                ))
        actions = torch.zeros((env.num_stations, env.max_ev_per_station), dtype=torch.float32, device=env.soc.device)
        n = len(evs)
        if n == 0:
            return actions
        dep = [i for i, ev in enumerate(evs) if ev[6]]
        m = len(dep)
        # variables: x (n), u (m), e (1)
        nv = n + m + 1
        norm = max(1.0, float(n))
        c = np.zeros(nv)
        c[n:n + m] = W_SOC / (100.0 * norm)
        c[-1] = W_AG / (MAX_EV_POWER_KW * norm)
        bounds = []
        for st, col, soc, target, pct_per_kwh, pmax, _ in evs:
            gain = POWER_TO_ENERGY * pct_per_kwh  # % per kW for one step
            lo = max(-pmax, (0.0 - soc) / gain) if gain > 0 else -pmax
            hi = min(pmax, (100.0 - soc) / gain) if gain > 0 else pmax
            bounds.append((min(lo, hi), max(lo, hi)))
        bounds += [(0.0, None)] * m + [(0.0, None)]
        a_ub = np.zeros((m + 2, nv))
        b_ub = np.zeros(m + 2)
        for r, i in enumerate(dep):
            st, col, soc, target, pct_per_kwh, pmax, _ = evs[i]
            # target - (soc + gain x) <= u  ->  -gain x - u <= soc - target
            a_ub[r, i] = -POWER_TO_ENERGY * pct_per_kwh
            a_ub[r, n + r] = -1.0
            b_ub[r] = soc - target
        # sum x - command <= e ; command - sum x <= e
        a_ub[m, :n] = 1.0
        a_ub[m, -1] = -1.0
        b_ub[m] = command
        a_ub[m + 1, :n] = -1.0
        a_ub[m + 1, -1] = -1.0
        b_ub[m + 1] = -command
        self.solves += 1
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0 or res.x is None:
            self.solve_failures += 1
            return actions
        for i, (st, col, _soc, _target, _pct, pmax, _) in enumerate(evs):
            if pmax > 0:
                actions[st, col] = float(np.clip(res.x[i] / pmax, -1.0, 1.0))
        return actions

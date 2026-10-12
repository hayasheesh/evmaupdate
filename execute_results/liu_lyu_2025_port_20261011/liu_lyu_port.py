"""Liu, Lyu ら（IEEE PowerTech 2025、https://github.com/Rick10119/Integrating-Fast-response-Capability-into-Virtual-Power-Plant-Operation-for-Ancillary-Services）
の入札・計画と実時間の配分を、このリポジトリの環境（5分刻み、30分コマ、EVだけ）に移したもの。

元のコードとの対応（元は MATLAB + YALMIP + Gurobi。ここは scipy の HiGHS）:
  - plan_block      : maxProfit_t.m（コマの初めに、残りのコマのシナリオ別の電力と電池残量の計画を解き直す。
                      入札は固定。今の電池残量の等式の双対変数を、配分の順位に使う）
  - day_ahead_bid   : maxProfit_1.m（日前に、電力量と上下対称の調整容量を決める。ランプの機会制約つき）
  - Allocator       : fastControl_prepare.m + fastControl_implement.m（各時刻、計画のシナリオの範囲を
                      各EVの上下限にし、「費用 − η·λ」の安い順に、指令までを詰める）

元のまま残したもの: シナリオは指令を0.1刻みにした22通りと基準（0）。各シナリオで群の電力が指令に一致すること。
電池残量は確率で重みを付けた期待値で進める。10分間の持続（delta_t_req = 1/6 h）。費用（放電100 $/MWh、充電0）。
電池残量の上下限を外したときの罰（1e4 $/MWh と 1e2 $/MWh）。配分の上下限（式14）と、充放電を同時に行わない規則、
電池残量が下限を割ったときの順位の上書き（λ = −10、上限を越えたとき λ = 1e3）。

この環境に寄せたもの（理由は各所の注記）:
  - 時間: 元の1時間のスロットを30分コマに、2秒の配分を5分に。計画はコマの初めに1回（元の main_seperate.m と同じ頻度）
  - 符号: 元は放電（発電）を正とする。環境は充電を正とするので、出入口で反転する
  - EV: 環境の各EV（定格7.7 kW、充放電とも効率1、電池残量の上限100%、目標は出発時のSoC）。
        元は電池残量の上下限を容量の10%と90%に置き、出発の1〜2時間前から下限を目標まで上げる。ここでは環境のEVに合わせ、
        下限を「残りの時間ずっと定格で充電して目標に届く最低の量」（元の1〜2時間前の段を、出発までの全時間に延ばしたもの）、
        上限を容量（100%）にする
  - 将来のEV: 元は全日のEVの到着・出発を前もって知っている。ここではまだ到着していないEVを、入札器が前提にした
        EVの3通りの実現（bid_ev_scenario_bank）で補い、それぞれ重み1/3で入れる
  - 指令の分布: 元は過去14日のRegDの1時間ごとの分布。ここでは入札器の学習用の指令（design_activation_scenario_payload、
        評価に使う指令とは別）から30分コマごとの分布を作る。指令がちょうど0の時刻は基準のシナリオ（0）に入れる
        （元のコードでは0は −0.05 の区間に入るが、この環境の指令は0の時刻が多いため）
  - コマの途中で到着したEVは計画を持たないので、上下限を定格の範囲、λ = 0 にする
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

STEPS_PER_BLOCK = 6
N_BLOCKS = 48
STEP_H = 5.0 / 60.0
BLOCK_H = 0.5
DT_REQ_H = 1.0 / 6.0
PR_DIS = 100.0 / 1000.0     # $/kWh（元 100 $/MWh）
PR_CH = 0.0
PEN_E12 = 1e4 / 1000.0      # $/kWh（元 1e4 $/MWh、電池残量の上下限）
PEN_E34 = 1e2 / 1000.0      # $/kWh（元 1e2 $/MWh、10分の持続）
PEN_BALANCE = 1e3           # $/kW・コマ。元は等式だが、実際のEVと予測のEVが食い違うと解けないので罰つきにする
LAMBDA_ABOVE = 1e3          # 元の fastControl_prepare.m
LAMBDA_BELOW = -1e1

D_S = np.concatenate([[-1.0], np.arange(-0.95, 0.951, 0.1), [1.0]])   # 22通り
SCEN = np.concatenate([D_S[:11], [0.0], D_S[11:]])                   # 基準（0）を真ん中に足して23通り
BASE_S = 11


def signal_bins(y):
    """指令の正規化値（−1〜1）を元のコードの22区間に入れる。0ちょうどは基準（0）。"""
    y = np.asarray(y, dtype=float)
    idx = np.empty(y.shape, dtype=int)
    zero = np.abs(y) <= 1e-9
    pos = (y > 0) & ~zero
    neg = (y < 0) & ~zero
    # 元: s>=0 なら ceil(s/0.1)+1/0.1+1（1始まり）, s<0 なら floor(s/0.1)+1/0.1+2
    ip = np.ceil(y[pos] / 0.1).astype(int) + 10 + 1 - 1   # 0始まりの D_S の番号
    ineg = np.floor(y[neg] / 0.1).astype(int) + 10 + 2 - 1
    ip[y[pos] > 0.9999] = 21
    ineg[y[neg] < -0.9999] = 0
    d_idx = np.empty(y.shape, dtype=int)
    d_idx[pos] = np.clip(ip, 0, 21)
    d_idx[neg] = np.clip(ineg, 0, 21)
    # D_S の番号を SCEN の番号へ（11 以上は基準の分だけずれる）
    idx[pos] = np.where(d_idx[pos] >= 11, d_idx[pos] + 1, d_idx[pos])
    idx[neg] = np.where(d_idx[neg] >= 11, d_idx[neg] + 1, d_idx[neg])
    idx[zero] = BASE_S
    return idx


def scenario_probabilities(payloads):
    """学習用の指令から、30分コマごとのシナリオの確率（48×23）を作る。"""
    counts = np.zeros((N_BLOCKS, SCEN.size))
    for p in payloads:
        y = np.asarray(p['down_proxy'], float)[:N_BLOCKS * STEPS_PER_BLOCK] - np.asarray(p['up_proxy'], float)[:N_BLOCKS * STEPS_PER_BLOCK]
        b = signal_bins(y)
        for k in range(N_BLOCKS):
            np.add.at(counts[k], b[k * STEPS_PER_BLOCK:(k + 1) * STEPS_PER_BLOCK], 1.0)
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


def ramp_quantiles(payloads, c_up=0.95, c_dn=0.05):
    """元の data_prepare_ramp.m: 1分あたりの指令の変化の分位点。ここは5分刻みなので、1ステップの変化を5で割る。"""
    up = np.zeros(N_BLOCKS)
    dn = np.zeros(N_BLOCKS)
    ys = np.stack([np.asarray(p['down_proxy'], float)[:288] - np.asarray(p['up_proxy'], float)[:288] for p in payloads])
    d = np.diff(ys, axis=1, prepend=ys[:, :1]) / 5.0
    for k in range(N_BLOCKS):
        v = d[:, k * STEPS_PER_BLOCK:(k + 1) * STEPS_PER_BLOCK].reshape(-1)
        up[k] = np.quantile(v, c_up)
        dn[k] = np.quantile(v, c_dn)
    return up, dn


def reg_kw(baseline, up, down, k, s_vals):
    """指令（充電を正）。target = baseline + down·y (y>0) / up·y (y<0)。"""
    s_vals = np.asarray(s_vals, float)
    return baseline[k] + np.where(s_vals > 0, down[k] * s_vals, up[k] * s_vals)


class _Builder:
    def __init__(self):
        self.n = 0
        self.lb, self.ub, self.c = [], [], []
        self.eq_r, self.eq_c, self.eq_v, self.eq_b = [], [], [], []
        self.ub_r, self.ub_c, self.ub_v, self.ub_b = [], [], [], []
        self.n_eq = 0
        self.n_ub = 0

    def var(self, size, lb=0.0, ub=np.inf, cost=0.0):
        i0 = self.n
        self.n += int(size)
        self.lb.append(np.broadcast_to(np.asarray(lb, float), (size,)).copy())
        self.ub.append(np.broadcast_to(np.asarray(ub, float), (size,)).copy())
        self.c.append(np.broadcast_to(np.asarray(cost, float), (size,)).copy())
        return np.arange(i0, i0 + size)

    def eq(self, cols, vals, rhs):
        r = self.n_eq
        self.n_eq += 1
        self.eq_r.extend([r] * len(cols))
        self.eq_c.extend(cols)
        self.eq_v.extend(vals)
        self.eq_b.append(rhs)
        return r

    def le(self, cols, vals, rhs):
        r = self.n_ub
        self.n_ub += 1
        self.ub_r.extend([r] * len(cols))
        self.ub_c.extend(cols)
        self.ub_v.extend(vals)
        self.ub_b.append(rhs)
        return r

    def solve(self):
        c = np.concatenate(self.c)
        bounds = np.stack([np.concatenate(self.lb), np.concatenate(self.ub)], axis=1)
        A_eq = sp.csr_matrix((self.eq_v, (self.eq_r, self.eq_c)), shape=(self.n_eq, self.n)) if self.n_eq else None
        A_ub = sp.csr_matrix((self.ub_v, (self.ub_r, self.ub_c)), shape=(self.n_ub, self.n)) if self.n_ub else None
        res = linprog(c, A_ub=A_ub, b_ub=np.asarray(self.ub_b) if self.n_ub else None,
                      A_eq=A_eq, b_eq=np.asarray(self.eq_b) if self.n_eq else None,
                      bounds=[tuple(b) for b in bounds], method='highs')
        if res.status != 0:
            raise RuntimeError(f'LP failed: {res.message}')
        return res


def _lower_kwh(ev, step):
    """時刻 step（ループの番号）での電池残量の下限：残りずっと定格で充電して目標に届く最低の量。"""
    if not ev['target_required']:
        return 0.0
    left_h = max(ev['dep'] - step, 0) * STEP_H
    return max(0.0, ev['target_kwh'] - ev['P'] * left_h)


def _ev_blocks(ev, first_block):
    k0 = max(first_block, ev['arr'] // STEPS_PER_BLOCK)
    k1 = min((ev['dep'] - 1) // STEPS_PER_BLOCK, N_BLOCKS - 1)
    out = []
    for k in range(k0, k1 + 1):
        a = (min(ev['dep'], (k + 1) * STEPS_PER_BLOCK) - max(ev['arr'], k * STEPS_PER_BLOCK)) / STEPS_PER_BLOCK
        if a > 0:
            out.append((k, a))
    return out


def _add_fleet(B, evs, first_block, probs, ramp=False):
    """EVごとのシナリオ別の電力と電池残量の変数・制約を足し、コマ×シナリオの群の電力（発電を正）の係数を返す。"""
    S = SCEN.size
    fleet = {}      # (k) -> list of (col_dis[S], col_ch[S])
    per_ev = []
    for ev in evs:
        blocks = _ev_blocks(ev, first_block)
        if not blocks:
            per_ev.append(None)
            continue
        w = ev['w']
        rec = {'blocks': blocks, 'dis': {}, 'ch': {}, 'E': {}, 'row_e0': None}
        kfirst = blocks[0][0]
        E = B.var(len(blocks) + 1, lb=-np.inf, ub=np.inf)
        rec['E_cols'] = E
        e0 = w * ev['E0']
        rec['row_e0'] = B.eq([int(E[0])], [1.0], e0)
        for j, (k, a) in enumerate(blocks):
            pmax = w * ev['P'] * a
            xd = B.var(S, 0.0, pmax, cost=0.0)
            xc = B.var(S, 0.0, pmax, cost=0.0)
            # 期待費用（元: hourly_Distribution .* Cost_deg）
            for s in range(S):
                B.c[-2][s] = probs[k, s] * BLOCK_H * PR_DIS
                B.c[-1][s] = probs[k, s] * BLOCK_H * PR_CH
            rec['dis'][k] = xd
            rec['ch'][k] = xc
            fleet.setdefault(k, []).append((xd, xc))
            # 期待値で電池残量を進める（効率1、維持1）
            cols = [int(E[j + 1]), int(E[j])] + list(map(int, xc)) + list(map(int, xd))
            vals = [1.0, -1.0] + list(-BLOCK_H * probs[k]) + list(BLOCK_H * probs[k])
            B.eq(cols, vals, 0.0)
            # 上下限（罰つき、元の delta_E1, delta_E2）
            t_end = min((k + 1) * STEPS_PER_BLOCK, ev['dep'])
            lo = w * _lower_kwh(ev, t_end)
            hi = w * ev['cap']
            d1, d2 = B.var(2, 0.0, np.inf, cost=PEN_E12)
            B.le([int(E[j + 1]), int(d1)], [-1.0, -1.0], -lo)
            B.le([int(E[j + 1]), int(d2)], [1.0, -1.0], hi)
            # 10分の持続（元の Energy reserve ratio、罰つき delta_E3, delta_E4）
            t_start = max(k * STEPS_PER_BLOCK, ev['arr'])
            lo0 = w * _lower_kwh(ev, t_start)
            for s in range(S):
                d3, d4 = B.var(2, 0.0, np.inf, cost=PEN_E34)
                # E - dt_req*(dis - ch) >= lo0 - d3
                B.le([int(E[j]), int(xd[s]), int(xc[s]), int(d3)], [-1.0, DT_REQ_H, -DT_REQ_H, -1.0], -lo0)
                B.le([int(E[j]), int(xd[s]), int(xc[s]), int(d4)], [1.0, -DT_REQ_H, DT_REQ_H, -1.0], hi)
            if ramp:
                rec.setdefault('R', {})[k] = B.var(2, 0.0, 12.0 * pmax)   # 元: ランプは定格の12倍/分
        per_ev.append(rec)
    return fleet, per_ev


def plan_block(evs, block, baseline, up, down, probs):
    """入札を固定し、コマ block から最後までの計画を解く（元の maxProfit_t.m）。

    evs: dict(key, w, P, cap, E0, arr, dep, target_kwh, target_required, present)
    返り値: present なEVごとの、今のコマのシナリオ別の放電・充電（コマ平均の kW、重み w=1）と、
    電池残量の限界価値 v（= −λ。元のコードの λ は −v）。
    """
    B = _Builder()
    fleet, per_ev = _add_fleet(B, evs, block, probs)
    S = SCEN.size
    slack_used = {}
    for k in range(block, N_BLOCKS):
        target = reg_kw(baseline, up, down, k, SCEN)      # 充電を正
        for s in range(S):
            cols, vals = [], []
            for xd, xc in fleet.get(k, []):
                cols += [int(xd[s]), int(xc[s])]
                vals += [1.0, -1.0]
            sp_, sn_ = B.var(2, 0.0, np.inf, cost=PEN_BALANCE)
            cols += [int(sp_), int(sn_)]
            vals += [1.0, -1.0]
            B.eq(cols, vals, -float(target[s]))
            slack_used[(k, s)] = (sp_, sn_)
    res = B.solve()
    x = res.x
    marg = res.eqlin.marginals
    out = {}
    for ev, rec in zip(evs, per_ev):
        if rec is None or not ev['present']:
            continue
        v = -float(marg[rec['row_e0']])
        entry = {'v': v}
        if block in rec['dis']:
            a = dict(rec['blocks'])[block]
            entry['dis'] = x[rec['dis'][block]] / max(a, 1e-9)
            entry['ch'] = x[rec['ch'][block]] / max(a, 1e-9)
        out[ev['key']] = entry
    slack_kw = float(sum(x[a] + x[b] for (k, s), (a, b) in slack_used.items() if k == block and s != BASE_S))
    return out, {'objective': float(res.fun), 'slack_current_block_kw': slack_kw, 'n_vars': int(B.n)}


def day_ahead_bid(evs, probs, price_e, price_reg, price_mil_unused, eps_up, eps_dn, min_bid_kw, s_perf=0.984, ramp=True):
    """元の maxProfit_1.m を、予備力を除き、30分コマ・EVだけで解く。

    price_e, price_reg は $/MWh（48コマ）。発電を正で、Bid_p は電力量、Bid_reg は上下対称の調整容量。
    ランプの機会制約: 元の h_R = 2 / min(max(|eps_dn|, eps_up)) は、指令が0のコマがあると無限大になるので、
    指令が動くコマだけで最小を取る。
    返り値: 充電を正にした baseline_plan, up_plan, down_plan（48）。
    ramp=False でランプの機会制約を外す（5分の指令では h_R が意味を持たないため、比較用）。
    """
    B = _Builder()
    S = SCEN.size
    fleet, per_ev = _add_fleet(B, evs, 0, probs, ramp=ramp)
    bid_p = B.var(N_BLOCKS, -np.inf, np.inf)
    bid_reg = B.var(N_BLOCKS, 0.0, np.inf)
    pe = np.asarray(price_e, float) / 1000.0
    pr = np.asarray(price_reg, float) / 1000.0
    ds_mean = probs @ SCEN
    # 利益の最大化 = 費用の最小化（−利益）
    for k in range(N_BLOCKS):
        B.c[-2][k] = -pe[k] * BLOCK_H
        B.c[-1][k] = -(pr[k] * s_perf + ds_mean[k] * pe[k]) * BLOCK_H
    moving = np.maximum(np.abs(eps_dn), eps_up)
    h_r = 2.0 / float(np.min(moving[moving > 1e-9]))
    for k in range(N_BLOCKS):
        for s in range(S):
            cols, vals = [], []
            for xd, xc in fleet.get(k, []):
                cols += [int(xd[s]), int(xc[s])]
                vals += [1.0, -1.0]
            sp_, sn_ = B.var(2, 0.0, np.inf, cost=PEN_BALANCE)
            cols += [int(bid_p[k]), int(bid_reg[k]), int(sp_), int(sn_)]
            vals += [-1.0, -float(SCEN[s]), 1.0, -1.0]
            B.eq(cols, vals, 0.0)
        # ランプ（元の 10a と容量の制約、予備力なし）
        rcols_up, rcols_dn = [], []
        for rec in per_ev:
            if rec is None or k not in rec.get('R', {}):
                continue
            r_dn, r_up = rec['R'][k]
            xd, xc = rec['dis'][k], rec['ch'][k]
            # Cap_up + Cap_dn = P_1 - P_minus1 = (xd[-1]-xc[-1]) - (xd[0]-xc[0])
            for r in (r_dn, r_up):
                B.le([int(r), int(xd[-1]), int(xc[-1]), int(xd[0]), int(xc[0])], [h_r, -1.0, 1.0, 1.0, -1.0], 0.0)
            rcols_up.append(int(r_up))
            rcols_dn.append(int(r_dn))
        if ramp and rcols_up:
            B.le(rcols_up + [int(bid_reg[k])], [-1.0] * len(rcols_up) + [max(eps_up[k], 0.0)], 0.0)
            B.le(rcols_dn + [int(bid_reg[k])], [-1.0] * len(rcols_dn) + [max(-eps_dn[k], 0.0)], 0.0)
    res = B.solve()
    x = res.x
    p = x[bid_p]
    r = x[bid_reg]
    r = np.where(r >= min_bid_kw, r, 0.0)   # この市場の最低入札量（元のモデルには無い）
    return {'baseline_plan': -p, 'up_plan': r.copy(), 'down_plan': r.copy(), 'raw_bid_reg': x[bid_reg], 'h_R': h_r,
            'objective': float(res.fun), 'n_vars': int(B.n)}


class Allocator:
    """元の fastControl_prepare.m と fastControl_implement.m（発電を正）。"""

    def __init__(self):
        self.plan = {}
        self.p_cur = {}     # key -> (dis, ch) 直前の時刻の電力

    def set_plan(self, plan, block):
        self.plan = plan
        self.block = block
        # コマの初めは基準のシナリオの計画に戻す（元: delta_t_rest == 1 のとき P_bl）
        for key, e in plan.items():
            if 'dis' in e:
                bl = e['dis'][BASE_S] - e['ch'][BASE_S]
                self.p_cur[key] = (max(bl, 0.0), max(-bl, 0.0))

    def allocate(self, evs, step, request_gen_kw, last_block=False):
        """evs: 今いるEV（key, P, cap, E, dep, target_kwh, target_required）。返り値: EVごとの充電を正にした電力。"""
        seg = []
        for i, ev in enumerate(evs):
            key, P = ev['key'], ev['P']
            e = self.plan.get(key)
            if e is not None and 'dis' in e:
                pd_lo, pd_hi = float(np.min(e['dis'])), float(np.max(e['dis']))
                pc_lo, pc_hi = float(np.min(e['ch'])), float(np.max(e['ch']))
                v = 0.0 if last_block else e['v']     # 元: 最後のスロットは λ = 0
            else:
                pd_lo, pd_hi, pc_lo, pc_hi, v = 0.0, P, 0.0, P, 0.0
            pd_lo, pd_hi, pc_lo, pc_hi = (min(max(x, 0.0), P) for x in (pd_lo, pd_hi, pc_lo, pc_hi))
            dis_prev, ch_prev = self.p_cur.get(key, (0.0, 0.0))
            if ch_prev > 1e-9:
                pd_hi = 0.0
            if dis_prev > 1e-9:
                pc_hi = 0.0
            lo_end = _lower_kwh(ev, min((step // STEPS_PER_BLOCK + 1) * STEPS_PER_BLOCK, ev['dep']))
            if ev['E'] < 1e-1 * lo_end:          # 元: 下限を大きく割ったら最大で充電
                pc_lo = pc_hi
                pd_hi = pd_lo
            pd_lo = min(pd_lo, pd_hi)
            pc_lo = min(pc_lo, pc_hi)
            lam = -v
            if ev['E'] > 0.99 * ev['cap']:
                lam = LAMBDA_ABOVE
            if ev['E'] < lo_end:
                lam = LAMBDA_BELOW
            seg.append((i, 0, PR_DIS - lam, pd_lo, pd_hi))
            seg.append((i, 1, -PR_CH - lam, -pc_hi, -pc_lo))
        seg.sort(key=lambda r: r[2])
        alloc = np.array([r[3] for r in seg], float)
        delta = request_gen_kw - alloc.sum()
        kdx = 0
        while delta > 0 and kdx < len(seg):
            room = seg[kdx][4] - alloc[kdx]
            if room < delta:
                alloc[kdx] = seg[kdx][4]
                delta -= room
            else:
                alloc[kdx] += delta
                delta = 0.0
            kdx += 1
        dis = np.zeros(len(evs))
        ch = np.zeros(len(evs))
        for r, a in zip(seg, alloc):
            if r[1] == 0:
                dis[r[0]] = a
            else:
                ch[r[0]] = -a
        for i, ev in enumerate(evs):
            self.p_cur[ev['key']] = (dis[i], ch[i])
        return ch - dis

"""能力の定義を4通りで測り、過大約定のどこまでが自前の保守性かを分ける。

A 現行   : 全ブロック接続 かつ min(電力, エネルギー)   ← 入札器が使っている定義
B 電力のみ: 全ブロック接続、エネルギー要件を外す
C 開始時点: ブロック開始時に接続、電力のみ（直前計測型の基準値に近い）
D 重なり : ブロックに少しでも重なれば算入、電力のみ（上限側）

市場の式（合計基準値電力－合計需要抑制計画電力）には継続要件もエネルギー
要件もない。A→D で、86.4% の不足がどこから来ているかが分かる。
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

DAY, FORECAST_SEED, EVAL_BASE, N_DRAWS = "2024-12-04", 1_076_030, 910_000, 24

from market.sustained_capability import STEPS_PER_BLOCK, N_BLOCKS, BLOCK_HOURS, _energy_envelope
from market.physical_lp_bidding import sample_ev_specs_from_evenv
from environment.arrival_context import (
    ArrivalScenarioSampler, perturb_synthetic_arrival_probabilities)

fixed_bid = pickle.loads((HERE / "fixed_bid_2024-12-04.pkl").read_bytes())
up_award = np.asarray(fixed_bid["up_plan"], float)
down_award = np.asarray(fixed_bid["down_plan"], float)
part = (up_award > 1e-8) | (down_award > 1e-8)
print(f"参加ブロック {int(part.sum())}/48", flush=True)


def capabilities(evs):
    """4定義ぶんの (up, down) を返す。"""
    out = {k: (np.zeros(N_BLOCKS), np.zeros(N_BLOCKS)) for k in "ABCD"}
    for block in range(N_BLOCKS):
        s, e = block * STEPS_PER_BLOCK, block * STEPS_PER_BLOCK + STEPS_PER_BLOCK
        for ev in evs:
            a, d = int(ev.arrival_t), int(ev.departure_t)
            whole = (a <= s and d >= e)
            at_start = (a <= s < d)
            overlap = (a < e and d > s)
            if not overlap:
                continue
            ckw = max(float(ev.max_charge_kw), 0.0)
            dkw = max(float(ev.max_discharge_kw), 0.0)
            cap = max(float(ev.capacity_kwh), 0.0)
            if whole:
                e_lo_s, e_hi_s = _energy_envelope(ev, s)
                e_lo_e, _ = _energy_envelope(ev, e)
                up_e = max(e_hi_s - e_lo_e, 0.0) / BLOCK_HOURS
                dn_e = max(cap - e_lo_s, 0.0) / BLOCK_HOURS
                out["A"][0][block] += min(dkw, up_e)
                out["A"][1][block] += min(ckw, dn_e)
                out["B"][0][block] += dkw
                out["B"][1][block] += ckw
            if at_start:
                out["C"][0][block] += dkw
                out["C"][1][block] += ckw
            out["D"][0][block] += dkw
            out["D"][1][block] += ckw
    return out


scen = ArrivalScenarioSampler().scenario_for_day(DAY)
arr = getattr(scen, "arrival_probabilities_by_station", None)
ctx = getattr(scen, "day_context", None)

acc = {k: {"up": [], "down": []} for k in "ABCD"}
for s in range(N_DRAWS):
    evs = sample_ev_specs_from_evenv(seed=EVAL_BASE + 1000 * s,
                                     arrival_probabilities_by_station=arr,
                                     day_context=ctx)
    c = capabilities(evs)
    for k in "ABCD":
        acc[k]["up"].append(c[k][0]); acc[k]["down"].append(c[k][1])
    print(f"  draw {s+1}/{N_DRAWS} ({len(evs)} sessions)", flush=True)

LABEL = {"A": "A 現行(継続+エネルギー)", "B": "B 電力のみ(継続あり)",
         "C": "C 開始時点接続, 電力のみ", "D": "D 重なり, 電力のみ"}
res = {}
print()
for name, award in (("up", up_award), ("down", down_award)):
    m = part & (award > 1e-8)
    a = award[m]
    print(f"===== {'上げ' if name=='up' else '下げ'} =====")
    print(f"  {'定義':26s} {'能力中位/約定':>13s} {'約定の分位':>11s} {'不足割合':>9s}")
    for k in "ABCD":
        dr = np.vstack(acc[k][name])[:, m]
        ratio = float(np.median(np.median(dr, axis=0) / a))
        pct = float(np.median((dr < a[None, :]).mean(axis=0) * 100))
        short = float((dr < a[None, :]).mean())
        res[f"{name}_{k}"] = dict(ratio=ratio, pct=pct, short=short)
        print(f"  {LABEL[k]:26s} {ratio:13.3f} {pct:10.1f}% {short*100:8.1f}%")
    print()
(HERE / "cap_definitions.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("-> execute_results/ft_sweep/cap_definitions.json", flush=True)

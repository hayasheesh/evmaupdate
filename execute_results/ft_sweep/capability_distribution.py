"""制御器を通さずに、入札の約定量が当日の能力分布のどこにあるかを測る。

能力は入札器と同じ定義（sustained_capability_for_scenario、30分継続、
ブロック全体つなぎっぱなしの車両のみ寄与）。EV実現を評価と同じ24通り引く。
制御器が一切入らないので、出るのは入札の過大さだけ。
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

DAY, FORECAST_SEED, EVAL_BASE, N_DRAWS = "2024-12-04", 1_076_030, 910_000, 24

from market.physical_lp_bidding import sample_ev_specs_from_evenv
from market.sustained_capability import sustained_capability_for_scenario
from environment.arrival_context import (
    ArrivalScenarioSampler, perturb_synthetic_arrival_probabilities)

fixed_bid = pickle.loads((HERE / "fixed_bid_2024-12-04.pkl").read_bytes())
up = np.asarray(fixed_bid["up_plan"], float)
down = np.asarray(fixed_bid["down_plan"], float)
part = (up > 1e-8) | (down > 1e-8)
print(f"参加ブロック {int(part.sum())}/48", flush=True)

scen = ArrivalScenarioSampler().scenario_for_day(DAY)
arr = getattr(scen, "arrival_probabilities_by_station", None)
ctx = getattr(scen, "day_context", None)

def cap(evs):
    c = sustained_capability_for_scenario(evs, duration_hours=0.5)
    return np.asarray(c.up, float), np.asarray(c.down, float)

# 入札が根拠にした1通り（摂動あり、_sample_ev_scenario_bank と同じ）
cert_evs = sample_ev_specs_from_evenv(
    seed=FORECAST_SEED, day_context=ctx,
    arrival_probabilities_by_station=perturb_synthetic_arrival_probabilities(
        arr, seed=FORECAST_SEED) if arr is not None else None)
cu, cd = cap(cert_evs)
print(f"証明時EV実現 {len(cert_evs)}セッション", flush=True)

# 当日側（評価と同じ ensemble: 未摂動の確率から seed を変えて引く）
ups, downs, sizes = [], [], []
for s in range(N_DRAWS):
    evs = sample_ev_specs_from_evenv(seed=EVAL_BASE + 1000 * s,
                                     arrival_probabilities_by_station=arr,
                                     day_context=ctx)
    u, d = cap(evs)
    ups.append(u); downs.append(d); sizes.append(len(evs))
U = np.vstack(ups); D = np.vstack(downs)          # (N_DRAWS, 48)
print(f"当日EV実現 {N_DRAWS}通り, セッション数 {min(sizes)}-{max(sizes)}", flush=True)

out = {}
for name, award, cert, mat in (("up", up, cu, U), ("down", down, cd, D)):
    m = part & (award > 1e-8)
    a = award[m][None, :]
    ratio_cert = (cert[m] / award[m])
    ratio = mat[:, m] / a                          # (N_DRAWS, n_blocks)
    short = ratio < 1.0
    out[name] = {
        "blocks": int(m.sum()),
        "約定/証明時能力 の中位": round(float(np.median(1.0 / ratio_cert)), 3),
        "証明時能力が約定を下回るブロック": int((ratio_cert < 1.0).sum()),
        "当日能力/約定 中位": round(float(np.median(ratio)), 3),
        "同 10%点": round(float(np.quantile(ratio, 0.10)), 3),
        "同 90%点": round(float(np.quantile(ratio, 0.90)), 3),
        "能力<約定 の(ブロックx実現)割合": round(float(short.mean()), 3),
        "ブロック中位で不足する割合": round(float((np.median(ratio, axis=0) < 1.0).mean()), 3),
    }
    print(f"\n=== {name} ({int(m.sum())}ブロック x {N_DRAWS}実現) ===", flush=True)
    for k, v in out[name].items():
        if k != "blocks":
            print(f"  {k}: {v}", flush=True)

(HERE / "capability_distribution.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
np.savez(HERE / "capability_raw.npz", up_award=up, down_award=down,
         cert_up=cu, cert_down=cd, draws_up=U, draws_down=D, participating=part)
print("\n保存しました", flush=True)

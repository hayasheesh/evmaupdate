"""修正版。入札が証明に使ったEV実現を正しく再現し、対照を置く。

A0（対照）: 証明時のEV実現 x forecast(設計)指令 24本
            -> 入札の証明そのもの。ここが通らなければ再現が誤っている。
A1        : 証明時のEV実現 x holdout 指令 24本
            -> 指令が新しいことだけの影響。
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DAY, FORECAST_SEED, N = "2024-12-04", 1_076_030, 24
BID_CSV = ("archive/ft_a1_baseline_20260917_023911/results/"
           "submitted_day_ahead_bid.csv")

from market.physical_lp_bidding import BiddingLPConfig, sample_ev_specs_from_evenv
from market.physical_lp_bidding.data_classes import ActivationScenario
from training.lower_bid_training import (
    _activation_scenarios_for_day, _fixed_bid_scenario_task,
)
from environment.arrival_context import (
    ArrivalScenarioSampler, perturb_synthetic_arrival_probabilities,
)

bid = pd.read_csv(BID_CSV).sort_values("block")
baseline = bid["baseline_kw"].to_numpy(float)
up = bid["awarded_up_kw"].to_numpy(float)
down = bid["awarded_down_kw"].to_numpy(float)
cfg = BiddingLPConfig(assessment_band_fraction=0.10, apply_transition_band=False,
                      time_limit_s=600.0, colgen_cache_columns_per_ev=12)

scen = ArrivalScenarioSampler().scenario_for_day(DAY)
arr = getattr(scen, "arrival_probabilities_by_station", None)
ctx = getattr(scen, "day_context", None)

# _sample_ev_scenario_bank(count=1, seed=FORECAST_SEED, seed_offset=0) と同じ:
#   ev_seed = seed + 0 + 10007*0、arrival rate に予測誤差を注入してから抽出
ev_seed = FORECAST_SEED
perturbed = (perturb_synthetic_arrival_probabilities(arr, seed=ev_seed)
             if arr is not None else None)
certified_evs = sample_ev_specs_from_evenv(
    seed=ev_seed, arrival_probabilities_by_station=perturbed, day_context=ctx)
print(f"証明時EV実現(摂動あり): {len(certified_evs)}セッション", flush=True)

results = {}
for label, part in (("A0_設計指令", "forecast"), ("A1_holdout指令", "holdout")):
    cmds, mode = _activation_scenarios_for_day(DAY, FORECAST_SEED,
                                               n_scenarios=N,
                                               scenario_partition=part)
    rows, t0 = [], time.perf_counter()
    for s, p in enumerate(cmds):
        sc = ActivationScenario(
            name=str(p.get("name") or f"cmd{s:02d}"),
            up_signal=np.asarray(p["up_proxy"], float).reshape(-1).copy(),
            down_signal=np.asarray(p["down_proxy"], float).reshape(-1).copy(),
            evs=certified_evs)
        r = _fixed_bid_scenario_task((s, sc, cfg, baseline, up, down))
        rows.append(r)
        print(f"  [{label}] 指令{s:02d} feasible={bool(r['feasible'])}", flush=True)
    ok = sum(1 for r in rows if r["feasible"])
    results[label] = {"feasible": ok, "total": len(rows),
                      "seconds": round(time.perf_counter() - t0, 1)}
    print(f"[{label}] 実行可能 {ok}/{len(rows)} ({results[label]['seconds']}s)", flush=True)

Path("execute_results/ft_sweep/lp_probe2.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== まとめ ===")
for k, v in results.items():
    print(f"  {k}: 実行可能 {v['feasible']}/{v['total']}")

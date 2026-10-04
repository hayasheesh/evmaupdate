"""凍結した入札に、完全情報LPの実行可能な運転計画が存在するかを解く。

区別したいこと: 当日の追従不足は「入札が出せない量を約定している」のか
「制御器が事前にSoCを積まない」のか。前者なら完全情報LPでも解が無い。

2条件:
  A. 予測EV実現 x holdout 24指令  -- 指令が新しいことだけの影響
  B. 評価EV実現   x holdout 24指令  -- 指令もEVも新しい（評価と同じ条件）
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DAY = "2024-12-04"
FORECAST_SEED = 1_076_030
EVAL_BASE_SEED = 910_000
N_CMD = 24
BID_CSV = ("archive/ft_a1_baseline_20260917_023911/results/"
           "submitted_day_ahead_bid.csv")

from market.physical_lp_bidding import BiddingLPConfig, sample_ev_specs_from_evenv
from market.physical_lp_bidding.data_classes import ActivationScenario
from training.lower_bid_training import (
    _activation_scenarios_for_day, _fixed_bid_scenario_task,
)
from environment.arrival_context import ArrivalScenarioSampler

bid = pd.read_csv(BID_CSV).sort_values("block")
baseline = bid["baseline_kw"].to_numpy(float)
up = bid["awarded_up_kw"].to_numpy(float)
down = bid["awarded_down_kw"].to_numpy(float)
print(f"入札: 48ブロック 参加{int(((up>1e-8)|(down>1e-8)).sum())}  "
      f"平均 上{up.mean():.1f} 下{down.mean():.1f} kW", flush=True)

cfg = BiddingLPConfig(assessment_band_fraction=0.10, apply_transition_band=False,
                      time_limit_s=600.0, colgen_cache_columns_per_ev=12)

scen = ArrivalScenarioSampler().scenario_for_day(DAY)
arr = getattr(scen, "arrival_probabilities_by_station", None)
ctx = getattr(scen, "day_context", None)

cmds, mode = _activation_scenarios_for_day(DAY, FORECAST_SEED,
                                           n_scenarios=N_CMD,
                                           scenario_partition="holdout")
print(f"指令: {len(cmds)}本 ({mode}, holdout)", flush=True)

def ev_draw(seed: int):
    return sample_ev_specs_from_evenv(seed=int(seed),
                                      arrival_probabilities_by_station=arr,
                                      day_context=ctx)

forecast_evs = ev_draw(FORECAST_SEED)
print(f"予測EV実現: {len(forecast_evs)}セッション", flush=True)

conditions = {}
conditions["A_予測EV"] = [(s, forecast_evs) for s in range(len(cmds))]
eval_evs = {}
for s in range(len(cmds)):
    eval_evs[s] = ev_draw(EVAL_BASE_SEED + 1000 * s)
print(f"評価EV実現: {len(cmds)}通り, セッション数 "
      f"{min(len(v) for v in eval_evs.values())}-{max(len(v) for v in eval_evs.values())}",
      flush=True)
conditions["B_評価EV"] = [(s, eval_evs[s]) for s in range(len(cmds))]

results = {}
for label, pairs in conditions.items():
    rows = []
    t0 = time.perf_counter()
    for s, evs in pairs:
        p = cmds[s]
        sc = ActivationScenario(
            name=str(p.get("name") or f"cmd{s:02d}"),
            up_signal=np.asarray(p["up_proxy"], dtype=float).reshape(-1).copy(),
            down_signal=np.asarray(p["down_proxy"], dtype=float).reshape(-1).copy(),
            evs=evs,
        )
        r = _fixed_bid_scenario_task((s, sc, cfg, baseline, up, down))
        rows.append(r)
        print(f"  [{label}] 指令{s:02d} EV{len(evs):4d} "
              f"feasible={bool(r['feasible'])} rounds={r.get('rounds')}", flush=True)
    ok = sum(1 for r in rows if r["feasible"])
    results[label] = {"feasible": ok, "total": len(rows),
                      "seconds": round(time.perf_counter() - t0, 1)}
    print(f"[{label}] 実行可能 {ok}/{len(rows)}  "
          f"({results[label]['seconds']}s)", flush=True)

out = Path("execute_results/ft_sweep/lp_feasibility_probe.json")
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== まとめ ===")
for k, v in results.items():
    print(f"  {k}: 実行可能 {v['feasible']}/{v['total']}")

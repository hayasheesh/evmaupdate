"""中央LPを同じ入札・同じholdout指令・同じEV乱数で走らせ、MARLと並べる。

MILPAgent は昨夜の整理で消えたので履歴(97dabe7^)から取り出し、追跡外の
ここに置く。Config の MILP_* 定数も同時に消えているので、削除時の値を
Config モジュールに注入してから読み込む（追跡ファイルは触らない）。
"""
from __future__ import annotations
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
DAY, FORECAST_SEED, N_CMD, N_SEEDS = "2024-12-04", 1_076_030, 24, 1
HORIZON = int(os.environ.get("PROBE_MILP_HORIZON", "1"))
PIPELINES = tuple(os.environ.get(
    "PROBE_PIPELINES", "central_lp_only,central_lp_bess").split(","))
TAG = f"h{HORIZON}"
PAIRED_SEED = 910_000          # MARL の zeroshot と同じ
HERE = Path(__file__).resolve().parent

import Config
# 97dabe7^ の Config から。値はそのまま。
for name, value in {
    "MILP_W_AG": 1.0, "MILP_W_SOC": 100.0, "MILP_W_SWITCH": 0.0,
    "MILP_AG_DEADBAND": 10, "MILP_HORIZON": HORIZON, "MILP_DEADBAND_PENALTY": 1.0,
    "MILP_SOC_PENALTY": 2.0, "MILP_SOLVER_TIME_LIMIT": None,
    "MILP_SOLVER_GAP_REL": None, "MILP_SOLVER_GAP_ABS": None,
    "MILP_SOLVER_THREADS": None, "MILP_SOLVER_PRESOLVE": "default",
    "MILP_SOLVER_CUTS": "default", "MILP_SOLVER_HEURISTIC": "default",
    "MILP_SOLVER_STRONG": None,
}.items():
    setattr(Config, name, value)

spec = importlib.util.spec_from_file_location("milp_agent", HERE / "milp_agent.py")
milp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(milp_mod)
MILPAgent = milp_mod.MILPAgent
print("MILPAgent 読み込み完了 (horizon=%d, W_AG=%.1f, W_SOC=%.1f)"
      % (Config.MILP_HORIZON, Config.MILP_W_AG, Config.MILP_W_SOC), flush=True)

from training.lower_bid_training import (
    build_fixed_upper_bid_for_day, _activation_scenarios_for_day,
)
from training.evaluate_controller_precision import evaluate_controller_precision
from environment.arrival_context import ArrivalScenarioSampler
from training.run_after_day_ahead_bid import _select_payload, _coerce_series
from environment.readcsv import load_multiple_demand_files_with_labels
from Config import EPISODE_STEPS

CACHE = HERE / "fixed_bid_2024-12-04.pkl"
import pickle
if CACHE.is_file():
    fixed_bid = pickle.loads(CACHE.read_bytes())
    print("入札をキャッシュから読み込み", flush=True)
else:
    all_data = load_multiple_demand_files_with_labels(train_split=25)
    payload = _select_payload(all_data, "all", DAY, 0)
    base_series, service_date = _coerce_series(payload, int(EPISODE_STEPS))
    scen = ArrivalScenarioSampler().scenario_for_day(DAY)
    t0 = time.perf_counter()
    fixed_bid = build_fixed_upper_bid_for_day(
        base_series, DAY, arrival_scenario=scen, forecast_seed=FORECAST_SEED)
    print(f"入札を解いた ({time.perf_counter()-t0:.0f}s)", flush=True)
    CACHE.write_bytes(pickle.dumps(fixed_bid))

cap = float(np.sum(np.asarray(fixed_bid["up_plan"]) + np.asarray(fixed_bid["down_plan"])))
print(f"入札: capacity={cap:,.0f} kW-block", flush=True)

cmds, mode = _activation_scenarios_for_day(DAY, FORECAST_SEED, n_scenarios=N_CMD,
                                           scenario_partition="holdout")
eval_bid = dict(fixed_bid)
eval_bid.update({"activation_scenario_payload": list(cmds),
                 "activation_scenarios": len(cmds), "activation_mode": mode})
print(f"評価指令: {len(cmds)}本 (holdout)", flush=True)

results = {}
for pipeline in PIPELINES:
    agent = MILPAgent(horizon=HORIZON)
    agent.set_test_mode(True)
    t0 = time.perf_counter()
    s = evaluate_controller_precision(
        agent, eval_bid, n_seeds=N_SEEDS, base_seed=PAIRED_SEED,
        evaluation_pipeline=pipeline,
        out_dir=str(HERE / f"central_lp_{TAG}_{pipeline}"), visualize=False)
    results[pipeline] = {k: s.get(k) for k in (
        "global_tracking_rate", "controller_pre_system_tracking_rate",
        "up_pass_rate", "down_pass_rate", "soc_hit_rate",
        "controller_pre_system_mae_kw", "post_bess_mae_kw")}
    results[pipeline]["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"[{pipeline}] {json.dumps(results[pipeline], ensure_ascii=False)}", flush=True)

(HERE / f"central_lp_eval_{TAG}.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== 中央LP 対 MARL (holdout 24指令, seed 910000) ===")
print(f"{'':28}{'追従':>9}{'補正前':>9}{'上げ合格':>10}{'SoC':>8}")
for k, v in results.items():
    print(f"{k:28}{(v['global_tracking_rate'] or 0)*100:9.2f}"
          f"{(v['controller_pre_system_tracking_rate'] or 0)*100:9.2f}"
          f"{(v['up_pass_rate'] or 0)*100:10.2f}{(v['soc_hit_rate'] or 0)*100:8.2f}")
print(f"{'MARL zeroshot (system)':28}{95.36:9.2f}{81.03:9.2f}{88.03:10.2f}{100.00:8.2f}")

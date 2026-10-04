"""中央LPと同一パイプラインで MARL(warm start) を測り直す。

marl_raw と central_lp_only は force/central/bess/response_source が完全に同一。
central_lp_bess は BESS のみ有効。同じ入札・同じholdout24指令・同じEV乱数で
測れば、制御器だけの差になる。あわせて指令の同一性も照合する。
"""
from __future__ import annotations
import sys, json, time, pickle
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

DAY, FORECAST_SEED, N_CMD, N_SEEDS, PAIRED_SEED = "2024-12-04", 1_076_030, 24, 1, 910_000
WARM = ROOT / "archive/direct_bid_256cmd_25d_evcount_12hbid_7station_20260916_002104"

from training.lower_bid_training import _activation_scenarios_for_day
from training.evaluate_controller_precision import evaluate_controller_precision
from training.system_controller import build_agent
from training.agent_checkpoint import find_latest_checkpoint
from environment.normalize import load_observation_normalization_for_archive
from environment.EVEnv import EVEnv
from tools.evaluator import set_env_seed

fixed_bid = pickle.loads((HERE / "fixed_bid_2024-12-04.pkl").read_bytes())
cmds, mode = _activation_scenarios_for_day(DAY, FORECAST_SEED, n_scenarios=N_CMD,
                                           scenario_partition="holdout")

# --- 指令の同一性照合: fine-tune の評価が引いたものと同じか ---
ident = [(str(c.get("source_date")), str(c.get("source_bmu"))) for c in cmds]
print(f"holdout指令 {len(cmds)}本 先頭3: {ident[:3]}", flush=True)
(HERE / "holdout_command_ids.json").write_text(
    json.dumps(ident, ensure_ascii=False, indent=1), encoding="utf-8")

eval_bid = dict(fixed_bid)
eval_bid.update({"activation_scenario_payload": list(cmds),
                 "activation_scenarios": len(cmds), "activation_mode": mode})

profile = load_observation_normalization_for_archive(str(WARM))
ckpt, ep = find_latest_checkpoint(str(WARM))
print(f"warm start: {ckpt} ep{ep}", flush=True)

set_env_seed(PAIRED_SEED)
env = EVEnv()
env.reset(net_demand_series=np.zeros(288, dtype=np.float32))
agent = build_agent(env)
agent.load_actors(ckpt, ep, map_location=None if torch.cuda.is_available() else "cpu")
agent.set_test_mode(True)

results = {}
for pipeline in ("marl_raw", "central_lp_bess", "system"):
    t0 = time.perf_counter()
    s = evaluate_controller_precision(
        agent, eval_bid, n_seeds=N_SEEDS, base_seed=PAIRED_SEED,
        evaluation_pipeline=pipeline,
        out_dir=str(HERE / f"marl_{pipeline}"), visualize=False)
    results[pipeline] = {k: s.get(k) for k in (
        "global_tracking_rate", "controller_pre_system_tracking_rate",
        "up_pass_rate", "down_pass_rate", "soc_hit_rate", "post_bess_mae_kw")}
    results[pipeline]["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"[MARL {pipeline}] {json.dumps(results[pipeline], ensure_ascii=False)}", flush=True)

(HERE / "marl_matched_eval.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("完了", flush=True)

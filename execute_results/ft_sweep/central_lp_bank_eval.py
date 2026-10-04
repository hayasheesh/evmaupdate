"""中央MILP(現在ステップのみ)を、MARLと同じ入札バンク・同じ乱数で測る。

既存の central_lp_eval.py は単一日固定でバンクを使わないので、今日の
5日 × 6指令の測定と並べられない。日ごとの指令シードと EV シードを
tools/evaluate_final_system_on_bid_bank.py と同じ式に合わせる。
"""
from __future__ import annotations
import sys, os, json, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

BANK = os.environ.get("LP_BANK", "execute_results/bid_banks/validation_5_7s_f500_dense8")
HORIZON = int(os.environ.get("LP_HORIZON", "1"))
N_CMD = int(os.environ.get("LP_CMD", "6"))
N_SEEDS = int(os.environ.get("LP_SEEDS", "1"))
BASE_SEED = int(os.environ.get("LP_BASE_SEED", "1422090"))
PIPELINES = tuple(os.environ.get("LP_PIPELINES", "central_lp_only,central_lp_bess").split(","))

import Config
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
milp_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(milp_mod)
MILPAgent = milp_mod.MILPAgent
print(f"MILPAgent 読み込み (horizon={HORIZON})", flush=True)

from training.bid_bank import BidBank
from training.lower_bid_training import _activation_scenarios_for_day
from training.evaluate_controller_precision import evaluate_controller_precision
from tools.evaluator import set_env_seed

bank = BidBank(BANK)
entries = list(bank.entries)
print(f"バンク {BANK}: {len(entries)} 日", flush=True)

out = {}
for pipeline in PIPELINES:
    set_env_seed(BASE_SEED)
    rows = []
    t_all = time.perf_counter()
    for day_index, entry in enumerate(entries):
        fixed_bid = dict(bank.load_entry(entry))
        command_seed = int(fixed_bid.get("forecast_seed", 0)) + 512_209
        payloads, mode = _activation_scenarios_for_day(
            fixed_bid.get("service_date"), command_seed,
            n_scenarios=N_CMD, scenario_partition="holdout")
        fixed_bid["activation_scenario_payload"] = list(payloads)
        fixed_bid["activation_scenarios"] = len(payloads)
        fixed_bid["activation_mode"] = mode
        agent = MILPAgent(horizon=HORIZON); agent.set_test_mode(True)
        t0 = time.perf_counter()
        s = evaluate_controller_precision(
            agent, fixed_bid, n_seeds=N_SEEDS,
            base_seed=BASE_SEED + day_index * 100_000,
            evaluation_pipeline=pipeline,
            out_dir=str(HERE / f"lp_h{HORIZON}_{pipeline}" / f"day_{day_index:03d}"),
            visualize=False)
        rows.append(s)
        print(f"  [{pipeline}] day{day_index} {entry['service_date']} "
              f"up={s.get('up_pass_rate',0)*100:.1f}% soc={s.get('soc_hit_rate',0)*100:.1f}% "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    m = lambda k: float(np.mean([r.get(k) or 0.0 for r in rows]))
    out[pipeline] = {k: m(k) for k in ("up_pass_rate","down_pass_rate","idle_pass_rate",
        "global_tracking_rate","controller_pre_system_tracking_rate","soc_hit_rate",
        "post_bess_mae_kw")}
    out[pipeline]["seconds"] = round(time.perf_counter()-t_all, 1)
    print(f"[{pipeline}] {json.dumps(out[pipeline], ensure_ascii=False)}", flush=True)

(HERE / f"central_lp_bank_h{HORIZON}.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== 中央MILP / 同一バンク・同一乱数 ===")
print(f"{'':22}{'上げ合格':>10}{'追従':>9}{'SoC達成':>9}{'連系点MAE':>11}")
for k, v in out.items():
    print(f"{k:22}{v['up_pass_rate']*100:10.2f}{v['global_tracking_rate']*100:9.2f}"
          f"{v['soc_hit_rate']*100:9.2f}{v['post_bess_mae_kw']:11.2f}")

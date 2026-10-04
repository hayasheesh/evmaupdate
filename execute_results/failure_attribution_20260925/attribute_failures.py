"""Which blocks the proposed system failed could a controller that knew the whole day have passed?

The proposed system (MARL + force charging + PCC BESS) is evaluated exactly as
tools/evaluate_final_system_on_bid_bank.py does. While it runs, every accepted
EV session, the dispatch target, the band and the PCC response of each rollout
are recorded from the environment itself, so the LP sees the same vehicles the
controller saw.

For each rollout the same fleet (plus the BESS as one always-connected,
bidirectional unit over its usable 10-90 % range, efficiency 1) is given to
certify_direct_phase_one, the bidder's exact per-EV LP with departure targets
as hard constraints:

  joint     all assessed steps at once, minimizing total distance outside the
            band. A block is passed when none of its steps is outside.
  isolated  for each block the system failed, only that block is assessed.
            Feasible means an omniscient controller could have passed that
            block if it were willing to fail any other block.

A block fails when any of its steps leaves the band, as in the evaluation
(stay rate >= 0.90 with six steps per block).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(r"C:\Users\admin\Desktop\EVMALOCALUPDATE")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Each run's environment (its launcher in execute_results/ft_sweep/) plus what
# the evaluation tool sets. EnvConfig reads these on import, so the profile is
# picked from argv before any project import.
COMMON_ENV = {
    "EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR": "0",
    "EVMA_ACTOR_EV_COUNT": "1",
    "EVMA_GLOBAL_BALANCE_REWARD_MODE": "bounded_absolute_error",
    "EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW": "150",
    "EVMA_ACTIVATION_SCENARIO_DIR": "data/aemo/nem/processed_5min/dense8",
    "EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW": "500",
    "EVMA_LOWER_BID_CONTEXT_OBS": "1",
    "EVMA_USE_RESIDUAL_BESS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
PROFILES = {
    # run_pretrain_G.sh: a side experiment (does tracking reach 100% if SoC is ignored?)
    "abg": {"EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW": "60", "EVMA_LOCAL_FLEET_RESIDUAL_OBS": "1"},
    # run_pretrain_noalloc_dense8.sh: main line, purely local actor inputs at execution
    # The checkpoint's actor input is 333 wide: 24 blocks of day-ahead bid lookahead.
    "noalloc": {"EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW": "0", "EVMA_LOCAL_FLEET_RESIDUAL_OBS": "0",
                "EVMA_LOWER_BID_LOOKAHEAD_BLOCKS": "24"},
}
PROFILE = sys.argv[sys.argv.index("--profile") + 1] if "--profile" in sys.argv else "abg"
RUN_ENV = {**COMMON_ENV, **PROFILES[PROFILE]}
os.environ.update(RUN_ENV)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

import tools.evaluator as evaluator_module  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from market.physical_lp_bidding.colgen_feasibility import certify_direct_phase_one  # noqa: E402
from market.physical_lp_bidding.data_classes import EVSpec  # noqa: E402

STEPS, SPB, BLOCKS, DT = 288, 6, 48, 5.0 / 60.0
FREE = 1e9
LOGS: list[dict] = []
LAST_SEED = {"value": None}


def _capture_sessions(env, log) -> None:
    """Same session identity and fields as market.physical_lp_bidding.evenv_adapter."""
    for station in range(env.num_stations):
        for slot_tensor in torch.nonzero(env.ev_mask[station], as_tuple=False).squeeze(-1):
            slot = int(slot_tensor.item())
            ev_id = int(env.ev_ids[station, slot].item())
            env_arrival = int(env.arrival_step[station, slot].item())
            key = (ev_id, env_arrival)
            if key in log["seen"]:
                continue
            log["seen"].add(key)
            arrival = max(env_arrival - 1, 0)
            raw_departure = int(env.depart[station, slot].item())
            log["sessions"].append(dict(
                arrival_t=int(np.clip(arrival, 0, STEPS - 1)),
                departure_t=max(raw_departure, arrival + 1),
                initial_soc=float(env.soc[station, slot].item()),
                target_soc=float(env.target[station, slot].item()),
                capacity_kwh=float(env.ev_capacity_kwh[station, slot].item()),
                max_charge_kw=float(env.ev_max_power_kw[station, slot].item()),
                max_discharge_kw=float(env.ev_max_power_kw[station, slot].item()),
                station_id=int(station),
                ev_id=f"{ev_id}@{env_arrival}",
                target_required=bool(raw_departure <= STEPS),
            ))


_orig_reset, _orig_begin, _orig_apply = EVEnv.reset, EVEnv.begin_step, EVEnv.apply_action
_orig_seed = evaluator_module.set_env_seed


def _seed(seed, *args, **kwargs):
    LAST_SEED["value"] = int(seed)
    return _orig_seed(seed, *args, **kwargs)


def _reset(self, *args, **kwargs):
    out = _orig_reset(self, *args, **kwargs)
    log = {
        "seed": LAST_SEED["value"],
        "target": np.asarray(kwargs.get("net_demand_series"), dtype=float).reshape(-1)[:STEPS],
        "tol": np.asarray(kwargs.get("tol_narrow_series"), dtype=float).reshape(-1)[:STEPS],
        "sessions": [], "seen": set(), "disp": [], "resp": [],
        "ev_kw": [], "actor_kw": [], "bess_kw": [], "bess_soc_pct": [], "forced_kw": [],
        "bess": dict(
            capacity_kwh=float(self.bess_energy_capacity_kwh),
            energy_kwh=float(self.bess_energy_kwh),
            power_kw=float(self.bess_power_limit_kw),
            min_soc_pct=float(self.bess_min_soc_pct),
            max_soc_pct=float(self.bess_max_soc_pct),
            enabled=bool(self.use_residual_bess),
        ),
    }
    self._attribution_log = log
    LOGS.append(log)
    _capture_sessions(self, log)
    return out


def _begin(self, *args, **kwargs):
    out = _orig_begin(self, *args, **kwargs)
    log = getattr(self, "_attribution_log", None)
    if log is not None:
        _capture_sessions(self, log)
    return out


def _apply(self, *args, **kwargs):
    out = _orig_apply(self, *args, **kwargs)
    log = getattr(self, "_attribution_log", None)
    if log is not None:
        info = out[4] if isinstance(out, tuple) and len(out) >= 5 and isinstance(out[4], dict) else {}
        log["disp"].append(float(self.current_net_demand))
        log["resp"].append(float(info.get("pcc_power_kw", np.nan)))
        log["ev_kw"].append(float(info.get("total_ev_transport", np.nan)))
        log["actor_kw"].append(float(info.get("raw_actor_total_power_kw", np.nan)))
        log["bess_kw"].append(float(info.get("bess_power_kw", np.nan)))
        log["bess_soc_pct"].append(float(info.get("bess_soc_pct", np.nan)))
        from training.system_controller import apply_force_charging
        log["forced_kw"].append(float(getattr(apply_force_charging, "last_forced_kw", 0.0)))
    return out


def _bess_spec(bess: dict) -> EVSpec | None:
    if not bess["enabled"] or bess["power_kw"] <= 0.0:
        return None
    lo = bess["capacity_kwh"] * bess["min_soc_pct"] / 100.0
    hi = bess["capacity_kwh"] * bess["max_soc_pct"] / 100.0
    usable = max(hi - lo, 1e-9)
    return EVSpec(
        arrival_t=0, departure_t=STEPS,
        initial_soc=float(np.clip((bess["energy_kwh"] - lo) / usable, 0.0, 1.0)),
        target_soc=0.0, capacity_kwh=usable,
        max_charge_kw=bess["power_kw"], max_discharge_kw=bess["power_kw"],
        station_id=-1, ev_id="bess", target_required=False,
    )


def _failed_blocks(passed: np.ndarray, participating: np.ndarray) -> list[int]:
    return [b for b in range(BLOCKS) if participating[b] and not passed[b * SPB:(b + 1) * SPB].all()]


def attribute(log: dict, participating: np.ndarray, time_limit_s: float) -> dict:
    disp = np.asarray(log["disp"], dtype=float)
    tol = log["tol"]
    evs = [EVSpec(**s) for s in log["sessions"]]
    bess = _bess_spec(log["bess"])
    fleet = evs + ([bess] if bess is not None else [])
    assessed = np.repeat(participating, SPB)
    lower = np.where(assessed, disp - tol, -FREE)
    upper = np.where(assessed, disp + tol, FREE)

    t0 = time.perf_counter()
    feasible, _, info = certify_direct_phase_one(fleet, lower, upper, steps=STEPS, dt=DT, time_limit_s=time_limit_s)
    joint_seconds = time.perf_counter() - t0
    if not info.get("complete", False):
        return {"joint_status": info.get("reason", "incomplete")}
    slack = np.asarray(info.get("step_slack_kw", np.zeros(STEPS)), dtype=float)
    step_tol = float(info.get("step_violation_tol_kw", 1e-5))
    lp_joint_failed = _failed_blocks(slack <= step_tol, participating)

    system_passed = np.abs(disp - np.asarray(log["resp"], dtype=float)) <= tol + 1e-6
    system_failed = _failed_blocks(system_passed, participating)

    isolated = {}
    for b in system_failed:
        lo = np.full(STEPS, -FREE)
        hi = np.full(STEPS, FREE)
        sl = slice(b * SPB, (b + 1) * SPB)
        lo[sl], hi[sl] = lower[sl], upper[sl]
        ok, _, binfo = certify_direct_phase_one(fleet, lo, hi, steps=STEPS, dt=DT, time_limit_s=time_limit_s)
        isolated[b] = None if ok is None else bool(ok)
    return {
        "joint_status": "complete",
        "joint_objective_kw_steps": float(info.get("objective", 0.0)),
        "joint_seconds": round(joint_seconds, 2),
        "lp_joint_failed": lp_joint_failed,
        "system_failed": system_failed,
        "isolated": isolated,
        "sessions": len(evs),
        "departing_with_target": int(sum(1 for e in evs if e.target_required)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="archive/prod_f500_dense8_ABG_7station_20260922_112241")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="abg")
    parser.add_argument("--episode", type=int, default=200, help="-1 lets the evaluation tool choose the best checkpoint")
    parser.add_argument("--bid-bank-dir", default="execute_results/bid_banks/validation_5_7s_f500_dense8")
    parser.add_argument("--command-scenarios", type=int, default=12)
    parser.add_argument("--ev-seeds", type=int, default=1)
    parser.add_argument("--pipeline", default="marl_force_bess")
    parser.add_argument("--time-limit-s", type=float, default=600.0)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--output-dir", default="execute_results/failure_attribution_20260925/abg_ep200")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    EVEnv.reset, EVEnv.begin_step, EVEnv.apply_action = _reset, _begin, _apply
    evaluator_module.set_env_seed = _seed
    from tools.evaluate_final_system_on_bid_bank import main as evaluate_main
    from training.bid_bank import BidBank

    t0 = time.perf_counter()
    episode_args = [] if args.episode < 0 else ["--episode", str(args.episode)]
    evaluate_main([
        "--model-dir", args.model_dir, *episode_args,
        "--bid-bank-dir", args.bid_bank_dir,
        "--command-scenarios", str(args.command_scenarios), "--ev-seeds", str(args.ev_seeds),
        "--pipeline", args.pipeline, "--max-days", str(args.max_days),
        "--output-dir", str(out / "evaluation"),
    ])
    print(f"[attribution] evaluation done in {time.perf_counter() - t0:.0f}s", flush=True)

    rows = pd.read_csv(out / "evaluation" / "all_rollouts.csv")
    rollouts = [log for log in LOGS if len(log["disp"]) == STEPS]
    if len(rollouts) != len(rows):
        raise RuntimeError(f"recorded {len(rollouts)} rollouts but the evaluation wrote {len(rows)}")
    bank = BidBank(Path(args.bid_bank_dir))
    entries = list(bank.entries)

    block_rows, rollout_rows = [], []
    for i, (log, row) in enumerate(zip(rollouts, rows.itertuples(index=False))):
        if log["seed"] != int(row.realized_seed):
            raise RuntimeError(f"rollout {i}: seed {log['seed']} != {row.realized_seed}")
        bid = dict(bank.load_entry(entries[int(row.bid_day_index)]))
        up = np.asarray(bid["up_plan"], dtype=float)[:BLOCKS]
        down = np.asarray(bid["down_plan"], dtype=float)[:BLOCKS]
        participating = (up > 1e-6) | (down > 1e-6)
        t1 = time.perf_counter()
        res = attribute(log, participating, args.time_limit_s)
        recorded = sorted(int(b) for b in str(row.failed_tracking_blocks).split(";") if b not in ("", "nan"))
        rollout_rows.append({
            "rollout": i, "bid_day_index": int(row.bid_day_index), "service_date": row.service_date,
            "scenario": int(row.scenario), "realized_seed": int(row.realized_seed),
            "participating_blocks": int(participating.sum()),
            "system_failed_recorded": len(recorded),
            "system_failed_recomputed": len(res.get("system_failed", [])),
            "reproduced": recorded == res.get("system_failed"),
            "lp_joint_failed": len(res.get("lp_joint_failed", [])),
            "joint_status": res.get("joint_status"),
            "joint_objective_kw_steps": res.get("joint_objective_kw_steps"),
            "sessions": res.get("sessions"), "seconds": round(time.perf_counter() - t1, 1),
        })
        for b in res.get("system_failed", []):
            block_rows.append({
                "rollout": i, "bid_day_index": int(row.bid_day_index), "block": b,
                "up_kw": float(up[b]), "down_kw": float(down[b]),
                "lp_joint_passes": b not in res["lp_joint_failed"],
                "lp_isolated_passes": res["isolated"].get(b),
            })
        print(f"[attribution] rollout {i + 1}/{len(rows)} failed={len(res.get('system_failed', []))} "
              f"lp_joint_failed={len(res.get('lp_joint_failed', []))} reproduced={rollout_rows[-1]['reproduced']} "
              f"{rollout_rows[-1]['seconds']}s", flush=True)

    trace_keys = ("disp", "resp", "ev_kw", "actor_kw", "bess_kw", "bess_soc_pct", "forced_kw")
    np.savez_compressed(
        out / "step_traces.npz",
        tol=np.stack([log["tol"] for log in rollouts]),
        **{k: np.asarray([log[k] for log in rollouts], dtype=float) for k in trace_keys},
    )
    rdf, bdf = pd.DataFrame(rollout_rows), pd.DataFrame(block_rows)
    rdf.to_csv(out / "rollouts.csv", index=False)
    bdf.to_csv(out / "failed_blocks.csv", index=False)
    n = len(bdf)
    summary = {
        "model_dir": args.model_dir, "episode": args.episode, "profile": args.profile,
        "bid_bank_dir": args.bid_bank_dir,
        "pipeline": args.pipeline, "rollouts": len(rdf),
        "reproduced_rollouts": int(rdf["reproduced"].sum()),
        "participating_blocks": int(rdf["participating_blocks"].sum()),
        "system_failed_blocks": n,
        "lp_joint_failed_blocks": int(rdf["lp_joint_failed"].sum()),
        "failed_blocks_lp_joint_passes": int(bdf["lp_joint_passes"].sum()) if n else 0,
        "failed_blocks_lp_isolated_passes": int((bdf["lp_isolated_passes"] == True).sum()) if n else 0,  # noqa: E712
        "failed_blocks_lp_isolated_fails": int((bdf["lp_isolated_passes"] == False).sum()) if n else 0,  # noqa: E712
        "failed_blocks_lp_unsolved": int(bdf["lp_isolated_passes"].isna().sum()) if n else 0,
        "run_env": RUN_ENV,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

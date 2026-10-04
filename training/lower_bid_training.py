"""Support the blockwise upper bid and fixed-bid lower training episodes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import os
import random
import time

import numpy as np
import torch

from EnvConfig import (
    EPISODE_STEPS,
    LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS,
    LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,
    LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW,
    LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW,
    LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT,
    LOWER_TRAIN_UPPER_BID_DOWN_MAX_KW,
    LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_MIN_DIRECTION_BID_KW,
    LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_TIME_LIMIT_S,
    LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC,
    LOWER_TRAIN_UPPER_BID_SEED,
    LOWER_TRAIN_UPPER_BID_UP_MAX_KW,
    LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES,
    LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_BID,
    LOWER_TRAIN_UPPER_BID_VERBOSE_BID_BUILD,
    LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW,
    LOWER_BID_LOOKAHEAD_BLOCKS,
)
from market.activation_scenarios import (
    LOWER_CONTROL_POOL,
    activation_library_signature,
    build_activation_scenario_set,
    lower_control_pool_label,
    scenarios_to_solver_payload,
)
from market.bid_participation import (
    participation_by_block,
    participation_by_step,
    zero_instruction_tolerance_by_block,
)
from market.bid_env import (
    N_BLOCKS,
    STEPS_PER_BLOCK,
    baseline_aware_target_series,
)
from market.physical_lp_bidding import (
    USABLE_SOLVER_STATUSES,
    ActivationScenario,
    BiddingLPConfig,
    BiddingSolution,
    JointBiddingProblem,
    sample_ev_specs_from_evenv,
    solve_joint_hard_bidding_benders,
    solve_natural_baseline_lp,
    stratified_ev_activation_scenarios,
    validate_joint_solution,
)
from market.physical_lp_bidding.colgen_feasibility import (
    certify as certify_colgen_feasibility,
)
from market.physical_lp_bidding.joint_validation import (
    ev_labels,
    fixed_bid_tracking_bands,
)


_BID_PROGRESS_LOG_PATH: str | None = None
def _fixed_bid_scenario_task(payload):
    """Pickle-safe frozen-bid column-generation worker."""

    scenario_idx, scenario, cfg, baseline, up_kw, down_kw = payload
    _targets, _tolerances, lower, upper = fixed_bid_tracking_bands(
        cfg,
        baseline,
        up_kw,
        down_kw,
        np.asarray(scenario.up_signal, dtype=float),
        np.asarray(scenario.down_signal, dtype=float),
        apply_transition_band=bool(cfg.apply_transition_band),
    )
    started = time.perf_counter()
    feasible, rounds, info = certify_colgen_feasibility(
        scenario.evs,
        lower,
        upper,
        steps=int(cfg.steps),
        dt=float(cfg.dt_hours),
        eta_ch=float(cfg.eta_ch),
        return_dispatch=True,
    )
    return {
        "scenario_index": int(scenario_idx),
        "feasible": feasible,
        "rounds": int(rounds),
        "info": info,
        "runtime_s": float(time.perf_counter() - started),
    }


@dataclass
class SubmittedBidResult:
    """Small result object for the submitted blockwise bid."""

    solved: bool
    feasible: bool
    status: str
    objective_capacity_kw_block: float
    method: str = "uninitialized"
    n_scenarios: int = 0
    mean_baseline_kw: float = 0.0
    mean_up_kw: float = 0.0
    mean_down_kw: float = 0.0
    up_pass_rate: float = 1.0
    down_pass_rate: float = 1.0
    global_tracking_rate: float = 1.0
    soc_hit_rate: float = 1.0

    def as_dict(self, prefix: str = "submitted_bid") -> dict:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}method": self.method,
            f"{p}solved": bool(self.solved),
            f"{p}feasible": bool(self.feasible),
            f"{p}status": self.status,
            f"{p}objective_capacity_kw_block": float(
                self.objective_capacity_kw_block
            ),
            f"{p}n_scenarios": int(self.n_scenarios),
            f"{p}mean_baseline_kw": float(self.mean_baseline_kw),
            f"{p}mean_up_kw": float(self.mean_up_kw),
            f"{p}mean_down_kw": float(self.mean_down_kw),
            f"{p}up_pass_rate": float(self.up_pass_rate),
            f"{p}down_pass_rate": float(self.down_pass_rate),
            f"{p}global_tracking_rate": float(self.global_tracking_rate),
            f"{p}soc_hit_rate": float(self.soc_hit_rate),
        }


def _capture_rng_state() -> dict:
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])




def _capacity_objective_kw_block(up_plan, down_plan) -> float:
    """Total offered EV regulation capacity across the 48 blocks."""

    up = np.asarray(up_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    down = np.asarray(down_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    return float(np.sum(up + down))


UPPER_BID_BUILDER_KIND = "outer_participation_colgen_benders"




def _sample_ev_candidate(payload) -> list:
    """One seeded EVEnv rollout; a process-pool task of _sample_ev_scenario_bank."""

    ev_seed, arrival_probs, day_context, service_date = payload
    return sample_ev_specs_from_evenv(
        seed=ev_seed,
        arrival_probabilities_by_station=arrival_probs,
        day_context=day_context,
        service_date=service_date,
    )


def _sample_ev_scenario_bank(
    *,
    count: int,
    seed: int,
    seed_offset: int,
    arrival_probs,
    day_context,
    label: str,
    service_date=None,
    workers: int = 1,
) -> list[list]:
    """Draw ``count`` EV realizations of one service day from its arrival forecast.

    Each realization is one seeded EVEnv rollout with zero power: the same
    arrival probabilities, session pools and previous-day run the lower
    controller later meets. No forecast error is applied. Every realization
    seeds the environment itself, so ``workers`` processes return the same
    bank, in the same order, as one.
    """

    count = max(int(count), 1)
    payloads = [
        (int(seed) + int(seed_offset) + 10007 * ev_idx, arrival_probs, day_context, service_date)
        for ev_idx in range(count)
    ]
    worker_count = max(1, min(int(workers), count))
    if worker_count == 1:
        bank: list[list] = []
        for ev_idx, payload in enumerate(payloads):
            bank.append(_sample_ev_candidate(payload))
            _bid_build_log(f"physical-joint: sampled {label} EV scenario {ev_idx + 1}/{count} evs={len(bank[-1])}")
        return bank
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        bank = list(executor.map(_sample_ev_candidate, payloads, chunksize=1))
    for ev_idx, evs in enumerate(bank):
        _bid_build_log(f"physical-joint: sampled {label} EV scenario {ev_idx + 1}/{count} evs={len(evs)}")
    return bank


def _select_ev_scenarios_by_count(ev_candidates: list[list]):
    """Select one seeded realization or min-, median-, and max-size cases.

    Candidates are ranked by accepted session count, with candidate index as a
    stable tie-break. For an even-sized pool, use the lower median rank. The
    normal robust bank is ordered minimum, median, maximum. A one-candidate
    pool is retained as a diagnostic single-realization case.
    """

    candidates = list(ev_candidates)
    if not candidates:
        raise ValueError("at least one candidate EV realization is required")
    if len(candidates) == 2:
        raise ValueError(
            "use exactly one EV candidate for a single-realization diagnostic, "
            "or at least three candidates for min/median/max"
        )
    ranked_indices = sorted(
        range(len(candidates)), key=lambda index: (len(candidates[index]), index)
    )
    if len(ranked_indices) == 1:
        ranks = (0,)
        labels = ("single",)
    else:
        ranks = (0, (len(ranked_indices) - 1) // 2, len(ranked_indices) - 1)
        labels = ("minimum", "median", "maximum")
    selected = []
    metadata = []
    for label, rank in zip(labels, ranks):
        candidate_index = ranked_indices[rank]
        evs = candidates[candidate_index]
        selected.append(evs)
        metadata.append(
            {
                "label": label,
                "candidate_index": int(candidate_index),
                "rank_zero_based": int(rank),
                "candidate_count": int(len(candidates)),
                "ev_count": int(len(evs)),
            }
        )
    return selected, metadata


def _solve_fixed_bid_scenarios_decomposed(
    problem: JointBiddingProblem,
    *,
    precomputed_task_results: list[dict] | None = None,
) -> BiddingSolution:
    """Certify a frozen bid with independent column-generated recourse LPs.

    The submitted baseline and widths are already fixed, and the production
    contract permits no failed market block.  Participation and miss decisions
    therefore do not belong in these scenario checks. Each scenario is a
    block-angular full-horizon EV/SoC LP: vehicle paths are generated independently
    and the restricted master carries only aggregate tracking rows.
    """

    started = time.perf_counter()
    scenario_power: dict[str, np.ndarray] = {}
    ev_power: dict[str, dict[str, np.ndarray]] = {}
    ev_energy: dict[str, dict[str, np.ndarray]] = {}
    statuses: list[str] = []
    solvers: list[str] = []
    objectives: list[float] = []
    runtimes: list[float] = []
    weights: list[float] = []
    names: list[str] = []
    required_statuses: list[str] = []
    cfg = problem.config
    baseline = np.asarray(problem.fixed_baseline, dtype=float).reshape(cfg.blocks)
    up_kw = np.asarray(problem.fixed_up, dtype=float).reshape(cfg.blocks)
    down_kw = np.asarray(problem.fixed_down, dtype=float).reshape(cfg.blocks)

    reused_precomputed = precomputed_task_results is not None
    if reused_precomputed:
        task_results = list(precomputed_task_results)
        expected_indices = list(range(len(problem.scenarios)))
        actual_indices = sorted(
            int(task.get("scenario_index", -1)) for task in task_results
        )
        dispatch_keys = {
            "scenario_power_kw",
            "ev_power_kw",
            "ev_energy_kwh",
        }
        if actual_indices != expected_indices or not all(
            task.get("feasible") is True
            and dispatch_keys.issubset(task.get("info", {}))
            for task in task_results
        ):
            raise ValueError(
                "precomputed recourse does not cover every scenario with dispatch"
            )
        task_results.sort(key=lambda task: int(task["scenario_index"]))
    else:
        payloads = [
            (index, scenario, cfg, baseline, up_kw, down_kw)
            for index, scenario in enumerate(problem.scenarios)
        ]
        worker_count = max(1, min(
            int(getattr(cfg, "scenario_workers", 1)),
            len(payloads) or 1,
        ))
        if worker_count > 1:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                task_results = list(
                    executor.map(_fixed_bid_scenario_task, payloads, chunksize=1)
                )
        else:
            task_results = [
                _fixed_bid_scenario_task(payload) for payload in payloads
            ]

    for task in task_results:
        scenario_idx = int(task["scenario_index"])
        scenario = problem.scenarios[scenario_idx]
        feasible = task["feasible"]
        rounds = int(task["rounds"])
        info = task["info"]
        runtime = float(task["runtime_s"])
        if feasible is True:
            scenario_status = "optimal"
        elif feasible is False:
            scenario_status = "infeasible"
        else:
            scenario_status = "colgen_incomplete"
        required_statuses.append(scenario_status)
        statuses.append(scenario_status)
        solvers.append("scipy-highs-colgen")
        objectives.append(
            float(np.dot(
                np.asarray(problem.objective_weights, dtype=float).reshape(-1),
                up_kw + down_kw,
            ))
            if feasible is True
            else 0.0
        )
        runtimes.append(runtime)
        weights.append(max(float(scenario.weight), 0.0))
        name = str(scenario.name)
        names.append(name)
        if feasible is True:
            scenario_power[name] = np.asarray(
                info["scenario_power_kw"], dtype=float
            )
            ev_power[name] = {}
            ev_energy[name] = {}
            one_problem = replace(problem, scenarios=[scenario])
            labels = ev_labels(one_problem)
            for ev_idx, (power, energy) in enumerate(zip(
                info["ev_power_kw"], info["ev_energy_kwh"]
            )):
                label = labels[(0, ev_idx)]
                ev_power[name][label] = np.asarray(power, dtype=float)
                ev_energy[name][label] = np.asarray(energy, dtype=float)
        if (scenario_idx + 1) % 8 == 0 or scenario_idx + 1 == len(problem.scenarios):
            _bid_build_log(
                "physical-joint: certified recourse "
                f"{scenario_idx + 1}/{len(problem.scenarios)} "
                f"rounds={rounds} status={scenario_status}"
            )

    successful = all(
        status.lower() in USABLE_SOLVER_STATUSES for status in statuses
    )
    if successful and all(status.lower() == "optimal" for status in statuses):
        status = "optimal"
    elif successful:
        status = "time_limit_feasible"
    else:
        status = "failed"
    weight_arr = np.asarray(weights, dtype=float)
    if float(weight_arr.sum()) <= 0.0:
        weight_arr = np.ones(len(objectives), dtype=float)
    objective = float(np.dot(weight_arr / weight_arr.sum(), np.asarray(objectives, dtype=float)))
    return BiddingSolution(
        status=status,
        solver="decomposed-column-generation:" + ",".join(sorted(set(solvers))),
        objective_value=objective,
        baseline_kw=np.asarray(problem.fixed_baseline, dtype=float).copy(),
        up_kw=np.asarray(problem.fixed_up, dtype=float).copy(),
        down_kw=np.asarray(problem.fixed_down, dtype=float).copy(),
        scenario_power_kw=scenario_power,
        ev_power_kw=ev_power,
        ev_energy_kwh=ev_energy,
        metadata={
            "runtime_s": float(time.perf_counter() - started),
            "sum_solver_runtime_s": float(np.nansum(np.asarray(runtimes, dtype=float))),
            "scenario_name_by_index": names,
            "scenario_statuses": statuses,
            "required_rate_statuses": required_statuses,
            "required_rate_fallback_indices": [],
            "required_rate_all_solved": bool(successful),
            "hard_recourse_lp": True,
            "column_generation": True,
            "decomposed_scenarios": int(len(problem.scenarios)),
            "precomputed_recourse_reused": bool(reused_precomputed),
        },
    )


def _mean_natural_baseline(
    *,
    ev_bank: list[list],
    config: BiddingLPConfig,
    baseline_min_kw: float,
    baseline_max_kw: float,
) -> np.ndarray:
    """Average early-feasible no-activation schedules across EV draws."""

    baselines: list[np.ndarray] = []
    zeros = np.zeros(config.steps, dtype=float)
    for ev_idx, evs in enumerate(ev_bank):
        scenario = ActivationScenario(
            name=f"natural_ev_{ev_idx:02d}",
            up_signal=zeros,
            down_signal=zeros,
            evs=evs,
        )
        baselines.append(solve_natural_baseline_lp(
            scenario,
            config=config,
            baseline_min_kw=float(baseline_min_kw),
            baseline_max_kw=float(baseline_max_kw),
        ))
    if not baselines:
        raise ValueError("natural baseline requires at least one EV scenario")
    return np.mean(np.stack(baselines, axis=0), axis=0)
























def _activation_scenarios_for_day(
    service_date,
    forecast_seed: int,
    n_scenarios: int | None = None,
    scenario_partition: str | None = "forecast",
    exclude_sources=None,
):
    scenarios = build_activation_scenario_set(
        service_date=service_date,
        n_scenarios=int(
            LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS
            if n_scenarios is None
            else n_scenarios
        ),
        seed=int(forecast_seed),
        proxy_shape_dir=str(LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR),
        scenario_partition=scenario_partition,
        require_unique=True,
        exclude_sources=exclude_sources,
    )
    signature = activation_library_signature(LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR)
    return scenarios_to_solver_payload(scenarios), str(signature["activation_signal_mode"])


def upper_bid_bank_settings() -> dict[str, object]:
    """Return every upper-bid input that makes persisted bids incompatible.

    This is shared by the one-day runner and the parallel bid-bank builder.
    The activation CSV content is fingerprinted so an old signal library cannot
    silently be mixed into lower-controller training.
    """

    candidate_count = int(LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES)
    fixed_ev_scenarios = 1 if candidate_count == 1 else 3
    single_realization = candidate_count == 1
    return {
        "upper_bid_contract_version": 30,
        **activation_library_signature(LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR),
        "builder_kind": UPPER_BID_BUILDER_KIND,
        "bid_objective": "total_ev_regulation_capacity_kw_block",
        "award_assumption": "full_award",
        "activation_scenarios": int(LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS),
        "bank_bid_contract": (
            "fixed_single_ev_all_commands_k0"
            if single_realization
            else "fixed_min_median_max_ev_all_commands_k0"
        ),
        "ev_information_regime": "clairvoyant",
        "fixed_ev_scenarios": fixed_ev_scenarios,
        "ev_scenario_candidate_count": candidate_count,
        "ev_scenario_selection": (
            "single_seeded_realization"
            if single_realization
            else "minimum_lower_median_maximum_by_session_count"
        ),
        "all_command_scenarios": True,
        "allowed_command_failures": 0,
        "physical_lp_tracking_contract": "all_assessed_blocks_hard",
        # training.bid_bank.zero_idle_baseline: blocks without bid width.
        "idle_baseline_kw": 0.0,
        "initial_bid_rule": (
            "aggregate_energy_robust_lp"
            if LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_BID
            else "assessment_i_caps"
        ),
        "physical_lp_min_direction_bid_kw": float(LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_MIN_DIRECTION_BID_KW),
        "solver": "outer_participation_colgen_benders",
        "research_minimum_bid_quantity_kw": str(
            LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW
        ),
        "physical_lp_time_limit_s": float(LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_TIME_LIMIT_S),
        "baseline_min_kw": float(LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW),
        "baseline_max_kw": float(LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW),
        "up_max_kw": float(LOWER_TRAIN_UPPER_BID_UP_MAX_KW),
        "down_max_kw": float(LOWER_TRAIN_UPPER_BID_DOWN_MAX_KW),
        "reward_band_frac": float(LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC),
        # The bid holds every assessed step to its own instruction band; the
        # 5-minute response allowance of Secondary-2 is not used.
        "five_minute_response_band": False,
        "baseline_step_weight": float(LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT),
    }


def set_upper_bid_progress_log(path: str | Path | None, reset: bool = True) -> str | None:
    """Set an optional file sink for upper-bid progress messages."""

    global _BID_PROGRESS_LOG_PATH
    if path is None:
        _BID_PROGRESS_LOG_PATH = None
        return None
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        out_path.write_text("", encoding="utf-8")
    _BID_PROGRESS_LOG_PATH = str(out_path)
    return _BID_PROGRESS_LOG_PATH




def _bid_build_log(message: str) -> None:
    line = f"[upper-bid] {message}"
    if bool(LOWER_TRAIN_UPPER_BID_VERBOSE_BID_BUILD):
        print(line, flush=True)
    if _BID_PROGRESS_LOG_PATH:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with Path(_BID_PROGRESS_LOG_PATH).open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {line}\n")


def _bid_to_target_and_tolerance(
    base_series,
    baseline_plan,
    up_plan,
    down_plan,
    *,
    band_fraction: float | None = None,
):
    base = np.asarray(base_series, dtype=np.float32).reshape(-1)[:EPISODE_STEPS]
    if base.size < EPISODE_STEPS:
        base = np.pad(base, (0, EPISODE_STEPS - base.size))

    baseline = np.asarray(baseline_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    up = np.asarray(up_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    down = np.asarray(down_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    if baseline.size < N_BLOCKS:
        baseline = np.pad(baseline, (0, N_BLOCKS - baseline.size))
    if up.size < N_BLOCKS:
        up = np.pad(up, (0, N_BLOCKS - up.size))
    if down.size < N_BLOCKS:
        down = np.pad(down, (0, N_BLOCKS - down.size))

    target = np.zeros(EPISODE_STEPS, dtype=np.float32)
    tol = np.zeros(EPISODE_STEPS, dtype=np.float32)
    regulation_all = np.zeros(EPISODE_STEPS, dtype=np.float32)
    frac = max(float(
        LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC
        if band_fraction is None
        else band_fraction
    ), 0.0)
    participating = participation_by_block(up, down)
    zero_instruction_tol = zero_instruction_tolerance_by_block(
        up,
        down,
        band_fraction=frac,
    )
    for b in range(N_BLOCKS):
        s0 = b * STEPS_PER_BLOCK
        s1 = s0 + STEPS_PER_BLOCK
        if not participating[b]:
            # Zero award means no market instruction. Zero is only a neutral
            # observation placeholder; EVEnv disables global tracking here.
            continue
        block_target, regulation = baseline_aware_target_series(base[s0:s1], baseline[b], up[b], down[b])
        target[s0:s1] = block_target
        regulation_all[s0:s1] = regulation
        direction_tol = np.where(regulation < -1e-6, frac * up[b], frac * down[b])
        idle_tol = zero_instruction_tol[b]
        tol[s0:s1] = np.where(np.abs(regulation) <= 1e-6, idle_tol, direction_tol)
    tol = np.maximum(tol, 1e-6).astype(np.float32)
    return target, tol, regulation_all


def _bid_to_target_tol_from_activation(
    baseline_plan,
    up_plan,
    down_plan,
    up_proxy,
    down_proxy,
    *,
    band_fraction: float | None = None,
):
    baseline = np.asarray(baseline_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    up = np.asarray(up_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    down = np.asarray(down_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    if baseline.size < N_BLOCKS:
        baseline = np.pad(baseline, (0, N_BLOCKS - baseline.size))
    if up.size < N_BLOCKS:
        up = np.pad(up, (0, N_BLOCKS - up.size))
    if down.size < N_BLOCKS:
        down = np.pad(down, (0, N_BLOCKS - down.size))
    up_p = np.asarray(up_proxy, dtype=float).reshape(-1)[:EPISODE_STEPS]
    down_p = np.asarray(down_proxy, dtype=float).reshape(-1)[:EPISODE_STEPS]
    if up_p.size < EPISODE_STEPS:
        up_p = np.pad(up_p, (0, EPISODE_STEPS - up_p.size))
    if down_p.size < EPISODE_STEPS:
        down_p = np.pad(down_p, (0, EPISODE_STEPS - down_p.size))

    target = np.zeros(EPISODE_STEPS, dtype=np.float32)
    tol = np.zeros(EPISODE_STEPS, dtype=np.float32)
    regulation_all = np.zeros(EPISODE_STEPS, dtype=np.float32)
    frac = max(float(
        LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC
        if band_fraction is None
        else band_fraction
    ), 0.0)
    participating = participation_by_block(up, down)
    zero_instruction_tol = zero_instruction_tolerance_by_block(
        up,
        down,
        band_fraction=frac,
    )
    for b in range(N_BLOCKS):
        s0 = b * STEPS_PER_BLOCK
        s1 = s0 + STEPS_PER_BLOCK
        if not participating[b]:
            continue
        # The instruction is the award times the utilization fraction the
        # source unit saw: the area requirement is allocated pro rata over
        # contracted capacity, so a smaller award receives a smaller kW.
        regulation = down[b] * down_p[s0:s1] - up[b] * up_p[s0:s1]
        target[s0:s1] = float(baseline[b]) + regulation
        regulation_all[s0:s1] = regulation
        direction_tol = np.where(regulation < -1e-6, frac * up[b], frac * down[b])
        idle_tol = zero_instruction_tol[b]
        tol[s0:s1] = np.where(np.abs(regulation) <= 1e-6, idle_tol, direction_tol)
    tol = np.maximum(tol, 1e-6).astype(np.float32)
    return target.astype(np.float32), tol, regulation_all.astype(np.float32)


def bid_instruction_scale_kw(baseline_plan, up_plan, down_plan) -> float:
    """The widest instruction this day's submitted bid can produce, in kW.

    Fixed the day before, so normalising the instruction by it uses nothing the
    operating day has not already revealed -- unlike the realised min/max,
    which is only known once the day is over. The optional bid lookahead also
    uses this value to encode its already-known schedule as dimensionless ratios.
    """

    baseline = np.asarray(baseline_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    up = np.asarray(up_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    down = np.asarray(down_plan, dtype=float).reshape(-1)[:N_BLOCKS]
    live = participation_by_block(up, down)
    if not np.any(live):
        return 1.0
    reach = np.concatenate([
        np.abs(baseline[live] - up[live]),
        np.abs(baseline[live] + down[live]),
    ])
    return float(max(np.max(reach), 1.0))


def bid_lookahead_context_series(
    baseline_plan,
    up_plan,
    down_plan,
    *,
    instruction_scale_kw: float,
    lookahead_blocks: int,
) -> dict[str, np.ndarray]:
    """Encode the known blockwise bid schedule at every five-minute step.

    Offset zero is the current 30-minute block. Later offsets advance through
    the submitted baseline/up/down schedule. Values beyond the service day are
    zero because this one-day bank contains no next-day commitment. Everything
    is divided by the known instruction envelope, keeping these features
    dimensionless and preventing any realized command from leaking into the
    lookahead.
    """

    horizon = max(int(lookahead_blocks), 0)
    if horizon == 0:
        return {}
    scale = max(float(instruction_scale_kw), 1.0)
    plans = {
        "baseline": np.asarray(baseline_plan, dtype=np.float32).reshape(-1)[:N_BLOCKS],
        "up": np.asarray(up_plan, dtype=np.float32).reshape(-1)[:N_BLOCKS],
        "down": np.asarray(down_plan, dtype=np.float32).reshape(-1)[:N_BLOCKS],
    }
    current_block = np.arange(EPISODE_STEPS, dtype=np.int64) // STEPS_PER_BLOCK
    context: dict[str, np.ndarray] = {}
    for offset in range(horizon):
        indices = current_block + offset
        valid = indices < N_BLOCKS
        for field, plan in plans.items():
            values = np.zeros(EPISODE_STEPS, dtype=np.float32)
            values[valid] = plan[indices[valid]] / scale
            context[f"bid_{field}_lookahead_{offset}"] = values
    return context


def _submitted_bid_info(
    result,
    baseline_plan,
    up_plan,
    down_plan,
    target,
    regulation,
    activation_mode,
    activation_scenarios,
    seed,
    submitted_up_plan=None,
    submitted_down_plan=None,
    award_metadata=None,
):
    participating = participation_by_block(up_plan, down_plan)
    submitted_up = np.asarray(
        up_plan if submitted_up_plan is None else submitted_up_plan,
        dtype=float,
    )
    submitted_down = np.asarray(
        down_plan if submitted_down_plan is None else submitted_down_plan,
        dtype=float,
    )
    submitted_participating = participation_by_block(submitted_up, submitted_down)
    return {
        "source": "upper_bid",
        "method": str(result.method),
        "activation_mode": str(activation_mode),
        "activation_scenarios": int(len(activation_scenarios or [])),
        "activation_scenario_sources": [
            str(payload.get("source", ""))
            for payload in (activation_scenarios or [])
        ],
        "forecast_seed": int(seed),
        "bid_solved": bool(result.solved),
        "bid_feasible": bool(result.feasible),
        "bid_status": str(result.status),
        "bid_objective_capacity_kw_block": (
            float(result.objective_capacity_kw_block)
            if np.isfinite(result.objective_capacity_kw_block)
            else float("nan")
        ),
        "submitted_mean_up_kw": float(np.mean(submitted_up)) if len(submitted_up) else 0.0,
        "submitted_mean_down_kw": float(np.mean(submitted_down)) if len(submitted_down) else 0.0,
        "submitted_max_up_kw": float(np.max(submitted_up)) if len(submitted_up) else 0.0,
        "submitted_max_down_kw": float(np.max(submitted_down)) if len(submitted_down) else 0.0,
        "submitted_participating_blocks": int(np.count_nonzero(submitted_participating)),
        "awarded_mean_up_kw": float(np.mean(up_plan)) if len(up_plan) else 0.0,
        "awarded_mean_down_kw": float(np.mean(down_plan)) if len(down_plan) else 0.0,
        "awarded_max_up_kw": float(np.max(up_plan)) if len(up_plan) else 0.0,
        "awarded_max_down_kw": float(np.max(down_plan)) if len(down_plan) else 0.0,
        "market_award": dict(award_metadata or {}),
        "bid_mean_baseline_kw": float(np.mean(baseline_plan)) if len(baseline_plan) else 0.0,
        "bid_mean_up_kw": float(np.mean(up_plan)) if len(up_plan) else 0.0,
        "bid_mean_down_kw": float(np.mean(down_plan)) if len(down_plan) else 0.0,
        "bid_max_up_kw": float(np.max(up_plan)) if len(up_plan) else 0.0,
        "bid_max_down_kw": float(np.max(down_plan)) if len(down_plan) else 0.0,
        "bid_participating_blocks": int(np.count_nonzero(participating)),
        "bid_participating_block_indices": np.flatnonzero(participating).astype(int).tolist(),
        "bid_global_tracking_rate": float(result.global_tracking_rate),
        "target_min_kw": float(np.min(target)) if target.size else 0.0,
        "target_max_kw": float(np.max(target)) if target.size else 0.0,
        "regulation_abs_mean_kw": float(np.mean(np.abs(regulation))) if regulation.size else 0.0,
    }


def build_upper_bid_training_episode(
    base_series,
    service_date: str | None,
    episode_idx: int,
    arrival_scenario=None,
):
    """Build one canonical bid and select its episode command scenario."""

    forecast_seed = int(LOWER_TRAIN_UPPER_BID_SEED) + int(episode_idx) * 1009
    fixed_bid = build_fixed_upper_bid_for_day(
        base_series,
        service_date,
        arrival_scenario=arrival_scenario,
        forecast_seed=forecast_seed,
    )
    return build_fixed_upper_bid_training_episode(
        fixed_bid,
        episode_idx=episode_idx,
    )



BID_SOLVE_CACHE_DIR = Path(__file__).resolve().parents[1] / "execute_results" / "bid_solve_cache"


def _arrival_scenario_signature(arrival_scenario):
    """Stable bytes for one day's ArrivalScenario, or None if its shape is unknown.

    The scenario is a small dataclass of two arrays plus three scalars. Hashing
    the arrays keeps the key sensitive to the arrivals the bid was solved
    against; an unexpected field type disables the cache rather than dropping
    that field out of the key.
    """

    import hashlib

    if arrival_scenario is None:
        return None
    out = {}
    for field in (
        "service_date",
        "arrival_probabilities_by_station",
        "day_context",
        "source",
        "day_class",
    ):
        if not hasattr(arrival_scenario, field):
            return None
        value = getattr(arrival_scenario, field)
        if isinstance(value, np.ndarray):
            array = np.asarray(value, dtype=np.float64)
            if not np.isfinite(array).all():
                return None
            out[field] = {
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(array).tobytes()
                ).hexdigest(),
                "shape": list(array.shape),
            }
        elif isinstance(value, (str, bool, int, float, type(None))):
            out[field] = value
        else:
            return None
    return out


def _bid_solve_cache_key(
    base_series,
    service_date,
    arrival_scenario,
    seed: int,
    assessment_band_fraction,
) -> str | None:
    """Hash every input that decides the solved bid, or None if one is unhashable.

    A wrong key would hand back a bid solved for different inputs without
    saying so, so anything that cannot be reduced to stable bytes disables the
    cache instead of being quietly left out of the key.
    """

    import hashlib
    import json as _json

    try:
        series = np.asarray(base_series, dtype=np.float64)
        if not np.isfinite(series).all():
            return None
        arrival_sig = _arrival_scenario_signature(arrival_scenario)
        if arrival_scenario is not None and arrival_sig is None:
            return None
        payload = {
            "service_date": str(service_date),
            "seed": int(seed),
            "assessment_band_fraction": (
                None if assessment_band_fraction is None
                else round(float(assessment_band_fraction), 12)
            ),
            "base_series_sha256": hashlib.sha256(
                np.ascontiguousarray(series).tobytes()
            ).hexdigest(),
            "base_series_shape": list(series.shape),
            "arrival": arrival_sig,
            "settings": upper_bid_bank_settings(),
        }
        blob = _json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError, AttributeError, OSError):
        return None
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_cached_bid_solve(key):
    if not key:
        return None
    path = BID_SOLVE_CACHE_DIR / f"{key}.pt"
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    bid = payload.get("bid")
    return bid if isinstance(bid, dict) else None


def _store_cached_bid_solve(key, bid) -> None:
    if not key or not isinstance(bid, dict):
        return
    try:
        BID_SOLVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = BID_SOLVE_CACHE_DIR / f"{key}.pt"
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with open(tmp, "wb") as handle:
            torch.save({"key": key, "bid": bid}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        pass


def build_fixed_upper_bid_for_day(
    base_series,
    service_date: str | None,
    arrival_scenario=None,
    forecast_seed: int | None = None,
    assessment_band_fraction: float | None = None,
    scenario_workers: int = 1,
) -> dict:
    """Build one bid robust to three EV counts and every design command.

    The result is cached under a hash of the day, seed, arrival scenario, and
    upper-bid settings. EVMA_BID_SOLVE_CACHE=0 always solves.
    """

    from training.blockwise_bid import build_blockwise_bid_for_day

    seed = int(
        LOWER_TRAIN_UPPER_BID_SEED
        if forecast_seed is None
        else forecast_seed
    )
    cache_enabled = os.environ.get("EVMA_BID_SOLVE_CACHE", "1").strip() not in (
        "0", "false", "False",
    )
    cache_key = (
        _bid_solve_cache_key(
            base_series, service_date, arrival_scenario, seed, assessment_band_fraction
        )
        if cache_enabled
        else None
    )
    cached = _load_cached_bid_solve(cache_key)
    if cached is not None:
        print(
            f"[upper-bid] cache hit day={service_date} seed={seed} "
            f"key={cache_key[:12]} (solve skipped)",
            flush=True,
        )
        return cached
    fixed_bid = build_blockwise_bid_for_day(
        base_series,
        service_date,
        arrival_scenario=arrival_scenario,
        forecast_seed=seed,
        assessment_band_fraction=assessment_band_fraction,
        scenario_workers=max(1, int(scenario_workers)),
    )
    design_commands = list(fixed_bid.get("activation_scenario_payload") or [])
    expected_commands = int(LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS)
    if len(design_commands) != expected_commands:
        raise RuntimeError(
            f"bid LP used {len(design_commands)} commands; expected "
            f"all configured {expected_commands}"
        )
    training_commands, training_mode = _activation_scenarios_for_day(
        service_date,
        seed + 830_027,
        n_scenarios=expected_commands,
        scenario_partition="feedback",
    )

    def command_identity(row) -> tuple[str, str]:
        return str(row.get("source_date", "")), str(row.get("source_bmu", ""))

    design_ids = {command_identity(row) for row in design_commands}
    overlap = [
        command_identity(row)
        for row in training_commands
        if command_identity(row) in design_ids
    ]
    if overlap:
        raise RuntimeError(
            "feedback commands overlap the design commands; "
            f"examples={overlap[:3]}"
        )
    if len(training_commands) != expected_commands:
        raise RuntimeError(
            f"training pool contains {len(training_commands)} commands; "
            f"expected {expected_commands}"
        )
    fixed_bid["design_activation_scenario_payload"] = design_commands
    fixed_bid["design_activation_scenarios"] = len(design_commands)
    fixed_bid["activation_scenario_payload"] = list(training_commands)
    fixed_bid["activation_scenarios"] = len(training_commands)
    fixed_bid["training_command_partition"] = "feedback"
    fixed_bid["training_activation_mode"] = str(training_mode)
    _store_cached_bid_solve(cache_key, fixed_bid)
    return fixed_bid


def build_fixed_upper_bid_training_episode(
    fixed_bid: dict,
    episode_idx: int = 0,
    *,
    activation_scenario: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Return one submitted-bid training episode.

    The submitted bid is fixed, while activation rotates across the scenario
    payload. The submitted and awarded widths are used exactly as persisted;
    pretraining, fine-tuning, and evaluation therefore solve the same task.
    """

    result = fixed_bid["result"]
    baseline = np.asarray(fixed_bid["baseline_plan"], dtype=float)
    up = np.asarray(fixed_bid["up_plan"], dtype=float)
    down = np.asarray(fixed_bid["down_plan"], dtype=float)
    scenarios = fixed_bid.get("activation_scenario_payload") or []
    if activation_scenario is not None:
        s_idx = 0
        scenario = activation_scenario
    elif scenarios:
        s_idx = int(episode_idx) % len(scenarios)
        scenario = scenarios[s_idx]
    else:
        scenario = None
    if scenario is not None:
        target, tol, regulation = _bid_to_target_tol_from_activation(
            baseline,
            up,
            down,
            scenario.get("up_proxy"),
            scenario.get("down_proxy"),
            band_fraction=fixed_bid.get("assessment_band_fraction"),
        )
        scenario_name = str(scenario.get("name", f"scenario_{s_idx}"))
    else:
        s_idx = 0
        target = np.asarray(fixed_bid["target_series"], dtype=np.float32).reshape(-1)[:EPISODE_STEPS]
        tol = np.asarray(fixed_bid["tol_series"], dtype=np.float32).reshape(-1)[:EPISODE_STEPS]
        regulation = np.asarray(fixed_bid["regulation_series"], dtype=float)
        scenario_name = "base_proxy"
    info = _submitted_bid_info(
        result,
        baseline,
        up,
        down,
        target,
        regulation,
        fixed_bid.get("activation_mode", "local_5min_random"),
        scenarios,
        int(fixed_bid.get("forecast_seed", 0)),
        submitted_up_plan=fixed_bid.get("submitted_up_plan"),
        submitted_down_plan=fixed_bid.get("submitted_down_plan"),
        award_metadata=fixed_bid.get("award_metadata"),
    )
    info.update({
        "source": "fixed_upper_bid",
        "activation_scenario_index": int(s_idx),
        "activation_scenario_name": scenario_name,
        "activation_scenario_source": (
            str(scenario.get("source", "")) if scenario is not None else ""
        ),
        "activation_scenario_source_date": (
            str(scenario.get("source_date", "")) if scenario is not None else ""
        ),
        "activation_scenario_source_bmu": (
            str(scenario.get("source_bmu", "")) if scenario is not None else ""
        ),
        "service_date": fixed_bid.get("service_date"),
    })
    instruction_scale_kw = bid_instruction_scale_kw(baseline, up, down)
    cache_key = (int(LOWER_BID_LOOKAHEAD_BLOCKS), float(instruction_scale_kw))
    cached_context = fixed_bid.get("_runtime_market_context_cache")
    if not isinstance(cached_context, dict) or cached_context.get("key") != cache_key:
        market_context_series = {
            "instruction_scale_kw": np.full(
                EPISODE_STEPS, instruction_scale_kw, dtype=np.float32
            ),
            **bid_lookahead_context_series(
                baseline,
                up,
                down,
                instruction_scale_kw=instruction_scale_kw,
                lookahead_blocks=int(LOWER_BID_LOOKAHEAD_BLOCKS),
            ),
        }
        fixed_bid["_runtime_market_context_cache"] = {
            "key": cache_key,
            "series": market_context_series,
        }
    else:
        market_context_series = cached_context["series"]
    return target, tol, {
        "arrival_probabilities_by_station": fixed_bid.get("arrival_probabilities_by_station"),
        "day_context": fixed_bid.get("day_context"),
        "service_date": fixed_bid.get("service_date"),
        "baseline_series": np.repeat(
            np.asarray(baseline, dtype=float).reshape(-1)[:N_BLOCKS], STEPS_PER_BLOCK
        )[:EPISODE_STEPS],
        "tracking_enabled_series": participation_by_step(
            up,
            down,
            steps_per_block=STEPS_PER_BLOCK,
            steps=EPISODE_STEPS,
        ),
        "instruction_scale_kw": instruction_scale_kw,
        "market_context_series": market_context_series,
    }, info


def sample_random_historical_activation(
    fixed_bid: dict,
    episode_idx: int,
    *,
    stream: str,
    exclude_sources=None,
) -> dict:
    """Draw one reproducible command from every command outside the bid design pool.

    The upper bid remains certified against its persisted design-command set,
    drawn from the train partition. This draw is only the lower-controller
    command for one rollout, taken from LOWER_CONTROL_POOL: the validation and
    test partitions of a partitioned library, or the whole of a library without
    partitions. The stream label separates training and validation while a
    stable episode-based seed makes checkpoint comparisons and exact resume
    reproducible. ``exclude_sources`` removes commands from the pool before the
    draw; the training stream excludes the interim-test commands.
    """

    stream_name, seed = _lower_command_seed(fixed_bid, episode_idx, stream)
    commands, _mode = _activation_scenarios_for_day(
        str(fixed_bid.get("service_date") or ""),
        seed,
        n_scenarios=1,
        scenario_partition=LOWER_CONTROL_POOL,
        exclude_sources=exclude_sources,
    )
    if len(commands) != 1:
        raise RuntimeError(f"expected one historical command, got {len(commands)}")
    command = dict(commands[0])
    command["lower_command_sampling_pool"] = lower_control_pool_label(
        LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR
    )
    command["lower_command_stream"] = stream_name
    command["lower_command_episode_index"] = int(episode_idx)
    return command


def _lower_command_seed(fixed_bid: dict, episode_idx: int, stream: str) -> tuple[str, int]:
    stream_name = str(stream).strip().lower()
    if stream_name not in {"train", "validation"}:
        raise ValueError(f"unknown lower-command stream: {stream!r}")
    stream_offset = 1_700_003 if stream_name == "train" else 2_700_003
    seed = (
        int(fixed_bid.get("forecast_seed", 0))
        + stream_offset
        + max(int(episode_idx), 0) * 1_000_003
    )
    return stream_name, seed


def interim_test_command_sources(
    test_entries: list[dict],
    *,
    interim_test_episodes: int,
    library_dir,
) -> set[str]:
    """Sources of the commands every interim test draws, rebuilt from seeds.

    Each interim test calls the validation-stream preparer for indices
    0..interim_test_episodes - 1 with test entry index mod the test bank size,
    so every interim test uses the same commands. The training stream excludes
    them, which keeps the interim tests on commands training never drew.
    """

    sources: set[str] = set()
    for idx in range(int(interim_test_episodes)):
        entry = test_entries[idx % len(test_entries)]
        _stream, seed = _lower_command_seed(entry, idx, "validation")
        scenarios = build_activation_scenario_set(
            service_date=str(entry.get("service_date") or ""),
            n_scenarios=1,
            seed=int(seed),
            proxy_shape_dir=str(library_dir),
            scenario_partition=LOWER_CONTROL_POOL,
            require_unique=True,
        )
        sources.update(str(s.source) for s in scenarios)
    return sources


def lower_commands_seen_in_pretrain(
    train_entries: list[dict],
    test_entries: list[dict],
    *,
    environment_episodes: int,
    interim_test_episodes: int,
    library_dir,
) -> set[str]:
    """Sources of every rollout command a bank pretrain drew, rebuilt from seeds.

    training.train calls the train-stream preparer for environment episodes
    0..environment_episodes (warmup included), with bank entry
    (episode - 1) mod the bank size and entry 0 for episode 0. Every interim
    test calls the validation-stream preparer for indices
    0..interim_test_episodes - 1 with test entry index mod the test bank size,
    so all interim tests share the same commands. Each draw depends only on the
    entry's service date and forecast seed, so the set is exact.
    """

    seen: set[str] = set()
    interim = interim_test_command_sources(
        test_entries, interim_test_episodes=interim_test_episodes, library_dir=library_dir
    )

    def draw(entry: dict, idx: int, stream: str) -> None:
        _stream, seed = _lower_command_seed(entry, idx, stream)
        scenarios = build_activation_scenario_set(
            service_date=str(entry.get("service_date") or ""),
            n_scenarios=1,
            seed=int(seed),
            proxy_shape_dir=str(library_dir),
            scenario_partition=LOWER_CONTROL_POOL,
            require_unique=True,
            exclude_sources=interim if stream == "train" else None,
        )
        seen.update(str(s.source) for s in scenarios)

    for idx in range(int(environment_episodes) + 1):
        draw(train_entries[max(idx - 1, 0) % len(train_entries)], idx, "train")
    for idx in range(int(interim_test_episodes)):
        draw(test_entries[idx % len(test_entries)], idx, "validation")
    return seen

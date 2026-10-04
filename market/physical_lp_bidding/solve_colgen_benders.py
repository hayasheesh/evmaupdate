"""Fixed-participation upper-bid refinement by column generation and Benders.

Participation is decided by the outer active set in
``training.blockwise_bid``.  This module contains only the continuous problem
for one fixed pattern: a 48-block baseline plus up/down quantities.  Frozen
EV/activation scenarios are generated vehicle-by-vehicle, and infeasibility
certificates become cuts in the small first-stage master.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import linprog

from .colgen_feasibility import (
    affine_tracking_bands,
    certify as certify_colgen_feasibility,
    certify_direct_phase_one,
    feasibility_cut_from_certificate,
)
from .data_classes import (
    BiddingSolution,
    JointBiddingProblem,
    assessment_i_rows,
    baseline_step_matrix,
    project_assessment_i,
)
from .prefix_circulation import (
    master_cut as circulation_master_cut,
    scenario_circulation,
    solve_circulation,
)
from .joint_validation import problem_arrays


_WORKER_COLUMN_POOLS: dict[tuple, list[list[np.ndarray]]] = {}


def _fleet_fingerprint(evs) -> tuple:
    """Identify a physical EV bank for worker-local path-column reuse."""

    return tuple(
        (
            int(ev.arrival_t),
            int(ev.departure_t),
            float(ev.initial_kwh),
            float(ev.target_kwh),
            float(ev.capacity_kwh),
            float(ev.max_charge_kw),
            float(ev.max_discharge_kw),
            bool(getattr(ev, "target_required", False)),
        )
        for ev in evs
    )


def _compact_worker_column_pool(
    column_pool: list[list[np.ndarray]],
    max_columns_per_ev: int,
) -> tuple[list[list[np.ndarray]], dict[str, int]]:
    """Bound a worker warm-start without changing oracle correctness.

    The quiet first path and the newest distinct priced paths are retained.
    Dropped paths remain discoverable by the exact pricing oracle, so this only
    changes warm-start size, never the feasible set or its certificate.
    """

    limit = int(max_columns_per_ev)
    before = int(sum(len(columns) for columns in column_pool))
    if limit <= 0:
        return column_pool, {
            "before": before,
            "after": before,
            "dropped": 0,
            "limit_per_ev": 0,
        }

    compacted: list[list[np.ndarray]] = []
    for vehicle_columns in column_pool:
        if len(vehicle_columns) <= limit:
            compacted.append(list(vehicle_columns))
            continue
        keep_indices = [0]
        first = np.round(
            np.asarray(vehicle_columns[0], dtype=np.float64), decimals=8
        )
        first[first == 0.0] = 0.0
        fingerprints = {first.tobytes()}
        for index in range(len(vehicle_columns) - 1, 0, -1):
            candidate = np.round(
                np.asarray(vehicle_columns[index], dtype=np.float64), decimals=8
            )
            candidate[candidate == 0.0] = 0.0
            key = candidate.tobytes()
            if key in fingerprints:
                continue
            fingerprints.add(key)
            keep_indices.append(index)
            if len(keep_indices) >= limit:
                break
        keep_indices.sort()
        compacted.append([vehicle_columns[index] for index in keep_indices])
    after = int(sum(len(columns) for columns in compacted))
    return compacted, {
        "before": before,
        "after": after,
        "dropped": int(before - after),
        "limit_per_ev": limit,
    }


def _scenario_oracle_task(payload):
    """Pickle-safe independent recourse oracle for one Benders scenario."""

    (
        scenario_index,
        scenario,
        cfg,
        candidate,
        up_active,
        down_active,
        colgen_max_rounds,
        first_stage_size,
        return_dispatch,
        *optional_time_limit,
    ) = payload
    oracle_time_limit_s = (
        optional_time_limit[0]
        if optional_time_limit
        else cfg.time_limit_s
    )
    direct_threshold = int(getattr(cfg, "direct_oracle_max_evs", 0))
    if direct_threshold > 0 and len(scenario.evs) <= direct_threshold:
        result = _scenario_direct_oracle_task(payload)
        # Say which solver actually answered.  Overwriting this unconditionally
        # hid whether the circulation or the LP produced the result, which is
        # the one thing a runtime breakdown needs to know.
        solver = str(result.get("oracle_solver") or "direct_sparse_phase1")
        result["oracle_solver"] = solver
        result["fallback_solver"] = ""
        result["certificate"]["primary_oracle_solver"] = solver
        return result
    fleet_key = _fleet_fingerprint(scenario.evs)
    lower, upper, lower_rows, upper_rows = affine_tracking_bands(
        cfg,
        candidate[: cfg.blocks],
        candidate[cfg.blocks : 2 * cfg.blocks],
        candidate[2 * cfg.blocks :],
        up_active,
        down_active,
        np.asarray(scenario.up_signal, dtype=float),
        np.asarray(scenario.down_signal, dtype=float),
    )
    started = time.perf_counter()
    decided = _circulation_decision(
        scenario_index, scenario, cfg, lower, upper, lower_rows, upper_rows,
        candidate, started, return_dispatch=bool(return_dispatch),
    )
    if decided is not None:
        return decided
    feasible, colgen_rounds, certificate = certify_colgen_feasibility(
        scenario.evs,
        lower,
        upper,
        steps=int(cfg.steps),
        dt=float(cfg.dt_hours),
        eta_ch=float(cfg.eta_ch),
        max_rounds=int(colgen_max_rounds),
        return_dispatch=bool(return_dispatch),
        initial_columns=_WORKER_COLUMN_POOLS.get(fleet_key),
        return_column_pool=True,
        time_limit_s=oracle_time_limit_s,
    )
    runtime = float(time.perf_counter() - started)
    certificate = dict(certificate)
    column_pool = certificate.pop("column_pool", None)
    if column_pool is not None:
        column_pool, cache_stats = _compact_worker_column_pool(
            column_pool,
            int(getattr(cfg, "colgen_cache_columns_per_ev", 12)),
        )
        _WORKER_COLUMN_POOLS[fleet_key] = column_pool
        certificate["column_cache_before"] = int(cache_stats["before"])
        certificate["column_cache_after"] = int(cache_stats["after"])
        certificate["column_cache_dropped"] = int(cache_stats["dropped"])
        certificate["column_cache_limit_per_ev"] = int(
            cache_stats["limit_per_ev"]
        )
    certificate["rounds"] = int(colgen_rounds)
    cut = None
    if feasible is False:
        cut = feasibility_cut_from_certificate(
            certificate,
            lower_rows,
            upper_rows,
            candidate,
        )
        if cut is None and certificate.get("reason") == (
            "a vehicle has no feasible dispatch"
        ):
            cut = {
                "coefficient": np.zeros(first_stage_size, dtype=float),
                "constant": -1.0,
                "value_at_candidate": -1.0,
                "raw_farkas_value": -1.0,
                "cut_method": "colgen_intrinsic_infeasibility",
            }
    return {
        "scenario_index": int(scenario_index),
        "feasible": feasible,
        "colgen_rounds": int(colgen_rounds),
        "certificate": certificate,
        "cut": cut,
        "runtime_s": runtime,
        "oracle_solver": "column_generation",
    }


def _scenario_direct_oracle_task(payload):
    """Exact sparse fallback for one unfinished column-generation oracle."""

    (
        scenario_index,
        scenario,
        cfg,
        candidate,
        up_active,
        down_active,
        _colgen_max_rounds,
        first_stage_size,
        return_dispatch,
        *optional_time_limit,
    ) = payload
    oracle_time_limit_s = (
        optional_time_limit[0]
        if optional_time_limit
        else cfg.time_limit_s
    )
    lower, upper, lower_rows, upper_rows = affine_tracking_bands(
        cfg,
        candidate[: cfg.blocks],
        candidate[cfg.blocks : 2 * cfg.blocks],
        candidate[2 * cfg.blocks :],
        up_active,
        down_active,
        np.asarray(scenario.up_signal, dtype=float),
        np.asarray(scenario.down_signal, dtype=float),
    )
    started = time.perf_counter()
    decided = _circulation_decision(
        scenario_index, scenario, cfg, lower, upper, lower_rows, upper_rows,
        candidate, started, return_dispatch=bool(return_dispatch),
    )
    if decided is not None:
        return decided
    feasible, direct_rounds, certificate = certify_direct_phase_one(
        scenario.evs,
        lower,
        upper,
        steps=int(cfg.steps),
        dt=float(cfg.dt_hours),
        eta_ch=float(cfg.eta_ch),
        return_dispatch=bool(return_dispatch),
        time_limit_s=oracle_time_limit_s,
    )
    runtime = float(time.perf_counter() - started)
    certificate = dict(certificate)
    certificate["rounds"] = int(direct_rounds)
    cut = None
    if feasible is False:
        cut = feasibility_cut_from_certificate(
            certificate,
            lower_rows,
            upper_rows,
            candidate,
        )
        if cut is None and certificate.get("reason") == (
            "a vehicle has no feasible dispatch"
        ):
            cut = {
                "coefficient": np.zeros(first_stage_size, dtype=float),
                "constant": -1.0,
                "value_at_candidate": -1.0,
                "raw_farkas_value": -1.0,
                "cut_method": "direct_intrinsic_infeasibility",
            }
    return {
        "scenario_index": int(scenario_index),
        "feasible": feasible,
        "colgen_rounds": 0,
        "certificate": certificate,
        "cut": cut,
        "runtime_s": runtime,
        "oracle_solver": "direct_sparse_phase1",
        "fallback_solver": "direct_sparse_phase1",
    }


def _screened_result(scenario_index, runtime, solver):
    return {
        "scenario_index": int(scenario_index),
        "feasible": True,
        "colgen_rounds": 0,
        "certificate": {
            "complete": True,
            "prefix_circulation_screen": True,
        },
        "cut": None,
        "runtime_s": float(runtime),
        "oracle_solver": solver,
        "fallback_solver": "",
    }


def _circulation_decision(
    scenario_index,
    scenario,
    cfg,
    lower,
    upper,
    lower_rows,
    upper_rows,
    candidate,
    started,
    *,
    return_dispatch: bool,
):
    """Answer the scenario from one circulation solve, whichever way it goes.

    With unit charge/discharge efficiency the scenario recourse is a
    prefix-bounded matrix feasibility problem, so one feasible circulation
    answers it. The screen rounds the arc bounds in both directions and asks a
    compiled max flow: a feasible answer comes from a restriction of the real
    problem and an infeasible one from a relaxation, so neither can be wrong.
    An infeasible answer hands its Hoffman inequality to the Benders master in
    place of an LP dual. Measured at seven and twenty stations, on pools of
    128, 256 and 512 commands and on a second EV draw, the capacity came out
    identical to the LP dual every time, and the solve took a third to an
    eighth of the time. None means the exact LP has to answer.
    """

    if return_dispatch:
        return None
    circulation = scenario_circulation(
        scenario.evs,
        lower,
        upper,
        steps=int(cfg.steps),
        dt=float(cfg.dt_hours),
        eta_ch=float(cfg.eta_ch),
    )
    if circulation is None:
        return None
    feasible, _flow, violated = solve_circulation(circulation)
    if feasible:
        return _screened_result(
            scenario_index,
            time.perf_counter() - started,
            "prefix_circulation",
        )
    if violated is None:
        return None
    cut = circulation_master_cut(
        circulation, violated, lower_rows, upper_rows, candidate
    )
    if cut is None:
        return None
    return {
        "scenario_index": int(scenario_index),
        "feasible": False,
        "colgen_rounds": 0,
        "certificate": {
            "complete": True,
            "prefix_circulation_cut": True,
            "hoffman_slack": float(cut["hoffman_slack"]),
        },
        "cut": cut,
        "runtime_s": float(time.perf_counter() - started),
        "oracle_solver": "prefix_circulation_cut",
        "fallback_solver": "",
    }


def _dump_slow_scenario(payload, result, runtime_s: float) -> None:
    """Save the exact input of an unexpectedly slow oracle call.

    A batch is as slow as its slowest scenario, and a scenario that takes ten
    times what the same call takes on a bench cannot be explained from the
    outside.  Set EVMA_BENDERS_SLOW_SCENARIO_DUMP_S to a threshold in seconds
    and EVMA_BENDERS_SLOW_SCENARIO_DUMP_DIR to a directory to capture those
    calls for replay.  Off unless both are set.
    """

    import os

    threshold = os.environ.get("EVMA_BENDERS_SLOW_SCENARIO_DUMP_S")
    directory = os.environ.get("EVMA_BENDERS_SLOW_SCENARIO_DUMP_DIR")
    if not threshold or not directory:
        return
    try:
        if float(runtime_s) < float(threshold):
            return
    except (TypeError, ValueError):
        return
    import pickle
    from pathlib import Path as _Path

    target = _Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    existing = len(list(target.glob("slow_*.pkl")))
    if existing >= 40:
        return
    record = {
        "payload": payload,
        "runtime_s": float(runtime_s),
        "oracle_solver": str(result.get("oracle_solver", "")),
        "feasible": result.get("feasible"),
    }
    with (target / f"slow_{existing:03d}.pkl").open("wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _stratified_failure_selection(failures, limit):
    """Select cuts across both EV and activation scenario axes.

    Scenario pools are EV-major, so taking the first failures repeatedly spends
    a cut budget on one EV draw.  Greedy balancing keeps the small cut set
    representative without changing the final all-scenario acceptance test.
    A non-positive limit preserves the historical uncapped behaviour.
    """

    if limit is None or int(limit) <= 0 or len(failures) <= int(limit):
        return list(failures)

    def axes(scenario):
        metadata = getattr(scenario, "metadata", None) or {}
        return (
            metadata.get("ev_scenario"),
            metadata.get("activation_scenario"),
        )

    remaining = list(failures)
    selected: list = []
    ev_counts: dict = {}
    activation_counts: dict = {}
    while remaining and len(selected) < int(limit):
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                ev_counts.get(axes(remaining[index][0])[0], 0),
                activation_counts.get(axes(remaining[index][0])[1], 0),
                index,
            ),
        )
        scenario, recourse = remaining.pop(best_index)
        ev_key, activation_key = axes(scenario)
        ev_counts[ev_key] = ev_counts.get(ev_key, 0) + 1
        activation_counts[activation_key] = (
            activation_counts.get(activation_key, 0) + 1
        )
        selected.append((scenario, recourse))
    return selected


@dataclass
class _MasterResult:
    """One continuous first-stage master solve."""

    success: bool
    message: str
    x: np.ndarray | None = None
    fun: float = float("nan")
    active_matrix: np.ndarray | None = None
    active_rhs: np.ndarray | None = None
    termination_kind: str = "optimal"


def _solver_termination_kind(result) -> str:
    """Separate proven infeasibility from a limit or solver interruption."""

    if bool(getattr(result, "success", False)):
        return "optimal"
    status = int(getattr(result, "status", -1))
    if status == 1:
        return "limit"
    if status == 2:
        return "infeasible"
    if status == 3:
        return "unbounded"
    message = str(getattr(result, "message", "")).lower()
    if "time limit" in message or "iteration limit" in message:
        return "limit"
    if "infeasible" in message:
        return "infeasible"
    if "unbounded" in message:
        return "unbounded"
    return "solver_failure"


def _with_baseline_steps(
    matrix: np.ndarray,
    rhs: np.ndarray,
    step_matrix: np.ndarray | None,
    step_weight: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Append step variables z >= |D x| as the last columns.

    ``step_matrix`` D has the same columns as ``matrix``. Returns the widened
    rows, their right-hand side, and the number of step variables (0 when the
    weight or the participating pairs are absent).
    """

    if step_matrix is None or not len(step_matrix) or float(step_weight) <= 0.0:
        return matrix, rhs, 0
    count = int(step_matrix.shape[0])
    identity = np.eye(count)
    widened = np.vstack([
        np.hstack([matrix, np.zeros((matrix.shape[0], count))]),
        np.hstack([step_matrix, -identity]),
        np.hstack([-step_matrix, -identity]),
    ])
    return widened, np.concatenate([rhs, np.zeros(2 * count)]), count


def _solve_first_stage_master(
    *,
    objective: np.ndarray,
    cuts: list[dict],
    bounds: list[tuple[float, float]],
    time_limit_s: float | None,
    static_matrix: np.ndarray | None = None,
    static_rhs: np.ndarray | None = None,
    step_matrix: np.ndarray | None = None,
    step_weight: float = 0.0,
) -> _MasterResult:
    """Solve the continuous baseline/up/down master LP.

    ``static_matrix``/``static_rhs`` are rows every candidate must meet from
    the first round (Assessment I); the feasibility cuts are appended to them.
    ``step_matrix`` rows (``baseline_step_matrix``) put ``step_weight`` per kW
    of baseline step into the objective. ``fun`` includes that term;
    ``active_matrix`` holds only the rows over x.
    """

    rows = [-np.asarray(cut["coefficient"], dtype=float) for cut in cuts]
    rhs = [float(cut["constant"]) for cut in cuts]
    if static_matrix is not None and len(static_matrix):
        rows = list(np.asarray(static_matrix, dtype=float)) + rows
        rhs = list(np.asarray(static_rhs, dtype=float)) + rhs
    matrix = np.vstack(rows)
    rhs_vector = np.asarray(rhs, dtype=float)
    lp_matrix, lp_rhs, step_count = _with_baseline_steps(
        matrix, rhs_vector, step_matrix, step_weight
    )
    lp_objective = np.concatenate([objective, np.full(step_count, float(step_weight))])
    options = {"time_limit": float(time_limit_s)} if time_limit_s is not None else None
    result = linprog(
        lp_objective,
        A_ub=lp_matrix,
        b_ub=lp_rhs,
        bounds=list(bounds) + [(0.0, None)] * step_count,
        method="highs",
        options=options,
    )
    return _MasterResult(
        success=bool(result.success and result.x is not None),
        message=str(result.message),
        x=None if result.x is None else np.asarray(result.x, dtype=float)[: objective.size],
        fun=float(result.fun) if result.fun is not None else float("nan"),
        active_matrix=matrix,
        active_rhs=rhs_vector,
        termination_kind=_solver_termination_kind(result),
    )




def solve_joint_hard_bidding_benders(
    problem: JointBiddingProblem,
    scenarios: list,
    *,
    initial_solution: BiddingSolution,
    minimum_active_width_kw: float = 1.0,
    max_rounds: int = 20,
    cuts_per_round: int = 0,
    max_total_cuts: int = 0,
    allow_widen: bool = False,
    stabilize_master: bool = True,
    enforce_minimum_active_width: bool = True,
    colgen_max_rounds: int = 60,
    final_recourse_out: list[dict] | None = None,
    scenario_executor=None,
    progress=None,
) -> tuple[BiddingSolution, dict]:
    """Refine baseline and quantities for one participation pattern.

    Each scenario is certified by vehicle-wise column generation.  Its
    phase-one certificate becomes a cut in the 144-variable bid master.  If no
    common bid exists at the configured non-zero floor, ``master_infeasible``
    is returned to the outer active set.  With
    ``enforce_minimum_active_width=False``, the same cuts instead expose which
    active direction wants to fall below that floor.
    """

    cfg = problem.config
    objective_weights, base_min, base_max, u_cap, d_cap, *_ = problem_arrays(problem)
    assessment_matrix, assessment_rhs = assessment_i_rows(problem)
    # The first candidate must already meet Assessment I: a seed that the
    # scenarios happen to accept is otherwise certified without the master.
    initial_up, initial_down = project_assessment_i(
        problem,
        initial_solution.baseline_kw,
        initial_solution.up_kw,
        initial_solution.down_kw,
    )
    threshold = max(float(minimum_active_width_kw), cfg.eps * 10.0)
    up_active = initial_up >= threshold
    down_active = initial_down >= threshold
    first_stage_size = cfg.blocks * 3
    objective = np.zeros(first_stage_size, dtype=float)
    objective[cfg.blocks : 2 * cfg.blocks] = -objective_weights
    objective[2 * cfg.blocks :] = -objective_weights
    step_weight = max(float(getattr(cfg, "baseline_step_weight", 0.0)), 0.0)
    step_matrix = baseline_step_matrix(up_active | down_active, cfg.blocks)
    master_bounds: list[tuple[float, float]] = [
        (float(base_min[block]), float(base_max[block]))
        for block in range(cfg.blocks)
    ]
    master_bounds.extend(
        [
            (
                threshold
                if up_active[block] and enforce_minimum_active_width
                else 0.0,
                (
                    float(u_cap[block])
                    if allow_widen
                    else max(float(initial_up[block]), threshold)
                )
                if up_active[block]
                else 0.0,
            )
            for block in range(cfg.blocks)
        ]
    )
    master_bounds.extend(
        [
            (
                threshold
                if down_active[block] and enforce_minimum_active_width
                else 0.0,
                (
                    float(d_cap[block])
                    if allow_widen
                    else max(float(initial_down[block]), threshold)
                )
                if down_active[block]
                else 0.0,
            )
            for block in range(cfg.blocks)
        ]
    )
    cuts: list[dict] = []
    round_rows: list[dict] = []
    started = time.perf_counter()
    candidate = np.concatenate(
        [
            np.asarray(initial_solution.baseline_kw, dtype=float),
            initial_up,
            initial_down,
        ]
    )
    complete = False
    stop_reason = "round_limit"
    total_oracle_runtime = 0.0
    total_cut_runtime = 0.0
    floor_relaxed_candidate: np.ndarray | None = None
    floor_relaxed_master_status = "not_needed"
    floor_relaxed_master_runtime_s = 0.0

    def project_master_bounds(
        values: np.ndarray,
        bounds_to_apply: list[tuple[float | None, float | None]],
    ) -> np.ndarray:
        """Project harmless HiGHS bound-tolerance leakage before LP reuse."""

        projected = np.asarray(values, dtype=float).copy()
        for index, (lower, upper) in enumerate(bounds_to_apply):
            if lower is not None:
                projected[index] = max(projected[index], float(lower))
            if upper is not None:
                projected[index] = min(projected[index], float(upper))
        return projected

    # Spawning this repository's worker process imports the full research
    # package (including torch on Windows). Keep one pool alive across Benders
    # rounds; rebuilding it for every cut round changes no mathematical work.
    worker_count = max(1, min(
        int(getattr(cfg, "scenario_workers", 1)),
        len(scenarios) or 1,
    ))
    owns_executor = False
    if scenario_executor is not None:
        executor = scenario_executor
    elif worker_count > 1:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        owns_executor = True
    else:
        executor = None

    def run_oracle_batch(task, payloads, label: str):
        """Run one oracle wave in input order while exposing stragglers."""

        batch_started = time.perf_counter()
        if executor is None:
            results = [task(payload) for payload in payloads]
        else:
            results = [None] * len(payloads)
            futures = {
                executor.submit(task, payload): position
                for position, payload in enumerate(payloads)
            }
            for future in as_completed(futures):
                position = futures[future]
                result = future.result()
                results[position] = result
                runtime_s = float(result.get("runtime_s", 0.0))
                _dump_slow_scenario(payloads[position], result, runtime_s)
                if progress is not None and runtime_s >= 60.0:
                    progress(
                        f"benders: slow {label} scenario "
                        f"{result.get('scenario_index', position)} finished in "
                        f"{runtime_s:.1f}s"
                    )
        if progress is not None and results:
            slowest = max(results, key=lambda row: float(row.get("runtime_s", 0.0)))
            dropped = int(sum(
                int(row.get("certificate", {}).get("column_cache_dropped", 0))
                for row in results
            ))
            progress(
                f"benders: {label} batch completed {len(results)} scenarios "
                f"in {time.perf_counter() - batch_started:.1f}s wall; "
                f"slowest={slowest.get('scenario_index')} "
                f"{float(slowest.get('runtime_s', 0.0)):.1f}s; "
                f"cached-columns-dropped={dropped}"
            )
        return results

    def remaining_oracle_budget(oracle) -> float | None:
        if cfg.time_limit_s is None or float(cfg.time_limit_s) <= 0.0:
            return None
        return float(cfg.time_limit_s) - float(oracle.get("runtime_s", 0.0))

    def mark_oracle_budget_exhausted(oracle) -> None:
        certificate = dict(oracle.get("certificate", {}))
        certificate.update({
            "complete": False,
            "timed_out": True,
            "reason": "total scenario oracle time limit reached",
        })
        oracle["certificate"] = certificate
        oracle["feasible"] = None
        oracle["cut"] = None
    if final_recourse_out is not None:
        final_recourse_out.clear()

    all_scenario_indices = list(range(len(scenarios)))

    for round_index in range(max(int(max_rounds), 0) + 1):
        failures: list[tuple[object, dict]] = []
        scenario_rows: list[dict] = []
        oracle_incomplete = False
        watched_indices = all_scenario_indices
        oracle_payloads = [
            (
                scenario_index,
                scenarios[scenario_index],
                cfg,
                candidate,
                up_active,
                down_active,
                int(colgen_max_rounds),
                first_stage_size,
                # The trajectory is wanted only for the round that certifies,
                # and asking for it here costs every round the fast oracles,
                # which cannot produce one.  Collect it once, below, from the
                # round that actually succeeds.
                False,
                cfg.time_limit_s,
            )
            for scenario_index in watched_indices
        ]
        oracle_results = run_oracle_batch(
            _scenario_oracle_task,
            oracle_payloads,
            f"round-{round_index}-column-generation",
        )
        retry_positions = [
            index
            for index, oracle in enumerate(oracle_results)
            if oracle["feasible"] is None
            and "round limit" in str(
                oracle.get("certificate", {}).get("reason", "")
            ).lower()
        ]
        if retry_positions:
            if progress is not None:
                progress(
                    "benders: retrying "
                    f"{len(retry_positions)} unfinished column-generation "
                    f"oracles with {int(colgen_max_rounds) * 3} rounds"
                )
            retry_payloads = []
            eligible_retry_positions = []
            for position in retry_positions:
                remaining = remaining_oracle_budget(oracle_results[position])
                if remaining is not None and remaining <= 1e-3:
                    mark_oracle_budget_exhausted(oracle_results[position])
                    continue
                payload = list(oracle_payloads[position])
                payload[6] = int(colgen_max_rounds) * 3
                payload[9] = remaining
                retry_payloads.append(tuple(payload))
                eligible_retry_positions.append(position)
            if retry_payloads and executor is not None:
                retry_results = run_oracle_batch(
                    _scenario_oracle_task,
                    retry_payloads,
                    f"round-{round_index}-column-generation-retry",
                )
            elif retry_payloads:
                retry_results = run_oracle_batch(
                    _scenario_oracle_task,
                    retry_payloads,
                    f"round-{round_index}-column-generation-retry",
                )
            else:
                retry_results = []
            for position, retried in zip(
                eligible_retry_positions, retry_results
            ):
                original = oracle_results[position]
                retried["runtime_s"] = float(
                    original["runtime_s"] + retried["runtime_s"]
                )
                retried["colgen_rounds"] = int(
                    original["colgen_rounds"] + retried["colgen_rounds"]
                )
                retried["certificate"]["rounds"] = int(
                    retried["colgen_rounds"]
                )
                oracle_results[position] = retried
        direct_positions = [
            index
            for index, oracle in enumerate(oracle_results)
            if oracle["feasible"] is None
            and str(oracle.get("oracle_solver", "")) != "direct_sparse_phase1"
        ]
        if direct_positions:
            incomplete_reasons = sorted({
                str(
                    oracle_results[position]
                    .get("certificate", {})
                    .get("reason", "unknown")
                )
                for position in direct_positions
            })
            if progress is not None:
                progress(
                    "benders: exact direct phase-I fallback for "
                    f"{len(direct_positions)} unfinished oracle(s); reasons="
                    + " | ".join(incomplete_reasons)
                )
            direct_payloads = []
            eligible_direct_positions = []
            for position in direct_positions:
                remaining = remaining_oracle_budget(oracle_results[position])
                if remaining is not None and remaining <= 1e-3:
                    mark_oracle_budget_exhausted(oracle_results[position])
                    continue
                payload = list(oracle_payloads[position])
                payload[9] = remaining
                direct_payloads.append(tuple(payload))
                eligible_direct_positions.append(position)
            if direct_payloads and executor is not None:
                direct_results = run_oracle_batch(
                    _scenario_direct_oracle_task,
                    direct_payloads,
                    f"round-{round_index}-direct-fallback",
                )
            elif direct_payloads:
                direct_results = run_oracle_batch(
                    _scenario_direct_oracle_task,
                    direct_payloads,
                    f"round-{round_index}-direct-fallback",
                )
            else:
                direct_results = []
            for position, direct in zip(
                eligible_direct_positions, direct_results
            ):
                original = oracle_results[position]
                direct["runtime_s"] = float(
                    original["runtime_s"] + direct["runtime_s"]
                )
                direct["colgen_rounds"] = int(original["colgen_rounds"])
                direct["certificate"]["column_generation_rounds"] = int(
                    original["colgen_rounds"]
                )
                direct["certificate"]["column_generation_reason"] = str(
                    original.get("certificate", {}).get("reason", "")
                )
                oracle_results[position] = direct

        round_exact_slowest_runtime_s = max(
            (
                float(oracle.get("runtime_s", 0.0))
                for oracle in oracle_results
            ),
            default=0.0,
        )
        for oracle in oracle_results:
            scenario = scenarios[int(oracle["scenario_index"])]
            feasible = oracle["feasible"]
            colgen_rounds = int(oracle["colgen_rounds"])
            certificate = dict(oracle["certificate"])
            oracle_runtime = float(oracle["runtime_s"])
            total_oracle_runtime += oracle_runtime
            scenario_rows.append(
                {
                    "scenario": str(scenario.name),
                    "passed": feasible is True,
                    "status": (
                        "feasible"
                        if feasible is True
                        else "infeasible"
                        if feasible is False
                        else "incomplete"
                    ),
                    "colgen_rounds": int(colgen_rounds),
                    "colgen_columns": int(certificate.get("columns", 0)),
                    "column_cache_before": int(
                        certificate.get("column_cache_before", 0)
                    ),
                    "column_cache_after": int(
                        certificate.get("column_cache_after", 0)
                    ),
                    "column_cache_dropped": int(
                        certificate.get("column_cache_dropped", 0)
                    ),
                    "timed_out": bool(certificate.get("timed_out", False)),
                    "runtime_s": oracle_runtime,
                    "reason": str(certificate.get("reason", "")),
                    "column_generation_reason": str(
                        certificate.get("column_generation_reason", "")
                    ),
                    "fallback_solver": str(
                        oracle.get("fallback_solver", "")
                    ),
                    "oracle_solver": str(oracle.get("oracle_solver", "")),
                }
            )
            if feasible is None:
                oracle_incomplete = True
            if feasible is False:
                failures.append(
                    (
                        scenario,
                        {
                            "scenario_index": int(oracle["scenario_index"]),
                            "certificate": certificate,
                            "cut": oracle.get("cut"),
                        },
                    )
                )

        feasible_count = int(sum(
            oracle["feasible"] is True for oracle in oracle_results
        ))
        infeasible_count = int(sum(
            oracle["feasible"] is False for oracle in oracle_results
        ))
        incomplete_count = int(sum(
            oracle["feasible"] is None for oracle in oracle_results
        ))
        row = {
            "round": int(round_index),
            "failures": int(len(failures)),
            "feasible_scenarios": feasible_count,
            "infeasible_scenarios": infeasible_count,
            "incomplete_scenarios": incomplete_count,
            "exact_oracle_scenarios": int(len(oracle_results)),
            "exact_oracle_slowest_runtime_s": float(
                round_exact_slowest_runtime_s
            ),
            "pass_rate": float(
                feasible_count / len(scenarios)
                if scenarios
                else 1.0
            ),
            "cuts_before": int(len(cuts)),
            "scenario_rows": scenario_rows,
        }
        round_rows.append(row)
        if progress is not None:
            progress(
                f"benders: round {round_index}/{int(max_rounds)} "
                f"feasible/infeasible/unknown "
                f"{feasible_count}/{infeasible_count}/{incomplete_count} "
                f"cuts {len(cuts)} oracle {total_oracle_runtime:.0f}s"
            )
        if oracle_incomplete:
            stop_reason = "colgen_oracle_incomplete"
            break
        if not failures:
            complete = True
            stop_reason = "pool_certified"
            if final_recourse_out is not None:
                dispatch_keys = {
                    "scenario_power_kw",
                    "ev_power_kw",
                    "ev_energy_kwh",
                }
                if not all(
                    dispatch_keys.issubset(oracle["certificate"])
                    for oracle in oracle_results
                ):
                    # This round is already decided; re-run it only to obtain
                    # the trajectories the caller stores. One extra wave per
                    # certified build is enough. The normal oracle selector
                    # keeps the direct sparse LP for small fleets and column
                    # generation for large fleets; both return a dispatch when
                    # requested.
                    oracle_results = run_oracle_batch(
                        _scenario_oracle_task,
                        [
                            tuple(list(payload[:8]) + [True] + list(payload[9:]))
                            for payload in oracle_payloads
                        ],
                        f"round-{round_index}-dispatch",
                    )
                if all(
                    dispatch_keys.issubset(oracle["certificate"])
                    for oracle in oracle_results
                ):
                    final_recourse_out.extend(
                        {
                            "scenario_index": int(oracle["scenario_index"]),
                            "feasible": oracle["feasible"],
                            "rounds": int(oracle["colgen_rounds"]),
                            "info": oracle["certificate"],
                            "runtime_s": float(oracle["runtime_s"]),
                        }
                        for oracle in oracle_results
                    )
            break
        if round_index >= max(int(max_rounds), 0):
            stop_reason = "round_limit"
            break
        if max_total_cuts and len(cuts) >= int(max_total_cuts):
            stop_reason = "cut_budget_exhausted"
            break

        remaining_budget = (
            max(int(max_total_cuts) - len(cuts), 0)
            if max_total_cuts
            else 0
        )
        per_round = int(cuts_per_round)
        if remaining_budget:
            per_round = (
                min(per_round, remaining_budget)
                if per_round
                else remaining_budget
            )
        selected = _stratified_failure_selection(failures, per_round)
        new_cuts: list[dict] = []
        for scenario, failure in selected:
            cut_started = time.perf_counter()
            cut = failure.get("cut")
            total_cut_runtime += time.perf_counter() - cut_started
            if cut is not None:
                cut["scenario"] = str(scenario.name)
                cut["scenario_index"] = int(failure["scenario_index"])
                new_cuts.append(cut)
        row["cuts_added"] = int(len(new_cuts))
        row["cut_scenarios"] = [cut["scenario"] for cut in new_cuts]
        row["failures_considered"] = int(len(failures))
        row["cut_ev_scenarios"] = sorted(
            {
                (getattr(scenario, "metadata", None) or {}).get(
                    "ev_scenario"
                )
                for scenario, _ in selected
            },
            key=lambda value: (value is None, value),
        )
        row["cut_activation_scenarios"] = sorted(
            {
                (getattr(scenario, "metadata", None) or {}).get(
                    "activation_scenario"
                )
                for scenario, _ in selected
            },
            key=lambda value: (value is None, value),
        )
        if not new_cuts:
            stop_reason = "no_farkas_cut"
            break
        cuts.extend(new_cuts)

        master_started = time.perf_counter()
        master = _solve_first_stage_master(
            objective=objective,
            cuts=cuts,
            bounds=master_bounds,
            time_limit_s=cfg.time_limit_s,
            static_matrix=assessment_matrix,
            static_rhs=assessment_rhs,
            step_matrix=step_matrix,
            step_weight=step_weight,
        )
        row["master_runtime_s"] = float(
            time.perf_counter() - master_started
        )
        row["master_status"] = str(master.message)
        row["master_termination_kind"] = str(master.termination_kind)
        if not master.success:
            # Only a proven infeasible primary master can justify lifting the
            # participation floor and nominating a direction for retirement.
            # A time/iteration limit or solver failure is incomplete evidence.
            if master.termination_kind != "infeasible":
                stop_reason = (
                    "master_limit"
                    if master.termination_kind == "limit"
                    else "master_incomplete"
                )
                break
            if enforce_minimum_active_width:
                relaxed_bounds = list(master_bounds)
                for block in range(cfg.blocks):
                    if up_active[block]:
                        _lower, upper_bound = relaxed_bounds[
                            cfg.blocks + block
                        ]
                        relaxed_bounds[cfg.blocks + block] = (
                            0.0,
                            upper_bound,
                        )
                    if down_active[block]:
                        _lower, upper_bound = relaxed_bounds[
                            2 * cfg.blocks + block
                        ]
                        relaxed_bounds[2 * cfg.blocks + block] = (
                            0.0,
                            upper_bound,
                        )
                relaxed_started = time.perf_counter()
                relaxed_master = _solve_first_stage_master(
                    objective=objective,
                    cuts=cuts,
                    bounds=relaxed_bounds,
                    time_limit_s=cfg.time_limit_s,
                    static_matrix=assessment_matrix,
                    static_rhs=assessment_rhs,
                    step_matrix=step_matrix,
                    step_weight=step_weight,
                )
                floor_relaxed_master_runtime_s += float(
                    time.perf_counter() - relaxed_started
                )
                floor_relaxed_master_status = str(relaxed_master.message)
                row["floor_relaxed_master_status"] = (
                    floor_relaxed_master_status
                )
                if relaxed_master.success:
                    floor_relaxed_candidate = project_master_bounds(
                        relaxed_master.x,
                        relaxed_bounds,
                    )
            stop_reason = "master_infeasible"
            break

        master_matrix = master.active_matrix
        master_rhs = master.active_rhs
        primary_candidate = project_master_bounds(
            np.asarray(master.x, dtype=float), master_bounds
        )
        if stabilize_master:
            # The L1 term is a proximal step, not cosmetics.  Replacing it with
            # a path-independent tie-break -- take the deepest point of the cut
            # region at the optimal capacity -- was measured and it destroys
            # convergence: the analytic centre of a partial cut set is not near
            # the scenario-feasible set, so every round after the first came
            # back 0/32 feasible and the build died at the 800-cut budget.
            # Depth in the cuts says nothing about feasibility, because the
            # cuts describe only the part of the constraint set already found.
            # Which capacity-tied bid comes out therefore still depends on the
            # search path; removing that needs an explicit description of the
            # feasible region, not a different objective on the same iteration.
            # Variables (x, t, z): t >= |x - previous| is the proximal move and
            # z the baseline steps, which keep their weight in the objective
            # row so this step does not undo the step tie-break.
            previous_candidate = candidate.copy()
            zero_cut_block = np.zeros(
                (master_matrix.shape[0], first_stage_size), dtype=float
            )
            cut_rows, cut_rhs, step_count = _with_baseline_steps(
                np.hstack([master_matrix, zero_cut_block]),
                master_rhs,
                np.hstack([step_matrix, np.zeros_like(step_matrix)]),
                step_weight,
            )
            secondary_size = 2 * first_stage_size + step_count
            secondary_objective = np.zeros(secondary_size, dtype=float)
            secondary_objective[first_stage_size : 2 * first_stage_size] = 1.0
            objective_row = np.zeros(secondary_size, dtype=float)
            objective_row[:first_stage_size] = objective
            objective_row[2 * first_stage_size :] = step_weight
            identity = np.eye(first_stage_size, dtype=float)
            step_filler = np.zeros((first_stage_size, step_count), dtype=float)
            objective_tolerance = max(
                1e-7, abs(float(master.fun)) * 1e-10
            )
            secondary_matrix = np.vstack(
                [
                    cut_rows,
                    objective_row[None, :],
                    np.hstack([identity, -identity, step_filler]),
                    np.hstack([-identity, -identity, step_filler]),
                ]
            )
            secondary_rhs = np.concatenate(
                [
                    cut_rhs,
                    np.asarray([float(master.fun) + objective_tolerance]),
                    previous_candidate,
                    -previous_candidate,
                ]
            )
            secondary_started = time.perf_counter()
            secondary = linprog(
                secondary_objective,
                A_ub=secondary_matrix,
                b_ub=secondary_rhs,
                bounds=master_bounds
                + [(0.0, None)] * (first_stage_size + step_count),
                method="highs",
                options=(
                    {"time_limit": float(cfg.time_limit_s)}
                    if cfg.time_limit_s is not None
                    else None
                ),
            )
            row["stabilization_runtime_s"] = float(
                time.perf_counter() - secondary_started
            )
            row["stabilization_status"] = str(secondary.message)
            row["tie_break_rule"] = "proximal_l1_to_previous_candidate"
            if secondary.success and secondary.x is not None:
                candidate = project_master_bounds(
                    np.asarray(
                        secondary.x[:first_stage_size], dtype=float
                    ),
                    master_bounds,
                )
                row["master_l1_move_kw"] = float(
                    np.sum(np.abs(candidate - previous_candidate))
                )
            else:
                candidate = primary_candidate
        else:
            candidate = primary_candidate
        row["capacity_objective_kw_block"] = float(
            -np.dot(objective, primary_candidate)
        )
        row["baseline_step_kw"] = float(
            np.sum(np.abs(step_matrix @ candidate)) if len(step_matrix) else 0.0
        )

    if executor is not None and owns_executor:
        executor.shutdown(wait=True, cancel_futures=True)

    solution = BiddingSolution(
        status="optimal" if complete else "benders_incomplete",
        solver="scipy-highs-benders",
        objective_value=float(
            np.dot(
                objective_weights,
                candidate[cfg.blocks : 2 * cfg.blocks]
                + candidate[2 * cfg.blocks :],
            )
        ),
        baseline_kw=candidate[: cfg.blocks].copy(),
        up_kw=candidate[cfg.blocks : 2 * cfg.blocks].copy(),
        down_kw=candidate[2 * cfg.blocks :].copy(),
        metadata={},
    )
    summary = {
        "enabled": True,
        "method": "colgen_benders_fixed_participation",
        "complete": bool(complete),
        "stop_reason": stop_reason,
        "scenarios": int(len(scenarios)),
        "cuts": int(len(cuts)),
        "cut_methods": {
            method: int(
                sum(cut.get("cut_method") == method for cut in cuts)
            )
            for method in sorted(
                {
                    str(cut.get("cut_method", "unknown"))
                    for cut in cuts
                }
            )
        },
        "rounds": round_rows,
        "runtime_s": float(time.perf_counter() - started),
        "oracle_runtime_s": float(total_oracle_runtime),
        "cut_runtime_s": float(total_cut_runtime),
        "active_up_directions": int(np.count_nonzero(up_active)),
        "active_down_directions": int(np.count_nonzero(down_active)),
        "allow_widen": bool(allow_widen),
        "stabilize_master": bool(stabilize_master),
        "capacity_tie_break": (
            "proximal_l1_to_previous_candidate"
            if stabilize_master
            else "solver_choice"
        ),
        "enforce_minimum_active_width": bool(
            enforce_minimum_active_width
        ),
        "minimum_active_width_kw": float(threshold),
        "colgen_max_rounds": int(colgen_max_rounds),
        "colgen_cache_columns_per_ev": int(
            getattr(cfg, "colgen_cache_columns_per_ev", 12)
        ),
        "direct_oracle_max_evs": int(
            getattr(cfg, "direct_oracle_max_evs", 0)
        ),
        "scenario_oracle_time_limit_s": (
            None if cfg.time_limit_s is None else float(cfg.time_limit_s)
        ),
        "scenario_oracle_timeouts": int(sum(
            bool(scenario_row.get("timed_out", False))
            for round_row in round_rows
            for scenario_row in round_row.get("scenario_rows", [])
        )),
        "cached_columns_dropped": int(sum(
            int(scenario_row.get("column_cache_dropped", 0))
            for round_row in round_rows
            for scenario_row in round_row.get("scenario_rows", [])
        )),
        "shared_scenario_executor": bool(scenario_executor is not None),
        "exact_oracle_slowest_runtime_s": float(max(
            (
                float(row.get("exact_oracle_slowest_runtime_s", 0.0))
                for row in round_rows
            ),
            default=0.0,
        )),
        "floor_relaxed_master_status": floor_relaxed_master_status,
        "floor_relaxed_master_runtime_s": float(
            floor_relaxed_master_runtime_s
        ),
        "floor_relaxed_candidate": (
            None
            if floor_relaxed_candidate is None
            else floor_relaxed_candidate.astype(float).tolist()
        ),
    }
    solution.metadata["benders"] = summary
    return solution, summary

"""Blockwise day-ahead bid search for pretraining the lower controller.

The previous rule submitted ``fraction * reference_width`` in every block. That
creates many small non-zero awards. Since Assessment-II tolerance is 10 % of
the award, those blocks can demand single-digit-kW precision while adding very
little physical regulation capacity.

The production search keeps only direction participation outside the continuous
model. For a fixed participation pattern, a 144-variable master chooses every
block's baseline and up/down quantity; fixed candidates are certified by
vehicle-wise column generation. If the pattern cannot support the configured
non-zero floor, the outer active set retires the least valuable direction and
repeats. Every submitted point is finally certified under the full-award
assumption.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from EnvConfig import (
    LOWER_TRAIN_UPPER_BID_BENDERS_CUTS_PER_ROUND,
    LOWER_TRAIN_UPPER_BID_BENDERS_MAX_ROUNDS,
    LOWER_TRAIN_UPPER_BID_BENDERS_MAX_TOTAL_CUTS,
    LOWER_TRAIN_UPPER_BID_COLGEN_CACHE_COLUMNS_PER_EV,
    LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW,
    LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW,
    LOWER_TRAIN_UPPER_BID_DOWN_MAX_KW,
    LOWER_TRAIN_UPPER_BID_DIRECT_ORACLE_MAX_EVS,
    LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_TIME_LIMIT_S,
    LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC,
    LOWER_TRAIN_UPPER_BID_SEED,
    LOWER_TRAIN_UPPER_BID_UP_MAX_KW,
    LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES,
    LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_BID,
    LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_TIME_LIMIT_S,
    LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW,
    LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW_FRACTION,
    LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_MIN_DIRECTION_BID_KW,
)
from market.bid_env import N_BLOCKS, STEPS_PER_BLOCK
from market.physical_lp_bidding import BiddingLPConfig
from market.physical_lp_bidding.aggregate_energy import energy_feasible_initial_bid
from market.sustained_capability import (
    sustained_capability_quantile,
)


REVIEWED_FIXED_EV_30MIN_CAPABILITY = "reviewed_fixed_ev_bank_30min"


def _observed_activation_directions_by_block(
    activation_scenarios: list[dict],
    *,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return blocks in which each offered direction is actually exercised.

    A one-sided empirical library must not certify capacity in the absent
    direction merely because no scenario ever asks the fleet to deliver it.
    The mask is also useful for sparse libraries: capacity is offered only
    where at least one design scenario tests that direction.
    """

    observed_up = np.zeros(N_BLOCKS, dtype=bool)
    observed_down = np.zeros(N_BLOCKS, dtype=bool)
    expected_steps = N_BLOCKS * STEPS_PER_BLOCK
    for payload in activation_scenarios:
        for key, observed in (
            ("up_proxy", observed_up),
            ("down_proxy", observed_down),
        ):
            values = np.asarray(payload.get(key), dtype=float).reshape(-1)
            if values.size < expected_steps:
                raise ValueError(
                    f"activation scenario {key} has {values.size} steps; "
                    f"expected at least {expected_steps}"
                )
            active = values[:expected_steps].reshape(N_BLOCKS, STEPS_PER_BLOCK)
            observed |= np.any(active > float(eps), axis=1)
    return observed_up, observed_down


def minimum_bid_quantity_kw() -> float:
    """Return the minimum submitted delta-kW quantity, not the baseline."""

    explicit = str(LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW or "").strip()
    if explicit:
        research_floor = float(explicit)
    else:
        research_floor = (
            float(LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW_FRACTION)
            * min(
                float(LOWER_TRAIN_UPPER_BID_UP_MAX_KW),
                float(LOWER_TRAIN_UPPER_BID_DOWN_MAX_KW),
            )
        )
    value = max(
        research_floor,
        float(LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_MIN_DIRECTION_BID_KW),
    )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("minimum bid quantity must be finite and positive")
    return float(value)






def _dump_uncertified_candidate(certified_solution, validation, summary) -> None:
    """Save the bid a failed certification was about, for replay.

    The Benders loop and the final certification can disagree: the loop can
    stop with every command feasible while the certification, which recomputes
    SoC and tracking from the primal trajectories, still finds one that misses.
    The exception alone cannot say which side is right, because the bid it was
    about is gone by then.  Set EVMA_BLOCKWISE_UNCERTIFIED_DUMP_DIR to keep it.
    Off unless that is set.
    """

    import os

    directory = os.environ.get("EVMA_BLOCKWISE_UNCERTIFIED_DUMP_DIR")
    if not directory:
        return
    import pickle
    import time as _time
    from pathlib import Path as _Path

    target = _Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if len(list(target.glob("uncertified_*.pkl"))) >= 20:
        return
    record = {
        "baseline_kw": np.asarray(certified_solution.baseline_kw, dtype=float),
        "up_kw": np.asarray(certified_solution.up_kw, dtype=float),
        "down_kw": np.asarray(certified_solution.down_kw, dtype=float),
        "validation": validation,
        "benders": summary,
    }
    path = target / f"uncertified_{int(_time.time() * 1000)}.pkl"
    with path.open("wb") as handle:
        pickle.dump(record, handle)


def split_scenario_rows(rows) -> tuple[list[str], list[dict]]:
    """Split a validation report into rows nobody judged and rows that missed.

    A row with no recorded dispatch carries ``None`` for every rate, and the
    key is present, so ``row.get(key, 0.0)`` hands back that ``None`` rather
    than the default -- which is how a build died on ``float(None)``.  Both
    callers went through the same three lines to work this out, and fixing one
    of them is how the second was missed, so there is one of them now.
    """

    rows = list(rows)
    missing = [
        str(row.get("scenario", "")) for row in rows if row.get("dispatch_missing")
    ]
    failing = [
        {
            "scenario": str(row.get("scenario", "")),
            "tracking": float(row.get("global_step_pass_rate") or 0.0),
            "soc_ok": bool(row.get("soc_ok", False)),
            "failed_blocks": list(row.get("failed_block_indices", [])),
        }
        for row in rows
        if not bool(row.get("all_ok", False)) and not row.get("dispatch_missing")
    ]
    return missing, failing


def _missing_dispatch_reasons(recourse_solution, missing: list[str]) -> dict:
    """Count the recourse statuses of the scenarios that produced no dispatch.

    The decomposed recourse records one status per scenario.  Reading them back
    separates a command the oracle proved unholdable from one it simply did not
    finish, which the absence of a dispatch alone cannot distinguish.
    """

    metadata = getattr(recourse_solution, "metadata", None) or {}
    names = list(metadata.get("scenario_name_by_index") or [])
    statuses = list(metadata.get("scenario_statuses") or [])
    by_name = dict(zip((str(name) for name in names), statuses))
    counts: dict[str, int] = {}
    for name in missing:
        status = str(by_name.get(str(name), "unrecorded"))
        counts[status] = counts.get(status, 0) + 1
    return counts






def _reviewed_fixed_ev_assessment_i_bounds(lbt, ev_bank) -> tuple[np.ndarray, np.ndarray]:
    """Compute 30-minute Assessment-I limits across the selected EV bank."""

    bank = list(ev_bank)
    if not bank:
        raise ValueError("the reviewed Assessment-I bank must not be empty")
    up_kw, down_kw = sustained_capability_quantile(
        bank,
        quantile=0.0,
        duration_hours=0.5,
    )
    lbt._bid_build_log(
        "blockwise-bid-search: reviewed Assessment-I capability from the "
        f"fixed {len(bank)}-realization EV bank, 30min product duration: "
        f"up median {float(np.median(up_kw)):.0f}kW "
        f"down median {float(np.median(down_kw)):.0f}kW "
        "(no auxiliary capability draws)"
    )
    return up_kw, down_kw




def _apply_capability_bounds(
    capability,
    *,
    up_limit: np.ndarray,
    down_limit: np.ndarray,
    baseline_floor: np.ndarray,
    baseline_ceiling: np.ndarray,
    bound_baseline: bool = True,
):
    """Bound the whole day-ahead plan by the fleet's sustained capability.

    Both parts of the plan are 30-minute sustained commitments:

    - the submitted widths, because Assessment-I requires the awarded delta-kW
      to be deliverable for the full product duration. The reserve battery is
      reserved for the lower controller's tracking error rather than standby
      capacity, so the EV fleet has to prove this width itself;
    - the baseline, which the fleet holds inside the idle band whenever no
      instruction is present -- 237 of the 258 obligated steps on a typical
      2024-12-02 scenario.

    ``capability`` of ``None`` leaves every bound untouched.
    """

    if capability is None:
        return up_limit, down_limit, baseline_floor, baseline_ceiling
    capable_up, capable_down = capability
    capable_up = np.maximum(np.asarray(capable_up, dtype=float), 0.0)
    capable_down = np.maximum(np.asarray(capable_down, dtype=float), 0.0)
    bounded_floor = (
        np.maximum(baseline_floor, -capable_up)
        if bound_baseline
        else baseline_floor.copy()
    )
    bounded_ceiling = (
        np.minimum(baseline_ceiling, capable_down)
        if bound_baseline
        else baseline_ceiling.copy()
    )
    return (
        np.minimum(up_limit, capable_up),
        np.minimum(down_limit, capable_down),
        bounded_floor,
        bounded_ceiling,
    )


def _failed_market_block_indices(validation: dict) -> list[int]:
    """Return every 30-minute block that failed Assessment-II in any scenario."""

    failing_rows = [
        row
        for row in validation.get("scenario_rows", [])
        if not bool(row.get("all_ok", False))
    ]
    failed_frequency = dict(validation.get("failed_block_frequency", {}))
    return sorted(
        {
            int(block)
            for row in failing_rows
            for block in row.get("failed_block_indices", [])
            if 0 <= int(block) < N_BLOCKS
        }
        | {
            int(block)
            for block in failed_frequency
            if 0 <= int(block) < N_BLOCKS
        }
    )






def _jointly_certify_full_award_bid(
    lbt,
    *,
    baseline_plan: np.ndarray,
    submitted_up: np.ndarray,
    submitted_down: np.ndarray,
    activation_scenarios: list[dict],
    ev_bank: list,
    config: BiddingLPConfig,
    minimum_bid_quantity_kw: float,
    baseline_min_kw=None,
    baseline_max_kw=None,
    scenario_executor=None,
):
    """Return a jointly feasible full-award bid."""

    if baseline_min_kw is None:
        baseline_min_kw = float(LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW)
    if baseline_max_kw is None:
        baseline_max_kw = float(LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW)

    if not activation_scenarios:
        raise RuntimeError("blockwise bid cannot be certified without activation scenarios")
    if not ev_bank or not any(ev_bank):
        raise RuntimeError("blockwise bid cannot be certified without EV scenarios")

    forecast_payloads = list(activation_scenarios)
    forecast_scenarios = lbt.stratified_ev_activation_scenarios(
        ev_specs_by_scenario=ev_bank,
        activation_payloads=forecast_payloads,
        activations_per_ev=len(forecast_payloads),
    )
    initial_bid_up = np.asarray(
        submitted_up, dtype=float
    ).reshape(N_BLOCKS).copy()
    initial_bid_down = np.asarray(
        submitted_down, dtype=float
    ).reshape(N_BLOCKS).copy()
    problem = lbt.JointBiddingProblem(
        objective_weights=np.ones(N_BLOCKS, dtype=float),
        scenarios=forecast_scenarios,
        # The repair loop re-optimises the baseline, so it needs the same
        # sustained-capability box as the initial search; otherwise it can
        # walk the baseline back out to the nominal fleet rating.
        baseline_min_kw=baseline_min_kw,
        baseline_max_kw=baseline_max_kw,
        u_cap=initial_bid_up.copy(),
        d_cap=initial_bid_down.copy(),
        # Final certification rejects any failed market block. Model that same
        # hard contract in every Benders/column-generation solve.
        global_pass_rate=1.0,
        minimum_direction_bid_kw=float(minimum_bid_quantity_kw),
        config=config,
    )
    initial_solution = lbt.BiddingSolution(
        status="optimal",
        solver="rule_based_cap_fraction",
        objective_value=float(
            np.sum(np.asarray(submitted_up, dtype=float))
            + np.sum(np.asarray(submitted_down, dtype=float))
        ),
        baseline_kw=np.asarray(baseline_plan, dtype=float).reshape(N_BLOCKS).copy(),
        up_kw=np.asarray(submitted_up, dtype=float).reshape(N_BLOCKS).copy(),
        down_kw=np.asarray(submitted_down, dtype=float).reshape(N_BLOCKS).copy(),
    )

    # The same complete scenario set defines and certifies the robust problem.
    feedback_ev_bank = ev_bank
    feedback_scenarios = forecast_scenarios
    seed_validation = {
        "all_scenarios_ok": False,
        "soc_all_ok": False,
        "min_global_step_pass_rate": 0.0,
        "scenario_rows": [],
        "reason": "uncertified outer-participation seed",
    }
    (
        certified_solution,
        recourse_solution,
        validation,
        hard_refit_summary,
    ) = _certify_by_hard_lp_refit(
        lbt,
        problem=problem,
        initial_solution=initial_solution,
        feedback_scenarios=feedback_scenarios,
        initial_validation=seed_validation,
        scenario_executor=scenario_executor,
    )
    recourse_ok = bool(
        recourse_solution is not None and recourse_solution.success
    )
    required_rate_solver_ok = bool(
        recourse_ok
        and recourse_solution.metadata.get("required_rate_all_solved", False)
    )
    certified_capacity_kw_block = float(
        np.sum(np.asarray(certified_solution.up_kw, dtype=float))
        + np.sum(np.asarray(certified_solution.down_kw, dtype=float))
    )
    feasible = bool(
        certified_solution.success
        and recourse_ok
        and validation.get("all_scenarios_ok", False)
        and validation.get("soc_all_ok", False)
        and not _failed_market_block_indices(validation)
    )
    if not feasible:
        if bool(
            recourse_solution.metadata.get("final_recourse_skipped", False)
        ):
            raise RuntimeError(
                "blockwise full-award bid optimization was not certified; "
                "the final recourse was intentionally skipped, so the empty "
                "validation result must not be interpreted as an SoC miss. "
                f"benders={hard_refit_summary}"
            )
        rows = list(validation.get("scenario_rows", []))
        # A scenario with no recorded dispatch was never judged, so it must not
        # be listed among the ones that missed: the two need different fixes.
        missing, failing = split_scenario_rows(rows)
        if missing:
            _dump_uncertified_candidate(
                certified_solution, validation, hard_refit_summary
            )
            # Why there is no dispatch decides what to do about it.  The
            # recourse says "infeasible" when its pricing converged and the
            # command really cannot be held, and "colgen_incomplete" when it
            # ran out of rounds or time, which proves nothing either way.
            by_status = _missing_dispatch_reasons(recourse_solution, missing)
            raise RuntimeError(
                "blockwise full-award bid could not be certified: the recourse "
                f"produced no dispatch for {len(missing)} of {len(rows)} "
                "scenarios, so they were never judged. This is not a tracking "
                f"failure. reasons={by_status} scenarios={missing[:5]} "
                f"other_failures={len(failing)} "
                f"benders={hard_refit_summary.get('stop_reason')}"
            )
        _dump_uncertified_candidate(
            certified_solution, validation, hard_refit_summary
        )
        raise RuntimeError(
            "blockwise full-award bid failed joint certification: "
            f"min_tracking={validation.get('min_global_step_pass_rate', 0.0):.6f} "
            f"soc_all_ok={validation.get('soc_all_ok', False)} "
            f"required_rate_solver_ok={required_rate_solver_ok} failures={failing[:5]}"
        )
    feedback_summary = {
        "enabled": True,
        "method": "outer_participation_colgen_benders",
        "scenario_feedback_complete": bool(hard_refit_summary.get("complete", False)),
        "hard_lp_refit": hard_refit_summary,
        "required_rate_solver_all_solved": required_rate_solver_ok,
        "allowed_command_failures": 0,
        "certified_capacity_kw_block": certified_capacity_kw_block,
        "capacity_contract_ok": True,
        "certification_scorer": str(
            hard_refit_summary.get(
                "certification_scorer", "strict_frozen_bid_recourse"
            )
        ),
        "complete": feasible,
    }
    return (
        certified_solution,
        validation,
        feedback_summary,
        len(feedback_scenarios),
        feedback_ev_bank,
    )


def _certify_by_hard_lp_refit(
    lbt,
    *,
    problem,
    initial_solution,
    feedback_scenarios,
    initial_validation,
    scenario_executor=None,
):
    """Certify by continuous robust refits without inferring fake block failures.

    For a fixed up/down participation pattern, baseline and both directional
    quantities form a continuous robust problem.  Farkas-Benders keeps its
    144-variable master separate from the scenario EV/SoC recourse LPs. If the
    fixed pattern has no common feasible point, the same decomposed Benders
    model is solved with the configured active-direction floor temporarily
    relaxed. Directions below the floor become retirement candidates. The
    direction with the smallest offered-capacity loss per released kW is
    removed before re-solving.

    An infeasible recourse solution has no dispatch to score. Passing it to the
    validator fills the missing trajectory with zeros, which can make every
    submitted block look failed. Such rows are diagnostics, not evidence that
    all those market products should be retired.
    """

    del initial_validation  # An infeasible recourse has no block-level primal evidence.
    minimum_bid = max(float(problem.minimum_direction_bid_kw), 1.0)
    robust_problem = replace(
        problem,
        fixed_baseline=None,
        fixed_up=None,
        fixed_down=None,
        global_pass_rate=1.0,
    )
    up_caps = (
        np.full(N_BLOCKS, np.inf, dtype=float)
        if problem.u_cap is None
        else np.broadcast_to(
            np.asarray(problem.u_cap, dtype=float), (N_BLOCKS,)
        ).copy()
    )
    down_caps = (
        np.full(N_BLOCKS, np.inf, dtype=float)
        if problem.d_cap is None
        else np.broadcast_to(
            np.asarray(problem.d_cap, dtype=float), (N_BLOCKS,)
        ).copy()
    )
    baseline_min = np.broadcast_to(
        np.asarray(problem.baseline_min_kw, dtype=float), (N_BLOCKS,)
    ).copy()
    baseline_max = np.broadcast_to(
        np.asarray(problem.baseline_max_kw, dtype=float), (N_BLOCKS,)
    ).copy()
    solution = initial_solution
    active_set_rounds: list[dict] = []
    active_set_complete = False
    stop_reason = "direction_limit"
    accepted_recourse_tasks: list[dict] | None = None

    def compact(summary: dict) -> dict:
        benders_rounds = list(summary.get("rounds") or [])
        last_round = benders_rounds[-1] if benders_rounds else {}
        incomplete_rows = [
            {
                "scenario": str(row.get("scenario", "")),
                "reason": str(row.get("reason", "")),
                "column_generation_reason": str(
                    row.get("column_generation_reason", "")
                ),
                "fallback_solver": str(row.get("fallback_solver", "")),
            }
            for row in last_round.get("scenario_rows", [])
            if str(row.get("status", "")) == "incomplete"
        ]
        return {
            "complete": bool(summary.get("complete", False)),
            "stop_reason": str(summary.get("stop_reason", "")),
            "cuts": int(summary.get("cuts", 0)),
            "runtime_s": float(summary.get("runtime_s", 0.0)),
            "oracle_runtime_s": float(summary.get("oracle_runtime_s", 0.0)),
            "scenario_oracle_time_limit_s": summary.get(
                "scenario_oracle_time_limit_s"
            ),
            "scenario_oracle_timeouts": int(
                summary.get("scenario_oracle_timeouts", 0)
            ),
            "colgen_cache_columns_per_ev": int(
                summary.get("colgen_cache_columns_per_ev", 0)
            ),
            "direct_oracle_max_evs": int(
                summary.get("direct_oracle_max_evs", 0)
            ),
            "cached_columns_dropped": int(
                summary.get("cached_columns_dropped", 0)
            ),
            "cut_runtime_s": float(summary.get("cut_runtime_s", 0.0)),
            "exact_oracle_slowest_runtime_s": float(
                summary.get("exact_oracle_slowest_runtime_s", 0.0)
            ),
            "round_oracle_breakdown": [
                {
                    "round": int(row.get("round", 0)),
                    "exact": int(row.get("exact_oracle_scenarios", 0)),
                    "exact_slowest_runtime_s": float(
                        row.get("exact_oracle_slowest_runtime_s", 0.0)
                    ),
                }
                for row in benders_rounds
            ],
            "master_time_limit_scope": str(
                summary.get("master_time_limit_scope", "")
            ),
            "active_up_directions": int(summary.get("active_up_directions", 0)),
            "active_down_directions": int(summary.get("active_down_directions", 0)),
            "enforce_minimum_active_width": bool(
                summary.get("enforce_minimum_active_width", True)
            ),
            "last_round_counts": {
                "feasible": int(last_round.get("feasible_scenarios", 0)),
                "infeasible": int(last_round.get("infeasible_scenarios", 0)),
                "incomplete": int(last_round.get("incomplete_scenarios", 0)),
            },
            "incomplete_scenarios": incomplete_rows,
        }

    # There are at most 96 directional products. One direction is retired per
    # unsuccessful strict round, so this loop has a structural finite bound.
    for round_idx in range(2 * N_BLOCKS + 1):
        lbt._bid_build_log(
            "physical-hard-refit: strict decomposed Benders "
            f"round={round_idx}"
        )
        strict_recourse_tasks: list[dict] = []
        strict_solution, strict_summary = lbt.solve_joint_hard_bidding_benders(
            robust_problem,
            list(feedback_scenarios),
            initial_solution=solution,
            minimum_active_width_kw=minimum_bid,
            max_rounds=int(LOWER_TRAIN_UPPER_BID_BENDERS_MAX_ROUNDS),
            cuts_per_round=int(LOWER_TRAIN_UPPER_BID_BENDERS_CUTS_PER_ROUND),
            max_total_cuts=int(LOWER_TRAIN_UPPER_BID_BENDERS_MAX_TOTAL_CUTS),
            progress=lambda m: lbt._bid_build_log(f"blockwise-bid-search: {m}"),
            allow_widen=True,
            stabilize_master=True,
            enforce_minimum_active_width=True,
            final_recourse_out=strict_recourse_tasks,
            scenario_executor=scenario_executor,
        )
        round_row = {
            "round": int(round_idx),
            "strict": compact(strict_summary),
        }
        active_set_rounds.append(round_row)
        lbt._bid_build_log(
            "physical-hard-refit: strict result "
            f"round={round_idx} complete={strict_summary.get('complete', False)} "
            f"stop={strict_summary.get('stop_reason', '')} "
            f"cuts={strict_summary.get('cuts', 0)}"
        )
        if strict_summary.get("complete", False) and strict_solution.success:
            solution = strict_solution
            accepted_recourse_tasks = strict_recourse_tasks or None
            active_set_complete = True
            stop_reason = "strict_pattern_certified"
            break

        strict_floor_candidate = strict_summary.get("floor_relaxed_candidate")
        candidate_vector = (
            np.asarray(strict_floor_candidate, dtype=float).reshape(-1)
            if strict_floor_candidate is not None
            else np.asarray([], dtype=float)
        )
        relaxed_recourse_tasks: list[dict] = []
        if (
            candidate_vector.size == 3 * N_BLOCKS
            and np.all(np.isfinite(candidate_vector))
        ):
            relaxed_solution = replace(
                solution,
                status="benders_incomplete",
                solver="strict_cut_floor_relaxed_master",
                objective_value=float(np.sum(
                    candidate_vector[N_BLOCKS:]
                )),
                baseline_kw=candidate_vector[:N_BLOCKS].copy(),
                up_kw=candidate_vector[N_BLOCKS : 2 * N_BLOCKS].copy(),
                down_kw=candidate_vector[2 * N_BLOCKS :].copy(),
                scenario_power_kw={},
                ev_power_kw={},
                ev_energy_kwh={},
                metadata={},
            )
            relaxed_summary = {
                "complete": False,
                "stop_reason": "reused_strict_feasibility_cuts",
                "cuts": int(strict_summary.get("cuts", 0)),
                "runtime_s": float(
                    strict_summary.get("floor_relaxed_master_runtime_s", 0.0)
                ),
                "enforce_minimum_active_width": False,
            }
            round_row["floor_relaxed_source"] = "strict_master_cuts"
        else:
            lbt._bid_build_log(
                "physical-hard-refit: floor-relaxed diagnostic Benders "
                f"round={round_idx}"
            )
            relaxed_solution, relaxed_summary = lbt.solve_joint_hard_bidding_benders(
                robust_problem,
                list(feedback_scenarios),
                initial_solution=solution,
                minimum_active_width_kw=minimum_bid,
                # One cut/master update is enough to nominate a direction. The
                # next strict solve and final full EV x command recourse, not this
                # diagnostic, carry the feasibility guarantee.
                max_rounds=1,
                cuts_per_round=int(LOWER_TRAIN_UPPER_BID_BENDERS_CUTS_PER_ROUND),
                allow_widen=True,
                stabilize_master=True,
                enforce_minimum_active_width=False,
                final_recourse_out=relaxed_recourse_tasks,
                scenario_executor=scenario_executor,
            )
            round_row["floor_relaxed_source"] = "fresh_diagnostic_cuts"
        round_row["floor_relaxed"] = compact(relaxed_summary)
        lbt._bid_build_log(
            "physical-hard-refit: floor-relaxed result "
            f"round={round_idx} complete={relaxed_summary.get('complete', False)} "
            f"stop={relaxed_summary.get('stop_reason', '')} "
            f"cuts={relaxed_summary.get('cuts', 0)}"
        )
        relaxed_arrays_finite = bool(
            np.all(np.isfinite(np.asarray(relaxed_solution.baseline_kw, dtype=float)))
            and np.all(np.isfinite(np.asarray(relaxed_solution.up_kw, dtype=float)))
            and np.all(np.isfinite(np.asarray(relaxed_solution.down_kw, dtype=float)))
        )
        diagnostic_usable = bool(
            relaxed_arrays_finite
            and (
                relaxed_summary.get("complete", False)
                or int(relaxed_summary.get("cuts", 0)) > 0
                or round_row.get("floor_relaxed_source")
                == "strict_master_cuts"
            )
        )
        if not diagnostic_usable:
            stop_reason = "floor_relaxed_diagnostic_unavailable"
            break

        current_up = np.asarray(solution.up_kw, dtype=float).reshape(N_BLOCKS)
        current_down = np.asarray(solution.down_kw, dtype=float).reshape(N_BLOCKS)
        relaxed_up = np.asarray(relaxed_solution.up_kw, dtype=float).reshape(N_BLOCKS)
        relaxed_down = np.asarray(relaxed_solution.down_kw, dtype=float).reshape(N_BLOCKS)
        candidates: list[tuple[float, float, int, str, float, float]] = []
        tolerance = max(float(problem.config.eps) * 10.0, 1e-6)
        for direction, current, relaxed in (
            ("up", current_up, relaxed_up),
            ("down", current_down, relaxed_down),
        ):
            for block in range(N_BLOCKS):
                if current[block] <= tolerance:
                    continue
                if relaxed[block] + tolerance >= minimum_bid:
                    continue
                released_kw = max(minimum_bid - float(relaxed[block]), tolerance)
                capacity_loss_kw = float(current[block])
                candidates.append((
                    capacity_loss_kw / released_kw,
                    capacity_loss_kw,
                    int(block),
                    direction,
                    float(current[block]),
                    float(relaxed[block]),
                ))

        if not candidates and relaxed_summary.get("complete", False):
            # A certified floor-relaxed point whose active widths all meet the
            # floor is itself a valid strict point; this also handles tiny
            # numerical differences between the two masters.
            solution = relaxed_solution
            accepted_recourse_tasks = relaxed_recourse_tasks or None
            active_set_complete = True
            stop_reason = "floor_relaxed_solution_meets_floor"
            break
        if not candidates:
            stop_reason = "floor_relaxed_diagnostic_has_no_retirement_candidate"
            break

        selected = [min(candidates)]
        (
            score,
            capacity_loss_kw,
            block,
            direction,
            previous_kw,
            relaxed_kw,
        ) = selected[0]
        retired_keys = {(entry[2], entry[3]) for entry in selected}
        # Preserve participation for every non-selected direction, but seed
        # the next oracle at the floor-relaxed robust width. A below-floor
        # non-selected direction is clipped to the submission floor, not reset
        # to its old (often maximum) width. This avoids regenerating the same
        # infeasibility cuts after every one-direction active-set change.
        next_up = np.where(
            current_up > tolerance,
            np.maximum(relaxed_up, minimum_bid),
            0.0,
        )
        next_down = np.where(
            current_down > tolerance,
            np.maximum(relaxed_down, minimum_bid),
            0.0,
        )
        # HiGHS may return a point a few ulps outside a declared bound.  The
        # next frozen recourse validates fixed bids strictly, so project every
        # reused master value back to the physical box.  This is not an action
        # projection or feasibility repair: it only removes solver-tolerance
        # leakage from the exact same LP bounds.
        next_up = np.clip(next_up, 0.0, up_caps)
        next_down = np.clip(next_down, 0.0, down_caps)
        next_baseline = np.clip(
            np.asarray(relaxed_solution.baseline_kw, dtype=float),
            baseline_min,
            baseline_max,
        )
        for entry in selected:
            if entry[3] == "up":
                next_up[entry[2]] = 0.0
            else:
                next_down[entry[2]] = 0.0
        retired_rows = [
            {
                "direction": str(entry[3]),
                "block": int(entry[2]),
                "previous_kw": float(entry[4]),
                "floor_relaxed_kw": float(entry[5]),
                "offered_capacity_loss_kw": float(entry[1]),
                "loss_per_released_kw": float(entry[0]),
            }
            for entry in selected
        ]
        round_row["retired_directions"] = retired_rows
        round_row["retirement_candidate_count"] = int(len(candidates))
        round_row["other_directions_reseeded_at_floor"] = int(sum(
            1
            for candidate in candidates
            if (candidate[2], candidate[3]) not in retired_keys
        ))
        lbt._bid_build_log(
            "physical-hard-refit: retire "
            f"{len(selected)} direction(s) round={round_idx} "
            f"of {len(candidates)} below-floor candidates: "
            + ", ".join(
                f"{row['direction']}[{row['block']}] "
                f"{row['previous_kw']:.0f}->{row['floor_relaxed_kw']:.0f}kW"
                for row in retired_rows
            )
        )
        solution = replace(
            solution,
            status="optimal",
            solver="decomposed_benders_active_set_seed",
            objective_value=float(np.sum(next_up) + np.sum(next_down)),
            baseline_kw=next_baseline.copy(),
            up_kw=next_up,
            down_kw=next_down,
            scenario_power_kw={},
            ev_power_kw={},
            ev_energy_kwh={},
            metadata={},
        )

    fixed_problem = replace(
        problem,
        scenarios=list(feedback_scenarios),
        fixed_baseline=np.asarray(solution.baseline_kw, dtype=float).copy(),
        fixed_up=np.asarray(solution.up_kw, dtype=float).copy(),
        fixed_down=np.asarray(solution.down_kw, dtype=float).copy(),
        global_pass_rate=1.0,
        minimum_direction_bid_kw=minimum_bid,
    )

    if not active_set_complete:
        # An unfinished column-generation oracle is neither feasible nor
        # infeasible evidence. Re-solving all scenarios here cannot turn the
        # unaccepted first-stage bid into a certified one, so it is skipped.
        recourse_solution = replace(
            solution,
            status="benders_incomplete",
            solver="unaccepted_first_stage_no_final_recourse",
            scenario_power_kw={},
            ev_power_kw={},
            ev_energy_kwh={},
            metadata={"final_recourse_skipped": True},
        )
    elif accepted_recourse_tasks is None:
        recourse_solution = lbt._solve_fixed_bid_scenarios_decomposed(
            fixed_problem
        )
    else:
        recourse_solution = lbt._solve_fixed_bid_scenarios_decomposed(
            fixed_problem,
            precomputed_task_results=accepted_recourse_tasks,
        )
    validation = lbt.validate_joint_solution(fixed_problem, recourse_solution)
    reused_recourse_valid = bool(
        recourse_solution.success
        and validation.get("all_scenarios_ok", False)
        and validation.get("soc_all_ok", False)
        and not _failed_market_block_indices(validation)
    )
    if accepted_recourse_tasks is not None and not reused_recourse_valid:
        # The Benders oracle uses a conservative affine transition band, so its
        # dispatch should also pass the exact frozen-bid validator. Keep a
        # fail-safe: any numerical or future semantic drift triggers a fresh
        # independent recourse solve rather than accepting a cached trajectory.
        lbt._bid_build_log(
            "physical-hard-refit: cached final recourse failed validation; "
            "falling back to independent frozen-bid recourse"
        )
        recourse_solution = lbt._solve_fixed_bid_scenarios_decomposed(
            fixed_problem
        )
        validation = lbt.validate_joint_solution(
            fixed_problem, recourse_solution
        )
    complete = bool(
        active_set_complete
        and
        solution.success
        and recourse_solution.success
        and validation.get("all_scenarios_ok", False)
        and validation.get("soc_all_ok", False)
        and not _failed_market_block_indices(validation)
    )
    return solution, recourse_solution, validation, {
        "enabled": True,
        "method": "outer_participation_colgen_benders_active_set",
        "complete": complete,
        "stop_reason": stop_reason,
        "allowed_command_failures": 0,
        "certification_scorer": "strict_frozen_bid_recourse",
        "rounds": active_set_rounds,
        "retired_direction_count": int(sum(
            len(row.get("retired_directions") or []) for row in active_set_rounds
        )),
        "retirement_round_count": int(sum(
            bool(row.get("retired_directions")) for row in active_set_rounds
        )),
    }


def build_blockwise_bid_for_day(
    base_series,
    service_date: str | None,
    arrival_scenario=None,
    forecast_seed: int | None = None,
    assessment_band_fraction: float | None = None,
    scenario_workers: int = 1,
    scenario_executor=None,
) -> dict:
    """Maximize one 48-block bid over three EV-count cases and all commands."""

    from training import lower_bid_training as lbt

    arrival_probs = None
    day_context = None
    if arrival_scenario is not None:
        arrival_probs = getattr(arrival_scenario, "arrival_probabilities_by_station", None)
        day_context = getattr(arrival_scenario, "day_context", None)
    seed = int(LOWER_TRAIN_UPPER_BID_SEED if forecast_seed is None else forecast_seed)
    name = "outer_participation_colgen_benders"
    information_regime = "clairvoyant"
    activation_scenarios, activation_mode = lbt._activation_scenarios_for_day(
        service_date, seed
    )

    selected_band_fraction = float(
        LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC
        if assessment_band_fraction is None
        else assessment_band_fraction
    )
    if not np.isfinite(selected_band_fraction) or selected_band_fraction < 0.0:
        raise ValueError("assessment_band_fraction must be finite and non-negative")
    cfg = BiddingLPConfig(
        assessment_band_fraction=selected_band_fraction,
        scenario_workers=max(1, int(scenario_workers)),
        time_limit_s=(
            float(LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_TIME_LIMIT_S)
            if float(LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_TIME_LIMIT_S) > 0.0
            else None
        ),
        colgen_cache_columns_per_ev=max(
            int(LOWER_TRAIN_UPPER_BID_COLGEN_CACHE_COLUMNS_PER_EV), 0
        ),
        direct_oracle_max_evs=max(
            int(LOWER_TRAIN_UPPER_BID_DIRECT_ORACLE_MAX_EVS), 0
        ),
        apply_transition_band=False,
    )
    rng_state = lbt._capture_rng_state()
    try:
        candidate_count = int(LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES)
        if candidate_count < 1 or candidate_count == 2:
            raise ValueError(
                "LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES must be 1 for a "
                "single-realization diagnostic or at least 3 for min/median/max"
            )
        ev_candidates = lbt._sample_ev_scenario_bank(
            count=candidate_count,
            seed=seed,
            seed_offset=0,
            arrival_probs=arrival_probs,
            day_context=day_context,
            label=f"blockwise bid candidate {name}",
        )
        ev_bank, ev_scenario_selection = lbt._select_ev_scenarios_by_count(
            ev_candidates
        )
        for selected in ev_scenario_selection:
            lbt._bid_build_log(
                "physical-joint: selected EV scenario "
                f"{selected['label']} candidate="
                f"{selected['candidate_index'] + 1}/{candidate_count} "
                f"rank={selected['rank_zero_based'] + 1}/{candidate_count} "
                f"evs={selected['ev_count']}"
            )
        baseline_plan = lbt._mean_natural_baseline(
            ev_bank=ev_bank,
            config=cfg,
            baseline_min_kw=float(LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW),
            baseline_max_kw=float(LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW),
        )
    finally:
        lbt._restore_rng_state(rng_state)

    up_limit = np.full(
        N_BLOCKS, max(float(LOWER_TRAIN_UPPER_BID_UP_MAX_KW), 0.0), dtype=float
    )
    down_limit = np.full(
        N_BLOCKS, max(float(LOWER_TRAIN_UPPER_BID_DOWN_MAX_KW), 0.0), dtype=float
    )
    baseline_floor = np.full(
        N_BLOCKS, float(LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW), dtype=float
    )
    baseline_ceiling = np.full(
        N_BLOCKS, float(LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW), dtype=float
    )
    capability = _reviewed_fixed_ev_assessment_i_bounds(lbt, ev_bank)
    up_limit, down_limit, baseline_floor, baseline_ceiling = _apply_capability_bounds(
        capability,
        up_limit=up_limit,
        down_limit=down_limit,
        baseline_floor=baseline_floor,
        baseline_ceiling=baseline_ceiling,
        # The exact scenario recourse already constrains baseline power, SoC,
        # and departure energy. The reviewed Assessment-I check concerns the
        # offered 30-minute directional quantity, not an extra baseline box.
        bound_baseline=False,
    )
    natural_baseline = np.clip(
        np.asarray(baseline_plan, dtype=float).reshape(N_BLOCKS),
        baseline_floor,
        baseline_ceiling,
    )
    minimum_delta_kw = minimum_bid_quantity_kw()
    forecast_payloads = list(activation_scenarios)
    observed_up, observed_down = _observed_activation_directions_by_block(
        forecast_payloads
    )
    up_limit = np.where(observed_up, up_limit, 0.0)
    down_limit = np.where(observed_down, down_limit, 0.0)
    initial_activations_per_ev = len(forecast_payloads)
    initial_scenario_count = len(ev_bank) * initial_activations_per_ev
    lbt._bid_build_log(
        "blockwise-bid-search: outer participation active-set with "
        "column-generation feasibility cuts for baseline and per-block "
        "up/down delta-kW quantities "
        f"(minimum bid quantity={minimum_delta_kw:.1f}kW)"
    )
    initial_up_active = up_limit >= minimum_delta_kw - cfg.eps
    initial_down_active = down_limit >= minimum_delta_kw - cfg.eps
    initial_submitted_up = np.where(initial_up_active, up_limit, 0.0)
    initial_submitted_down = np.where(initial_down_active, down_limit, 0.0)
    energy_initial_summary = {"rule": "assessment_i_caps"}
    if LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_BID:
        energy_seed = energy_feasible_initial_bid(
            lbt.stratified_ev_activation_scenarios(
                ev_specs_by_scenario=ev_bank,
                activation_payloads=forecast_payloads,
                activations_per_ev=len(forecast_payloads),
            ),
            cfg,
            baseline_min=baseline_floor,
            baseline_max=baseline_ceiling,
            up_cap=initial_submitted_up,
            down_cap=initial_submitted_down,
            minimum_bid_kw=minimum_delta_kw,
            time_limit_s=(
                float(LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_TIME_LIMIT_S)
                if float(LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_TIME_LIMIT_S) > 0.0
                else None
            ),
            progress=lbt._bid_build_log,
        )
        if energy_seed is None:
            energy_initial_summary = {
                "rule": "assessment_i_caps",
                "aggregate_energy_seed": "no_aggregate_feasible_pattern",
            }
        else:
            natural_baseline = energy_seed.baseline_kw.copy()
            initial_submitted_up = energy_seed.up_kw.copy()
            initial_submitted_down = energy_seed.down_kw.copy()
            energy_initial_summary = dict(energy_seed.summary)
    initial_solution = lbt.BiddingSolution(
        status="optimal",
        solver="outer-participation-seed",
        objective_value=lbt._capacity_objective_kw_block(
            initial_submitted_up, initial_submitted_down
        ),
        baseline_kw=natural_baseline.copy(),
        up_kw=initial_submitted_up.copy(),
        down_kw=initial_submitted_down.copy(),
        metadata={
            "tracking_contract": "uncertified_seed",
            "n_binary_vars": 0,
            "runtime_s": 0.0,
            "stop_reason": "outer_participation_seed",
            "wall_clock_bound_result": False,
        },
    )
    initial_validation = {
        "all_scenarios_ok": False,
        "soc_all_ok": False,
        "min_global_step_pass_rate": 0.0,
        "scenario_rows": [],
        "reason": "uncertified outer-participation seed",
    }
    baseline_plan = natural_baseline.copy()
    initial_capacity = lbt._capacity_objective_kw_block(
        initial_submitted_up, initial_submitted_down
    )
    (
        certified,
        validation,
        feedback_summary,
        certified_scenarios,
        certification_ev_bank,
    ) = _jointly_certify_full_award_bid(
        lbt,
        baseline_plan=baseline_plan,
        submitted_up=initial_submitted_up,
        submitted_down=initial_submitted_down,
        activation_scenarios=list(activation_scenarios or []),
        ev_bank=ev_bank,
        config=cfg,
        minimum_bid_quantity_kw=minimum_delta_kw,
        baseline_min_kw=baseline_floor.copy(),
        baseline_max_kw=baseline_ceiling.copy(),
        scenario_executor=scenario_executor,
    )
    baseline_plan = np.asarray(certified.baseline_kw, dtype=float).reshape(N_BLOCKS)
    up_plan = np.asarray(certified.up_kw, dtype=float).reshape(N_BLOCKS)
    down_plan = np.asarray(certified.down_kw, dtype=float).reshape(N_BLOCKS)
    for label, values in (("up", up_plan), ("down", down_plan)):
        too_small = (values > 1e-8) & (values < minimum_delta_kw - 1e-5)
        if np.any(too_small):
            raise RuntimeError(
                f"certified {label} submission fell below participation floor "
                f"in blocks {np.flatnonzero(too_small).astype(int).tolist()}"
            )

    target, tol, regulation = lbt._bid_to_target_and_tolerance(
        base_series,
        baseline_plan,
        up_plan,
        down_plan,
        band_fraction=selected_band_fraction,
    )
    offered_capacity = lbt._capacity_objective_kw_block(up_plan, down_plan)
    result = lbt.SubmittedBidResult(
        solved=True,
        feasible=True,
        status="outer_participation_colgen_benders_certified",
        objective_capacity_kw_block=float(offered_capacity),
        method="outer_participation_colgen_benders_certified",
        n_scenarios=int(certified_scenarios),
        mean_baseline_kw=float(np.mean(baseline_plan)),
        mean_up_kw=float(np.mean(up_plan)),
        mean_down_kw=float(np.mean(down_plan)),
        up_pass_rate=float(validation.get("min_up_step_pass_rate", 0.0)),
        down_pass_rate=float(validation.get("min_down_step_pass_rate", 0.0)),
        global_tracking_rate=float(
            validation.get("min_global_step_pass_rate", 0.0)
        ),
        soc_hit_rate=1.0 if bool(validation.get("soc_all_ok", False)) else 0.0,
    )
    validation_summary = {
        key: value
        for key, value in validation.items()
        if key != "scenario_rows"
    }
    initial_validation_summary = {
        key: value
        for key, value in initial_validation.items()
        if key != "scenario_rows"
    }
    positive_up = up_plan[up_plan > 1e-8]
    positive_down = down_plan[down_plan > 1e-8]
    summary = {
        "method": "outer_participation_colgen_benders_certified",
        "condition": str(name),
        "minimum_bid_quantity_kw": float(minimum_delta_kw),
        "minimum_bid_quantity_mode": "configured_floor",
        "award_assumption": "full_award",
        "assessment_band_fraction": float(selected_band_fraction),
        "ev_information_regime": information_regime,
        "all_command_scenarios": True,
        "allowed_command_failures": 0,
        "assessment_i_capability_policy": REVIEWED_FIXED_EV_30MIN_CAPABILITY,
        "assessment_i_capability_ev_scenarios": int(len(ev_bank)),
        "ev_scenario_candidate_count": int(candidate_count),
        "ev_scenario_selection": list(ev_scenario_selection),
        "assessment_i_capability_duration_minutes": 30.0,
        "apply_transition_band": False,
        "per_block_baseline_search": True,
        "per_block_delta_kw_quantity_search": True,
        "directional_participation_search": True,
        "observed_up_activation_blocks": np.flatnonzero(observed_up).astype(int).tolist(),
        "observed_down_activation_blocks": np.flatnonzero(observed_down).astype(int).tolist(),
        "initial_search_validation": initial_validation_summary,
        "initial_search_scenarios": int(initial_scenario_count),
        "initial_search_ev_scenarios": int(len(ev_bank)),
        "initial_search_activation_scenarios_per_ev": int(
            initial_activations_per_ev
        ),
        "initial_search_solver": str(initial_solution.solver),
        "initial_search_status": str(initial_solution.status),
        "initial_search_tracking_contract": str(
            initial_solution.metadata.get("tracking_contract", "")
        ),
        "initial_search_n_binary_vars": 0,
        "initial_search_runtime_s": float(
            initial_solution.metadata.get("runtime_s", np.nan)
        ),
        "initial_search_stop_reason": str(
            initial_solution.metadata.get("stop_reason", "unknown")
        ),
        "initial_search_wall_clock_bound_result": bool(
            initial_solution.metadata.get("wall_clock_bound_result", False)
        ),
        "submitted_up_participating_blocks": int(np.count_nonzero(up_plan > 1e-8)),
        "submitted_down_participating_blocks": int(
            np.count_nonzero(down_plan > 1e-8)
        ),
        "submitted_up_nonzero_min_kw": (
            float(np.min(positive_up)) if positive_up.size else 0.0
        ),
        "submitted_up_nonzero_max_kw": (
            float(np.max(positive_up)) if positive_up.size else 0.0
        ),
        "submitted_down_nonzero_min_kw": (
            float(np.min(positive_down)) if positive_down.size else 0.0
        ),
        "submitted_down_nonzero_max_kw": (
            float(np.max(positive_down)) if positive_down.size else 0.0
        ),
        "bid_quantity_upper_bound": "configured_physical_limit",
        "offered_capacity_kw_block": float(offered_capacity),
        "mean_offered_capacity_kw": float(offered_capacity / N_BLOCKS),
        "initial_offered_capacity_kw_block": float(initial_capacity),
        "initial_bid": energy_initial_summary,
        "joint_certification": {
            "scenarios": int(certified_scenarios),
            "validation": validation_summary,
            "feedback": feedback_summary,
            # In-sample: the repair loop was driven by these same scenarios.
            "in_sample": True,
        },
        "searched": True,
        "search_objective": "maximize_total_ev_regulation_capacity_kw_block",
        "block_allocation_rule": (
            "unit capacity weights; physical Assessment-I, EV availability, "
            "SoC and all command-scenario recourse decide placement; L1 "
            "stabilization preserves the prior feasible shape among equal-"
            "capacity optima"
        ),
        "participation_rule": (
            "for each block and direction independently: binary participate, "
            "then optimize quantity continuously between the non-zero floor "
            "and the configured market-facing limit; sampled command recourse "
            "and departure SoC determine EV feasibility"
        ),
    }
    participating = (up_plan > 1e-8) | (down_plan > 1e-8)
    award_metadata = {
        "method": "full_award_assumption",
        "submitted_blocks": int(np.count_nonzero(participating)),
        "awarded_blocks": int(np.count_nonzero(participating)),
        "sampled_awarded_block_rate": 1.0 if np.any(participating) else 0.0,
        "sampled_awarded_quantity_rate": 1.0 if np.any(participating) else 0.0,
        "awarded_block_indices": np.flatnonzero(participating).astype(int).tolist(),
    }
    final_payload = {
        "feasibility_mode": "outer_participation_colgen_benders_certified",
        "bid_feasibility": summary,
        "service_date": service_date,
        "forecast_seed": int(seed),
        "assessment_band_fraction": float(selected_band_fraction),
        "bid_objective": "total_ev_regulation_capacity_kw_block",
        "ev_information_regime": information_regime,
        "all_command_scenarios": True,
        "allowed_command_failures": 0,
        "assessment_i_capability_policy": REVIEWED_FIXED_EV_30MIN_CAPABILITY,
        "assessment_i_capability_ev_scenarios": int(len(ev_bank)),
        "assessment_i_capability_duration_minutes": 30.0,
        "apply_transition_band": False,
        "bid_ev_scenario_bank": certification_ev_bank,
        "ev_scenario_candidate_count": int(candidate_count),
        "ev_scenario_selection": list(ev_scenario_selection),
        # Assessment I is a physical capability gate, independent of the
        # Assessment-II tracking-band counterfactual.  These are the sustained
        # 30-minute directional limits applied before the B/U/D search.
        "assessment_i_up_limit_kw": up_limit.copy(),
        "assessment_i_down_limit_kw": down_limit.copy(),
        "base_series": np.asarray(base_series, dtype=np.float32).reshape(-1)[:len(target)],
        "arrival_probabilities_by_station": arrival_probs,
        "day_context": day_context,
        "result": result,
        "baseline_plan": baseline_plan,
        "up_plan": up_plan,
        "down_plan": down_plan,
        "target_series": target,
        "tol_series": tol,
        "regulation_series": regulation,
        "activation_mode": str(activation_mode),
        "activation_scenarios": int(len(activation_scenarios or [])),
        "activation_scenario_payload": activation_scenarios,
        "fixed_ev_scenario_bank_used": True,
        "submitted_up_plan": up_plan.copy(),
        "submitted_down_plan": down_plan.copy(),
        "awarded_fraction_by_block": participating.astype(float),
        "award_metadata": award_metadata,
        "submitted_capacity_kw_block": float(offered_capacity),
        "awarded_capacity_kw_block": float(offered_capacity),
    }
    return final_payload

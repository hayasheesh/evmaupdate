"""Shared input and result validation for the current upper-bid LP path."""

from __future__ import annotations

import numpy as np

from market.bid_participation import (
    masked_pass_rate,
    participation_by_block,
    zero_instruction_tolerance_by_block,
)

from .colgen_feasibility import step_violation_tolerance_kw
from .data_classes import BiddingSolution, JointBiddingProblem


def _block_array(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.full(size, float(array[0]), dtype=float)
    if array.size != size:
        raise ValueError(
            f"{name} must have shape ({size},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def unique_scenario_names(problem: JointBiddingProblem) -> list[str]:
    used: set[str] = set()
    names: list[str] = []
    for index, scenario in enumerate(problem.scenarios):
        base = str(scenario.name or f"scenario_{index}")
        name = base
        suffix = 1
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)
    return names


def ev_labels(problem: JointBiddingProblem) -> dict[tuple[int, int], str]:
    labels: dict[tuple[int, int], str] = {}
    for scenario_index, scenario in enumerate(problem.scenarios):
        used: set[str] = set()
        for ev_index, ev in enumerate(scenario.evs):
            base = str(
                ev.ev_id
                if ev.ev_id is not None
                else f"st{ev.station_id}_ev{ev_index}"
            )
            label = base
            suffix = 1
            while label in used:
                label = f"{base}_{suffix}"
                suffix += 1
            used.add(label)
            labels[(scenario_index, ev_index)] = label
    return labels


def problem_arrays(
    problem: JointBiddingProblem,
    *,
    validate_fixed_minimum: bool = True,
):
    """Validate one first-stage problem and return normalized block arrays."""

    config = problem.config
    if config.steps != config.blocks * config.steps_per_block:
        raise ValueError("steps must equal blocks * steps_per_block")
    if not problem.scenarios:
        raise ValueError("at least one EV/activation scenario is required")
    if (
        abs(float(config.eta_ch) - 1.0) > 1e-9
        or abs(float(config.eta_dis) - 1.0) > 1e-9
    ):
        raise ValueError(
            "signed-power recourse requires eta_ch=eta_dis=1.0"
        )

    objective_weights = _block_array(
        problem.objective_weights, config.blocks, "objective_weights"
    )
    if np.any(objective_weights < 0.0):
        raise ValueError("objective_weights must be non-negative")
    baseline_min = _block_array(
        problem.baseline_min_kw, config.blocks, "baseline_min_kw"
    )
    baseline_max = _block_array(
        problem.baseline_max_kw, config.blocks, "baseline_max_kw"
    )
    if np.any(baseline_min > baseline_max):
        raise ValueError("baseline_min_kw exceeds baseline_max_kw")

    max_fleet_kw = max(
        (
            sum(
                max(
                    float(ev.max_charge_kw),
                    float(ev.max_discharge_kw),
                    0.0,
                )
                for ev in scenario.evs
            )
            for scenario in problem.scenarios
        ),
        default=0.0,
    )
    default_cap = max(
        max_fleet_kw + float(np.max(np.abs(baseline_min))),
        max_fleet_kw + float(np.max(np.abs(baseline_max))),
        0.0,
    )
    up_cap = _block_array(
        problem.u_cap if problem.u_cap is not None else default_cap,
        config.blocks,
        "u_cap",
    )
    down_cap = _block_array(
        problem.d_cap if problem.d_cap is not None else default_cap,
        config.blocks,
        "d_cap",
    )
    if np.any(up_cap < 0.0) or np.any(down_cap < 0.0):
        raise ValueError("reserve caps must be non-negative")

    fixed_baseline = (
        None
        if problem.fixed_baseline is None
        else _block_array(
            problem.fixed_baseline, config.blocks, "fixed_baseline"
        )
    )
    fixed_up = (
        None
        if problem.fixed_up is None
        else _block_array(problem.fixed_up, config.blocks, "fixed_up")
    )
    fixed_down = (
        None
        if problem.fixed_down is None
        else _block_array(problem.fixed_down, config.blocks, "fixed_down")
    )
    if fixed_baseline is not None and (
        np.any(fixed_baseline < baseline_min - 1e-9)
        or np.any(fixed_baseline > baseline_max + 1e-9)
    ):
        raise ValueError("fixed_baseline is outside baseline bounds")
    if fixed_up is not None and (
        np.any(fixed_up < -1e-9) or np.any(fixed_up > up_cap + 1e-9)
    ):
        raise ValueError("fixed_up is outside up caps")
    if fixed_down is not None and (
        np.any(fixed_down < -1e-9)
        or np.any(fixed_down > down_cap + 1e-9)
    ):
        raise ValueError("fixed_down is outside down caps")

    minimum_bid = max(float(problem.minimum_direction_bid_kw), 1.0)
    if (
        validate_fixed_minimum
        and fixed_up is not None
        and np.any(
            (fixed_up > config.eps)
            & (fixed_up < minimum_bid - 1e-9)
        )
    ):
        raise ValueError(
            "fixed_up contains a non-zero award below "
            "minimum_direction_bid_kw"
        )
    if (
        validate_fixed_minimum
        and fixed_down is not None
        and np.any(
            (fixed_down > config.eps)
            & (fixed_down < minimum_bid - 1e-9)
        )
    ):
        raise ValueError(
            "fixed_down contains a non-zero award below "
            "minimum_direction_bid_kw"
        )

    pass_rate = float(problem.global_pass_rate)
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError("global_pass_rate must be in [0, 1]")
    for scenario_index, scenario in enumerate(problem.scenarios):
        up_signal = np.asarray(scenario.up_signal, dtype=float).reshape(-1)
        down_signal = np.asarray(
            scenario.down_signal, dtype=float
        ).reshape(-1)
        if (
            up_signal.size != config.steps
            or down_signal.size != config.steps
        ):
            raise ValueError(
                f"scenario {scenario_index} signal length must be "
                f"{config.steps}"
            )
        if np.any(~np.isfinite(up_signal)) or np.any(
            ~np.isfinite(down_signal)
        ):
            raise ValueError(
                f"scenario {scenario_index} contains non-finite activation"
            )
        if np.any(
            (up_signal > config.eps) & (down_signal > config.eps)
        ):
            raise ValueError(
                f"scenario {scenario.name!r} has simultaneous "
                "up/down activation"
            )
    return (
        objective_weights,
        baseline_min,
        baseline_max,
        up_cap,
        down_cap,
        fixed_baseline,
        fixed_up,
        fixed_down,
    )


def fixed_bid_tracking_bands(
    config,
    baseline: np.ndarray,
    up_kw: np.ndarray,
    down_kw: np.ndarray,
    up_signal: np.ndarray,
    down_signal: np.ndarray,
    *,
    apply_transition_band: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact numeric Assessment-II bands for one frozen bid.

    The optimizer's counterpart is ``colgen_feasibility.affine_tracking_bands``,
    which returns coefficients rather than numbers because the bid is still
    unknown there. This one is for a bid already chosen, and is the stricter of
    the two: it applies the response-interval widening at a block boundary,
    which the affine form cannot express without disjunctions.
    """

    baseline = np.asarray(baseline, dtype=float).reshape(config.blocks)
    up_kw = np.asarray(up_kw, dtype=float).reshape(config.blocks)
    down_kw = np.asarray(down_kw, dtype=float).reshape(config.blocks)
    up_signal = np.asarray(up_signal, dtype=float).reshape(config.steps)
    down_signal = np.asarray(down_signal, dtype=float).reshape(config.steps)
    participating = participation_by_block(
        up_kw, down_kw, eps=config.eps
    )
    zero_tolerance = zero_instruction_tolerance_by_block(
        up_kw,
        down_kw,
        band_fraction=config.assessment_band_fraction,
        eps=config.eps,
    )
    targets = np.zeros(config.steps, dtype=float)
    tolerances = np.full(config.steps, np.inf, dtype=float)
    band_lower = np.full(config.steps, -np.inf, dtype=float)
    band_upper = np.full(config.steps, np.inf, dtype=float)
    assessed = np.zeros(config.steps, dtype=bool)
    for step in range(config.steps):
        block = min(
            config.blocks - 1, step // config.steps_per_block
        )
        if not participating[block]:
            continue
        if (
            up_signal[step] > config.eps
            and up_kw[block] > config.eps
        ):
            targets[step] = (
                baseline[block] - up_kw[block] * up_signal[step]
            )
            tolerances[step] = (
                config.assessment_band_fraction * up_kw[block]
            )
        elif (
            down_signal[step] > config.eps
            and down_kw[block] > config.eps
        ):
            targets[step] = (
                baseline[block] + down_kw[block] * down_signal[step]
            )
            tolerances[step] = (
                config.assessment_band_fraction * down_kw[block]
            )
        elif config.enforce_idle_baseline:
            targets[step] = baseline[block]
            tolerances[step] = zero_tolerance[block]
        else:
            continue
        assessed[step] = True
        band_lower[step] = targets[step] - tolerances[step]
        band_upper[step] = targets[step] + tolerances[step]

    if apply_transition_band:
        for step in range(1, config.steps):
            if not (assessed[step] and assessed[step - 1]):
                continue
            if (
                abs(targets[step] - targets[step - 1]) <= config.eps
            ):
                continue
            band_lower[step] = min(
                band_lower[step],
                targets[step - 1] - tolerances[step - 1],
            )
            band_upper[step] = max(
                band_upper[step],
                targets[step - 1] + tolerances[step - 1],
            )
    return targets, tolerances, band_lower, band_upper


def _missing_dispatch_row(name: str, config) -> dict:
    """Return a row that says no dispatch was recorded, and claims nothing else.

    Every rate is left as ``None`` rather than filled with a number computed
    from an assumed trajectory: there is no trajectory to compute from, and a
    fabricated one would be indistinguishable from a real, terrible result.
    """

    return {
        "scenario": str(name),
        "dispatch_missing": True,
        "ev_energy_missing": True,
        "all_ok": False,
        "soc_ok": None,
        "min_soc_margin_kwh": None,
        "global_step_pass_rate": None,
        "market_tracking_rate": None,
        "battery_peak_kw": None,
        "battery_energy_kwh": None,
        "battery_throughput_kwh": None,
        "up_step_pass_rate": None,
        "down_step_pass_rate": None,
        "idle_step_pass_rate": None,
        "up_delivery_ratio_by_block": [None] * int(config.blocks),
        "down_delivery_ratio_by_block": [None] * int(config.blocks),
        "assessed_steps": None,
        "participating_blocks": None,
        "miss_steps": None,
        "failed_blocks": None,
        "failed_block_indices": [],
        "free_power_abs_mean_kw": None,
        "delivered_capacity_kw_block": None,
    }


def validate_joint_solution(
    problem: JointBiddingProblem,
    solution: BiddingSolution,
    *,
    apply_transition_band: bool | None = None,
) -> dict:
    """Recompute SoC and tracking outcomes from primal trajectories."""

    config = problem.config
    if apply_transition_band is None:
        apply_transition_band = bool(config.apply_transition_band)
    baseline = np.asarray(
        solution.baseline_kw, dtype=float
    ).reshape(config.blocks)
    up_kw = np.asarray(solution.up_kw, dtype=float).reshape(config.blocks)
    down_kw = np.asarray(
        solution.down_kw, dtype=float
    ).reshape(config.blocks)
    participating_blocks = participation_by_block(
        up_kw, down_kw, eps=config.eps
    )
    participating_steps = np.repeat(
        participating_blocks, config.steps_per_block
    )[: config.steps]
    up_awarded_steps = np.repeat(
        up_kw > config.eps, config.steps_per_block
    )[: config.steps]
    down_awarded_steps = np.repeat(
        down_kw > config.eps, config.steps_per_block
    )[: config.steps]
    names = solution.metadata.get(
        "scenario_name_by_index"
    ) or unique_scenario_names(problem)
    labels = ev_labels(problem)
    rows: list[dict] = []
    up_delivery_samples: list[list[float]] = [
        [] for _ in range(config.blocks)
    ]
    down_delivery_samples: list[list[float]] = [
        [] for _ in range(config.blocks)
    ]

    for scenario_index, scenario in enumerate(problem.scenarios):
        name = str(names[scenario_index])
        # A scenario the solve never produced a dispatch for has no tracking
        # outcome to recompute.  Substituting a fleet that draws nothing would
        # report it as a catastrophic miss -- most bands exclude zero -- and
        # that reading is about the missing record, not about the bid.  Say so
        # instead, and let the caller decide what a missing answer means.
        stored_power = solution.scenario_power_kw.get(name)
        if stored_power is None:
            rows.append(_missing_dispatch_row(name, config))
            continue
        power = np.asarray(stored_power, dtype=float)
        up_signal = np.asarray(
            scenario.up_signal, dtype=float
        ).reshape(config.steps)
        down_signal = np.asarray(
            scenario.down_signal, dtype=float
        ).reshape(config.steps)
        targets, tolerances, band_lower, band_upper = (
            fixed_bid_tracking_bands(
                config,
                baseline,
                up_kw,
                down_kw,
                up_signal,
                down_signal,
                apply_transition_band=bool(apply_transition_band),
            )
        )
        assessed_band = np.isfinite(band_lower) & np.isfinite(band_upper)
        # The same tolerance the certifier uses, so a bid it passes is
        # not then failed here over a difference far below the width of
        # the band being tested.
        step_tol = step_violation_tolerance_kw(
            0.5 * (band_upper[assessed_band] - band_lower[assessed_band])
        )
        passed = np.ones(config.steps, dtype=bool)
        passed[assessed_band] = (
            power[assessed_band] >= band_lower[assessed_band] - step_tol
        ) & (
            power[assessed_band] <= band_upper[assessed_band] + step_tol
        )

        shortfall = np.zeros(config.steps, dtype=float)
        assessed = (
            participating_steps
            & np.isfinite(band_upper)
            & np.isfinite(band_lower)
        )
        if np.any(assessed):
            shortfall[assessed] = np.maximum(
                np.maximum(
                    band_lower[assessed] - power[assessed],
                    power[assessed] - band_upper[assessed],
                ),
                0.0,
            )
        run_energy = 0.0
        worst_run_energy = 0.0
        for step in range(config.steps):
            if shortfall[step] > config.eps:
                run_energy += shortfall[step] * config.dt_hours
                worst_run_energy = max(worst_run_energy, run_energy)
            else:
                run_energy = 0.0
        battery_peak_kw = (
            float(np.max(shortfall)) if shortfall.size else 0.0
        )
        battery_energy_kwh = float(worst_run_energy)
        battery_throughput_kwh = float(
            np.sum(shortfall) * config.dt_hours
        )

        soc_ok = True
        min_soc_margin = float("inf")
        energies = solution.ev_energy_kwh.get(name, {})
        missing_energy = False
        for ev_index, ev in enumerate(scenario.evs):
            energy = energies.get(labels[(scenario_index, ev_index)])
            if energy is None:
                # Same reasoning as the power above: no record is not a
                # violation, and reporting it as one hides which it was.
                missing_energy = True
                soc_ok = False
                min_soc_margin = float("-inf")
                continue
            actual_departure = int(ev.departure_t)
            if ev.target_required and actual_departure <= config.steps:
                departure = int(
                    np.clip(actual_departure, 0, config.steps)
                )
                required = float(ev.target_kwh)
            elif actual_departure > config.steps:
                departure = int(config.steps)
                required = ev.terminal_min_kwh(
                    config.steps,
                    dt_hours=float(config.dt_hours),
                    eta_ch=float(config.eta_ch),
                )
            else:
                continue
            margin = float(energy[departure] - required)
            min_soc_margin = min(min_soc_margin, margin)
            if margin < -1e-5:
                soc_ok = False
        if not scenario.evs or min_soc_margin == float("inf"):
            min_soc_margin = 0.0

        up_active = (
            participating_steps
            & up_awarded_steps
            & (up_signal > config.eps)
        )
        down_active = (
            participating_steps
            & down_awarded_steps
            & (down_signal > config.eps)
        )
        idle_active = participating_steps & ~(up_active | down_active)
        scenario_up_delivery: dict[str, float] = {}
        scenario_down_delivery: dict[str, float] = {}
        for block in range(config.blocks):
            start = block * config.steps_per_block
            stop = min(
                start + config.steps_per_block, config.steps
            )
            up_calibration = up_active[start:stop] & (
                up_signal[start:stop]
                > config.assessment_band_fraction + config.eps
            )
            if np.any(up_calibration):
                requested = (
                    up_kw[block]
                    * up_signal[start:stop][up_calibration]
                )
                delivered = (
                    baseline[block] - power[start:stop][up_calibration]
                )
                ratios = np.clip(
                    delivered / np.maximum(requested, config.eps),
                    0.0,
                    1.0,
                )
                up_delivery_samples[block].extend(
                    ratios.astype(float).tolist()
                )
                scenario_up_delivery[str(block)] = float(
                    np.quantile(ratios, 0.10)
                )
            down_calibration = down_active[start:stop] & (
                down_signal[start:stop]
                > config.assessment_band_fraction + config.eps
            )
            if np.any(down_calibration):
                requested = (
                    down_kw[block]
                    * down_signal[start:stop][down_calibration]
                )
                delivered = (
                    power[start:stop][down_calibration]
                    - baseline[block]
                )
                ratios = np.clip(
                    delivered / np.maximum(requested, config.eps),
                    0.0,
                    1.0,
                )
                down_delivery_samples[block].extend(
                    ratios.astype(float).tolist()
                )
                scenario_down_delivery[str(block)] = float(
                    np.quantile(ratios, 0.10)
                )

        global_rate = masked_pass_rate(passed, participating_steps)
        block_pass = np.ones(config.blocks, dtype=bool)
        for block in range(config.blocks):
            if not participating_blocks[block]:
                continue
            start = block * config.steps_per_block
            stop = min(
                start + config.steps_per_block, config.steps
            )
            block_pass[block] = bool(
                np.mean(passed[start:stop]) >= 0.90
            )
        delivered_capacity_kw_block = float(
            np.sum(block_pass.astype(float) * (up_kw + down_kw))
        )
        failed_blocks = np.flatnonzero(
            participating_blocks & ~block_pass
        ).astype(int)
        free_steps = ~participating_steps
        rows.append(
            {
                "scenario": name,
                "soc_ok": bool(soc_ok),
                "min_soc_margin_kwh": float(min_soc_margin),
                "global_step_pass_rate": global_rate,
                "market_tracking_rate": global_rate,
                "battery_peak_kw": battery_peak_kw,
                "battery_energy_kwh": battery_energy_kwh,
                "battery_throughput_kwh": battery_throughput_kwh,
                "up_step_pass_rate": (
                    float(np.mean(passed[up_active]))
                    if np.any(up_active)
                    else 1.0
                ),
                "down_step_pass_rate": (
                    float(np.mean(passed[down_active]))
                    if np.any(down_active)
                    else 1.0
                ),
                "idle_step_pass_rate": (
                    float(np.mean(passed[idle_active]))
                    if np.any(idle_active)
                    else 1.0
                ),
                "up_delivery_ratio_by_block": scenario_up_delivery,
                "down_delivery_ratio_by_block": scenario_down_delivery,
                "assessed_steps": int(
                    np.count_nonzero(participating_steps)
                ),
                "participating_blocks": int(
                    np.count_nonzero(participating_blocks)
                ),
                "miss_steps": int(
                    np.count_nonzero(participating_steps & ~passed)
                ),
                "failed_blocks": int(failed_blocks.size),
                "failed_block_indices": failed_blocks.tolist(),
                "free_power_abs_mean_kw": (
                    float(np.mean(np.abs(power[free_steps])))
                    if np.any(free_steps)
                    else 0.0
                ),
                "delivered_capacity_kw_block": delivered_capacity_kw_block,
                "all_ok": bool(
                    soc_ok
                    and global_rate + 1e-12
                    >= float(problem.global_pass_rate)
                ),
                "dispatch_missing": False,
                "ev_energy_missing": bool(missing_energy),
            }
        )

    if not solution.success:
        return {
            "scenario_rows": rows,
            "all_scenarios_ok": False,
            "soc_all_ok": False,
            "min_global_step_pass_rate": 0.0,
            "mean_global_step_pass_rate": 0.0,
            "min_up_step_pass_rate": 0.0,
            "min_down_step_pass_rate": 0.0,
            "min_idle_step_pass_rate": 0.0,
            "participating_block_count": int(
                np.count_nonzero(participating_blocks)
            ),
            "mean_delivered_capacity_kw_block": 0.0,
            "offered_capacity_kw_block": float(np.sum(up_kw + down_kw)),
            "up_delivery_ratio_by_block": {},
            "down_delivery_ratio_by_block": {},
        }

    # Rows with no recorded dispatch carry no rates.  They are counted, and
    # reported, but they must not enter an average as if they were a measured
    # outcome: the summary would then mix "this bid missed" with "nobody
    # looked".
    missing = [row for row in rows if row.get("dispatch_missing")]
    scored = [row for row in rows if not row.get("dispatch_missing")]
    rates = np.asarray(
        [row["global_step_pass_rate"] for row in scored], dtype=float
    )
    total_assessed = int(sum(row["assessed_steps"] for row in scored))
    total_misses = int(sum(row["miss_steps"] for row in scored))
    failed_block_frequency = {
        str(block): int(
            sum(
                block in row["failed_block_indices"] for row in rows
            )
        )
        for block in range(config.blocks)
        if any(block in row["failed_block_indices"] for row in rows)
    }
    return {
        "scenario_rows": rows,
        "all_scenarios_ok": bool(
            not missing and all(row["all_ok"] for row in scored)
        ),
        "soc_all_ok": bool(
            not missing and all(row["soc_ok"] for row in scored)
        ),
        "scenarios_without_a_dispatch": len(missing),
        "scenarios_without_a_dispatch_names": [
            str(row["scenario"]) for row in missing
        ],
        "min_global_step_pass_rate": float(
            min(
                (row["global_step_pass_rate"] for row in scored),
                default=1.0,
            )
        ),
        "mean_global_step_pass_rate": (
            float(
                np.mean(
                    [row["global_step_pass_rate"] for row in scored]
                )
            )
            if scored
            else 1.0
        ),
        "p10_global_step_pass_rate": (
            float(np.quantile(rates, 0.10)) if rates.size else 1.0
        ),
        "aggregate_global_step_pass_rate": (
            float(1.0 - total_misses / total_assessed)
            if total_assessed > 0
            else 1.0
        ),
        "battery_peak_kw": float(
            max(
                (row["battery_peak_kw"] for row in scored),
                default=0.0,
            )
        ),
        "battery_energy_kwh": float(
            max(
                (row["battery_energy_kwh"] for row in scored),
                default=0.0,
            )
        ),
        "battery_throughput_kwh_mean": (
            float(
                np.mean(
                    [row["battery_throughput_kwh"] for row in scored]
                )
            )
            if scored
            else 0.0
        ),
        "min_up_step_pass_rate": float(
            min(
                (row["up_step_pass_rate"] for row in scored),
                default=1.0,
            )
        ),
        "min_down_step_pass_rate": float(
            min(
                (row["down_step_pass_rate"] for row in scored),
                default=1.0,
            )
        ),
        "min_idle_step_pass_rate": float(
            min(
                (row["idle_step_pass_rate"] for row in scored),
                default=1.0,
            )
        ),
        "participating_block_count": int(
            np.count_nonzero(participating_blocks)
        ),
        "participating_block_indices": np.flatnonzero(
            participating_blocks
        ).astype(int).tolist(),
        "failed_block_frequency": failed_block_frequency,
        "up_delivery_ratio_by_block": {
            str(block): float(np.quantile(samples, 0.10))
            for block, samples in enumerate(up_delivery_samples)
            if samples
        },
        "down_delivery_ratio_by_block": {
            str(block): float(np.quantile(samples, 0.10))
            for block, samples in enumerate(down_delivery_samples)
            if samples
        },
        "mean_delivered_capacity_kw_block": (
            float(np.mean([
                row["delivered_capacity_kw_block"] for row in scored
            ]))
            if scored
            else 0.0
        ),
        "min_delivered_capacity_kw_block": float(
            min(
                (row["delivered_capacity_kw_block"] for row in scored),
                default=0.0,
            )
        ),
        "offered_capacity_kw_block": float(np.sum(up_kw + down_kw)),
    }

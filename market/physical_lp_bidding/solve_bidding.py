"""Natural no-activation baseline LP."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .data_classes import ActivationScenario, BiddingLPConfig


def solve_natural_baseline_lp(
    scenario: ActivationScenario,
    config: BiddingLPConfig | None = None,
    baseline_min_kw: float = 0.0,
    baseline_max_kw: float | None = None,
) -> np.ndarray:
    """Return the block-average charging that meets every reachable target.

    Discharge is intentionally disabled: without a market instruction there is
    no operational reason to spend energy.  The result is only the initial
    baseline seed; the Benders master may subsequently move it.

    Each block's baseline is the mean fleet power over its steps, not a power
    held at every step: an EV that must charge at its rating from the moment it
    arrives (one staying past midnight, see EVEnv._horizon_obligation) makes
    the fleet power step up inside a block, which no constant power can match.
    """

    config = config or BiddingLPConfig()
    upper_baseline = (
        sum(
            max(0.0, float(ev.max_charge_kw)) for ev in scenario.evs
        )
        if baseline_max_kw is None
        else float(baseline_max_kw)
    )
    bounds: list[tuple[float | None, float | None]] = [
        (float(baseline_min_kw), upper_baseline)
        for _ in range(config.blocks)
    ]
    columns_by_step: list[list[int]] = [
        [] for _ in range(config.steps)
    ]
    columns_by_ev: list[list[int]] = []
    for ev_index, ev in enumerate(scenario.evs):
        arrival = int(np.clip(ev.arrival_t, 0, config.steps))
        departure = int(
            np.clip(ev.departure_t, arrival + 1, config.steps)
        )
        ev_columns: list[int] = []
        for step in range(arrival, departure):
            column = len(bounds)
            bounds.append((0.0, max(0.0, float(ev.max_charge_kw))))
            columns_by_step[step].append(column)
            ev_columns.append(column)
        columns_by_ev.append(ev_columns)

    upper_rows: list[int] = []
    upper_columns: list[int] = []
    upper_data: list[float] = []
    upper_rhs: list[float] = []
    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_data: list[float] = []
    equality_rhs: list[float] = []

    def add_upper(
        coefficients: list[tuple[int, float]], upper: float
    ) -> None:
        row = len(upper_rhs)
        for column, value in coefficients:
            if value != 0.0:
                upper_rows.append(row)
                upper_columns.append(int(column))
                upper_data.append(float(value))
        upper_rhs.append(float(upper))

    def add_equality(
        coefficients: list[tuple[int, float]], right_hand_side: float
    ) -> None:
        row = len(equality_rhs)
        for column, value in coefficients:
            if value != 0.0:
                equality_rows.append(row)
                equality_columns.append(int(column))
                equality_data.append(float(value))
        equality_rhs.append(float(right_hand_side))

    for block in range(config.blocks):
        first = block * config.steps_per_block
        last = config.steps if block == config.blocks - 1 else first + config.steps_per_block
        add_equality(
            [
                (column, 1.0)
                for step in range(first, min(last, config.steps))
                for column in columns_by_step[step]
            ]
            + [(block, -float(min(last, config.steps) - first))],
            0.0,
        )

    dt_hours = float(config.dt_hours)
    for ev_index, ev in enumerate(scenario.evs):
        capacity = max(float(ev.capacity_kwh), 1e-9)
        initial = float(np.clip(ev.initial_kwh, 0.0, capacity))
        target = float(np.clip(ev.target_kwh, 0.0, capacity))
        actual_departure = int(ev.departure_t)
        ev_columns = columns_by_ev[ev_index]
        # Power is charging-only. Its cumulative sum is monotone, so the final
        # capacity inequality implies every intermediate energy upper bound;
        # the intermediate lower bounds are automatic from p >= 0.
        add_upper(
            [(column, dt_hours) for column in ev_columns],
            capacity - initial,
        )
        if ev.target_required and actual_departure <= config.steps:
            required = target
        elif actual_departure > config.steps:
            required = ev.terminal_min_kwh(
                config.steps,
                dt_hours=dt_hours,
                eta_ch=float(config.eta_ch),
            )
        else:
            required = 0.0
        if required > config.eps:
            add_upper(
                [(column, -dt_hours) for column in ev_columns],
                -(required - initial),
            )

    variable_count = len(bounds)
    objective = np.zeros(variable_count, dtype=float)
    tie_break = 1e-6 * np.arange(config.blocks, dtype=float)
    # Minimize total baseline energy with a deterministic early-block tie
    # break.  This baseline uses no external weighting signal.
    objective[: config.blocks] = (
        (1.0 + tie_break) * config.steps_per_block * config.dt_hours
    )
    upper_matrix = coo_matrix(
        (upper_data, (upper_rows, upper_columns)),
        shape=(len(upper_rhs), variable_count),
    ).tocsr()
    equality_matrix = coo_matrix(
        (equality_data, (equality_rows, equality_columns)),
        shape=(len(equality_rhs), variable_count),
    ).tocsr()
    result = linprog(
        objective,
        A_ub=upper_matrix,
        b_ub=np.asarray(upper_rhs, dtype=float),
        A_eq=equality_matrix,
        b_eq=np.asarray(equality_rhs, dtype=float),
        bounds=bounds,
        # This transportation-style LP is highly degenerate at fleet scale.
        # HiGHS IPM returns the same optimum much faster than dual simplex on
        # the 500-station case while preserving the exact LP formulation.
        method="highs-ipm",
        options=(
            {"time_limit": float(config.time_limit_s)}
            if config.time_limit_s is not None
            else None
        ),
    )
    if not result.success:
        raise RuntimeError(
            f"natural baseline LP failed: {result.message}"
        )
    return np.asarray(result.x[: config.blocks], dtype=float)

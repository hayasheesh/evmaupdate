"""Energy-aware initial bid from an aggregate relaxation of the EV fleet.

The Assessment-I seed looks at one 30-minute block at a time. A command that
keeps one direction for hours moves far more energy than any single block, so a
seed built block by block can ask for widths no fleet can hold over the day.

This module treats each EV realization as one aggregate battery. Per step it
bounds the fleet's total power by the connected chargers, and bounds the
fleet's cumulative energy change by the sum of what each vehicle can reach at
that step (``_ev_arrays`` semantics: battery limits, discharge rate, and the
energy the departure target still needs). Every bid the per-vehicle recourse
accepts satisfies these bounds, so the aggregate set contains the exact one.

One LP then chooses baseline, up, and down widths that every EV realization x
command scenario can track in aggregate, using the same affine tracking rows as
the Benders oracle. The result is an upper envelope and a starting point for the
exact search, not a certified bid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .colgen_feasibility import _ev_arrays, affine_tracking_bands
from .data_classes import ActivationScenario, BiddingLPConfig, baseline_step_matrix


@dataclass(frozen=True)
class FleetEnvelope:
    """Aggregate per-step bounds of one EV realization, in kW and kWh."""

    power_min_kw: np.ndarray
    power_max_kw: np.ndarray
    energy_lo_kwh: np.ndarray
    energy_hi_kwh: np.ndarray
    unreachable_targets: int


@dataclass
class InitialBid:
    baseline_kw: np.ndarray
    up_kw: np.ndarray
    down_kw: np.ndarray
    summary: dict = field(default_factory=dict)


def fleet_energy_envelope(evs, cfg: BiddingLPConfig) -> FleetEnvelope:
    """Sum each vehicle's reachable power and cumulative-energy interval.

    ``energy_*`` bound the fleet's cumulative energy change after each step,
    relative to the day start. A vehicle contributes nothing before arrival and
    keeps its final change after departure.
    """

    steps = int(cfg.steps)
    dt = float(cfg.dt_hours)
    p_min = np.zeros(steps)
    p_max = np.zeros(steps)
    e_lo = np.zeros(steps)
    e_hi = np.zeros(steps)
    unreachable = 0
    for ev in evs:
        arr, dep, cap, init, target = _ev_arrays(ev, steps, dt, cfg.eta_ch)
        n = dep - arr
        charge = max(float(ev.max_charge_kw), 0.0)
        discharge = max(float(ev.max_discharge_kw), 0.0)
        elapsed = np.arange(1, n + 1, dtype=float)
        hi = np.minimum(cap - init, charge * dt * elapsed)
        lo = np.maximum(-init, -discharge * dt * elapsed)
        if target is not None:
            lo = np.maximum(lo, (target - init) - charge * dt * (n - elapsed))
        if np.any(lo > hi + 1e-9):
            unreachable += 1
        lo = np.minimum(lo, hi)
        p_min[arr:dep] -= discharge
        p_max[arr:dep] += charge
        e_lo[arr:dep] += lo
        e_hi[arr:dep] += hi
        e_lo[dep:] += lo[-1]
        e_hi[dep:] += hi[-1]
    return FleetEnvelope(p_min, p_max, e_lo, e_hi, unreachable)


def _solve_aggregate_lp(
    scenarios: list[ActivationScenario],
    envelopes: dict[int, FleetEnvelope],
    cfg: BiddingLPConfig,
    *,
    baseline_min: np.ndarray,
    baseline_max: np.ndarray,
    up_cap: np.ndarray,
    down_cap: np.ndarray,
    up_active: np.ndarray,
    down_active: np.ndarray,
    elastic: bool,
    time_limit_s: float | None,
    sustained_power_min: np.ndarray | None = None,
    sustained_power_max: np.ndarray | None = None,
) -> dict:
    """Maximize total width subject to aggregate tracking in every scenario.

    With ``elastic`` each tracking row gets a non-negative slack at a price far
    above any width, and the returned ``slack_by_block`` says which blocks
    could not be tracked at all under this participation pattern. With
    ``cfg.baseline_step_weight`` the baseline steps between adjacent
    participating blocks carry that weight, so among equal-capacity seeds the
    one with the smaller steps is returned.
    """

    blocks = int(cfg.blocks)
    steps = int(cfg.steps)
    dt = float(cfg.dt_hours)
    n_x = 3 * blocks
    per_scenario = 2 * steps
    n_slack = len(scenarios) * steps if elastic else 0
    step_weight = max(float(getattr(cfg, "baseline_step_weight", 0.0)), 0.0)
    step_rows = (
        baseline_step_matrix(np.asarray(up_active) | np.asarray(down_active), blocks)
        if step_weight > 0.0
        else np.zeros((0, n_x))
    )
    n_step = int(step_rows.shape[0])
    slack_start = n_x + len(scenarios) * per_scenario
    step_start = slack_start + n_slack
    n_var = step_start + n_step
    slack_price = 1e3

    ub_rows: list[np.ndarray] = []
    ub_cols: list[np.ndarray] = []
    ub_vals: list[np.ndarray] = []
    ub_rhs: list[np.ndarray] = []
    eq_rows: list[np.ndarray] = []
    eq_cols: list[np.ndarray] = []
    eq_vals: list[np.ndarray] = []
    ub_count = 0
    eq_count = 0
    lower = np.full(n_var, 0.0)
    upper = np.full(n_var, np.inf)
    lower[:blocks] = baseline_min
    upper[:blocks] = baseline_max
    upper[blocks:2 * blocks] = np.where(up_active, up_cap, 0.0)
    upper[2 * blocks:n_x] = np.where(down_active, down_cap, 0.0)
    zero = np.zeros(blocks)
    step_index = np.arange(steps)

    for s, scenario in enumerate(scenarios):
        envelope = envelopes[id(scenario.evs)]
        p0 = n_x + s * per_scenario
        c0 = p0 + steps
        lower[p0:c0] = envelope.power_min_kw
        upper[p0:c0] = envelope.power_max_kw
        lower[c0:c0 + steps] = envelope.energy_lo_kwh
        upper[c0:c0 + steps] = envelope.energy_hi_kwh

        # C[t] - C[t-1] - dt * P[t] = 0, with C[-1] = 0.
        rows = eq_count + step_index
        eq_rows += [rows, rows, rows[1:]]
        eq_cols += [c0 + step_index, p0 + step_index, c0 + step_index[:-1]]
        eq_vals += [np.ones(steps), np.full(steps, -dt), -np.ones(steps - 1)]
        eq_count += steps

        _lo, _hi, lower_rows, upper_rows = affine_tracking_bands(
            cfg, zero, zero, zero, up_active, down_active,
            scenario.up_signal[:steps], scenario.down_signal[:steps],
        )
        tracked = np.flatnonzero(np.any(lower_rows[:, :blocks] != 0.0, axis=1))
        slack0 = slack_start + s * steps
        # sign=+1: lower_row @ x - P <= 0;  sign=-1: P - upper_row @ x <= 0.
        for sign, rows_matrix in ((1.0, lower_rows), (-1.0, upper_rows)):
            sub = rows_matrix[tracked]
            local, cols = np.nonzero(sub)
            row_ids = ub_count + np.arange(tracked.size)
            ub_rows += [row_ids[local], row_ids]
            ub_cols += [cols, p0 + tracked]
            ub_vals += [sign * sub[local, cols], np.full(tracked.size, -sign)]
            if elastic:
                ub_rows.append(row_ids)
                ub_cols.append(slack0 + tracked)
                ub_vals.append(-np.ones(tracked.size))
            ub_rhs.append(np.zeros(tracked.size))
            ub_count += tracked.size

    # Assessment I: baseline - up >= sustained_power_min and
    # baseline + down <= sustained_power_max in every block.
    block_index = np.arange(blocks)
    if sustained_power_min is not None:
        row_ids = ub_count + block_index
        ub_rows += [row_ids, row_ids]
        ub_cols += [block_index, blocks + block_index]
        ub_vals += [-np.ones(blocks), np.ones(blocks)]
        ub_rhs.append(-np.asarray(sustained_power_min, dtype=float).reshape(blocks))
        ub_count += blocks
    if sustained_power_max is not None:
        row_ids = ub_count + block_index
        ub_rows += [row_ids, row_ids]
        ub_cols += [block_index, 2 * blocks + block_index]
        ub_vals += [np.ones(blocks), np.ones(blocks)]
        ub_rhs.append(np.asarray(sustained_power_max, dtype=float).reshape(blocks))
        ub_count += blocks

    # Baseline steps: z_j >= |baseline[k+1] - baseline[k]| for each adjacent
    # participating pair, as two rows D x - z <= 0 and -D x - z <= 0.
    if n_step:
        pair_rows, pair_cols = np.nonzero(step_rows)
        for sign in (1.0, -1.0):
            row_ids = ub_count + np.arange(n_step)
            ub_rows += [row_ids[pair_rows], row_ids]
            ub_cols += [pair_cols, step_start + np.arange(n_step)]
            ub_vals += [sign * step_rows[pair_rows, pair_cols], -np.ones(n_step)]
            ub_rhs.append(np.zeros(n_step))
            ub_count += n_step

    objective = np.zeros(n_var)
    objective[blocks:n_x] = -1.0
    if elastic:
        objective[slack_start:step_start] = slack_price
    objective[step_start:] = step_weight
    a_ub = sparse.csr_matrix(
        (np.concatenate(ub_vals), (np.concatenate(ub_rows), np.concatenate(ub_cols))),
        shape=(ub_count, n_var),
    )
    a_eq = sparse.csr_matrix(
        (np.concatenate(eq_vals), (np.concatenate(eq_rows), np.concatenate(eq_cols))),
        shape=(eq_count, n_var),
    )
    started = time.perf_counter()
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.concatenate(ub_rhs) if ub_rhs else np.zeros(ub_count),
        A_eq=a_eq,
        b_eq=np.zeros(eq_count),
        bounds=np.column_stack([lower, upper]),
        method="highs",
        options=None if time_limit_s is None else {"time_limit": float(time_limit_s)},
    )
    out = {
        "success": bool(result.success and result.x is not None),
        "status": int(result.status),
        "message": str(result.message),
        "runtime_s": float(time.perf_counter() - started),
        "variables": int(n_var),
        "rows": int(ub_count + eq_count),
    }
    if out["success"]:
        x = np.asarray(result.x, dtype=float)
        out["baseline"] = x[:blocks].copy()
        out["up"] = x[blocks:2 * blocks].copy()
        out["down"] = x[2 * blocks:n_x].copy()
        if elastic:
            slack = x[slack_start:step_start].reshape(len(scenarios), steps)
            by_step = slack.max(axis=0)
            by_block = by_step[: blocks * int(cfg.steps_per_block)].reshape(
                blocks, int(cfg.steps_per_block)
            ).max(axis=1)
            out["slack_by_block"] = by_block
    return out


def energy_feasible_initial_bid(
    scenarios: list[ActivationScenario],
    cfg: BiddingLPConfig,
    *,
    baseline_min,
    baseline_max,
    up_cap,
    down_cap,
    minimum_bid_kw: float,
    time_limit_s: float | None = None,
    max_iterations: int = 20,
    progress=None,
    sustained_power_min=None,
    sustained_power_max=None,
) -> InitialBid | None:
    """Return a seed that every scenario can track in aggregate, or ``None``.

    Directions start active where the Assessment-I cap reaches the floor.
    Blocks the elastic LP cannot track under any width lose both directions;
    directions the LP sets below the floor are retired together. The loop ends
    when the LP solution respects the floor. ``None`` means no aggregate-feasible
    pattern was found, and the caller keeps its previous seed.
    """

    blocks = int(cfg.blocks)
    floor = max(float(minimum_bid_kw), 0.0)
    tolerance = max(float(cfg.eps) * 10.0, 1e-6)
    baseline_min = np.broadcast_to(np.asarray(baseline_min, dtype=float), (blocks,)).copy()
    baseline_max = np.broadcast_to(np.asarray(baseline_max, dtype=float), (blocks,)).copy()
    up_cap = np.asarray(up_cap, dtype=float).reshape(blocks)
    down_cap = np.asarray(down_cap, dtype=float).reshape(blocks)
    up_active = up_cap >= floor - tolerance
    down_active = down_cap >= floor - tolerance
    initial_capacity = float(np.sum(np.where(up_active, up_cap, 0.0)) + np.sum(np.where(down_active, down_cap, 0.0)))

    envelopes: dict[int, FleetEnvelope] = {}
    for scenario in scenarios:
        key = id(scenario.evs)
        if key not in envelopes:
            envelopes[key] = fleet_energy_envelope(scenario.evs, cfg)
    unreachable = int(sum(env.unreachable_targets for env in envelopes.values()))

    def log(message: str) -> None:
        if progress is not None:
            progress(f"energy-initial-bid: {message}")

    log(
        f"{len(scenarios)} scenarios over {len(envelopes)} EV realization(s); "
        f"Assessment-I seed capacity {initial_capacity:.0f} kW-block; "
        f"vehicles with unreachable targets={unreachable}"
    )
    history: list[dict] = []
    started = time.perf_counter()
    for iteration in range(int(max_iterations)):
        common = dict(
            baseline_min=baseline_min,
            baseline_max=baseline_max,
            up_cap=up_cap,
            down_cap=down_cap,
            up_active=up_active,
            down_active=down_active,
            time_limit_s=time_limit_s,
            sustained_power_min=sustained_power_min,
            sustained_power_max=sustained_power_max,
        )
        strict = _solve_aggregate_lp(scenarios, envelopes, cfg, elastic=False, **common)
        row = {
            "iteration": iteration,
            "active_up": int(np.count_nonzero(up_active)),
            "active_down": int(np.count_nonzero(down_active)),
            "strict_status": strict["message"],
            "strict_runtime_s": strict["runtime_s"],
        }
        if not strict["success"]:
            if strict["status"] != 2:
                row["stop"] = "strict_lp_not_solved"
                history.append(row)
                log(f"iteration {iteration}: LP stopped without a verdict ({strict['message']})")
                break
            elastic = _solve_aggregate_lp(scenarios, envelopes, cfg, elastic=True, **common)
            row["elastic_status"] = elastic["message"]
            row["elastic_runtime_s"] = elastic["runtime_s"]
            if not elastic["success"]:
                row["stop"] = "elastic_lp_not_solved"
                history.append(row)
                log(f"iteration {iteration}: elastic LP failed ({elastic['message']})")
                break
            untrackable = elastic["slack_by_block"] > 1e-6
            untrackable &= up_active | down_active
            row["untrackable_blocks"] = np.flatnonzero(untrackable).astype(int).tolist()
            history.append(row)
            log(
                f"iteration {iteration}: infeasible in aggregate; dropping both "
                f"directions in blocks {row['untrackable_blocks']}"
            )
            if not np.any(untrackable):
                row["stop"] = "infeasible_without_slack_blocks"
                break
            up_active &= ~untrackable
            down_active &= ~untrackable
            continue

        up = np.where(up_active, strict["up"], 0.0)
        down = np.where(down_active, strict["down"], 0.0)
        below_up = up_active & (up < floor - tolerance)
        below_down = down_active & (down < floor - tolerance)
        row["capacity_kw_block"] = float(np.sum(up) + np.sum(down))
        row["below_floor_up"] = np.flatnonzero(below_up).astype(int).tolist()
        row["below_floor_down"] = np.flatnonzero(below_down).astype(int).tolist()
        history.append(row)
        log(
            f"iteration {iteration}: capacity {row['capacity_kw_block']:.0f} kW-block, "
            f"below floor up={len(row['below_floor_up'])} down={len(row['below_floor_down'])}, "
            f"LP {strict['runtime_s']:.1f}s ({strict['variables']} vars)"
        )
        if not (np.any(below_up) or np.any(below_down)):
            summary = {
                "rule": "aggregate_energy_robust_lp",
                "complete": True,
                "assessment_i_capacity_kw_block": initial_capacity,
                "capacity_kw_block": row["capacity_kw_block"],
                "iterations": history,
                "runtime_s": float(time.perf_counter() - started),
                "unreachable_targets": unreachable,
            }
            return InitialBid(
                baseline_kw=np.clip(strict["baseline"], baseline_min, baseline_max),
                up_kw=np.clip(up, 0.0, up_cap),
                down_kw=np.clip(down, 0.0, down_cap),
                summary=summary,
            )
        up_active &= ~below_up
        down_active &= ~below_down

    log("no aggregate-feasible pattern at the floor; keeping the Assessment-I seed")
    return None

"""Frozen-bid feasibility by Dantzig-Wolfe decomposition, with a readable certificate.

With the bid frozen, the only rows tying vehicles together are the per-step
aggregate power bands; the SoC chain, the power limits, and the departure target
all live inside one vehicle. That is a block-angular LP. The retired monolithic
certifier scaled as roughly the fleet to the 2.4th power -- 156k coupled columns
at 50 stations and 1.5M at the 500-station target, where one solve was projected
to take about nine hours.

Decomposing puts each vehicle in its own pricing problem and leaves a master
carrying only the coupling rows, so the fleet enters through the number of
independent subproblems rather than the size of one solve. Measured against
the monolithic certifier at 7, 15, 25 and 50 stations:

    monolithic   time proportional to (vehicles ** 2.47)
    decomposed   time proportional to (vehicles ** 1.29)

with the two agreeing on feasibility at 20 of 20 test points, including five
straddling the boundary at each of three fleet sizes -- which is where a bid
builder actually works. The round count did not grow with the fleet: four to
five rounds at every size tested.

A vehicle's dispatch is described by its power path alone, since stored energy is
the running sum. That turns the SoC limits into prefix-sum bounds and the
departure target into a bound on the total, halving the pricing problem and
keeping it an LP on an interval matrix.

The two band limits are written as explicit rows rather than as bounds on an
aggregate variable. Bounds would hide the bid from the duals -- the phase-one
variables come back pinned at their own bounds and say nothing about which part of
the bid was impossible. As rows, the bid sits on the right-hand side and LP
duality returns

    sum_t alpha_t * L_t  -  sum_t beta_t * U_t  +  sum_i sigma_i  >  0

as a certificate: one linear inequality in the commanded trajectory that no
dispatch of this fleet satisfies. ``alpha`` and ``beta`` say *where* the day
broke, which is what makes the certificate interpretable rather than merely
valid. Measured localisation is sharp: perturbing a three-hour window put the
whole of alpha's mass inside it, narrowing to two five-minute steps just past
the boundary.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

STEPS = 288
DT_HOURS = 5.0 / 60.0
# How far outside its tracking band a five-minute value may sit and still
# count as inside it.  Two terms, because neither alone is right.
#
# An absolute floor alone cannot serve every bid: the same 1e-5 kW is a
# millionth of a 19 kW band and a ten-thousandth of the 0.1 kW band a 1 kW
# award carries, so one number is either meaningless or decisive depending on
# the bid.  A relative term alone cannot serve a band of zero width.
#
# The certifier and validate_joint_solution both use this, so they answer the
# same question; step_violation_tolerance_kw is the one place it is decided.
STEP_VIOLATION_TOL_KW = 1e-5
# One ten-thousandth of a band, which is where the absolute floor stops
# discriminating: at the 0.1 kW band a 1 kW award carries -- the narrowest this
# experiment uses -- the two terms meet exactly, so the relative term takes
# over only above the width where 1e-5 kW is still a meaningful share.
STEP_VIOLATION_TOL_FRACTION = 1e-4


def step_violation_tolerance_kw(band_half_width_kw) -> float:
    """Return the tolerance for the tightest band among these.

    The narrowest band is the reference, not the widest: one tolerance is
    applied to every step, and scaling it to a wide band elsewhere in the day
    would let a narrow one be missed outright.
    """

    import numpy as np

    widths = np.asarray(band_half_width_kw, dtype=float)
    finite = widths[np.isfinite(widths) & (widths > 0.0)]
    reference = float(np.min(finite)) if finite.size else 0.0
    return float(
        max(STEP_VIOLATION_TOL_KW, STEP_VIOLATION_TOL_FRACTION * reference)
    )


def _ev_arrays(ev, steps: int, dt: float, eta_ch: float):
    arr = int(np.clip(ev.arrival_t, 0, steps - 1))
    actual_dep = int(ev.departure_t)
    dep = int(np.clip(actual_dep, arr + 1, steps))
    cap = max(float(ev.capacity_kwh), 1e-9)
    init = float(np.clip(ev.initial_kwh, 0.0, cap))
    if bool(getattr(ev, "target_required", False)) and actual_dep <= steps:
        target = float(np.clip(ev.target_kwh, 0.0, cap))
    elif actual_dep > steps:
        # Match the monolithic certifier: an overnight EV must retain enough
        # energy at the truncated horizon to reach its real departure target by
        # charging flat-out after the modeled day.
        target = float(ev.terminal_min_kwh(
            steps,
            dt_hours=dt,
            eta_ch=eta_ch,
        ))
    else:
        target = None
    return arr, dep, cap, init, target


class _PathOracle:
    """One vehicle's pricing problem, built once and re-costed each round.

    Only the objective changes between rounds, so the prefix-sum constraint
    matrix and the bounds are assembled once and reused.
    """

    def __init__(self, ev, steps: int, dt: float, eta_ch: float):
        self.arr, self.dep, cap, init, target = _ev_arrays(
            ev, steps, dt, eta_ch
        )
        n = self.dep - self.arr
        self.n = n
        ch = max(0.0, float(ev.max_charge_kw))
        dis = max(0.0, float(ev.max_discharge_kw))
        self.dt = float(dt)
        self.ch = ch
        self.init = init
        self.target = target
        self.bounds = [(-dis, ch)] * n
        # Stored energy after k steps is init + dt * (sum of the first k powers),
        # so 0 <= e <= cap turns into a two-sided bound on every prefix sum.
        tri = sparse.tril(np.ones((n, n)), 0, format="csr")
        self.lo = np.full(n, (0.0 - init) / dt)
        self.hi = np.full(n, (cap - init) / dt)
        if target is not None:
            # The departure target only constrains the final prefix sum.
            self.lo[-1] = max(self.lo[-1], (target - init) / dt)
        self.A_ub = sparse.vstack([tri, -tri], format="csr")
        self.b_ub = np.concatenate([self.hi, -self.lo])
        self.feasible_bounds = bool(np.all(self.lo <= self.hi + 1e-9))
        self.last_solve_message = ""

    def solve(
        self,
        pi: np.ndarray,
        *,
        time_limit_s: float | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """Maximize the dual-weighted value of this vehicle's power path."""
        c = -np.asarray(pi[self.arr:self.dep], dtype=float)
        options = None
        if time_limit_s is not None:
            if float(time_limit_s) <= 0.0:
                self.last_solve_message = "oracle time limit reached before pricing"
                return None
            options = {"time_limit": max(float(time_limit_s), 1e-3)}
        res = linprog(
            c,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            bounds=self.bounds,
            method="highs",
            options=options,
        )
        self.last_solve_message = str(res.message)
        if not res.success:
            return None
        return np.asarray(res.x, dtype=float), float(-res.fun)


def _initial_column(oracle: _PathOracle) -> np.ndarray | None:
    """A quiet feasible path for one vehicle, used to start the master.

    Solving a zero-objective LP returns an arbitrary extreme point.  For an EV
    arriving empty that can be an almost-all-day charging path, after which
    column generation may need hundreds of degenerate rounds merely to recover
    the perfectly valid zero path.  A constant path that supplies exactly the
    individually required energy is feasible here because the only local
    constraints are power bounds and prefix-sum energy bounds.
    """

    if not oracle.feasible_bounds or oracle.n <= 0 or oracle.dt <= 0.0:
        return None
    required_kwh = max(
        0.0,
        (float(oracle.target) if oracle.target is not None else oracle.init)
        - oracle.init,
    )
    power_kw = required_kwh / (oracle.n * oracle.dt)
    if power_kw > oracle.ch + 1e-9:
        return None
    return np.full(oracle.n, min(power_kw, oracle.ch), dtype=float)


def _column_fingerprint(column: np.ndarray, *, decimals: int = 8) -> bytes:
    """Stable near-duplicate key for a vehicle path.

    HiGHS can return the same extreme path with a few last-bit differences on
    degenerate pricing problems.  Treating each copy as a fresh column was one
    source of unbounded restricted-master growth.
    """

    rounded = np.round(np.asarray(column, dtype=np.float64), decimals=decimals)
    rounded[rounded == 0.0] = 0.0  # normalize negative zero
    return rounded.tobytes()


def _assessed_column_block(
    oracle: _PathOracle,
    vehicle_columns: list[np.ndarray],
    assessed: np.ndarray,
) -> sparse.csr_matrix:
    """Build only the assessed rows of one vehicle's path matrix."""

    column_count = len(vehicle_columns)
    row_count = int(assessed.size)
    connected = (assessed >= oracle.arr) & (assessed < oracle.dep)
    if not np.any(connected):
        return sparse.csr_matrix((row_count, column_count))
    connected_rows = np.flatnonzero(connected)
    local_steps = assessed[connected] - int(oracle.arr)
    local_values = np.column_stack(vehicle_columns)[local_steps, :]
    local_sparse = sparse.coo_matrix(local_values)
    return sparse.csr_matrix(
        (
            local_sparse.data,
            (connected_rows[local_sparse.row], local_sparse.col),
        ),
        shape=(row_count, column_count),
    )


def _materialize_dispatch(
    oracles: list[_PathOracle],
    columns: list[list[np.ndarray]],
    weights: list[np.ndarray],
    *,
    steps: int,
) -> dict:
    """Recover one primal fleet dispatch from the master convex weights."""

    ev_power: list[np.ndarray] = []
    ev_energy: list[np.ndarray] = []
    total = np.zeros(steps, dtype=float)
    for oracle, vehicle_columns, vehicle_weights in zip(
        oracles, columns, weights
    ):
        local = np.zeros(oracle.n, dtype=float)
        for weight, column in zip(vehicle_weights, vehicle_columns):
            if abs(float(weight)) > 0.0:
                local += float(weight) * np.asarray(column, dtype=float)
        power = np.zeros(steps, dtype=float)
        power[oracle.arr:oracle.dep] = local
        energy = np.full(steps + 1, oracle.init, dtype=float)
        for t in range(oracle.arr, oracle.dep):
            energy[t + 1] = energy[t] + oracle.dt * power[t]
        if oracle.dep < steps:
            energy[oracle.dep + 1:] = energy[oracle.dep]
        ev_power.append(power)
        ev_energy.append(energy)
        total += power
    return {
        "scenario_power_kw": total,
        "ev_power_kw": ev_power,
        "ev_energy_kwh": ev_energy,
    }


def affine_tracking_bands(
    cfg,
    baseline: np.ndarray,
    up_kw: np.ndarray,
    down_kw: np.ndarray,
    up_active: np.ndarray,
    down_active: np.ndarray,
    up_signal: np.ndarray,
    down_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative affine bands for one fixed participation pattern.

    Not interchangeable with ``joint_validation.fixed_bid_tracking_bands``,
    which takes the bid as numbers and returns the exact band. This one takes
    the bid as unknowns and returns coefficient rows on
    ``[baseline, up, down]``, so the optimizer can carry the band as
    constraints. Use this one to decide a bid, that one to judge a decided bid.

    Rows are coefficients on ``[baseline, up, down]``. Within a product block
    the five-minute response interval has an ordering determined solely by the
    command signal, so it remains affine. At a block boundary both the bid and
    baseline may change; choosing the exact wider interval is disjunctive in
    the first-stage variables. The optimization oracle therefore keeps the new
    instruction's strict band there. This is conservative and makes every
    generated feasibility cut globally valid for the fixed pattern. Final
    frozen-bid certification may still use the exact wider transition band.
    """

    blocks = int(cfg.blocks)
    steps = int(cfg.steps)
    size = 3 * blocks
    baseline = np.asarray(baseline, dtype=float).reshape(blocks)
    up_kw = np.asarray(up_kw, dtype=float).reshape(blocks)
    down_kw = np.asarray(down_kw, dtype=float).reshape(blocks)
    up_active = np.asarray(up_active, dtype=bool).reshape(blocks)
    down_active = np.asarray(down_active, dtype=bool).reshape(blocks)
    up_signal = np.asarray(up_signal, dtype=float).reshape(steps)
    down_signal = np.asarray(down_signal, dtype=float).reshape(steps)
    if np.any((up_signal > cfg.eps) & (down_signal > cfg.eps)):
        raise ValueError("activation scenario has simultaneous up/down signal")

    candidate = np.concatenate([baseline, up_kw, down_kw])
    lower = np.full(steps, -np.inf, dtype=float)
    upper = np.full(steps, np.inf, dtype=float)
    lower_rows = np.zeros((steps, size), dtype=float)
    upper_rows = np.zeros((steps, size), dtype=float)
    band = max(float(cfg.assessment_band_fraction), 0.0)

    def endpoint(block: int, up_value: float, down_value: float):
        if not (up_active[block] or down_active[block]):
            return None
        lo = np.zeros(size, dtype=float)
        hi = np.zeros(size, dtype=float)
        lo[block] = 1.0
        hi[block] = 1.0
        effective_up = 0.0
        effective_down = 0.0
        if up_value > cfg.eps and up_active[block]:
            lo[blocks + block] = -(up_value + band)
            hi[blocks + block] = -(up_value - band)
            effective_up = float(up_value)
        elif down_value > cfg.eps and down_active[block]:
            lo[2 * blocks + block] = down_value - band
            hi[2 * blocks + block] = down_value + band
            effective_down = float(down_value)
        elif cfg.enforce_idle_baseline:
            # The idle band is band * (U + D), so both width columns enter the
            # same row and the row is exact: no supporting hyperplane has to be
            # chosen, and no later cut can find the block held to less.
            if up_active[block]:
                lo[blocks + block] = -band
                hi[blocks + block] = band
            if down_active[block]:
                lo[2 * blocks + block] = -band
                hi[2 * blocks + block] = band
        else:
            return None
        return lo, hi, effective_up, effective_down

    for t in range(steps):
        block = min(blocks - 1, t // int(cfg.steps_per_block))
        current = endpoint(block, float(up_signal[t]), float(down_signal[t]))
        if current is None:
            continue
        lo_row, hi_row, effective_up, effective_down = current
        if cfg.apply_transition_band and t > 0:
            previous_block = min(
                blocks - 1, (t - 1) // int(cfg.steps_per_block)
            )
            if previous_block == block:
                previous = endpoint(
                    previous_block,
                    float(up_signal[t - 1]),
                    float(down_signal[t - 1]),
                )
                if previous is not None:
                    prev_lo, prev_hi, prev_up, prev_down = previous
                    delta_up = effective_up - prev_up
                    delta_down = effective_down - prev_down
                    increasing = (
                        delta_up <= cfg.eps
                        and delta_down >= -cfg.eps
                        and (delta_up < -cfg.eps or delta_down > cfg.eps)
                    )
                    decreasing = (
                        delta_up >= -cfg.eps
                        and delta_down <= cfg.eps
                        and (delta_up > cfg.eps or delta_down < -cfg.eps)
                    )
                    if increasing:
                        lo_row = prev_lo
                    elif decreasing:
                        hi_row = prev_hi
        lower_rows[t] = lo_row
        upper_rows[t] = hi_row
        lower[t] = float(lo_row @ candidate)
        upper[t] = float(hi_row @ candidate)
    return lower, upper, lower_rows, upper_rows


def feasibility_cut_from_certificate(
    info: dict,
    lower_rows: np.ndarray,
    upper_rows: np.ndarray,
    candidate: np.ndarray,
) -> dict | None:
    """Translate a completed column-generation certificate into a master cut.

    The returned convention matches the bid master: every feasible first-stage
    point satisfies ``constant + coefficient @ x >= 0``.
    """

    direct_phase_one = bool(info.get("direct_phase_one", False))
    required = {"assessed", "alpha", "beta"}
    if not direct_phase_one:
        required.update({"sigma", "certificate_value"})
    if not bool(info.get("complete", False)) or not required.issubset(info):
        return None
    assessed = np.asarray(info["assessed"], dtype=np.int32)
    alpha = np.asarray(info["alpha"], dtype=float)
    beta = np.asarray(info["beta"], dtype=float)
    lo = np.asarray(lower_rows, dtype=float)[assessed]
    hi = np.asarray(upper_rows, dtype=float)[assessed]
    infeasibility_coefficient = alpha @ lo - beta @ hi
    coefficient = -np.asarray(infeasibility_coefficient, dtype=float)
    candidate = np.asarray(candidate, dtype=float).reshape(coefficient.size)
    if direct_phase_one:
        # The phase-I value function is convex in the tracking-band RHS.
        # Its optimal dual is therefore a subgradient. Every feasible bid x
        # has value zero and must satisfy
        #   objective(candidate) + g @ (x - candidate) <= 0.
        # Reorient that inequality to the Benders convention below.
        objective = float(info.get("objective", np.nan))
        if not np.isfinite(objective) or objective <= 0.0:
            return None
        constant = float(infeasibility_coefficient @ candidate - objective)
    else:
        sigma = np.asarray(info["sigma"], dtype=float)
        constant = -float(np.sum(sigma))
    value_at_candidate = float(constant + coefficient @ candidate)
    if value_at_candidate >= -1e-8:
        return None
    scale = max(float(np.max(np.abs(coefficient))), abs(constant), 1.0)
    return {
        "coefficient": coefficient / scale,
        "constant": constant / scale,
        "value_at_candidate": value_at_candidate / scale,
        "raw_farkas_value": (
            -float(info["certificate_value"])
            if "certificate_value" in info
            else -float(info.get("objective", np.nan))
        ),
        "phase1_violation_kw": float(info.get("objective", np.nan)),
        "cut_method": (
            "direct_phase1_fallback" if direct_phase_one else "colgen_phase1"
        ),
        "colgen_rounds": int(info.get("rounds", 0)),
        "colgen_columns": int(info.get("columns", 0)),
    }


def certify(
    evs,
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    *,
    steps: int = STEPS,
    dt: float = DT_HOURS,
    eta_ch: float = 1.0,
    max_rounds: int = 60,
    tol: float = STEP_VIOLATION_TOL_KW,
    free_magnitude: float = 1e8,
    return_dispatch: bool = False,
    initial_columns: list[list[np.ndarray]] | None = None,
    return_column_pool: bool = False,
    time_limit_s: float | None = None,
    fallback_to_direct: bool = True,
):
    """Decide whether the fleet can hold a frozen bid.

    Column generation answers first, because it is much the cheaper of the two
    for a full fleet.  It can also stop without an answer -- a round limit, a
    pricing solve that comes back with an unrecognised status, a duplicate
    column it cannot use -- and none of those say anything about the fleet.
    The direct phase-I LP is the exact fallback for exactly that case, so ask
    it rather than hand the caller no verdict: a caller with no verdict has no
    dispatch either, and a missing dispatch has been read as a tracking
    failure before.

    Returns ``(feasible, rounds, info)``.  ``feasible`` is ``None`` only when
    neither solver could decide inside the time allowed.
    """

    started = time.perf_counter()
    feasible, rounds, info = _certify_by_column_generation(
        evs,
        band_lo,
        band_hi,
        steps=steps,
        dt=dt,
        eta_ch=eta_ch,
        max_rounds=max_rounds,
        tol=tol,
        free_magnitude=free_magnitude,
        return_dispatch=return_dispatch,
        initial_columns=initial_columns,
        return_column_pool=return_column_pool,
        time_limit_s=time_limit_s,
    )
    if feasible is not None or not fallback_to_direct:
        return feasible, rounds, info

    remaining = None
    if time_limit_s is not None and float(time_limit_s) > 0.0:
        remaining = float(time_limit_s) - (time.perf_counter() - started)
        if remaining <= 0.0:
            return feasible, rounds, info

    direct_feasible, _direct_rounds, direct = certify_direct_phase_one(
        evs,
        band_lo,
        band_hi,
        steps=steps,
        dt=dt,
        eta_ch=eta_ch,
        tol=tol,
        free_magnitude=free_magnitude,
        return_dispatch=return_dispatch,
        time_limit_s=remaining,
    )
    direct["answered_after_column_generation_stopped"] = True
    direct["column_generation_reason"] = info.get("reason")
    direct["column_generation_rounds"] = int(rounds)
    if "column_pool" in info:
        direct["column_pool"] = info["column_pool"]
    return direct_feasible, rounds, direct


def _certify_by_column_generation(
    evs,
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    *,
    steps: int = STEPS,
    dt: float = DT_HOURS,
    eta_ch: float = 1.0,
    max_rounds: int = 60,
    tol: float = STEP_VIOLATION_TOL_KW,
    free_magnitude: float = 1e8,
    return_dispatch: bool = False,
    initial_columns: list[list[np.ndarray]] | None = None,
    return_column_pool: bool = False,
    time_limit_s: float | None = None,
):
    """Decide whether the fleet can hold a frozen bid.

    ``band_lo`` and ``band_hi`` are the per-step aggregate power limits the bid
    implies, as produced by ``fixed_bid_tracking_bands``; steps carrying no
    submitted instruction should arrive with limits beyond ``free_magnitude`` and
    are dropped, since they constrain nothing and would only add slack the
    certificate has to carry.

    Returns ``(feasible, rounds, info)``. Feasibility is read off the phase-one
    objective: the master may miss the band and pays for it, so a zero objective
    is a dispatch that tracks every assessed step. When infeasible, ``info``
    carries the certificate -- ``alpha`` and ``beta`` on the assessed steps,
    ``sigma`` per vehicle, and ``certificate_value``, which is positive exactly
    when the bid is provably unholdable. ``feasible`` is ``None`` when pricing
    did not finish, because a round limit or solver failure is not an
    infeasibility proof.
    """

    started = time.perf_counter()
    deadline = (
        None
        if time_limit_s is None or float(time_limit_s) <= 0.0
        else started + float(time_limit_s)
    )

    def remaining_time() -> float | None:
        if deadline is None:
            return None
        return float(deadline - time.perf_counter())

    def timed_out() -> bool:
        remaining = remaining_time()
        return bool(remaining is not None and remaining <= 0.0)

    band_lo = np.asarray(band_lo, dtype=float).reshape(-1)
    band_hi = np.asarray(band_hi, dtype=float).reshape(-1)
    if band_lo.size != steps or band_hi.size != steps:
        raise ValueError(f"tracking bands must each have {steps} entries")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    oracles = [_PathOracle(ev, steps, dt, eta_ch) for ev in evs]
    n_ev = len(oracles)
    columns: list[list[np.ndarray]] = []
    column_keys: list[set[bytes]] = []
    for index, oracle in enumerate(oracles):
        first = _initial_column(oracle)
        if first is None or not oracle.feasible_bounds:
            return False, 0, {
                "complete": True,
                "reason": "a vehicle has no feasible dispatch",
            }
        vehicle_columns = [first]
        vehicle_keys = {_column_fingerprint(first)}
        cached = (
            initial_columns[index]
            if initial_columns is not None and index < len(initial_columns)
            else ()
        )
        for candidate in cached:
            candidate = np.asarray(candidate, dtype=float).reshape(-1)
            if candidate.size != oracle.n or not np.all(np.isfinite(candidate)):
                continue
            prefix = np.cumsum(candidate)
            locally_feasible = bool(
                np.all(candidate >= np.asarray(oracle.bounds)[:, 0] - 1e-7)
                and np.all(candidate <= np.asarray(oracle.bounds)[:, 1] + 1e-7)
                and np.all(prefix >= oracle.lo - 1e-7)
                and np.all(prefix <= oracle.hi + 1e-7)
            )
            key = _column_fingerprint(candidate)
            if locally_feasible and key not in vehicle_keys:
                vehicle_columns.append(candidate.copy())
                vehicle_keys.add(key)
        columns.append(vehicle_columns)
        column_keys.append(vehicle_keys)

    def attach_column_pool(info: dict) -> dict:
        if return_column_pool:
            info["column_pool"] = columns
        return info

    assessed = np.flatnonzero(
        (np.abs(band_lo) < free_magnitude) & (np.abs(band_hi) < free_magnitude)
    )
    lower, upper = band_lo[assessed], band_hi[assessed]
    m = int(assessed.size)
    if m == 0:
        info = {
            "complete": True,
            "objective": 0.0,
            "reason": "no assessed step",
        }
        if return_dispatch:
            info.update(_materialize_dispatch(
                oracles,
                columns,
                [np.ones(1, dtype=float) for _ in oracles],
                steps=steps,
            ))
        return True, 0, attach_column_pool(info)
    if n_ev == 0:
        feasible = bool(np.all(lower <= tol) and np.all(upper >= -tol))
        return feasible, 0, attach_column_pool({
            "complete": True,
            "objective": 0.0 if feasible else float(
                np.maximum(lower, 0.0).sum() + np.maximum(-upper, 0.0).sum()
            ),
            "reason": "empty fleet",
        })
    if max_rounds <= 0:
        return None, 0, attach_column_pool({
            "complete": False,
            "reason": "round limit reached before pricing",
        })

    objective = np.inf
    alpha = beta = sigma = None
    n_columns = 0
    rnd = 0
    for rnd in range(1, max_rounds + 1):
        if timed_out():
            return None, rnd - 1, attach_column_pool({
                "complete": False,
                "timed_out": True,
                "reason": "oracle time limit reached before restricted master",
            })
        blocks, offsets, n_columns = [], [], 0
        for i, oracle in enumerate(oracles):
            offsets.append(n_columns)
            blocks.append(_assessed_column_block(oracle, columns[i], assessed))
            n_columns += len(columns[i])

        column_matrix = sparse.hstack(blocks, format="csr")
        eye = sparse.identity(m, format="csr")
        zero = sparse.csr_matrix((m, m))

        # -(P lam) - s <= -L  and  (P lam) - r <= U, so the band sits on the
        # right-hand side and its duals weight the bid itself.
        a_ub = sparse.vstack([
            sparse.hstack([-column_matrix, -eye, zero], format="csr"),
            sparse.hstack([column_matrix, zero, -eye], format="csr"),
        ], format="csr")
        b_ub = np.concatenate([-lower, upper])

        convexity = sparse.lil_matrix((n_ev, n_columns + 2 * m))
        for i in range(n_ev):
            convexity[i, offsets[i]:offsets[i] + len(columns[i])] = 1.0

        master_options = None
        remaining = remaining_time()
        if remaining is not None:
            if remaining <= 0.0:
                return None, rnd - 1, attach_column_pool({
                    "complete": False,
                    "timed_out": True,
                    "reason": "oracle time limit reached before restricted master",
                })
            master_options = {"time_limit": max(remaining, 1e-3)}
        res = linprog(
            np.concatenate([np.zeros(n_columns), np.ones(2 * m)]),
            A_ub=a_ub, b_ub=b_ub,
            A_eq=sparse.csr_matrix(convexity), b_eq=np.ones(n_ev),
            bounds=[(0.0, None)] * (n_columns + 2 * m),
            method="highs",
            options=master_options,
        )
        if not res.success:
            timeout = bool(
                timed_out()
                or "time limit" in str(res.message).lower()
            )
            return None, rnd, attach_column_pool({
                "complete": False,
                "timed_out": timeout,
                "reason": (
                    f"oracle time limit reached in restricted master: {res.message}"
                    if timeout
                    else f"master failed: {res.message}"
                ),
            })

        objective = float(res.fun)
        marginals = -np.asarray(res.ineqlin.marginals, dtype=float)
        alpha, beta = marginals[:m], marginals[m:]
        sigma = np.asarray(res.eqlin.marginals, dtype=float)
        # The restricted master carries one lower and one upper slack per
        # assessed step after the columns.  A command passes when no step sits
        # outside its band by more than the tolerance, which is what
        # validate_joint_solution asks; the objective is their sum over up to
        # 288 steps and answers a different question.  The sum stays in the
        # info because it is the value function the Benders cuts need.
        step_slacks = np.asarray(
            res.x[n_columns:n_columns + 2 * m], dtype=float
        )
        worst_violation = float(np.max(step_slacks)) if step_slacks.size else 0.0
        step_tol = max(tol, step_violation_tolerance_kw(0.5 * (upper - lower)))
        if worst_violation <= step_tol:
            info = {
                "complete": True,
                "objective": objective,
                "worst_step_violation_kw": worst_violation,
                "columns": n_columns,
            }
            if return_dispatch:
                info.update(_materialize_dispatch(
                    oracles,
                    columns,
                    [
                        np.asarray(
                            res.x[offsets[i]:offsets[i] + len(columns[i])],
                            dtype=float,
                        )
                        for i in range(n_ev)
                    ],
                    steps=steps,
                ))
            return True, rnd, attach_column_pool(info)

        pi = np.zeros(steps)
        pi[assessed] = alpha - beta
        added = 0
        duplicate_improving = 0
        max_duplicate_gain = -np.inf
        max_pricing_gain = -np.inf
        for i, oracle in enumerate(oracles):
            remaining = remaining_time()
            if remaining is not None and remaining <= 0.0:
                return None, rnd, attach_column_pool({
                    "complete": False,
                    "timed_out": True,
                    "objective": objective,
                    "columns": n_columns,
                    "reason": (
                        "oracle time limit reached before pricing "
                        f"vehicle {i}"
                    ),
                })
            out = oracle.solve(pi, time_limit_s=remaining)
            if out is None:
                timeout = bool(
                    timed_out()
                    or "time limit" in oracle.last_solve_message.lower()
                )
                return None, rnd, attach_column_pool({
                    "complete": False,
                    "timed_out": timeout,
                    "objective": objective,
                    "columns": n_columns,
                    "reason": (
                        f"oracle time limit reached pricing vehicle {i}: "
                        f"{oracle.last_solve_message}"
                        if timeout
                        else f"pricing failed for vehicle {i}: "
                        f"{oracle.last_solve_message}"
                    ),
                })
            column, value = out
            # A column's reduced cost is 0 - (p.pi + sigma_i), so it improves the
            # master exactly when that quantity is positive.
            gain = float(value + sigma[i])
            max_pricing_gain = max(max_pricing_gain, gain)
            if gain > 1e-7:
                key = _column_fingerprint(column)
                if key in column_keys[i]:
                    duplicate_improving += 1
                    max_duplicate_gain = max(max_duplicate_gain, gain)
                else:
                    columns[i].append(column)
                    column_keys[i].add(key)
                    added += 1
        if added == 0:
            if duplicate_improving and max_duplicate_gain > max(1e-6, 10.0 * tol):
                return None, rnd, attach_column_pool({
                    "complete": False,
                    "objective": objective,
                    "columns": n_columns,
                    "duplicate_improving_columns": int(duplicate_improving),
                    "max_pricing_gain": float(max_duplicate_gain),
                    "reason": "pricing produced only duplicate improving columns",
                })
            break

    if added > 0:
        # The restricted master is still missing improving columns.  Its dual
        # is not valid for the full master, hence neither "infeasible" nor the
        # apparent certificate may be reported.
        return None, rnd, attach_column_pool({
            "complete": False,
            "objective": objective,
            "columns": n_columns,
            "max_pricing_gain": max_pricing_gain,
            "reason": "round limit reached with improving columns remaining",
        })

    return False, rnd, attach_column_pool({
        "complete": True,
        "objective": objective,
        "columns": n_columns,
        "max_pricing_gain": max_pricing_gain,
        "assessed": assessed,
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "certificate_value": float(alpha @ lower - beta @ upper + sigma.sum()),
    })


def certify_direct_phase_one(
    evs,
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    *,
    steps: int = STEPS,
    dt: float = DT_HOURS,
    eta_ch: float = 1.0,
    tol: float = STEP_VIOLATION_TOL_KW,
    free_magnitude: float = 1e8,
    return_dispatch: bool = False,
    time_limit_s: float | None = None,
    free_baseline_blocks: int | None = None,
):
    """Exact sparse phase-I fallback for an unfinished column generator.

    This is deliberately not the normal path.  Column generation is much
    cheaper for a full fleet, but a solver/degeneracy failure must not leave a
    scenario classified as either feasible or infeasible without proof.  The
    fallback writes every connected EV power and energy state into one sparse
    LP and minimizes aggregate tracking-band slack.  A zero objective is an
    exact feasible dispatch; a positive objective supplies a valid subgradient
    cut for the Benders master.

    ``free_baseline_blocks`` answers a different question: it lets the baseline
    of each 30-minute block move, which slides that block's bands together
    without changing their width or their shape.  It is a diagnostic for how
    much a baseline revised close to delivery could recover, and it is an upper
    bound on any revision scheme, because the offset here is chosen knowing the
    whole command.  ``None`` keeps the baseline the bid declared.
    """

    deadline = (
        None
        if time_limit_s is None or float(time_limit_s) <= 0.0
        else time.perf_counter() + float(time_limit_s)
    )

    band_lo = np.asarray(band_lo, dtype=float).reshape(-1)
    band_hi = np.asarray(band_hi, dtype=float).reshape(-1)
    if band_lo.size != steps or band_hi.size != steps:
        raise ValueError(f"tracking bands must each have {steps} entries")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    oracles = [_PathOracle(ev, steps, dt, eta_ch) for ev in evs]
    if any(_initial_column(oracle) is None for oracle in oracles):
        return False, 0, {
            "complete": True,
            "direct_phase_one": True,
            "reason": "a vehicle has no feasible dispatch",
        }

    assessed = np.flatnonzero(
        (np.abs(band_lo) < free_magnitude)
        & (np.abs(band_hi) < free_magnitude)
    )
    lower = band_lo[assessed]
    upper = band_hi[assessed]
    assessed_position = {
        int(step): position for position, step in enumerate(assessed)
    }

    variable_bounds: list[tuple[float | None, float | None]] = []
    power_indices: list[np.ndarray] = []
    energy_indices: list[np.ndarray] = []
    power_by_assessed_step: list[list[int]] = [
        [] for _ in range(len(assessed))
    ]
    for oracle in oracles:
        power = np.arange(
            len(variable_bounds), len(variable_bounds) + oracle.n, dtype=np.int32
        )
        variable_bounds.extend(oracle.bounds)
        energy = np.arange(
            len(variable_bounds), len(variable_bounds) + oracle.n, dtype=np.int32
        )
        cap = float(oracle.hi[0] * oracle.dt + oracle.init)
        for local_step in range(oracle.n):
            lower_energy = 0.0
            if local_step == oracle.n - 1 and oracle.target is not None:
                lower_energy = max(lower_energy, float(oracle.target))
            variable_bounds.append((lower_energy, cap))
            global_step = oracle.arr + local_step
            position = assessed_position.get(global_step)
            if position is not None:
                power_by_assessed_step[position].append(int(power[local_step]))
        power_indices.append(power)
        energy_indices.append(energy)

    lower_slack_offset = len(variable_bounds)
    variable_bounds.extend([(0.0, None)] * len(assessed))
    upper_slack_offset = len(variable_bounds)
    variable_bounds.extend([(0.0, None)] * len(assessed))
    baseline_offset = None
    block_of_assessed: np.ndarray | None = None
    if free_baseline_blocks is not None:
        blocks = int(free_baseline_blocks)
        if blocks <= 0 or steps % blocks != 0:
            raise ValueError(
                "free_baseline_blocks must divide the step count "
                f"({blocks} does not divide {steps})"
            )
        steps_per_block = steps // blocks
        block_of_assessed = assessed // steps_per_block
        baseline_offset = len(variable_bounds)
        # Free: the diagnostic asks what any baseline could do, so it must not
        # be bounded by a level anyone happened to plan.
        variable_bounds.extend([(None, None)] * blocks)
    variable_count = len(variable_bounds)

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    eq_rhs: list[float] = []
    for oracle, power, energy in zip(oracles, power_indices, energy_indices):
        for local_step in range(oracle.n):
            row = len(eq_rhs)
            eq_rows.extend([row, row])
            eq_cols.extend([int(energy[local_step]), int(power[local_step])])
            eq_data.extend([1.0, -float(dt)])
            if local_step == 0:
                eq_rhs.append(float(oracle.init))
            else:
                eq_rows.append(row)
                eq_cols.append(int(energy[local_step - 1]))
                eq_data.append(-1.0)
                eq_rhs.append(0.0)

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    ub_rhs = np.empty(2 * len(assessed), dtype=float)
    for position, power_columns in enumerate(power_by_assessed_step):
        lower_row = position
        upper_row = len(assessed) + position
        for column in power_columns:
            ub_rows.extend([lower_row, upper_row])
            ub_cols.extend([column, column])
            ub_data.extend([-1.0, 1.0])
        ub_rows.extend([lower_row, upper_row])
        ub_cols.extend([
            lower_slack_offset + position,
            upper_slack_offset + position,
        ])
        ub_data.extend([-1.0, -1.0])
        if baseline_offset is not None:
            # sum(power) >= lower + beta  and  sum(power) <= upper + beta:
            # the pair of bands for this step slides by the same beta, so the
            # window keeps its width and only its centre moves.
            column = baseline_offset + int(block_of_assessed[position])
            ub_rows.extend([lower_row, upper_row])
            ub_cols.extend([column, column])
            ub_data.extend([1.0, -1.0])
        ub_rhs[lower_row] = -float(lower[position])
        ub_rhs[upper_row] = float(upper[position])

    equality_matrix = sparse.coo_matrix(
        (eq_data, (eq_rows, eq_cols)),
        shape=(len(eq_rhs), variable_count),
    ).tocsr()
    upper_matrix = sparse.coo_matrix(
        (ub_data, (ub_rows, ub_cols)),
        shape=(len(ub_rhs), variable_count),
    ).tocsr()
    objective_vector = np.zeros(variable_count, dtype=float)
    objective_vector[lower_slack_offset:upper_slack_offset + len(assessed)] = 1.0

    attempts: list[dict] = []
    result = None
    for method, base_options in (
        ("highs", None),
        ("highs-ds", {"presolve": False}),
        ("highs-ipm", {"presolve": False}),
    ):
        remaining = (
            None if deadline is None else float(deadline - time.perf_counter())
        )
        if remaining is not None and remaining <= 0.0:
            attempts.append({
                "method": method,
                "status": 1,
                "message": "oracle time limit reached before solver attempt",
            })
            break
        options = dict(base_options or {})
        if remaining is not None:
            options["time_limit"] = max(remaining, 1e-3)
        if not options:
            options = None
        candidate = linprog(
            objective_vector,
            A_ub=upper_matrix,
            b_ub=ub_rhs,
            A_eq=equality_matrix,
            b_eq=np.asarray(eq_rhs, dtype=float),
            bounds=variable_bounds,
            method=method,
            options=options,
        )
        attempts.append({
            "method": method,
            "status": int(candidate.status),
            "message": str(candidate.message),
        })
        if candidate.success and candidate.x is not None:
            result = candidate
            break

    if result is None:
        timeout = bool(
            deadline is not None
            and (
                time.perf_counter() >= deadline
                or any(
                    "time limit" in str(attempt.get("message", "")).lower()
                    for attempt in attempts
                )
            )
        )
        return None, 0, {
            "complete": False,
            "direct_phase_one": True,
            "timed_out": timeout,
            "reason": (
                "direct phase-I oracle time limit reached"
                if timeout
                else "direct phase-I LP failed"
            ),
            "solver_attempts": attempts,
        }

    objective_value = float(result.fun)
    # A command passes when no five-minute value sits outside its band by more
    # than the tolerance, which is what validate_joint_solution asks.  The
    # objective is the sum of those violations over up to 288 steps, so a sum
    # threshold answers a different question: it can reject a command whose
    # every step is inside the band the validator would accept.  The sum is
    # kept because it is the convex value function the Benders cuts need.
    worst_violation = 0.0
    # Distance outside the band at each step, zero where the step is not
    # assessed. Which steps carry it says which blocks a controller that knew
    # the whole day would still have failed under this minimum-violation plan.
    step_slack = np.zeros(steps, dtype=float)
    if len(assessed):
        slacks = np.asarray(
            result.x[lower_slack_offset:upper_slack_offset + len(assessed)],
            dtype=float,
        )
        worst_violation = float(np.max(slacks)) if slacks.size else 0.0
        step_slack[assessed] = slacks[:len(assessed)] + slacks[len(assessed):]
    tol = max(tol, step_violation_tolerance_kw(0.5 * (upper - lower)))
    info = {
        "complete": True,
        "direct_phase_one": True,
        "objective": objective_value,
        "worst_step_violation_kw": worst_violation,
        "step_slack_kw": step_slack,
        "step_violation_tol_kw": float(tol),
        "columns": int(variable_count),
        "assessed": assessed,
        "solver_attempts": attempts,
    }
    if baseline_offset is not None:
        # How far this command needed each block's baseline moved.  One
        # command's answer says nothing on its own; the spread of these across
        # commands says whether a single declared baseline could serve them
        # all, which is what a revision deadline is or is not needed for.
        info["baseline_offset_kw"] = np.asarray(
            result.x[baseline_offset:baseline_offset + int(free_baseline_blocks)],
            dtype=float,
        )
    if worst_violation <= tol:
        if return_dispatch:
            ev_power: list[np.ndarray] = []
            ev_energy: list[np.ndarray] = []
            total = np.zeros(steps, dtype=float)
            for oracle, power in zip(oracles, power_indices):
                local = np.asarray(result.x[power], dtype=float)
                power_trace = np.zeros(steps, dtype=float)
                power_trace[oracle.arr:oracle.dep] = local
                energy_trace = np.full(steps + 1, oracle.init, dtype=float)
                for step in range(oracle.arr, oracle.dep):
                    energy_trace[step + 1] = (
                        energy_trace[step] + float(dt) * power_trace[step]
                    )
                if oracle.dep < steps:
                    energy_trace[oracle.dep + 1:] = energy_trace[oracle.dep]
                ev_power.append(power_trace)
                ev_energy.append(energy_trace)
                total += power_trace
            info.update({
                "scenario_power_kw": total,
                "ev_power_kw": ev_power,
                "ev_energy_kwh": ev_energy,
            })
        return True, 0, info

    marginals = -np.asarray(result.ineqlin.marginals, dtype=float)
    count = len(assessed)
    info.update({
        "alpha": marginals[:count],
        "beta": marginals[count:],
        "reason": "direct phase-I positive tracking slack",
    })
    return False, 0, info

"""Data containers for the central physical LP reserve bidder.

The EV control LP code in this project uses one signed EV power variable:
positive means charging, negative means discharging.  This module follows the
same convention, so there is no separate simultaneous charge/discharge issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def soc_to_fraction(value: float) -> float:
    """Return an EVSpec SoC, which is a fraction of the battery in [0, 1].

    A percentage cannot be told apart from a fraction by its value: 1.0 % and
    a full battery are both 1.0. EVEnv works in percent, so the adapter divides
    by 100; anything above 1 here is a caller that did not, and is refused.
    """

    v = float(value)
    if not np.isfinite(v) or v < -1e-9 or v > 1.0 + 1e-9:
        raise ValueError(f"EVSpec SoC must be a fraction in [0, 1], got {v!r}")
    return float(np.clip(v, 0.0, 1.0))


@dataclass(frozen=True)
class EVSpec:
    """One EV session.

    ``departure_t`` is the first unavailable 5-minute step.  The EV can be
    controlled for ``arrival_t <= t < departure_t``. ``target_required`` is
    false for sessions whose actual departure lies beyond the modeled day.
    ``initial_soc`` and ``target_soc`` are fractions of the battery (0..1).
    """

    arrival_t: int
    departure_t: int
    initial_soc: float
    target_soc: float
    capacity_kwh: float = 100.0
    max_charge_kw: float = 27.5
    max_discharge_kw: float = 27.5
    station_id: int = 0
    ev_id: str | int | None = None
    target_required: bool = True

    @property
    def initial_kwh(self) -> float:
        return soc_to_fraction(self.initial_soc) * float(self.capacity_kwh)

    @property
    def target_kwh(self) -> float:
        return soc_to_fraction(self.target_soc) * float(self.capacity_kwh)

    def terminal_min_kwh(
        self,
        horizon_t: int,
        *,
        dt_hours: float = 5.0 / 60.0,
        eta_ch: float = 1.0,
    ) -> float:
        """Energy that must remain at a truncated horizon.

        Sessions departing after the modeled service day used to carry no
        target constraint at all.  That lets the bidder empty a long-stay EV at
        midnight even when there is not enough time to recharge it before its
        real departure.  Preserve exactly the energy from which flat-out
        charging after ``horizon_t`` can still reach the departure target.

        For an in-horizon session the ordinary departure constraint remains the
        authority.  ``target_required=False`` with an in-horizon departure is
        therefore left unconstrained for backwards-compatible synthetic cases.
        """

        horizon = int(horizon_t)
        departure = int(self.departure_t)
        if departure <= horizon:
            return 0.0
        capacity = max(float(self.capacity_kwh), 0.0)
        remaining_hours = max(departure - horizon, 0) * max(float(dt_hours), 0.0)
        future_charge = (
            max(float(self.max_charge_kw), 0.0)
            * max(float(eta_ch), 0.0)
            * remaining_hours
        )
        return float(np.clip(float(self.target_kwh) - future_charge, 0.0, capacity))

    def connected(self, t: int) -> bool:
        return int(self.arrival_t) <= int(t) < int(self.departure_t)


@dataclass(frozen=True)
class ActivationScenario:
    """One day-ahead uncertainty scenario.

    ``up_signal`` and ``down_signal`` are 5-minute signals in [0, 1].  At a
    given step only one side should be active.  With neither active, the LP
    enforces aggregate EV power equal to the submitted baseline.
    """

    name: str
    up_signal: np.ndarray
    down_signal: np.ndarray
    evs: list[EVSpec]
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BiddingLPConfig:
    steps: int = 288
    blocks: int = 48
    steps_per_block: int = 6
    dt_hours: float = 5.0 / 60.0
    assessment_band_fraction: float = 0.10
    # Secondary Reserve 2 allows the resource its five-minute response time
    # after a changed simple-dispatch instruction. During that response the
    # admissible interval spans the old and new instruction bands.
    apply_transition_band: bool = True
    eps: float = 1e-8
    enforce_idle_baseline: bool = True
    idle_tolerance_kw: float = 1e-6
    eta_ch: float = 1.0
    eta_dis: float = 1.0
    # Independent EV x command recourse scenarios may be solved in separate
    # processes.  Keep the library default serial; experiment CLI chooses a
    # bounded value from the host CPU count.
    scenario_workers: int = 1
    # LP wall-clock cap. An incomplete solve is rejected; a time-limited
    # incumbent is never treated as a certified upper bid.
    time_limit_s: float | None = None
    # Worker-local column pools are only warm starts.  Keeping every path ever
    # priced makes later EV x command scenarios progressively larger without
    # strengthening the mathematical certificate.  Zero disables the cap.
    colgen_cache_columns_per_ev: int = 12
    # Use the exact direct sparse phase-I oracle at or below this EV count and
    # column generation above it. Zero keeps column generation for every size.
    direct_oracle_max_evs: int = 0
    # Objective weight on the baseline step between two adjacent participating
    # blocks, in kW-block per kW of step. Capacity alone leaves the baseline
    # free wherever Assessment I and the recourse do not pin it, and the LP then
    # returns whichever vertex it reaches, which can put the whole band on one
    # side of a block and the next block's on the other. With a weight this
    # small the search still maximizes capacity first: removing a step of s kW
    # can cost at most weight x s kW-block. 0 leaves the baseline to the solver.
    baseline_step_weight: float = 0.0


@dataclass(frozen=True)
class JointBiddingProblem:
    """First-stage bid with scenario-specific physical recourse.

    ``baseline``, ``up`` and ``down`` are common day-ahead decisions.  EV
    dispatch is allowed to differ by scenario, while every EV departure target
    remains a hard constraint.  The production path uses
    ``global_pass_rate=1`` so every assessed block is hard. The field is kept
    in the validation object for stored-artifact compatibility, but the current
    builder does not solve a chance contract. A block with zero up and down
    award has no tracking obligation.
    """

    # Relative first-stage weights for U_b + D_b.  The precision/capacity
    # study uses unit weights, so every kW-block is valued equally and block
    # placement is decided only by physical recourse feasibility.
    objective_weights: np.ndarray
    scenarios: list[ActivationScenario]
    baseline_min_kw: float | np.ndarray = 0.0
    baseline_max_kw: float | np.ndarray = 0.0
    u_cap: np.ndarray | None = None
    d_cap: np.ndarray | None = None
    global_pass_rate: float = 1.0
    minimum_direction_bid_kw: float = 0.0
    fixed_baseline: np.ndarray | None = None
    fixed_up: np.ndarray | None = None
    fixed_down: np.ndarray | None = None
    # Assessment I, per block: the fleet power the EVs can hold for the whole
    # 30-minute block (market.sustained_capability). The bid must keep
    # baseline - up >= sustained_power_min_kw and
    # baseline + down <= sustained_power_max_kw. None leaves it unconstrained.
    sustained_power_min_kw: np.ndarray | None = None
    sustained_power_max_kw: np.ndarray | None = None
    config: BiddingLPConfig = field(default_factory=BiddingLPConfig)


def assessment_i_rows(problem: "JointBiddingProblem") -> tuple[np.ndarray, np.ndarray]:
    """Assessment I as rows ``A x <= b`` over x = (baseline, up, down)."""

    blocks = int(problem.config.blocks)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    if problem.sustained_power_min_kw is not None:
        floor = np.broadcast_to(np.asarray(problem.sustained_power_min_kw, dtype=float), (blocks,))
        for block in range(blocks):
            row = np.zeros(3 * blocks)
            row[block] = -1.0
            row[blocks + block] = 1.0
            rows.append(row)
            rhs.append(-float(floor[block]))
    if problem.sustained_power_max_kw is not None:
        ceiling = np.broadcast_to(np.asarray(problem.sustained_power_max_kw, dtype=float), (blocks,))
        for block in range(blocks):
            row = np.zeros(3 * blocks)
            row[block] = 1.0
            row[2 * blocks + block] = 1.0
            rows.append(row)
            rhs.append(float(ceiling[block]))
    if not rows:
        return np.zeros((0, 3 * blocks)), np.zeros(0)
    return np.vstack(rows), np.asarray(rhs, dtype=float)


def baseline_step_matrix(participating, blocks: int) -> np.ndarray:
    """Rows giving ``baseline[k+1] - baseline[k]`` over x = (baseline, up, down).

    One row per pair of adjacent blocks that both participate. A block with no
    award has no assessed baseline (its tracking is off), so a step into or
    out of it is not a step of the submitted bid.
    """

    blocks = int(blocks)
    active = np.asarray(participating, dtype=bool).reshape(blocks)
    pairs = np.flatnonzero(active[:-1] & active[1:])
    matrix = np.zeros((pairs.size, 3 * blocks))
    matrix[np.arange(pairs.size), pairs] = -1.0
    matrix[np.arange(pairs.size), pairs + 1] = 1.0
    return matrix


def project_assessment_i(problem: "JointBiddingProblem", baseline, up, down):
    """Shrink up/down widths so the bid meets Assessment I at its baseline."""

    blocks = int(problem.config.blocks)
    baseline = np.asarray(baseline, dtype=float).reshape(blocks)
    up = np.asarray(up, dtype=float).reshape(blocks).copy()
    down = np.asarray(down, dtype=float).reshape(blocks).copy()
    if problem.sustained_power_min_kw is not None:
        floor = np.broadcast_to(np.asarray(problem.sustained_power_min_kw, dtype=float), (blocks,))
        up = np.minimum(up, np.maximum(baseline - floor, 0.0))
    if problem.sustained_power_max_kw is not None:
        ceiling = np.broadcast_to(np.asarray(problem.sustained_power_max_kw, dtype=float), (blocks,))
        down = np.minimum(down, np.maximum(ceiling - baseline, 0.0))
    return up, down


# Solver statuses that carry a usable primal point. Kept in one place so the
# per-scenario recourse aggregation and BiddingSolution.success cannot drift
# apart and disagree about whether a solve produced something submittable.
USABLE_SOLVER_STATUSES = frozenset({
    "optimal",
    "suboptimal",
    "time_limit_feasible",
})


@dataclass
class BiddingSolution:
    status: str
    solver: str
    objective_value: float
    baseline_kw: np.ndarray
    up_kw: np.ndarray
    down_kw: np.ndarray
    scenario_power_kw: dict[str, np.ndarray] = field(default_factory=dict)
    ev_power_kw: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    ev_energy_kwh: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return str(self.status).lower() in USABLE_SOLVER_STATUSES

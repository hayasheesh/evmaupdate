"""Scenario feasibility as a feasible circulation, with a Hoffman certificate.

The scenario recourse asks one question: with the bid frozen, is there a
dispatch of the connected fleet whose per-step total stays inside the
Assessment-II band?  Written as a matrix ``X`` with one row per vehicle and one
column per five-minute step, that question is

    f(i,t) <= X(i,t) <= g(i,t)                      power limits
    lo(i,k) <= sum_{s<=k} X(i,s) <= hi(i,k)         SoC bounds, departure target
    L(t) <= sum_i X(i,t) <= U(t)                    the tracking band

which is exactly the *prefix-bounded matrix* feasibility problem of Borsik,
Frank, Madarasi and Takacs (arXiv:2505.10739): entry bounds, bounds on every
row prefix, and bounds on a column prefix -- here only the full column sum.
Section 7.1 of that paper builds a digraph whose ``(l, u)``-feasible
circulations correspond one-to-one to such matrices (their Theorem 7.2), so
feasibility is decided by one max-flow and Hoffman's circulation theorem
supplies the certificate when there is none.

This module builds that digraph in its contracted form.  Two contractions are
applied, both exact:

* the column prefixes are unbounded except at the last row, so the column chain
  collapses to a single node per assessed step, and an unassessed step
  collapses into the hub;
* the total-sum arc is unbounded, so the paper's two extra vertices merge into
  one hub.

The vehicle rows keep every bound, since ``_PathOracle`` bounds every prefix.
Entries outside a vehicle's connection window are fixed at zero and are simply
absent.

The construction is exact only because charge and discharge convert one for
one.  Stored energy is then the running sum of one signed power variable, which
is what makes the row constraints prefix bounds.  With a round-trip loss the
state is no longer a prefix sum of a single variable and the problem leaves this
class.  The same paper also records the boundary on the other side: bounding
arbitrary *segments* rather than prefixes is not a network-flow problem at all,
and is NP-complete once the segment bounds vary by row and column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colgen_feasibility import (
    STEPS,
    DT_HOURS,
    STEP_VIOLATION_TOL_KW,
    _PathOracle,
    step_violation_tolerance_kw,
)


BAND_LOWER = "band_lower"
BAND_UPPER = "band_upper"


@dataclass
class Circulation:
    """A digraph with arc bounds, plus where the tracking band entered it."""

    node_count: int
    tail: np.ndarray
    head: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    # Global five-minute step carried by each arc, or -1 for arcs that do not
    # carry the band.  The bid enters the network only through these.
    band_step: np.ndarray
    hub: int
    assessed_steps: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def arc_count(self) -> int:
        return int(self.tail.size)


def scenario_circulation(
    evs,
    band_lo,
    band_hi,
    *,
    steps: int = STEPS,
    dt: float = DT_HOURS,
    eta_ch: float = 1.0,
    free_magnitude: float = 1e8,
) -> Circulation | None:
    """Return the circulation network for one frozen-bid scenario.

    ``None`` means a vehicle cannot be dispatched at all, which is an
    infeasibility of the fleet rather than of the band.
    """

    band_lo = np.asarray(band_lo, dtype=float).reshape(-1)
    band_hi = np.asarray(band_hi, dtype=float).reshape(-1)
    if band_lo.size != steps or band_hi.size != steps:
        raise ValueError(f"tracking bands must each have {steps} entries")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not np.isclose(float(eta_ch), 1.0):
        # Stored energy is the running sum of one signed power variable only
        # when charge and discharge convert one for one.  With a round-trip
        # loss the row constraints stop being prefix bounds and the problem
        # leaves the class this construction relies on.
        raise ValueError(
            "the circulation model requires unit charge/discharge efficiency; "
            f"got eta_ch={float(eta_ch)!r}"
        )

    oracles = [_PathOracle(ev, int(steps), float(dt), float(eta_ch)) for ev in evs]
    if any(oracle.n <= 0 or not oracle.feasible_bounds for oracle in oracles):
        return None

    assessed = np.flatnonzero(
        (np.abs(band_lo) < free_magnitude) & (np.abs(band_hi) < free_magnitude)
    )
    column_node: dict[int, int] = {}

    # Node 0 is the hub: the paper's two extra vertices and every unassessed
    # column collapse into it.
    hub = 0
    next_node = 1
    for step in assessed:
        column_node[int(step)] = next_node
        next_node += 1

    tail: list[int] = []
    head: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    band_step: list[int] = []

    for oracle in oracles:
        span = int(oracle.n)
        first_node = next_node
        next_node += span
        # Row-prefix arcs.  Arc k carries the prefix sum of the first k+1
        # entries, exactly as in Claim 7.1 of the paper: the last one is fed
        # from the hub, the others run backwards along the row.
        for position in range(span):
            prefix_node = first_node + position
            if position == span - 1:
                tail.append(hub)
                head.append(prefix_node)
            else:
                tail.append(first_node + position + 1)
                head.append(prefix_node)
            lower.append(float(oracle.lo[position]))
            upper.append(float(oracle.hi[position]))
            band_step.append(-1)
        # Entry arcs: one signed power variable per connected step.
        low_power, high_power = oracle.bounds[0]
        for position in range(span):
            step = int(oracle.arr + position)
            tail.append(first_node + position)
            head.append(column_node.get(step, hub))
            lower.append(float(low_power))
            upper.append(float(high_power))
            band_step.append(-1)

    # Column arcs: the only place the bid enters.
    for step in assessed:
        tail.append(column_node[int(step)])
        head.append(hub)
        lower.append(float(band_lo[int(step)]))
        upper.append(float(band_hi[int(step)]))
        band_step.append(int(step))

    return Circulation(
        node_count=next_node,
        tail=np.asarray(tail, dtype=np.int64),
        head=np.asarray(head, dtype=np.int64),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        band_step=np.asarray(band_step, dtype=np.int64),
        hub=hub,
        assessed_steps=np.asarray(assessed, dtype=np.int64),
        metadata={
            "vehicles": len(oracles),
            "connected_steps": int(sum(int(o.n) for o in oracles)),
            "assessed_steps": int(assessed.size),
        },
    )


class _Dinic:
    """Max flow with real capacities.

    scipy's ``maximum_flow`` takes integer capacities only.  Rounding kilowatts
    to integers would decide feasibility differently exactly at the band scale
    where the capacity search spends its time, so the flow is computed in
    floating point here.
    """

    def __init__(self, node_count: int):
        self.node_count = int(node_count)
        self.graph: list[list[int]] = [[] for _ in range(self.node_count)]
        self.to: list[int] = []
        self.capacity: list[float] = []

    def add_arc(self, tail: int, head: int, capacity: float) -> int:
        index = len(self.to)
        self.graph[int(tail)].append(index)
        self.to.append(int(head))
        self.capacity.append(float(capacity))
        self.graph[int(head)].append(index + 1)
        self.to.append(int(tail))
        self.capacity.append(0.0)
        return index

    def _levels(self, source: int, sink: int, tol: float):
        level = [-1] * self.node_count
        level[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for arc in self.graph[node]:
                nxt = self.to[arc]
                if self.capacity[arc] > tol and level[nxt] < 0:
                    level[nxt] = level[node] + 1
                    queue.append(nxt)
        return level if level[sink] >= 0 else None

    def _augment(self, node: int, sink: int, pushed: float, level, iterator, tol):
        if node == sink:
            return pushed
        while iterator[node] < len(self.graph[node]):
            arc = self.graph[node][iterator[node]]
            nxt = self.to[arc]
            if self.capacity[arc] > tol and level[nxt] == level[node] + 1:
                sent = self._augment(
                    nxt, sink, min(pushed, self.capacity[arc]), level, iterator, tol
                )
                if sent > tol:
                    self.capacity[arc] -= sent
                    self.capacity[arc ^ 1] += sent
                    return sent
            iterator[node] += 1
        return 0.0

    def max_flow(self, source: int, sink: int, *, tol: float = 1e-9) -> float:
        total = 0.0
        while True:
            level = self._levels(source, sink, tol)
            if level is None:
                return total
            iterator = [0] * self.node_count
            while True:
                pushed = self._augment(
                    source, sink, float("inf"), level, iterator, tol
                )
                if pushed <= tol:
                    break
                total += pushed

    def reachable(self, source: int, *, tol: float = 1e-9) -> np.ndarray:
        seen = np.zeros(self.node_count, dtype=bool)
        seen[source] = True
        stack = [source]
        while stack:
            node = stack.pop()
            for arc in self.graph[node]:
                nxt = self.to[arc]
                if self.capacity[arc] > tol and not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        return seen


def hoffman_slack(circulation: Circulation, inside: np.ndarray) -> float:
    """Return ``rho_u(W) - delta_l(W)``; negative means ``W`` proves infeasibility."""

    inside = np.asarray(inside, dtype=bool)
    tail_in = inside[circulation.tail]
    head_in = inside[circulation.head]
    entering = head_in & ~tail_in
    leaving = tail_in & ~head_in
    return float(
        np.sum(circulation.upper[entering]) - np.sum(circulation.lower[leaving])
    )


# Returned in place of a verdict when igraph is missing, so that "no solver"
# stays distinguishable from "the solver could not decide".
NO_COMPILED_SOLVER = object()


def band_deficit_tolerance_kw(circulation: Circulation) -> float:
    """Return the flow deficit this circulation may carry and still pass.

    The bid enters the network only through the band arcs, so the band is what
    a deficit has to be measured against.  Every other scale in the network --
    above all the total supply, which grows with the fleet and with the day --
    is unrelated to the question being asked, and a tolerance built from one
    of those makes a large fleet pass a violation a small fleet would fail.
    """

    band = circulation.band_step >= 0
    if not np.any(band):
        return STEP_VIOLATION_TOL_KW
    return step_violation_tolerance_kw(
        0.5 * (circulation.upper[band] - circulation.lower[band])
    )


def _igraph_solve(
    circulation: Circulation, *, tol: float = 1e-7
):
    """Decide the circulation with a compiled float max flow.

    Returns ``(feasible, violated)``.  ``feasible`` is ``None`` when the flow
    falls short by more than the band allows and no Hoffman set accounts for
    it: that is not a proof either way, and saying "infeasible" there was
    wrong.  ``NO_COMPILED_SOLVER`` means igraph is not installed, so the
    caller keeps its own solver.

    A returned violated set is checked against :func:`hoffman_slack` before it
    is handed back, so a partition this function reads wrongly cannot become a
    cut.
    """

    try:
        import igraph
    except Exception:
        return NO_COMPILED_SOLVER

    nodes = circulation.node_count
    source, sink = nodes, nodes + 1
    excess = np.zeros(nodes, dtype=float)
    np.add.at(excess, circulation.head, circulation.lower)
    np.subtract.at(excess, circulation.tail, circulation.lower)
    supply = np.flatnonzero(excess > tol)
    demand = np.flatnonzero(excess < -tol)
    edges = (
        list(zip(circulation.tail.tolist(), circulation.head.tolist()))
        + [(source, int(node)) for node in supply]
        + [(int(node), sink) for node in demand]
    )
    capacities = np.concatenate(
        (
            circulation.upper - circulation.lower,
            excess[supply],
            -excess[demand],
        )
    ).tolist()
    required = float(np.sum(excess[supply]))
    graph = igraph.Graph(nodes + 2, edges, directed=True)
    # Only the value is wanted first.  igraph's maxflow() also materialises the
    # flow on every arc, which costs two orders of magnitude more than the
    # value or the cut on a network this size.
    value = float(graph.maxflow_value(source, sink, capacity=capacities))
    # `required` sums every lower bound in the network, so it grows with the
    # fleet and with the length of the day.  A deficit measured as a share of
    # it therefore has nothing to do with the band being decided: at 89
    # vehicles a share of 1e-7 came to about 1.1e-03 kW, which is more than
    # twice the 4.0e-04 kW the band itself allows, and a bid violating its
    # band by that much was certified.  Double precision on a network this
    # size is good to about 1e-12 kW, so the band's own tolerance is not
    # near the arithmetic either.
    if value >= required - band_deficit_tolerance_kw(circulation):
        return True, None
    cut = graph.st_mincut(source, sink, capacity=capacities)
    inside = np.zeros(nodes, dtype=bool)
    inside[[node for node in cut.partition[0] if node < nodes]] = True
    for candidate in (inside, ~inside):
        if hoffman_slack(circulation, candidate) < -tol:
            return False, candidate
    return None, None


def solve_circulation(
    circulation: Circulation, *, tol: float = 1e-7, need_flow: bool = False
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Return ``(feasible, flow, violated_set)`` for one circulation.

    The standard reduction is used: an arc ``[l, u]`` becomes an arc of
    capacity ``u - l`` plus a demand of ``l`` at its head and a supply of ``l``
    at its tail, and the circulation exists exactly when the max flow between
    the added terminals saturates every supply arc.  When it does not, the
    residual reachable set is the Hoffman-violating vertex set; the returned
    set is checked against :func:`hoffman_slack` before it is handed back.
    """

    compiled = (
        NO_COMPILED_SOLVER
        if need_flow
        else _igraph_solve(circulation, tol=tol)
    )
    if compiled is not NO_COMPILED_SOLVER:
        feasible, violated = compiled
        # The dispatch itself is only wanted by the probes, which keep using
        # the pure-Python path; the bidder needs the decision and the cut.
        return feasible, None, violated

    node_count = circulation.node_count
    source = node_count
    sink = node_count + 1
    solver = _Dinic(node_count + 2)
    arc_index = np.empty(circulation.arc_count, dtype=np.int64)
    excess = np.zeros(node_count, dtype=float)
    for arc in range(circulation.arc_count):
        low = circulation.lower[arc]
        high = circulation.upper[arc]
        if high < low - tol:
            return False, None, None
        arc_index[arc] = solver.add_arc(
            int(circulation.tail[arc]),
            int(circulation.head[arc]),
            float(high - low),
        )
        excess[int(circulation.head[arc])] += low
        excess[int(circulation.tail[arc])] -= low

    required = 0.0
    for node in range(node_count):
        if excess[node] > tol:
            solver.add_arc(source, node, float(excess[node]))
            required += float(excess[node])
        elif excess[node] < -tol:
            solver.add_arc(node, sink, float(-excess[node]))

    flow_value = solver.max_flow(source, sink, tol=tol * 1e-2)
    if flow_value >= required - band_deficit_tolerance_kw(circulation):
        flow = np.array(
            [
                circulation.lower[arc]
                + solver.capacity[int(arc_index[arc]) ^ 1]
                for arc in range(circulation.arc_count)
            ],
            dtype=float,
        )
        return True, flow, None

    reachable = solver.reachable(source, tol=tol * 1e-2)[:node_count]
    for candidate in (reachable, ~reachable):
        if hoffman_slack(circulation, candidate) < -tol:
            return False, None, candidate
    # The flow says infeasible but neither side of the residual cut is a
    # certificate; report it rather than guess.
    return None, None, None


def master_cut(
    circulation: Circulation,
    violated: np.ndarray,
    lower_rows,
    upper_rows,
    candidate,
) -> dict | None:
    """Turn a Hoffman set into a cut in the bid master's own variables.

    The master's convention is that every feasible first-stage point satisfies
    ``constant + coefficient @ x >= 0``.  Hoffman's condition on the violated
    vertex set says the band arcs entering it cannot supply what the arcs
    leaving it demand,

        sum_{t in upper} band_hi(t) - sum_{t in lower} band_lo(t) + const >= 0,

    and ``affine_tracking_bands`` gives each band as a row on
    ``[baseline, up, down]``, so the inequality is already linear in the bid.
    Unlike a dual vector read off one LP basis, this inequality describes the
    exact feasible region rather than one supporting hyperplane of a relaxation.

    ``None`` when the set does not actually cut the current candidate off.
    """

    cut = band_cut(circulation, violated)
    lower_rows = np.asarray(lower_rows, dtype=float)
    upper_rows = np.asarray(upper_rows, dtype=float)
    candidate = np.asarray(candidate, dtype=float).reshape(-1)
    coefficient = np.zeros(candidate.size, dtype=float)
    for step in cut["band_upper_steps"]:
        coefficient += upper_rows[int(step)]
    for step in cut["band_lower_steps"]:
        coefficient -= lower_rows[int(step)]
    constant = float(cut["constant"])
    value_at_candidate = float(constant + coefficient @ candidate)
    if value_at_candidate >= -1e-8:
        return None
    scale = max(float(np.max(np.abs(coefficient), initial=0.0)), abs(constant), 1.0)
    return {
        "coefficient": coefficient / scale,
        "constant": constant / scale,
        "value_at_candidate": value_at_candidate / scale,
        "raw_farkas_value": value_at_candidate,
        "cut_method": "prefix_circulation_hoffman",
        "hoffman_slack": float(cut["slack"]),
        "band_upper_steps": list(cut["band_upper_steps"]),
        "band_lower_steps": list(cut["band_lower_steps"]),
    }


def band_cut(circulation: Circulation, violated: np.ndarray) -> dict:
    """Return the violated Hoffman inequality as a linear form in the band.

    Feasibility of this vertex set requires

        sum_t upper_coefficient[t] * band_hi[t]
      - sum_t lower_coefficient[t] * band_lo[t]
      + constant  >=  0

    and the returned set makes the left side negative.  Both band arrays are
    affine in the bid, so this is a linear cut in the bid.
    """

    inside = np.asarray(violated, dtype=bool)
    tail_in = inside[circulation.tail]
    head_in = inside[circulation.head]
    entering = head_in & ~tail_in
    leaving = tail_in & ~head_in
    band = circulation.band_step >= 0

    upper_steps = circulation.band_step[entering & band]
    lower_steps = circulation.band_step[leaving & band]
    constant = float(
        np.sum(circulation.upper[entering & ~band])
        - np.sum(circulation.lower[leaving & ~band])
    )
    return {
        "band_upper_steps": upper_steps.astype(int).tolist(),
        "band_lower_steps": lower_steps.astype(int).tolist(),
        "constant": constant,
        "slack": hoffman_slack(circulation, inside),
    }

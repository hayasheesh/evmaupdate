"""The circulation model must decide exactly what the exact LP decides.

The scenario feasibility question is a prefix-bounded matrix feasibility
problem, so it can be answered by one feasible circulation instead of a coupled
LP.  These tests hold that claim to the existing exact certifier: same answer on
random fleets, same answer on both sides of the band scale where the answer
flips, a Hoffman set whenever the answer is no, and a cut that never removes a
band the LP accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

from market.physical_lp_bidding.colgen_feasibility import (
    _PathOracle,
    _initial_column,
    certify_direct_phase_one,
)
from market.physical_lp_bidding.data_classes import EVSpec
from market.physical_lp_bidding.prefix_circulation import (
    Circulation,
    band_cut,
    band_deficit_tolerance_kw,
    hoffman_slack,
    scenario_circulation,
    solve_circulation,
)

DT = 5.0 / 60.0


def _fleet(rng, count, steps):
    evs = []
    for _ in range(count):
        arrival = int(rng.integers(0, max(1, steps - 2)))
        departure = int(rng.integers(arrival + 1, steps + 1))
        capacity = float(rng.uniform(20.0, 60.0))
        initial = float(rng.uniform(0.1, 0.6))
        charge = float(rng.uniform(3.0, 11.0))
        # Keep the departure target reachable; otherwise the case is infeasible
        # for a reason that has nothing to do with the band.
        room = charge * (departure - arrival) * DT / capacity
        target = float(min(0.95, initial + rng.uniform(0.0, 1.0) * min(0.35, room)))
        evs.append(
            EVSpec(
                arrival_t=arrival,
                departure_t=departure,
                initial_soc=initial,
                target_soc=target,
                capacity_kwh=capacity,
                max_charge_kw=charge,
                max_discharge_kw=float(rng.uniform(0.0, 11.0)),
                target_required=True,
            )
        )
    return evs


def _anchor(evs, steps):
    """A dispatch every vehicle can follow, used to place the band."""

    total = np.zeros(steps, dtype=float)
    for ev in evs:
        oracle = _PathOracle(ev, steps, DT, 1.0)
        column = _initial_column(oracle)
        if column is None:
            return None
        total[oracle.arr : oracle.dep] += np.asarray(column, dtype=float)
    return total


def _case(rng):
    steps = int(rng.integers(5, 15))
    evs = _fleet(rng, int(rng.integers(2, 6)), steps)
    anchor = _anchor(evs, steps)
    if anchor is None:
        return None
    width = float(rng.choice([0.0, 0.05, 0.5, 3.0, 10.0]))
    centre = anchor + rng.normal(0.0, float(rng.choice([0.0, 1.0, 5.0])), steps)
    lower = centre - width
    upper = centre + width
    free = rng.random(steps) < 0.25
    lower[free] = -1e9
    upper[free] = 1e9
    return evs, lower, upper, steps


def test_the_circulation_decides_exactly_what_the_lp_decides():
    rng = np.random.default_rng(20260903)
    decided = feasible = 0
    for _ in range(400):
        made = _case(rng)
        if made is None:
            continue
        evs, lower, upper, steps = made
        reference, _rounds, _info = certify_direct_phase_one(
            evs, lower, upper, steps=steps, dt=DT
        )
        circulation = scenario_circulation(evs, lower, upper, steps=steps, dt=DT)
        answer = False
        if circulation is not None:
            answer, _flow, _violated = solve_circulation(circulation)
        assert bool(answer) is bool(reference)
        decided += 1
        feasible += int(bool(reference))
    # The generator has to produce both answers or the test proves nothing.
    assert decided > 200
    assert 0 < feasible < decided


def test_an_infeasible_scenario_comes_with_a_hoffman_set():
    rng = np.random.default_rng(4242)
    certificates = 0
    for _ in range(200):
        made = _case(rng)
        if made is None:
            continue
        evs, lower, upper, steps = made
        circulation = scenario_circulation(evs, lower, upper, steps=steps, dt=DT)
        if circulation is None:
            continue
        answer, _flow, violated = solve_circulation(circulation)
        if answer:
            continue
        assert violated is not None
        assert hoffman_slack(circulation, violated) < 0.0
        certificates += 1
    assert certificates > 20


def test_a_feasible_scenario_returns_a_dispatch_inside_every_bound():
    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(200):
        made = _case(rng)
        if made is None:
            continue
        evs, lower, upper, steps = made
        circulation = scenario_circulation(evs, lower, upper, steps=steps, dt=DT)
        if circulation is None:
            continue
        answer, flow, _violated = solve_circulation(circulation, need_flow=True)
        if not answer:
            continue
        assert flow is not None
        assert np.all(flow >= circulation.lower - 1e-6)
        assert np.all(flow <= circulation.upper + 1e-6)
        # Flow conservation at every node is what makes the arcs a dispatch.
        balance = np.zeros(circulation.node_count, dtype=float)
        np.add.at(balance, circulation.head, flow)
        np.subtract.at(balance, circulation.tail, flow)
        assert np.max(np.abs(balance)) < 1e-6
        checked += 1
    assert checked > 20


def test_the_two_methods_flip_at_the_same_band_scale():
    rng = np.random.default_rng(31337)
    flips = 0
    for _ in range(12):
        steps = int(rng.integers(6, 14))
        evs = _fleet(rng, int(rng.integers(2, 6)), steps)
        anchor = _anchor(evs, steps)
        if anchor is None:
            continue
        centre = anchor + rng.normal(0.0, 3.0, steps)
        low, high = 0.0, 60.0
        for _round in range(30):
            middle = 0.5 * (low + high)
            passed, _rounds, _info = certify_direct_phase_one(
                evs, centre - middle, centre + middle, steps=steps, dt=DT
            )
            if passed:
                high = middle
            else:
                low = middle
        if high >= 60.0 or low <= 0.0:
            continue
        for width in (low * 0.999, high * 1.001):
            reference, _rounds, _info = certify_direct_phase_one(
                evs, centre - width, centre + width, steps=steps, dt=DT
            )
            circulation = scenario_circulation(
                evs, centre - width, centre + width, steps=steps, dt=DT
            )
            answer = False
            if circulation is not None:
                answer, _flow, _violated = solve_circulation(circulation)
            assert bool(answer) is bool(reference)
        flips += 1
    assert flips >= 5


def test_a_derived_cut_never_removes_a_band_the_lp_accepts():
    rng = np.random.default_rng(2024)
    tested = 0
    for _ in range(40):
        steps = int(rng.integers(6, 14))
        evs = _fleet(rng, int(rng.integers(2, 5)), steps)
        anchor = _anchor(evs, steps)
        if anchor is None:
            continue
        # A narrow band around a shifted trajectory is reliably infeasible,
        # which is what a cut can be derived from.
        centre = anchor + rng.normal(0.0, 4.0, steps)
        lower = centre - 0.05
        upper = centre + 0.05
        circulation = scenario_circulation(evs, lower, upper, steps=steps, dt=DT)
        if circulation is None:
            continue
        answer, _flow, violated = solve_circulation(circulation)
        if answer or violated is None:
            continue
        cut = band_cut(circulation, violated)
        assert cut["slack"] < 0.0
        for _probe in range(40):
            width = float(rng.uniform(0.0, 30.0))
            shift = rng.normal(0.0, 3.0, steps)
            probe_lo = anchor + shift - width
            probe_hi = anchor + shift + width
            feasible, _rounds, _info = certify_direct_phase_one(
                evs, probe_lo, probe_hi, steps=steps, dt=DT
            )
            if not feasible:
                continue
            value = (
                float(np.sum(probe_hi[cut["band_upper_steps"]]))
                - float(np.sum(probe_lo[cut["band_lower_steps"]]))
                + cut["constant"]
            )
            assert value >= -1e-6
            tested += 1
    assert tested > 20


def test_a_round_trip_loss_is_refused_rather_than_approximated():
    rng = np.random.default_rng(5)
    evs = _fleet(rng, 3, 8)
    with pytest.raises(ValueError, match="unit charge/discharge efficiency"):
        scenario_circulation(
            evs, np.zeros(8), np.ones(8), steps=8, dt=DT, eta_ch=0.95
        )




def _small_config(steps, blocks):
    from market.physical_lp_bidding.data_classes import BiddingLPConfig

    return BiddingLPConfig(
        steps=int(steps),
        blocks=int(blocks),
        steps_per_block=int(steps // blocks),
        dt_hours=DT,
        assessment_band_fraction=0.10,
        apply_transition_band=False,
    )


def test_the_master_cut_is_linear_in_the_bid_and_removes_no_feasible_bid():
    from market.physical_lp_bidding.colgen_feasibility import affine_tracking_bands
    from market.physical_lp_bidding.prefix_circulation import (
        master_cut,
        scenario_circulation,
        solve_circulation,
    )

    steps, blocks = 12, 2
    cfg = _small_config(steps, blocks)
    rng = np.random.default_rng(515)
    tested = 0
    for _ in range(30):
        evs = _fleet(rng, int(rng.integers(2, 5)), steps)
        anchor = _anchor(evs, steps)
        if anchor is None:
            continue
        # Probe bids have to sit near what the fleet can actually hold, or
        # every probe is infeasible and the test checks nothing.
        block_anchor = anchor.reshape(blocks, -1).mean(axis=1)
        up_signal = np.zeros(steps)
        down_signal = np.zeros(steps)
        up_signal[: steps // 2] = rng.uniform(0.3, 1.0)
        down_signal[steps // 2 :] = rng.uniform(0.3, 1.0)
        active = np.ones(blocks, dtype=bool)

        def bands(x):
            return affine_tracking_bands(
                cfg,
                x[:blocks],
                x[blocks : 2 * blocks],
                x[2 * blocks :],
                active,
                active,
                up_signal,
                down_signal,
            )

        candidate = np.concatenate(
            [
                rng.uniform(-5.0, 20.0, blocks),
                rng.uniform(20.0, 60.0, blocks),
                rng.uniform(20.0, 60.0, blocks),
            ]
        )
        lower, upper, lower_rows, upper_rows = bands(candidate)
        # The rows have to reproduce the numbers, or the cut is in the wrong
        # coordinates.
        finite = np.abs(lower) < 1e8
        assert np.allclose(lower_rows[finite] @ candidate, lower[finite], atol=1e-9)
        assert np.allclose(upper_rows[finite] @ candidate, upper[finite], atol=1e-9)

        circulation = scenario_circulation(evs, lower, upper, steps=steps, dt=DT)
        if circulation is None:
            continue
        feasible, _flow, violated = solve_circulation(circulation)
        if feasible or violated is None:
            continue
        cut = master_cut(circulation, violated, lower_rows, upper_rows, candidate)
        if cut is None:
            continue
        assert cut["value_at_candidate"] < 0.0

        for _probe in range(60):
            probe = np.concatenate(
                [
                    block_anchor + rng.normal(0.0, 1.0, blocks),
                    rng.uniform(0.5, 6.0, blocks),
                    rng.uniform(0.5, 6.0, blocks),
                ]
            )
            probe_lower, probe_upper, _lr, _ur = bands(probe)
            passed, _rounds, _info = certify_direct_phase_one(
                evs, probe_lower, probe_upper, steps=steps, dt=DT
            )
            if not passed:
                continue
            value = float(cut["constant"] + cut["coefficient"] @ probe)
            # A cut that removed a bid this scenario can actually hold would
            # make the master's answer wrong, not merely different.
            assert value >= -1e-6
            tested += 1
    assert tested > 20


def _circulation_with(band_half_width_kw: float, other_arc_kw: float):
    """Return a circulation whose band arcs and bulk arcs are set separately.

    The shape is irrelevant here; what is under test is which arcs the
    tolerance is read from.
    """

    import numpy as np

    return Circulation(
        node_count=3,
        tail=np.array([0, 1, 2], dtype=np.int64),
        head=np.array([1, 2, 0], dtype=np.int64),
        lower=np.array(
            [-other_arc_kw, -other_arc_kw, -band_half_width_kw], dtype=float
        ),
        upper=np.array(
            [other_arc_kw, other_arc_kw, band_half_width_kw], dtype=float
        ),
        band_step=np.array([-1, -1, 0], dtype=np.int64),
        hub=0,
        assessed_steps=np.array([0], dtype=np.int64),
    )


def test_the_screen_tolerance_reads_the_band_and_not_the_rest_of_the_network():
    """A larger fleet must not buy a larger violation.

    The accepted flow deficit used to be a share of the network's total
    supply, which grows with the number of vehicles and the length of the day.
    At 89 vehicles that came to about 1.1e-03 kW against a band that allowed
    4.0e-04, and a bid violating its band by that much was certified.
    """

    narrow = band_deficit_tolerance_kw(_circulation_with(4.0, 1.0))
    with_a_bigger_fleet = band_deficit_tolerance_kw(
        _circulation_with(4.0, 100_000.0)
    )

    assert narrow == pytest.approx(with_a_bigger_fleet)
    # And it does follow the band.
    assert band_deficit_tolerance_kw(
        _circulation_with(4_000.0, 1.0)
    ) > narrow


def test_a_band_miss_above_the_tolerance_is_not_screened_as_feasible():
    """One vehicle, one step, and a band it cannot reach."""

    ev = EVSpec(
        arrival_t=0,
        departure_t=6,
        initial_soc=0.5,
        target_soc=0.5,
        capacity_kwh=1_000.0,
        max_charge_kw=10.0,
        max_discharge_kw=10.0,
        station_id=0,
        ev_id="ev",
        target_required=False,
    )
    steps = 6
    free = np.full(steps, np.inf)
    lower = free.copy()
    upper = free.copy()
    # The fleet can draw at most 10 kW, and the band starts above that.
    lower[0], upper[0] = 10.5, 12.5

    circulation = scenario_circulation(
        [ev], lower, upper, steps=steps, dt=DT, eta_ch=1.0
    )
    assert circulation is not None
    answer, _flow, _violated = solve_circulation(circulation)
    assert answer is not True

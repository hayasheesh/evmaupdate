"""Boundary and certificate tests for frozen-bid column generation."""

from __future__ import annotations

import numpy as np
import pytest

from market.physical_lp_bidding import colgen_feasibility as colgen_module
from market.physical_lp_bidding.colgen_feasibility import (
    certify,
    certify_direct_phase_one,
)
from market.physical_lp_bidding.data_classes import (
    BiddingLPConfig,
    EVSpec,
)
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands

STEPS, BLOCKS, SPB = 288, 48, 6


def _fleet(n=24, departure=STEPS):
    return [
        EVSpec(
            arrival_t=0,
            departure_t=departure,
            initial_soc=0.30,
            target_soc=0.60,
            capacity_kwh=60.0,
            max_charge_kw=11.0,
            max_discharge_kw=11.0,
            station_id=i % 4,
            ev_id=i,
            target_required=True,
        )
        for i in range(n)
    ]


def _bid(evs, depth_kw):
    """A frozen bid holding one down command across the middle of the day."""
    up_kw = np.zeros(BLOCKS)
    down_kw = np.zeros(BLOCKS)
    up_sig = np.zeros(STEPS)
    down_sig = np.zeros(STEPS)
    for b in range(20, 24):
        down_sig[b * SPB:(b + 1) * SPB] = 1.0
        down_kw[b] = depth_kw
    return up_kw, down_kw, 0.25 * down_kw, up_sig, down_sig


def _decomposed_verdict(evs, depth_kw):
    up_kw, down_kw, baseline, up_sig, down_sig = _bid(evs, depth_kw)
    cfg = BiddingLPConfig()
    _t, _tol, lo, hi = fixed_bid_tracking_bands(
        cfg, baseline, up_kw, down_kw, up_sig, down_sig,
        apply_transition_band=bool(cfg.apply_transition_band),
    )
    free = ~np.isfinite(lo) | ~np.isfinite(hi)
    decomposed, _rounds, _info = certify(
        evs,
        np.where(free, -1e9, lo),
        np.where(free, 1e9, hi),
        steps=STEPS,
        dt=float(cfg.dt_hours),
    )
    return decomposed


@pytest.mark.parametrize(
    ("depth_kw", "expected"),
    [
        (40.0, True),
        (120.0, True),
        (200.0, True),
        (260.0, False),
        (400.0, False),
    ],
)
def test_known_fleet_boundary(depth_kw, expected):
    evs = _fleet()
    assert _decomposed_verdict(evs, depth_kw) is expected


def test_certificate_is_positive_when_infeasible():
    """An infeasible bid must come back with a certificate that proves it."""
    evs = _fleet()
    up_kw, down_kw, baseline, up_sig, down_sig = _bid(evs, 5_000.0)
    cfg = BiddingLPConfig()
    _t, _tol, lo, hi = fixed_bid_tracking_bands(
        cfg, baseline, up_kw, down_kw, up_sig, down_sig,
        apply_transition_band=bool(cfg.apply_transition_band),
    )
    free = ~np.isfinite(lo) | ~np.isfinite(hi)
    feasible, _rounds, info = certify(
        evs, np.where(free, -1e9, lo), np.where(free, 1e9, hi),
        steps=STEPS, dt=float(cfg.dt_hours),
    )
    assert not feasible
    assert info["certificate_value"] > 0.0
    # The phase-one objective and the certificate measure the same shortfall.
    assert info["certificate_value"] == pytest.approx(info["objective"], rel=1e-6)


def test_a_fleet_that_cannot_meet_its_own_target_is_rejected():
    evs = [EVSpec(
        arrival_t=0, departure_t=6, initial_soc=0.10, target_soc=0.99,
        capacity_kwh=100.0, max_charge_kw=1.0, max_discharge_kw=1.0,
        station_id=0, ev_id=0, target_required=True,
    )]
    feasible, rounds, info = certify(evs, np.zeros(STEPS), np.zeros(STEPS))
    assert not feasible
    assert rounds == 0
    assert "reason" in info


def test_overnight_terminal_reserve_is_enforced():
    """Truncating the day must not erase an overnight EV's future obligation."""
    ev = EVSpec(
        arrival_t=0,
        departure_t=STEPS + 12,
        initial_soc=0.0,
        target_soc=1.0,
        capacity_kwh=100.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        station_id=0,
        ev_id=0,
        target_required=False,
    )
    cfg = BiddingLPConfig()
    decomposed, rounds, info = certify(
        [ev],
        np.full(STEPS, -np.inf),
        np.full(STEPS, np.inf),
        eta_ch=float(cfg.eta_ch),
    )

    assert decomposed is False
    assert rounds == 0
    assert info["complete"] is True


def test_quiet_initial_column_avoids_degenerate_zero_band_tail():
    ev = EVSpec(
        arrival_t=0,
        departure_t=STEPS,
        initial_soc=0.0,
        target_soc=1.0,
        capacity_kwh=100.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        station_id=0,
        ev_id=0,
        target_required=False,
    )
    feasible, rounds, info = certify([ev], np.zeros(STEPS), np.zeros(STEPS))
    assert feasible is True
    assert rounds == 1
    assert info["complete"] is True


def _one_slow_vehicle() -> EVSpec:
    return EVSpec(
        arrival_t=0,
        departure_t=STEPS,
        initial_soc=0.5,
        target_soc=0.5,
        capacity_kwh=100.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        station_id=0,
        ev_id=0,
        target_required=False,
    )


def test_round_limit_is_not_reported_as_an_infeasibility_proof():
    feasible, rounds, info = colgen_module._certify_by_column_generation(
        [_one_slow_vehicle()], np.ones(STEPS), np.ones(STEPS), max_rounds=1
    )
    assert feasible is None
    assert rounds == 1
    assert info["complete"] is False
    assert "certificate_value" not in info


def test_a_stopped_column_generator_is_answered_by_the_direct_oracle():
    """No verdict is not an answer, and the exact fallback has one.

    A caller that receives no verdict also receives no dispatch, and a missing
    dispatch has been read as a tracking failure before now.  One round is not
    enough for the column generator here, so this is that case.
    """

    ev = _one_slow_vehicle()
    feasible, _rounds, info = certify(
        [ev], np.ones(STEPS), np.ones(STEPS), max_rounds=1
    )

    assert feasible is True
    assert info["complete"] is True
    assert info["answered_after_column_generation_stopped"] is True
    assert info["direct_phase_one"] is True

    # The old behaviour is still available for a caller that wants the
    # column generator's own answer and nothing else.
    without, _r, plain = certify(
        [ev],
        np.ones(STEPS),
        np.ones(STEPS),
        max_rounds=1,
        fallback_to_direct=False,
    )
    assert without is None
    assert plain["complete"] is False


def test_oracle_time_limit_is_passed_to_highs_and_reported_as_incomplete(
    monkeypatch,
):
    seen_options = []

    def time_limited(*_args, **kwargs):
        seen_options.append(kwargs.get("options"))
        return type("Result", (), {
            "success": False,
            "status": 1,
            "message": "Time limit reached",
        })()

    monkeypatch.setattr(colgen_module, "linprog", time_limited)
    ev = _fleet(n=1)[0]
    feasible, _rounds, info = certify(
        [ev],
        np.zeros(STEPS),
        np.zeros(STEPS),
        max_rounds=1,
        time_limit_s=2.0,
    )

    assert feasible is None
    assert info["timed_out"] is True
    assert "time limit" in info["reason"].lower()
    assert seen_options and 0.0 < seen_options[0]["time_limit"] <= 2.0


@pytest.mark.parametrize(("depth_kw", "expected"), [(120.0, True), (400.0, False)])
def test_direct_phase_one_fallback_matches_known_fleet_boundary(
    depth_kw, expected
):
    evs = _fleet()
    up_kw, down_kw, baseline, up_sig, down_sig = _bid(evs, depth_kw)
    cfg = BiddingLPConfig()
    _target, _tol, lower, upper = fixed_bid_tracking_bands(
        cfg,
        baseline,
        up_kw,
        down_kw,
        up_sig,
        down_sig,
        apply_transition_band=bool(cfg.apply_transition_band),
    )

    feasible, rounds, info = certify_direct_phase_one(
        evs,
        lower,
        upper,
        steps=STEPS,
        dt=float(cfg.dt_hours),
        return_dispatch=bool(expected),
    )

    assert feasible is expected
    assert rounds == 0
    assert info["complete"] is True
    assert info["direct_phase_one"] is True
    if expected:
        assert "scenario_power_kw" in info
    else:
        assert info["objective"] > 0.0
        assert {"assessed", "alpha", "beta"}.issubset(info)


def test_the_step_tolerance_follows_the_tightest_band():
    """A fixed number of kW cannot serve every band width.

    The same 1e-5 kW is a millionth of a 19 kW band and a ten-thousandth of
    the 0.1 kW band a 1 kW award carries.  The tolerance therefore scales,
    and scales to the narrowest band in the day: one number is applied to
    every step, and scaling it to a wide band elsewhere would let a narrow
    one be missed outright.
    """

    import numpy as np

    from market.physical_lp_bidding.colgen_feasibility import (
        STEP_VIOLATION_TOL_FRACTION,
        STEP_VIOLATION_TOL_KW,
        step_violation_tolerance_kw,
    )

    # The two terms meet at the narrowest band the experiment uses.
    assert step_violation_tolerance_kw(np.array([0.1])) == pytest.approx(
        STEP_VIOLATION_TOL_KW
    )
    assert STEP_VIOLATION_TOL_FRACTION * 0.1 == pytest.approx(
        STEP_VIOLATION_TOL_KW
    )

    # Above it the relative term takes over, at a constant share of the band.
    for width in (1.0, 19.314, 64.0):
        tolerance = step_violation_tolerance_kw(np.array([width]))
        assert tolerance / width == pytest.approx(STEP_VIOLATION_TOL_FRACTION)

    # The narrowest band decides, not the widest.
    mixed = step_violation_tolerance_kw(np.array([64.0, 19.314, 1.0]))
    assert mixed == pytest.approx(step_violation_tolerance_kw(np.array([1.0])))

    # A band of no width leaves only the floor.
    assert step_violation_tolerance_kw(
        np.array([0.0, np.inf])
    ) == pytest.approx(STEP_VIOLATION_TOL_KW)


def test_a_violation_far_below_the_band_is_not_a_miss():
    """The certifier judges each step, not the sum over the day.

    A day of 288 steps each a hair inside its band sums to a number a fixed
    threshold on the sum would reject, while no step is outside anything.
    """

    import numpy as np

    from market.physical_lp_bidding.colgen_feasibility import (
        certify_direct_phase_one,
        step_violation_tolerance_kw,
    )
    from market.physical_lp_bidding.data_classes import EVSpec

    steps = 8
    fleet = [
        EVSpec(
            arrival_t=0, departure_t=steps, initial_soc=0.5, target_soc=0.0,
            # Large enough that the power limit binds and the energy limit
            # does not: the test is about the tolerance, not about SoC.
            capacity_kwh=10_000.0,
            max_charge_kw=100.0, max_discharge_kw=100.0,
            station_id=0, ev_id=0, target_required=False,
        )
    ]
    # A 20 kW half-width band the fleet can sit in.
    centre = 50.0
    half = 20.0
    lower = np.full(steps, centre - half)
    upper = np.full(steps, centre + half)
    assert certify_direct_phase_one(
        fleet, lower, upper, steps=steps, dt=0.5
    )[0] is True

    tolerance = step_violation_tolerance_kw(np.full(steps, half))
    # Push the band just past what the fleet can reach, by less than the
    # tolerance for a band this wide.  Summed over the day it is well above
    # any fixed threshold on the sum.
    unreachable = 100.0
    nudge = 0.5 * tolerance
    assert steps * nudge > 1e-5
    assert certify_direct_phase_one(
        fleet,
        np.full(steps, unreachable + nudge),
        np.full(steps, unreachable + nudge + 2 * half),
        steps=steps, dt=0.5,
    )[0] is True

    # A real miss, above the tolerance, still fails.
    assert certify_direct_phase_one(
        fleet,
        np.full(steps, unreachable + 1.0),
        np.full(steps, unreachable + 1.0 + 2 * half),
        steps=steps, dt=0.5,
    )[0] is False


def test_direct_phase_one_reports_the_violation_at_each_step():
    ev = EVSpec(
        arrival_t=0, departure_t=4, initial_soc=0.0, target_soc=0.0,
        capacity_kwh=100.0, max_charge_kw=10.0, max_discharge_kw=0.0,
        target_required=False,
    )
    free = 1e9
    lower = np.array([-free, -free, 15.0, -free])
    upper = np.array([free, free, 20.0, free])
    feasible, _rounds, info = certify_direct_phase_one(
        [ev], lower, upper, steps=4, dt=5.0 / 60.0,
    )
    assert feasible is False
    assert info["step_slack_kw"] == pytest.approx([0.0, 0.0, 5.0, 0.0], abs=1e-6)

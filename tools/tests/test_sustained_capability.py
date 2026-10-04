import numpy as np
import pytest

from market.physical_lp_bidding import EVSpec
from market.sustained_capability import (
    BLOCK_HOURS,
    STEPS_PER_BLOCK,
    sustained_capability_for_scenario,
    sustained_capability_quantile,
)


def _ev(**kwargs):
    # arrival_t / departure_t are 5-minute step indices, so a full day is 288.
    base = dict(
        arrival_t=0,
        departure_t=288,
        initial_soc=0.5,
        target_soc=0.5,
        capacity_kwh=100.0,
        max_charge_kw=10.0,
        max_discharge_kw=10.0,
    )
    base.update(kwargs)
    return EVSpec(**base)


def test_vehicle_absent_for_part_of_a_block_does_not_count():
    """A block's award must be deliverable for the whole 30 minutes."""

    leaves_midway = _ev(arrival_t=0, departure_t=STEPS_PER_BLOCK - 1)
    stays = _ev(arrival_t=0, departure_t=STEPS_PER_BLOCK)

    assert sustained_capability_for_scenario([leaves_midway], blocks=1).occupancy[0] == 0
    assert sustained_capability_for_scenario([stays], blocks=1).occupancy[0] == 1


def test_aggregate_is_the_exact_sum_over_vehicles():
    """Energy cannot move between batteries, so capability simply adds."""

    fleet = [
        _ev(initial_soc=0.4, target_soc=0.8, max_charge_kw=7.0, max_discharge_kw=7.0),
        _ev(initial_soc=0.6, target_soc=0.6, max_charge_kw=11.0, max_discharge_kw=11.0),
        _ev(initial_soc=0.2, target_soc=0.9, max_charge_kw=50.0, max_discharge_kw=50.0),
    ]
    together = sustained_capability_for_scenario(fleet)
    apart = [sustained_capability_for_scenario([ev]) for ev in fleet]

    np.testing.assert_allclose(together.up, sum(a.up for a in apart))
    np.testing.assert_allclose(together.down, sum(a.down for a in apart))


def test_energy_limit_binds_before_power_when_the_battery_is_nearly_full():
    """A nearly full battery cannot absorb its rated power for 30 minutes."""

    # 2 kWh of room, so at most 4 kW sustained over half an hour.
    nearly_full = _ev(initial_soc=0.98, target_soc=0.98, capacity_kwh=100.0,
                      max_charge_kw=50.0, max_discharge_kw=50.0)
    cap = sustained_capability_for_scenario([nearly_full], blocks=1)

    assert cap.down_power_limit[0] == pytest.approx(50.0)
    assert cap.down_energy_limit[0] == pytest.approx(2.0 / BLOCK_HOURS)
    assert cap.down[0] == pytest.approx(4.0)


def test_departure_target_limits_how_much_can_be_given_back():
    """Up capability is bounded by energy held above the departure need."""

    # Arrives empty, must leave full, and only just has time: nothing to give.
    tight = _ev(
        arrival_t=0, departure_t=STEPS_PER_BLOCK, initial_soc=0.0, target_soc=1.0,
        capacity_kwh=5.0, max_charge_kw=60.0, max_discharge_kw=60.0,
    )
    cap = sustained_capability_for_scenario([tight], blocks=1)

    assert cap.up[0] == pytest.approx(0.0, abs=1e-9)
    # The power limit alone would have claimed 60 kW was available.
    assert cap.up_power_limit[0] == pytest.approx(60.0)


def test_post_horizon_departure_keeps_a_reachable_terminal_reserve():
    """A long-stay EV cannot be emptied for free at the service-day boundary."""

    ev = _ev(
        departure_t=300,
        initial_soc=0.8,
        target_soc=0.8,
        capacity_kwh=100.0,
        max_charge_kw=10.0,
        max_discharge_kw=100.0,
        target_required=False,
    )

    # Twelve five-minute steps remain after the 288-step horizon: 1 hour at
    # 10 kW.  At least 70 kWh must therefore remain at midnight to reach 80.
    assert ev.terminal_min_kwh(288) == pytest.approx(70.0)
    cap = sustained_capability_for_scenario([ev])
    assert cap.up[-1] < cap.up_power_limit[-1]


def test_charging_early_creates_up_capability_a_snapshot_would_miss():
    """The envelope credits headroom built up during the stay.

    Reading arrival SoC alone reports no up capability for a vehicle that
    arrives below its target, even though it can charge past that target first
    and give the energy back later.
    """

    ev = _ev(
        arrival_t=0, departure_t=288, initial_soc=0.3, target_soc=0.5,
        capacity_kwh=100.0, max_charge_kw=10.0, max_discharge_kw=10.0,
    )
    cap = sustained_capability_for_scenario([ev])

    # At arrival it holds 30 kWh against a 50 kWh target: a snapshot sees zero.
    assert ev.initial_kwh < ev.target_kwh
    # Late in the stay it has had hours to charge, so it can give energy back.
    assert cap.up[-1] > 0.0
    assert cap.up[-1] == pytest.approx(cap.up_power_limit[-1])


def test_quantile_reduces_capability_across_scenarios():
    scenarios = [
        [_ev(max_charge_kw=10.0, max_discharge_kw=10.0)],
        [_ev(max_charge_kw=20.0, max_discharge_kw=20.0)],
        [_ev(max_charge_kw=30.0, max_discharge_kw=30.0)],
    ]
    low_up, low_down = sustained_capability_quantile(scenarios, quantile=0.0)
    high_up, high_down = sustained_capability_quantile(scenarios, quantile=1.0)

    assert np.all(low_down <= high_down)
    assert low_down[0] == pytest.approx(10.0)
    assert high_down[0] == pytest.approx(30.0)
    assert np.all(low_up <= high_up)


def test_quantile_rejects_an_empty_scenario_set():
    with pytest.raises(ValueError):
        sustained_capability_quantile([], quantile=0.1)

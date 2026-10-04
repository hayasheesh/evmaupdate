import numpy as np
import pytest

from market.physical_lp_bidding import ActivationScenario, BiddingLPConfig, EVSpec
from market.physical_lp_bidding.aggregate_energy import (
    energy_feasible_initial_bid,
    fleet_energy_envelope,
)


def _config(blocks: int) -> BiddingLPConfig:
    return BiddingLPConfig(
        steps=6 * blocks,
        blocks=blocks,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        assessment_band_fraction=0.10,
        apply_transition_band=False,
    )


def _down_all_day(evs, steps: int) -> ActivationScenario:
    return ActivationScenario(
        name="down",
        up_signal=np.zeros(steps),
        down_signal=np.ones(steps),
        evs=evs,
    )


def test_envelope_matches_hand_calculation():
    """One hour at 10 kW, half full, target unchanged."""

    cfg = _config(2)
    ev = EVSpec(0, 12, 0.5, 0.5, 100.0, 10.0, 10.0)
    envelope = fleet_energy_envelope([ev], cfg)
    step_kwh = 10.0 / 12.0
    elapsed = np.arange(1, 13)
    assert envelope.energy_hi_kwh == pytest.approx(step_kwh * elapsed)
    # Late enough, the target forces the lower bound back up to zero change.
    expected_lo = np.maximum(-step_kwh * elapsed, -step_kwh * (12 - elapsed))
    assert envelope.energy_lo_kwh == pytest.approx(expected_lo)
    assert envelope.power_max_kw == pytest.approx(np.full(12, 10.0))
    assert envelope.power_min_kw == pytest.approx(np.full(12, -10.0))


def test_down_width_is_limited_by_the_room_left_in_the_battery():
    """5 kWh of room over 30 minutes at 90 % of the award gives 100/9 kW."""

    cfg = _config(1)
    ev = EVSpec(0, 6, 0.95, 0.0, 100.0, 20.0, 20.0)
    seed = energy_feasible_initial_bid(
        [_down_all_day([ev], 6)],
        cfg,
        baseline_min=0.0,
        baseline_max=0.0,
        up_cap=np.zeros(1),
        down_cap=np.full(1, 20.0),
        minimum_bid_kw=1.0,
    )
    assert seed is not None
    assert seed.down_kw[0] == pytest.approx(100.0 / 9.0, rel=1e-6)
    assert seed.up_kw[0] == 0.0


def test_a_sustained_command_spreads_the_room_over_the_whole_day():
    """Each block alone could absorb 20 kW; twelve in a row share 10 kWh.

    The vehicle cannot discharge, so a block that drops out of the bid cannot
    make room for the others.
    """

    blocks = 12
    cfg = _config(blocks)
    ev = EVSpec(0, 6 * blocks, 0.90, 0.0, 100.0, 20.0, 0.0)
    seed = energy_feasible_initial_bid(
        [_down_all_day([ev], 6 * blocks)],
        cfg,
        baseline_min=0.0,
        baseline_max=0.0,
        up_cap=np.zeros(blocks),
        down_cap=np.full(blocks, 20.0),
        minimum_bid_kw=0.5,
    )
    assert seed is not None
    absorbed_kwh = float(np.sum(0.9 * seed.down_kw * 0.5))
    assert absorbed_kwh == pytest.approx(10.0, rel=1e-6)
    assert np.all(seed.down_kw <= 20.0 + 1e-9)

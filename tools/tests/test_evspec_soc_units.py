"""EVSpec SoC is a fraction; EVEnv SoC is a percentage."""

import numpy as np
import pytest

from market.physical_lp_bidding.data_classes import EVSpec, soc_to_fraction
from market.physical_lp_bidding.evenv_adapter import sample_ev_specs_from_evenv


def test_one_percent_is_not_read_as_a_full_battery():
    ev = EVSpec(0, 6, 0.01, 0.02, 60.0, 11.0, 11.0)
    assert ev.initial_kwh == pytest.approx(0.6)
    assert ev.target_kwh == pytest.approx(1.2)


def test_a_percentage_is_refused():
    with pytest.raises(ValueError):
        soc_to_fraction(45.0)
    with pytest.raises(ValueError):
        EVSpec(0, 6, 45.0, 60.0).initial_kwh


def test_sampled_sessions_are_fractions_and_reach_their_target():
    """Every target the environment sets is reachable in the stay at full power."""

    specs = sample_ev_specs_from_evenv(seed=12345, service_date="2024-04-02")
    assert specs
    dt = 5.0 / 60.0
    for ev in specs:
        assert 0.0 <= ev.initial_soc <= 1.0
        assert 0.0 <= ev.target_soc <= 1.0
        need = ev.target_kwh - ev.initial_kwh
        room = ev.max_charge_kw * dt * (ev.departure_t - ev.arrival_t)
        assert need <= room + 1e-6, ev

from __future__ import annotations

import numpy as np
import pytest

from market.physical_lp_bidding import ActivationScenario, BiddingLPConfig, EVSpec
from market.physical_lp_bidding.solve_bidding import solve_natural_baseline_lp


def test_natural_baseline_supplies_exact_required_energy() -> None:
    config = BiddingLPConfig(
        steps=6,
        blocks=1,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        time_limit_s=10.0,
    )
    scenario = ActivationScenario(
        name="one_ev",
        up_signal=np.zeros(6),
        down_signal=np.zeros(6),
        evs=[EVSpec(0, 6, 0.50, 0.56, 100.0, 20.0, 0.0)],
    )

    baseline = solve_natural_baseline_lp(
        scenario,
        config=config,
        baseline_min_kw=0.0,
        baseline_max_kw=100.0,
    )

    assert baseline.tolist() == pytest.approx([12.0])


def test_natural_baseline_is_the_block_average_when_an_ev_arrives_mid_block() -> None:
    """An EV arriving halfway through a block makes the fleet power step up;
    the seed baseline is that block's average power, not a constant power."""
    config = BiddingLPConfig(
        steps=6,
        blocks=1,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        time_limit_s=10.0,
    )
    scenario = ActivationScenario(
        name="late_ev",
        up_signal=np.zeros(6),
        down_signal=np.zeros(6),
        evs=[EVSpec(3, 6, 0.50, 0.53, 100.0, 20.0, 0.0)],
    )

    baseline = solve_natural_baseline_lp(
        scenario,
        config=config,
        baseline_min_kw=0.0,
        baseline_max_kw=100.0,
    )

    # 3 kWh over the 30-minute block.
    assert baseline.tolist() == pytest.approx([6.0])

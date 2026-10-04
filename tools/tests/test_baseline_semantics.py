"""Regression tests for the submitted baseline's operational meaning."""

from __future__ import annotations

import numpy as np
import pytest

from market.bid_env import BASE_DOWN_KW, BASE_UP_KW, baseline_aware_target_series
from market.physical_lp_bidding import BiddingLPConfig
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
from training.lower_bid_training import _bid_to_target_tol_from_activation


def test_zero_instruction_target_is_the_submitted_baseline() -> None:
    baseline = 120.0
    target, regulation = baseline_aware_target_series(
        np.zeros(6, dtype=float),
        baseline_kw=baseline,
        awarded_up_kw=40.0,
        awarded_down_kw=70.0,
    )
    assert np.allclose(regulation, 0.0)
    assert np.allclose(target, baseline)


def test_up_and_down_instructions_are_offsets_from_the_same_baseline() -> None:
    baseline = 100.0
    source = np.array([-BASE_UP_KW, 0.0, BASE_DOWN_KW], dtype=float)
    target, regulation = baseline_aware_target_series(
        source,
        baseline_kw=baseline,
        awarded_up_kw=40.0,
        awarded_down_kw=60.0,
    )
    assert regulation.tolist() == pytest.approx([-40.0, 0.0, 60.0])
    assert target.tolist() == pytest.approx([60.0, 100.0, 160.0])


def test_physical_validator_centers_idle_band_on_baseline() -> None:
    config = BiddingLPConfig(
        steps=6,
        blocks=1,
        steps_per_block=6,
        assessment_band_fraction=0.10,
        enforce_idle_baseline=True,
        apply_transition_band=False,
    )
    targets, tolerances, lower, upper = fixed_bid_tracking_bands(
        config,
        baseline=np.array([100.0]),
        up_kw=np.array([20.0]),
        down_kw=np.array([30.0]),
        up_signal=np.zeros(6),
        down_signal=np.zeros(6),
        apply_transition_band=False,
    )
    # With both directions offered, the zero-instruction band is set by the
    # whole contracted range: (20 + 30) * 10% = 5 kW around the submitted
    # baseline.  Only this row uses the sum; an activated step is still held
    # to ten percent of the award in the direction being called.
    assert np.allclose(targets, 100.0)
    assert np.allclose(tolerances, 5.0)
    assert np.allclose(lower, 95.0)
    assert np.allclose(upper, 105.0)


def test_counterfactual_band_is_carried_with_the_fixed_bid_episode() -> None:
    baseline = np.full(48, 100.0)
    up = np.full(48, 20.0)
    down = np.full(48, 40.0)
    up_proxy = np.zeros(288)
    down_proxy = np.ones(288)

    target, tolerance, _regulation = _bid_to_target_tol_from_activation(
        baseline,
        up,
        down,
        up_proxy,
        down_proxy,
        band_fraction=0.075,
    )

    assert np.allclose(target, 140.0)
    assert np.allclose(tolerance, 3.0)

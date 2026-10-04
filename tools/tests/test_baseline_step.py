"""Baseline steps between adjacent participating blocks as a capacity tie-break."""

import numpy as np
import pytest

from market.physical_lp_bidding import ActivationScenario, BiddingLPConfig, EVSpec
from market.physical_lp_bidding.aggregate_energy import energy_feasible_initial_bid
from market.physical_lp_bidding.data_classes import baseline_step_matrix
from market.physical_lp_bidding.solve_colgen_benders import _solve_first_stage_master


def test_steps_count_only_adjacent_participating_blocks():
    matrix = baseline_step_matrix([True, True, False, True, True], 5)
    assert matrix.shape == (2, 15)
    baseline = np.array([1.0, 4.0, 100.0, 7.0, 3.0])
    x = np.concatenate([baseline, np.zeros(10)])
    assert matrix @ x == pytest.approx([3.0, -4.0])


def _three_block_master(step_weight: float):
    """Each block holds u + d <= 6 under Assessment I for any baseline in [3, 7].

    Bounds: baseline in [0, 10], up and down in [0, 3]. Assessment I:
    baseline - up >= 0 and baseline + down <= 10. One cut pins the first
    baseline at 3, so every capacity-optimal bid has 18 kW-block and only the
    other two baselines are free.
    """

    blocks = 3
    objective = np.concatenate([np.zeros(blocks), -np.ones(2 * blocks)])
    bounds = [(0.0, 10.0)] * blocks + [(0.0, 3.0)] * (2 * blocks)
    assessment = []
    rhs = []
    for b in range(blocks):
        row = np.zeros(3 * blocks)
        row[b], row[blocks + b] = -1.0, 1.0
        assessment.append(row)
        rhs.append(0.0)
        row = np.zeros(3 * blocks)
        row[b], row[2 * blocks + b] = 1.0, 1.0
        assessment.append(row)
        rhs.append(10.0)
    # A cut is stored as coefficient . x >= -constant, i.e. -coefficient . x <= constant.
    cut_coefficient = np.zeros(3 * blocks)
    cut_coefficient[0] = -1.0
    cuts = [{"coefficient": cut_coefficient, "constant": 3.0}]
    return _solve_first_stage_master(
        objective=objective,
        cuts=cuts,
        bounds=bounds,
        time_limit_s=None,
        static_matrix=np.vstack(assessment),
        static_rhs=np.asarray(rhs),
        step_matrix=baseline_step_matrix([True, True, True], blocks),
        step_weight=step_weight,
    )


def test_master_picks_the_flattest_baseline_among_equal_capacity_bids():
    result = _three_block_master(1e-3)
    assert result.success
    baseline, up, down = result.x[:3], result.x[3:6], result.x[6:]
    assert np.sum(up + down) == pytest.approx(18.0)
    assert baseline == pytest.approx([3.0, 3.0, 3.0], abs=1e-7)
    # fun is capacity plus the step term, here zero.
    assert result.fun == pytest.approx(-18.0, abs=1e-7)


def test_master_without_weight_keeps_the_same_capacity():
    result = _three_block_master(0.0)
    assert result.success
    assert np.sum(result.x[3:]) == pytest.approx(18.0)


def _seed_config(blocks: int, step_weight: float) -> BiddingLPConfig:
    return BiddingLPConfig(
        steps=6 * blocks,
        blocks=blocks,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        assessment_band_fraction=0.10,
        apply_transition_band=False,
        baseline_step_weight=step_weight,
    )


def _seed(step_weight: float):
    """One EV with room both ways all day and no command.

    Assessment I holds baseline - up >= -20 and baseline + down <= 20 with up
    and down at most 30, so every baseline in [-10, 10] gives 40 kW per block.
    """

    blocks = 4
    steps = 6 * blocks
    cfg = _seed_config(blocks, step_weight)
    ev = EVSpec(0, steps, 0.5, 0.5, 100.0, 20.0, 20.0)
    idle = ActivationScenario(
        name="idle", up_signal=np.zeros(steps), down_signal=np.zeros(steps), evs=[ev]
    )
    return energy_feasible_initial_bid(
        [idle],
        cfg,
        baseline_min=-20.0,
        baseline_max=20.0,
        up_cap=np.full(blocks, 30.0),
        down_cap=np.full(blocks, 30.0),
        minimum_bid_kw=1.0,
        sustained_power_min=np.full(blocks, -20.0),
        sustained_power_max=np.full(blocks, 20.0),
    )


def test_seed_lp_returns_a_flat_baseline_at_full_capacity():
    seed = _seed(1e-3)
    assert seed is not None
    assert np.sum(seed.up_kw + seed.down_kw) == pytest.approx(160.0, rel=1e-6)
    assert np.max(np.abs(np.diff(seed.baseline_kw))) == pytest.approx(0.0, abs=1e-6)


def test_seed_lp_capacity_does_not_depend_on_the_weight():
    with_weight = _seed(1e-3)
    without = _seed(0.0)
    assert np.sum(with_weight.up_kw + with_weight.down_kw) == pytest.approx(
        np.sum(without.up_kw + without.down_kw), rel=1e-9
    )

from types import SimpleNamespace

import numpy as np
import pytest

from market.bid_env import N_BLOCKS
from training.blockwise_bid import (
    _direction_retirement_options,
    _failed_market_block_indices,
    _reviewed_fixed_ev_assessment_i_bounds,
    minimum_bid_quantity_kw,
    split_scenario_rows,
)


def test_failed_blocks_include_frequency_and_scenario_rows():
    validation = {
        "failed_block_frequency": {"7": 3},
        "scenario_rows": [{"all_ok": False, "failed_block_indices": [4]}],
    }
    assert _failed_market_block_indices(validation) == [4, 7]


def test_explicit_minimum_bid_quantity_is_enforced(monkeypatch):
    monkeypatch.setattr(
        "training.blockwise_bid.LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW", "320"
    )
    monkeypatch.setattr(
        "training.blockwise_bid.LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_MIN_DIRECTION_BID_KW",
        1000.0,
    )
    assert minimum_bid_quantity_kw() == pytest.approx(1000.0)


def test_reviewed_assessment_i_uses_fixed_bank_and_30_minutes(monkeypatch):
    from training import blockwise_bid as bidder

    seen = {}

    def capability(bank, *, quantile, duration_hours):
        seen.update(bank=list(bank), quantile=quantile, duration_hours=duration_hours)
        return np.full(N_BLOCKS, 321.0), np.full(N_BLOCKS, 654.0)

    monkeypatch.setattr(bidder, "sustained_capability_quantile", capability)
    bank = [["fixed EV draw"]]
    up, down = _reviewed_fixed_ev_assessment_i_bounds(
        SimpleNamespace(_bid_build_log=lambda _message: None), bank
    )
    assert seen == {"bank": bank, "quantile": 0.0, "duration_hours": 0.5}
    np.testing.assert_allclose(up, 321.0)
    np.testing.assert_allclose(down, 654.0)


def test_ev_scenario_selector_picks_minimum_lower_median_and_maximum():
    from training.lower_bid_training import _select_ev_scenarios_by_count

    candidates = [
        [f"candidate_{index}" for _ in range(count)]
        for index, count in enumerate((5, 1, 4, 2, 3, 6))
    ]

    selected, metadata = _select_ev_scenarios_by_count(candidates)

    assert [len(scenario) for scenario in selected] == [1, 3, 6]
    assert [row["label"] for row in metadata] == ["minimum", "median", "maximum"]
    assert [row["candidate_index"] for row in metadata] == [1, 4, 5]
    assert [row["rank_zero_based"] for row in metadata] == [0, 2, 5]


def test_three_ev_scenarios_cross_every_activation_command():
    from market.physical_lp_bidding import (
        EVSpec,
        stratified_ev_activation_scenarios,
    )

    ev_bank = [
        [EVSpec(arrival_t=index, departure_t=index + 1, initial_soc=0.5, target_soc=0.5)]
        for index in range(3)
    ]
    commands = [
        {
            "name": f"command_{index}",
            "source": "test",
            "up_proxy": np.array([float(index)]),
            "down_proxy": np.array([0.0]),
        }
        for index in range(128)
    ]

    scenarios = stratified_ev_activation_scenarios(
        ev_specs_by_scenario=ev_bank,
        activation_payloads=commands,
        activations_per_ev=len(commands),
    )

    assert len(scenarios) == 3 * 128
    for ev_index in range(3):
        group = [
            scenario
            for scenario in scenarios
            if scenario.metadata["ev_scenario"] == ev_index
        ]
        assert len(group) == 128
        assert {
            scenario.metadata["activation_scenario"] for scenario in group
        } == set(range(128))


def test_assessment_i_couples_baseline_and_width():
    from market.physical_lp_bidding import BiddingLPConfig, JointBiddingProblem
    from market.physical_lp_bidding.data_classes import (
        assessment_i_rows,
        project_assessment_i,
    )

    problem = JointBiddingProblem(
        objective_weights=np.ones(N_BLOCKS),
        scenarios=[],
        sustained_power_min_kw=np.full(N_BLOCKS, -251.0),
        sustained_power_max_kw=np.full(N_BLOCKS, 253.0),
        config=BiddingLPConfig(),
    )
    baseline = np.full(N_BLOCKS, 100.0)
    up, down = project_assessment_i(
        problem, baseline, np.full(N_BLOCKS, 1000.0), np.full(N_BLOCKS, 1000.0)
    )
    # Up may take the fleet from +100 kW down to -251 kW; down from +100 kW to +253 kW.
    np.testing.assert_allclose(up, 351.0)
    np.testing.assert_allclose(down, 153.0)
    matrix, rhs = assessment_i_rows(problem)
    x = np.concatenate([baseline, up, down])
    assert matrix.shape == (2 * N_BLOCKS, 3 * N_BLOCKS)
    assert np.all(matrix @ x <= rhs + 1e-9)
    x_wide = np.concatenate([baseline, up + 1.0, down])
    assert np.any(matrix @ x_wide > rhs + 1e-9)


def test_minimum_bid_quantity_requires_a_measured_band(monkeypatch):
    monkeypatch.setattr(
        "training.blockwise_bid.LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW", ""
    )
    with pytest.raises(ValueError, match="minimum bid quantity"):
        minimum_bid_quantity_kw()


def test_missing_dispatch_is_separate_from_a_judged_failure():
    rows = [
        {
            "scenario": "missed",
            "all_ok": False,
            "dispatch_missing": False,
            "global_step_pass_rate": 0.25,
            "soc_ok": False,
            "failed_block_indices": [3],
        },
        {
            "scenario": "not solved",
            "all_ok": False,
            "dispatch_missing": True,
            "global_step_pass_rate": None,
            "soc_ok": None,
            "failed_block_indices": [],
        },
    ]
    missing, failing = split_scenario_rows(rows)
    assert missing == ["not solved"]
    assert [row["scenario"] for row in failing] == ["missed"]


def test_active_directions_remain_retirement_options_above_the_floor():
    current_up = np.zeros(N_BLOCKS)
    current_down = np.zeros(N_BLOCKS)
    current_up[0] = 500.0
    current_down[1] = 300.0

    below_floor, active = _direction_retirement_options(
        current_up,
        current_down,
        current_up.copy(),
        current_down.copy(),
        minimum_bid=250.0,
        tolerance=1e-6,
    )

    assert below_floor == []
    assert {(entry[2], entry[3]) for entry in active} == {(0, "up"), (1, "down")}
    assert min(active)[2:4] == (1, "down")


def test_below_floor_direction_keeps_priority_over_other_active_directions():
    current_up = np.zeros(N_BLOCKS)
    current_down = np.zeros(N_BLOCKS)
    current_up[0] = 500.0
    current_down[1] = 300.0
    relaxed_up = current_up.copy()
    relaxed_up[0] = 100.0

    below_floor, active = _direction_retirement_options(
        current_up,
        current_down,
        relaxed_up,
        current_down.copy(),
        minimum_bid=250.0,
        tolerance=1e-6,
    )

    assert [(entry[2], entry[3]) for entry in below_floor] == [(0, "up")]
    assert len(active) == 2

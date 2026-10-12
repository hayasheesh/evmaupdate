import numpy as np
import pytest
import torch


def test_bounded_balance_reward_has_no_tolerance_flat_top() -> None:
    from environment.EVEnv import EVEnv

    env = EVEnv.__new__(EVEnv)
    env.balance_reward = 1.0
    env._balance_reward_error_scale_kw = 150.0
    env._balance_reward_linear_tail_kw = 0.0
    env._balance_reward_shape = "tanh_linear"
    env.tol_narrow_metrics = 75.0

    deviations = (0.0, 1.0, 10.0, 75.0, 150.0, 1000.0)
    rewards = [env._calculate_balance_reward(value) for value in deviations]

    assert rewards[0] == pytest.approx(1.0)
    assert all(left > right for left, right in zip(rewards, rewards[1:]))
    assert all(-1.0 <= value <= 1.0 for value in rewards)


def test_convex_balance_reward_charges_each_further_kw_more() -> None:
    from environment.EVEnv import EVEnv

    env = EVEnv.__new__(EVEnv)
    env.balance_reward = 1.0
    env._balance_reward_error_scale_kw = 150.0
    env._balance_reward_shape = "convex"

    deviations = np.arange(0.0, 601.0, 25.0)
    rewards = np.array([env._calculate_balance_reward(value) for value in deviations])
    cost_per_step = -np.diff(rewards)

    assert rewards[0] == pytest.approx(1.0)
    # No flat band: the first kW already costs what it costs under the tanh form.
    assert env._calculate_balance_reward(1.0) == pytest.approx(1.0 - 2.0 / 150.0, rel=1e-2)
    assert np.all(np.diff(cost_per_step) > 0)
    # Lighter than the squared error far out: 600 kW costs under 16 times 150 kW.
    assert (1.0 - rewards[-1]) < 16.0 * (1.0 - env._calculate_balance_reward(150.0))


def test_bank_profile_uses_training_quantiles_and_round_trips(tmp_path) -> None:
    from environment.normalize import (
        configure_observation_normalization,
        derive_observation_normalization_profile,
        get_observation_normalization_profile,
        load_observation_normalization_profile,
        normalize_ag_request,
        save_observation_normalization_profile,
    )

    previous_profile = get_observation_normalization_profile()
    try:
        samples = {
            "demand_kw": np.array([-10.0, 0.0, 100.0, 1000.0]),
        }
        profile = derive_observation_normalization_profile(
            samples, quantile=0.75, source="unit-test-bank"
        )
        assert profile["demand_center_kw"] == 0.0
        assert profile["demand_scale_kw"] == pytest.approx(325.0)

        path = tmp_path / "input" / "observation_normalization.json"
        save_observation_normalization_profile(path, profile)
        configure_observation_normalization(previous_profile)
        loaded = load_observation_normalization_profile(path)
        assert loaded == profile
        assert normalize_ag_request(torch.tensor([325.0])).item() == pytest.approx(1.0)
    finally:
        configure_observation_normalization(previous_profile)


def test_a_bank_profile_keeps_one_command_scale_for_every_day() -> None:
    from environment.normalize import (
        bid_lookahead_scale_kw,
        configure_observation_normalization,
        derive_observation_normalization_profile,
        get_observation_normalization_profile,
        use_instruction_scale,
    )

    previous_profile = get_observation_normalization_profile()
    try:
        samples = {"demand_kw": np.array([-10.0, 0.0, 100.0, 1000.0])}
        derive_observation_normalization_profile(
            samples, quantile=0.75, command_scale="bank"
        )
        use_instruction_scale(900.0)
        assert get_observation_normalization_profile()["demand_scale_kw"] == pytest.approx(325.0)
        assert bid_lookahead_scale_kw(900.0) == pytest.approx(325.0)

        derive_observation_normalization_profile(
            samples, quantile=0.75, command_scale="day"
        )
        use_instruction_scale(900.0)
        assert get_observation_normalization_profile()["demand_scale_kw"] == pytest.approx(900.0)
        assert bid_lookahead_scale_kw(900.0) == pytest.approx(900.0)
    finally:
        configure_observation_normalization(previous_profile)


def test_a_saved_profile_without_a_command_scale_reads_as_day() -> None:
    from environment.normalize import (
        configure_observation_normalization,
        get_observation_normalization_profile,
    )

    previous_profile = get_observation_normalization_profile()
    try:
        old = dict(previous_profile)
        old.pop("command_scale", None)
        assert configure_observation_normalization(old)["command_scale"] == "day"
    finally:
        configure_observation_normalization(previous_profile)

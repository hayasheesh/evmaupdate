import numpy as np
import pytest
import torch


def test_bounded_balance_reward_has_no_tolerance_flat_top() -> None:
    from environment.EVEnv import EVEnv

    env = EVEnv.__new__(EVEnv)
    env.balance_reward = 1.0
    env._balance_reward_error_scale_kw = 150.0
    env._balance_reward_linear_tail_kw = 0.0
    env.tol_narrow_metrics = 75.0

    deviations = (0.0, 1.0, 10.0, 75.0, 150.0, 1000.0)
    rewards = [env._calculate_balance_reward(value) for value in deviations]

    assert rewards[0] == pytest.approx(1.0)
    assert all(left > right for left, right in zip(rewards, rewards[1:]))
    assert all(-1.0 <= value <= 1.0 for value in rewards)


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

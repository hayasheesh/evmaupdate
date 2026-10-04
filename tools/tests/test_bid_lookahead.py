import numpy as np
import pytest


def test_half_day_bid_lookahead_uses_known_blocks_and_zero_pads() -> None:
    from market.bid_env import N_BLOCKS, STEPS_PER_BLOCK
    from training.lower_bid_training import bid_lookahead_context_series

    baseline = np.arange(N_BLOCKS, dtype=np.float32) * 10.0
    up = np.arange(N_BLOCKS, dtype=np.float32) + 100.0
    down = np.arange(N_BLOCKS, dtype=np.float32) + 200.0
    context = bid_lookahead_context_series(
        baseline,
        up,
        down,
        instruction_scale_kw=1000.0,
        lookahead_blocks=24,
    )

    assert len(context) == 24 * 3
    assert context["bid_baseline_lookahead_0"][0] == pytest.approx(0.0)
    assert context["bid_up_lookahead_1"][0] == pytest.approx(0.101)
    assert context["bid_down_lookahead_0"][STEPS_PER_BLOCK] == pytest.approx(0.201)
    assert context["bid_baseline_lookahead_1"][-1] == pytest.approx(0.0)


def test_zero_bid_lookahead_adds_no_features() -> None:
    from training.lower_bid_training import bid_lookahead_context_series

    context = bid_lookahead_context_series(
        np.zeros(48),
        np.zeros(48),
        np.zeros(48),
        instruction_scale_kw=1.0,
        lookahead_blocks=0,
    )
    assert context == {}


def test_lower_command_draw_uses_the_lower_control_pool_and_separate_streams(
    monkeypatch,
) -> None:
    import training.lower_bid_training as lower
    from market.activation_scenarios import LOWER_CONTROL_POOL

    calls = []

    def fake_commands(service_date, seed, *, n_scenarios, scenario_partition, exclude_sources=None):
        calls.append((service_date, seed, n_scenarios, scenario_partition))
        return ([{
            "name": f"draw-{seed}",
            "up_proxy": np.zeros(288),
            "down_proxy": np.zeros(288),
            "source": "test:all-history",
            "source_date": "2024-01-01",
            "source_bmu": "UNIT",
        }], "test-mode")

    monkeypatch.setattr(lower, "_activation_scenarios_for_day", fake_commands)
    monkeypatch.setattr(lower, "lower_control_pool_label", lambda _directory: "all_historical")
    fixed_bid = {"service_date": "2024-12-02", "forecast_seed": 93180}
    train = lower.sample_random_historical_activation(
        fixed_bid, 17, stream="train"
    )
    validation = lower.sample_random_historical_activation(
        fixed_bid, 17, stream="validation"
    )

    assert all(call[2] == 1 and call[3] == LOWER_CONTROL_POOL for call in calls)
    assert calls[0][1] != calls[1][1]
    assert train["lower_command_sampling_pool"] == "all_historical"
    assert train["lower_command_stream"] == "train"
    assert validation["lower_command_stream"] == "validation"

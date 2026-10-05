"""A day's EV scenarios and natural baseline are reused only for the same inputs."""

from __future__ import annotations

import numpy as np
import pytest

from training import lower_bid_training as lbt

DAY = "2024-04-02"


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    # These tests draw three candidates, so they pin a rule that picks three.
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION", "session_count")
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIOS", 3)
    monkeypatch.setattr(lbt, "EV_SCENARIO_CACHE_DIR", tmp_path)
    monkeypatch.delenv("EVMA_BID_SOLVE_CACHE", raising=False)
    lbt._code_and_settings_signature.cache_clear()
    yield tmp_path
    lbt._code_and_settings_signature.cache_clear()


def _bank(**overrides):
    kwargs = dict(count=3, seed=4242, arrival_probs=None, service_date=DAY,
                  label="test", workers=1)
    kwargs.update(overrides)
    return lbt._selected_ev_scenario_bank(**kwargs)


def _key(**overrides):
    kwargs = dict(count=3, seed=4242, seed_offset=0, arrival_probs=None,
                  service_date=DAY)
    kwargs.update(overrides)
    return lbt._ev_scenario_bank_key(**kwargs)


def _no_rollout(**_kwargs):
    raise AssertionError("the stored EV scenarios should have been used")


def test_the_second_call_reads_the_stored_scenarios(cache_dir, monkeypatch):
    bank, selection, key = _bank()
    assert key is not None
    assert (cache_dir / f"bank_{key}.pt").is_file()

    monkeypatch.setattr(lbt, "sample_ev_specs_from_evenv", _no_rollout)
    again, again_selection, again_key = _bank()
    assert again_key == key
    assert again == bank
    assert again_selection == selection


def test_the_stored_scenarios_equal_a_fresh_draw(cache_dir, monkeypatch):
    bank, selection, _key_ = _bank()
    monkeypatch.setenv("EVMA_BID_SOLVE_CACHE", "0")
    fresh, fresh_selection, fresh_key = _bank()
    assert fresh_key is None
    assert fresh == bank
    assert fresh_selection == selection


def test_the_key_follows_every_rollout_input(cache_dir, monkeypatch):
    import EnvConfig

    base = _key()
    assert base is not None
    assert _key(seed=4243) != base
    assert _key(service_date="2024-04-03") != base
    assert _key(count=4) != base
    assert _key(arrival_probs=np.ones((7, 288))) != base
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION", "low_connection_2_max_count")
    assert _key() != base
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION", "session_count")

    monkeypatch.setattr(
        EnvConfig,
        "EV_TARGET_REACHABLE_POWER_FRACTION",
        EnvConfig.EV_TARGET_REACHABLE_POWER_FRACTION + 0.01,
    )
    lbt._code_and_settings_signature.cache_clear()
    assert _key() != base


def test_the_key_does_not_follow_the_commands_or_the_minimum_bid(cache_dir, monkeypatch):
    import EnvConfig

    base = _key()
    monkeypatch.setattr(EnvConfig, "LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW", "1")
    monkeypatch.setattr(EnvConfig, "LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR", "/elsewhere")
    monkeypatch.setattr(EnvConfig, "ACTIVATION_SIGNAL_SET", "elsewhere")
    lbt._code_and_settings_signature.cache_clear()
    assert _key() == base


def test_the_key_covers_the_code_and_files_the_rollout_reads():
    inputs = lbt._code_and_settings_inputs("market.physical_lp_bidding.evenv_adapter")
    assert {
        "market.physical_lp_bidding.evenv_adapter",
        "market.physical_lp_bidding.data_classes",
        "environment.EVEnv",
        "environment.station_sessions",
        "environment.calendars",
        "environment.ev_info_loader",
    } <= set(inputs["modules"])
    assert "tools.evaluator.set_env_seed" in inputs["objects"]
    assert {"DEVICE", "EPISODE_STEPS"} <= set(inputs["settings"]["Config"])
    assert {
        "MAX_EV_PER_STATION",
        "NUM_STATIONS",
        "PER_STATION_SESSION_IDS",
        "EV_BATTERY_CAPACITY_OPTIONS_KWH",
    } <= set(inputs["settings"]["EnvConfig"])
    assert inputs["setting_contents"]["EnvConfig.STATION_SESSION_DIR"] is not None
    assert "EnvConfig.EV_SOC_ARRIVAL_DISTRIBUTION_PATH" in inputs["setting_contents"]


def test_reuse_off_stores_nothing(cache_dir, monkeypatch):
    monkeypatch.setenv("EVMA_BID_SOLVE_CACHE", "0")
    _bank_, _selection, key = _bank()
    assert key is None
    assert not any(cache_dir.iterdir())


def test_a_damaged_entry_is_drawn_again(cache_dir):
    bank, selection, key = _bank()
    (cache_dir / f"bank_{key}.pt").write_bytes(b"not a cache entry")
    again, again_selection, _again_key = _bank()
    assert again == bank
    assert again_selection == selection


def test_the_natural_baseline_is_reused_for_the_same_bank_and_lp(cache_dir, monkeypatch):
    from EnvConfig import (
        LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW,
        LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW,
    )
    from market.physical_lp_bidding import BiddingLPConfig

    bank, _selection, key = _bank()
    kwargs = dict(
        ev_bank=bank,
        bank_key=key,
        baseline_min_kw=LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW,
        baseline_max_kw=LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW,
    )
    first = lbt._natural_baseline_for_bank(config=BiddingLPConfig(), **kwargs)

    def no_lp(*_args, **_kwargs):
        raise AssertionError("the LP was solved")

    monkeypatch.setattr(lbt, "solve_natural_baseline_lp", no_lp)
    second = lbt._natural_baseline_for_bank(config=BiddingLPConfig(), **kwargs)
    np.testing.assert_array_equal(first, second)
    # Parallelism is not an input of the LP.
    lbt._natural_baseline_for_bank(config=BiddingLPConfig(scenario_workers=4), **kwargs)

    with pytest.raises(AssertionError, match="the LP was solved"):
        lbt._natural_baseline_for_bank(config=BiddingLPConfig(eta_ch=0.95), **kwargs)
    with pytest.raises(AssertionError, match="the LP was solved"):
        lbt._natural_baseline_for_bank(
            config=BiddingLPConfig(),
            **{**kwargs, "baseline_max_kw": LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW - 1.0},
        )
    with pytest.raises(AssertionError, match="the LP was solved"):
        lbt._natural_baseline_for_bank(
            config=BiddingLPConfig(), **{**kwargs, "bank_key": None}
        )

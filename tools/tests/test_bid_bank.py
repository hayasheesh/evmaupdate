from __future__ import annotations

import json

import numpy as np

from training.bid_bank import (
    BID_BANK_VERSION,
    BidBank,
    forecast_seed_for_day,
    manifest_settings_match,
    save_fixed_bid,
)


def test_bid_bank_loads_persisted_day_by_date(tmp_path) -> None:
    day_dir = tmp_path / "days" / "000_2024-08-01"
    bid_path = day_dir / "fixed_bid.pkl"
    save_fixed_bid(
        bid_path,
        {
            "service_date": "2024-08-01",
            "baseline_plan": np.arange(48, dtype=float),
            "up_plan": np.full(48, 500.0),
            "down_plan": np.full(48, 1000.0),
        },
    )
    entry = {
        "index": 0,
        "service_date": "2024-08-01",
        "forecast_seed": 73000,
        "bid_path": str(bid_path.relative_to(tmp_path)),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "version": BID_BANK_VERSION,
            "complete": True,
            "entries": [entry],
        }),
        encoding="utf-8",
    )

    bank = BidBank(tmp_path)
    loaded = bank.load_date("2024-08-01")

    assert len(bank) == 1
    assert np.array_equal(loaded["baseline_plan"], np.arange(48, dtype=float))


def test_forecast_seed_is_reproducible_and_day_specific() -> None:
    assert forecast_seed_for_day(73000, 0) == 73000
    assert forecast_seed_for_day(73000, 1) == 74009
    assert forecast_seed_for_day(73000, 1) == forecast_seed_for_day(73000, 1)


def test_manifest_settings_match_rejects_missing_or_stale_arrival_model() -> None:
    required = {"arrival_model": {"source": "future", "growth": 2.0}}
    assert manifest_settings_match({"settings": required}, required)
    assert not manifest_settings_match({"settings": {}}, required)
    assert not manifest_settings_match(
        {"settings": {"arrival_model": {"source": "legacy", "growth": 1.0}}},
        required,
    )


def test_manifest_settings_match_can_ignore_only_named_library_fields() -> None:
    required = {
        "arrival_model": {"source": "future"},
        "activation_source_dir": "train-library",
    }
    actual = {
        "arrival_model": {"source": "future"},
        "activation_source_dir": "held-out-library",
    }
    assert manifest_settings_match(
        {"settings": actual},
        required,
        ignored_keys={"activation_source_dir"},
    )
    assert not manifest_settings_match(
        {"settings": {**actual, "arrival_model": {"source": "legacy"}}},
        required,
        ignored_keys={"activation_source_dir"},
    )


def _idle_bid() -> dict:
    from training.lower_bid_training import SubmittedBidResult

    up = np.zeros(48)
    down = np.zeros(48)
    up[14:40] = 300.0
    down[20:44] = 250.0
    return {
        "baseline_plan": np.full(48, -2194.9375),
        "up_plan": up,
        "down_plan": down,
        "submitted_up_plan": up.copy(),
        "submitted_down_plan": down.copy(),
        "result": SubmittedBidResult(
            solved=True, feasible=True, status="ok",
            objective_capacity_kw_block=1.0, mean_baseline_kw=-2194.9375,
        ),
        "_runtime_market_context_cache": {"key": (24, 1.0), "series": {}},
    }


def test_blocks_without_bid_width_get_a_zero_baseline() -> None:
    from training.bid_bank import zero_idle_baseline

    bid = _idle_bid()
    assert zero_idle_baseline(bid) == 48 - 30  # blocks 14..43 carry width
    baseline = bid["baseline_plan"]
    assert np.all(baseline[:14] == 0.0) and np.all(baseline[44:] == 0.0)
    assert np.all(baseline[14:44] == -2194.9375)
    assert "_runtime_market_context_cache" not in bid
    assert bid["result"].mean_baseline_kw == float(np.mean(baseline))
    assert zero_idle_baseline(bid) == 0


def test_a_block_with_only_a_submitted_width_keeps_its_baseline() -> None:
    from training.bid_bank import zero_idle_baseline

    bid = _idle_bid()
    bid["submitted_up_plan"][5] = 100.0
    zero_idle_baseline(bid)
    assert bid["baseline_plan"][5] == -2194.9375
    assert bid["baseline_plan"][4] == 0.0


def test_the_bank_builder_saves_the_zeroed_baseline(tmp_path) -> None:
    from training.bid_bank import build_training_bid_bank, load_fixed_bid

    seen = {}

    def build_episode(fixed_bid, idx):
        seen["baseline"] = np.asarray(fixed_bid["baseline_plan"]).copy()
        return np.zeros(288), np.ones(288), {}, {"service_date": "2024-08-01"}

    class Sampler:
        def scenario_for_day(self, date):
            return None

    manifest = build_training_bid_bank(
        [{"date": "2024-08-01", "series": np.zeros(288)}],
        tmp_path,
        base_forecast_seed=73000,
        build_fixed_bid=lambda series, date, arrival_scenario, forecast_seed: _idle_bid(),
        build_episode=build_episode,
        arrival_sampler=Sampler(),
        set_progress_log=lambda path, reset: None,
    )
    saved = load_fixed_bid(tmp_path / manifest["entries"][0]["bid_path"])
    assert np.all(saved["baseline_plan"][:14] == 0.0)
    assert np.all(seen["baseline"][:14] == 0.0)
    assert saved["idle_baseline_kw"] == 0.0

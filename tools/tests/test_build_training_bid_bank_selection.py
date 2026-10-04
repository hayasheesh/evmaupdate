from __future__ import annotations

from datetime import date, timedelta
from tools.build_training_bid_bank import (
    _with_fixed_ev_all_command_bid,
    select_payloads_for_bank,
)
from training.run_after_day_ahead_bid import stratified_bank_day_selection


def _all_data() -> dict[str, list[dict]]:
    first = date(2024, 8, 1)
    payloads = [
        {"date": (first + timedelta(days=index)).isoformat(), "series": None}
        for index in range(40)
    ]
    return {"train": payloads[:25], "test": payloads[25:]}


def test_batch_builder_uses_the_training_drivers_stratified_dates() -> None:
    all_data = _all_data()
    expected_train, expected_test, _ = stratified_bank_day_selection(
        all_data["train"] + all_data["test"],
        train_count=10,
        test_count=4,
    )
    actual_train, train_info = select_payloads_for_bank(
        all_data,
        split="train",
        selected_days=10,
        paired_train_days=10,
        paired_test_days=4,
    )
    actual_test, test_info = select_payloads_for_bank(
        all_data,
        split="test",
        selected_days=4,
        paired_train_days=10,
        paired_test_days=4,
    )

    assert [p["date"] for p in actual_train] == [p["date"] for p in expected_train]
    assert [p["date"] for p in actual_test] == [p["date"] for p in expected_test]
    assert train_info["mode"] == test_info["mode"] == "stratified_proportional"


def test_marl_bank_bid_is_minmedmax_ev_all_128_commands_k0(monkeypatch) -> None:
    from training import lower_bid_training as lbt
    from training import blockwise_bid

    seen = {}

    def fake_builder(*args, **kwargs):
        seen.update(kwargs)
        return {
            "activation_scenario_payload": [
                {
                    "source_date": f"design-day-{index}",
                    "source_bmu": "design-unit",
                }
                for index in range(128)
            ]
        }

    def fake_commands(_day, _seed, *, n_scenarios, scenario_partition):
        assert n_scenarios == 128
        assert scenario_partition == "feedback"
        return [
            {"source_date": f"feedback-{index}", "source_bmu": "unit"}
            for index in range(n_scenarios)
        ], "test-mode"

    monkeypatch.setattr(blockwise_bid, "build_blockwise_bid_for_day", fake_builder)
    monkeypatch.setattr(lbt, "_activation_scenarios_for_day", fake_commands)
    wrapped = _with_fixed_ev_all_command_bid(scenario_workers=7)
    bid = wrapped(
        [0.0],
        "2024-01-01",
        arrival_scenario=object(),
        forecast_seed=123,
    )

    assert seen["scenario_workers"] == 7
    assert set(seen) == {
        "arrival_scenario",
        "forecast_seed",
        "assessment_band_fraction",
        "scenario_workers",
    }
    assert bid["design_activation_scenarios"] == 128
    assert bid["activation_scenarios"] == 128
    assert bid["training_command_partition"] == "feedback"

"""Picking the bid's EV realizations by connected power instead of session count."""

import numpy as np
import pytest

from market.physical_lp_bidding.data_classes import EVSpec
import training.lower_bid_training as lbt


def _fleet(n_day: int, n_night: int, n_evening: int, extra: int = 0) -> list[EVSpec]:
    """EVs connected 08:00-16:00, 00:00-07:00 and 18:00-23:00, plus extra short day sessions."""

    def evs(n, arrival, departure):
        return [EVSpec(arrival_t=arrival, departure_t=departure, initial_soc=0.5, target_soc=0.5,
                       capacity_kwh=50.0, max_charge_kw=10.0, max_discharge_kw=10.0)
                for _ in range(n)]

    return evs(n_day, 96, 192) + evs(n_night, 0, 84) + evs(n_evening, 216, 276) + evs(extra, 120, 126)


def test_connected_power_counts_each_ev_over_its_session():
    power = lbt._connected_charge_kw_by_block(_fleet(2, 1, 0))
    assert power.shape == (48,)
    assert power[0] == 10.0          # one night EV
    assert power[20] == 20.0         # two day EVs at 10:00
    assert power[40] == 0.0


def test_low_connection_picks_cover_different_blocks_then_most_sessions():
    typical = [_fleet(10, 10, 10) for _ in range(6)]
    few_at_night = _fleet(10, 4, 10)
    few_in_evening = _fleet(10, 10, 4)
    most_sessions = _fleet(10, 10, 10, extra=30)
    candidates = typical + [few_at_night, few_in_evening, most_sessions]
    selected, metadata = lbt._select_ev_scenarios_low_connection(candidates)
    assert [m["candidate_index"] for m in metadata] == [6, 7, 8]
    assert [m["label"] for m in metadata] == ["low_connection_1", "low_connection_2", "maximum"]
    assert selected[2] is most_sessions
    assert all(m["candidate_count"] == 9 and m["ev_count"] == len(evs) for m, evs in zip(metadata, selected))


def test_low_connection_maximum_skips_a_low_pick():
    # The session-count maximum is also the lowest at night; the third pick is the next largest.
    candidates = [_fleet(10, 10, 10) for _ in range(4)] + [_fleet(10, 2, 10, extra=40), _fleet(10, 10, 5)]
    _, metadata = lbt._select_ev_scenarios_low_connection(candidates)
    picks = [m["candidate_index"] for m in metadata]
    assert picks[:2] == [4, 5]
    assert picks[2] == 3              # ties among the typical fleets go to the highest index, as by count


def test_blocks_with_no_connected_ev_add_nothing():
    candidates = [_fleet(10, 0, 0) for _ in range(3)] + [_fleet(5, 0, 0)]
    _, metadata = lbt._select_ev_scenarios_low_connection(candidates)
    assert metadata[0]["candidate_index"] == 3
    assert metadata[0]["min_connected_ratio"] == 0.5
    assert all(np.isfinite(m["min_connected_ratio"]) for m in metadata)


def test_three_low_picks_then_maximum_returns_four_realizations():
    typical = [_fleet(10, 10, 10) for _ in range(6)]
    few_at_night = _fleet(10, 4, 10)
    few_in_evening = _fleet(10, 10, 4)
    few_by_day = _fleet(4, 10, 10)
    most_sessions = _fleet(10, 10, 10, extra=30)
    candidates = typical + [few_at_night, few_in_evening, few_by_day, most_sessions]
    selected, metadata = lbt._select_ev_scenarios_low_connection(
        candidates, low_picks=3, count_picks=("maximum",)
    )
    assert sorted(m["candidate_index"] for m in metadata[:3]) == [6, 7, 8]
    assert metadata[3]["candidate_index"] == 9
    assert [m["label"] for m in metadata] == [
        "low_connection_1", "low_connection_2", "low_connection_3", "maximum",
    ]
    assert selected[3] is most_sessions


def test_one_low_pick_then_lower_median_and_maximum_of_the_rest():
    # Session counts 30, 31, ..., 36; the night-short fleet has the fewest night EVs.
    candidates = [_fleet(10, 10, 10, extra=k) for k in range(6)] + [_fleet(10, 3, 10, extra=20)]
    _, metadata = lbt._select_ev_scenarios_low_connection(
        candidates, low_picks=1, count_picks=("median", "maximum")
    )
    assert [m["label"] for m in metadata] == ["low_connection_1", "median", "maximum"]
    assert metadata[0]["candidate_index"] == 6
    # Of the remaining six (30..35 sessions) the lower median is index 2 and the maximum index 5.
    assert [m["candidate_index"] for m in metadata[1:]] == [2, 5]


def test_too_few_candidates_are_rejected():
    with pytest.raises(ValueError):
        lbt._select_ev_scenarios_low_connection(
            [_fleet(10, 10, 10) for _ in range(3)], low_picks=3, count_picks=("maximum",)
        )


@pytest.mark.parametrize(
    "selection, contract, low_picks, count_picks",
    [
        ("low_connection_2_max_count", "fixed_low2_max_ev_all_commands_k0", 2, ("maximum",)),
        ("low_connection_1_median_max_count", "fixed_low1_median_max_ev_all_commands_k0", 1, ("median", "maximum")),
        ("low_connection_3_max_count", "fixed_low3_max_ev_all_commands_k0", 3, ("maximum",)),
    ],
)
def test_selection_setting_picks_the_rule_and_names_the_bank(monkeypatch, selection, contract, low_picks, count_picks):
    candidates = [_fleet(10, n, 10) for n in (3, 8, 5, 9, 1, 6, 7)]
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION", selection)
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIOS", low_picks + len(count_picks))
    expected = lbt._select_ev_scenarios_low_connection(
        candidates, low_picks=low_picks, count_picks=count_picks
    )[1]
    assert lbt._select_ev_scenarios(candidates)[1] == expected
    settings = lbt.upper_bid_bank_settings()
    assert settings["bank_bid_contract"] == contract
    assert settings["fixed_ev_scenarios"] == low_picks + len(count_picks)


def test_session_count_setting_keeps_the_count_rule(monkeypatch):
    candidates = [_fleet(10, n, 10) for n in (3, 8, 5, 9, 1)]
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION", "session_count")
    monkeypatch.setattr(lbt, "LOWER_TRAIN_UPPER_BID_EV_SCENARIOS", 3)
    assert lbt._select_ev_scenarios(candidates)[1] == lbt._select_ev_scenarios_by_count(candidates)[1]
    settings = lbt.upper_bid_bank_settings()
    assert settings["ev_scenario_selection"] == "minimum_lower_median_maximum_by_session_count"
    assert settings["bank_bid_contract"] == "fixed_min_median_max_ev_all_commands_k0"
    assert settings["fixed_ev_scenarios"] == 3


def test_default_selection_is_three_low_picks_then_maximum():
    import EnvConfig

    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION == "low_connection_3_max_count"
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIOS == 4
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT.startswith("low3max_4of")

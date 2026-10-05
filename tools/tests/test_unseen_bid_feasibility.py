"""指令×EV seedを別試行とし、未判定を確定率に含めない。"""

from tools.evaluate_unseen_bid_feasibility import counts, market_summary


def test_each_ev_seed_is_a_separate_trial_and_unknown_stays_unknown():
    day = {
        "ev_seeds": [11, 12, 13],
        "rows": [
            {
                "trials": [{"feasible": True}] * 3,
            },
            {
                "trials": [
                    {"feasible": False},
                    {"feasible": None},
                    {"feasible": True},
                ],
            },
            {
                "trials": [
                    {"feasible": True},
                    {"feasible": None},
                    {"feasible": True},
                ],
            },
        ]
    }
    result = counts(day)
    assert result == {
        "commands": 3,
        "trials": 9,
        "feasible": 6,
        "infeasible": 1,
        "unknown": 2,
    }
    summary = market_summary([day])
    assert summary["ev_seeds_per_command"] == 3
    assert summary["success_rate"] is None
    assert summary["success_rate_lower_bound"] == 6 / 9
    assert summary["success_rate_upper_bound"] == 8 / 9


def test_bank_arguments_name_the_market():
    import pytest

    from tools.evaluate_unseen_bid_feasibility import parse_banks

    banks = parse_banks(["PJM=some/bank", "GB=other/bank"])
    assert list(banks) == ["PJM", "GB"]
    assert banks["PJM"].name == "bank"
    with pytest.raises(ValueError):
        parse_banks(["no_market_given"])


def test_command_sources_name_a_library_partition():
    import pytest

    from tools.evaluate_unseen_bid_feasibility import parse_command_sources

    sources = parse_command_sources(["PJM=data/pjm/command_libraries/regd_calendar_day:validation"])
    assert sources["PJM"][1] == "validation"
    assert sources["PJM"][0].endswith("regd_calendar_day")
    with pytest.raises(ValueError):
        parse_command_sources(["PJM=data/lib:holdout"])


def test_feedback_draw_is_a_prefix_of_a_larger_draw():
    """A bid built with N design commands stores the first N feedback commands of
    one fixed permutation, so a fixed --unseen-commands count is the same set for
    every bid of the day whatever its design count."""

    import os

    import pytest

    from EnvConfig import _ACTIVATION_SIGNAL_DIRS
    from market.activation_scenarios import build_activation_scenario_set

    library = _ACTIVATION_SIGNAL_DIRS.get("aemo_plan_deviation")
    if not library or not os.path.isdir(library):
        pytest.skip("AEMO command library is not on this machine")

    def ids(n):
        return [
            (s.source_date, s.source_bmu)
            for s in build_activation_scenario_set(
                service_date="2024-04-02", n_scenarios=n, seed=73_000 + 830_027,
                proxy_shape_dir=library, scenario_partition="feedback", require_unique=True,
            )
        ]

    small, large = ids(16), ids(48)
    assert large[:16] == small
    assert len(set(large)) == 48

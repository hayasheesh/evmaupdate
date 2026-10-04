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

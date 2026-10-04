from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from market.physical_lp_bidding import (
    ActivationScenario,
    BiddingLPConfig,
    BiddingSolution,
    EVSpec,
    JointBiddingProblem,
    solve_joint_hard_bidding_benders,
    validate_joint_solution,
)
from market.physical_lp_bidding import solve_colgen_benders as benders_module
from market.physical_lp_bidding.colgen_feasibility import (
    certify as certify_colgen_feasibility,
)
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
from training.lower_bid_training import _solve_fixed_bid_scenarios_decomposed


def _one_block_problem(
    scenarios: list[ActivationScenario],
) -> JointBiddingProblem:
    config = BiddingLPConfig(
        steps=6,
        blocks=1,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        assessment_band_fraction=0.10,
        time_limit_s=30.0,
    )
    return JointBiddingProblem(
        objective_weights=np.ones(1),
        scenarios=scenarios,
        baseline_min_kw=0.0,
        baseline_max_kw=0.0,
        u_cap=np.zeros(1),
        d_cap=np.full(1, 20.0),
        global_pass_rate=1.0,
        config=config,
    )


def _limited_down_scenario(name: str) -> ActivationScenario:
    return ActivationScenario(
        name=name,
        up_signal=np.zeros(6),
        down_signal=np.ones(6),
        evs=[
            EVSpec(
                0,
                6,
                0.50,
                0.50,
                100.0,
                10.0,
                0.0,
                ev_id="ev",
            )
        ],
    )


def test_floor_relaxed_master_exposes_a_subminimum_direction() -> None:
    scenario = _limited_down_scenario("floor_relaxation")
    problem = _one_block_problem([scenario])
    initial = BiddingSolution(
        status="optimal",
        solver="test-seed",
        objective_value=20.0,
        baseline_kw=np.zeros(1),
        up_kw=np.zeros(1),
        down_kw=np.full(1, 20.0),
    )

    strict, strict_summary = solve_joint_hard_bidding_benders(
        problem,
        [scenario],
        initial_solution=initial,
        minimum_active_width_kw=15.0,
        max_rounds=6,
        allow_widen=True,
        enforce_minimum_active_width=True,
    )
    relaxed, relaxed_summary = solve_joint_hard_bidding_benders(
        problem,
        [scenario],
        initial_solution=initial,
        minimum_active_width_kw=15.0,
        max_rounds=6,
        allow_widen=True,
        enforce_minimum_active_width=False,
    )

    assert not strict.success
    assert strict_summary["stop_reason"] == "master_infeasible"
    strict_floor_candidate = np.asarray(
        strict_summary["floor_relaxed_candidate"], dtype=float
    )
    assert strict_floor_candidate.shape == (3,)
    assert strict_floor_candidate[2] == pytest.approx(
        100.0 / 9.0, abs=1e-5
    )
    assert relaxed.success
    assert relaxed_summary["complete"] is True
    assert relaxed_summary["enforce_minimum_active_width"] is False
    assert relaxed.down_kw[0] == pytest.approx(100.0 / 9.0, abs=1e-5)


def _without_the_circulation_cut(monkeypatch) -> None:
    """Take the LP dual for the cut, which is what this file used to get.

    The circulation's Hoffman inequality is the default now; both sources have
    to keep working, so one test pins each.
    """

    import EnvConfig

    monkeypatch.setattr(
        EnvConfig, "LOWER_TRAIN_UPPER_BID_CIRCULATION_CUT", False, raising=False
    )


def test_benders_moves_the_common_baseline(monkeypatch) -> None:
    _without_the_circulation_cut(monkeypatch)
    config = BiddingLPConfig(
        steps=6,
        blocks=1,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        assessment_band_fraction=0.10,
        time_limit_s=30.0,
    )
    up_signal = np.r_[np.ones(3), np.zeros(3)]
    down_signal = np.r_[np.zeros(3), np.ones(3)]
    easy = ActivationScenario(
        name="benders_easy_v2g",
        up_signal=up_signal,
        down_signal=down_signal,
        evs=[
            EVSpec(
                0,
                6,
                0.50,
                0.50,
                100.0,
                20.0,
                10.0,
                ev_id="v2g",
            )
        ],
    )
    charge_only = ActivationScenario(
        name="benders_charge_only",
        up_signal=up_signal,
        down_signal=down_signal,
        evs=[
            EVSpec(
                0,
                6,
                0.50,
                0.50,
                100.0,
                20.0,
                0.0,
                ev_id="charge",
            )
        ],
    )
    problem = JointBiddingProblem(
        objective_weights=np.ones(1),
        scenarios=[easy],
        baseline_min_kw=0.0,
        baseline_max_kw=10.0,
        u_cap=np.full(1, 10.0),
        d_cap=np.full(1, 10.0),
        global_pass_rate=1.0,
        config=config,
    )
    initial = BiddingSolution(
        status="optimal",
        solver="test-seed",
        objective_value=20.0,
        baseline_kw=np.zeros(1),
        up_kw=np.full(1, 10.0),
        down_kw=np.full(1, 10.0),
    )

    solution, summary = solve_joint_hard_bidding_benders(
        problem,
        [easy, charge_only],
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=8,
    )

    assert summary["complete"] is True
    assert summary["cuts"] >= 1
    assert summary["cut_methods"].get("colgen_phase1", 0) >= 1
    assert solution.baseline_kw[0] > 8.9
    assert solution.up_kw[0] > 9.9
    assert solution.down_kw[0] > 9.9


def test_stratified_cut_selection_spans_both_scenario_axes() -> None:
    class Scenario:
        def __init__(self, ev_index: int, activation_index: int):
            self.name = f"cross_ev{ev_index:02d}_a{activation_index:02d}"
            self.metadata = {
                "ev_scenario": ev_index,
                "activation_scenario": activation_index,
            }

    failures = [
        (Scenario(ev_index, activation_index), object())
        for ev_index in range(3)
        for activation_index in range(8)
    ]

    selected = benders_module._stratified_failure_selection(failures, 6)
    assert len(selected) == 6
    ev_indices = [
        scenario.metadata["ev_scenario"] for scenario, _ in selected
    ]
    activation_indices = [
        scenario.metadata["activation_scenario"]
        for scenario, _ in selected
    ]
    assert set(ev_indices) == {0, 1, 2}
    assert max(ev_indices.count(index) for index in set(ev_indices)) == 2
    assert len(set(activation_indices)) >= 4
    assert len(
        benders_module._stratified_failure_selection(failures, 0)
    ) == len(failures)
    assert len(
        benders_module._stratified_failure_selection(failures, 999)
    ) == len(failures)


def test_stratified_cut_selection_tolerates_missing_metadata() -> None:
    class Scenario:
        def __init__(self, index: int):
            self.name = f"plain_{index}"

    failures = [(Scenario(index), object()) for index in range(10)]
    selected = benders_module._stratified_failure_selection(failures, 4)
    assert len(selected) == 4
    assert len({scenario.name for scenario, _ in selected}) == 4




def test_parallel_scenario_oracles_match_serial_result() -> None:
    scenarios = [
        _limited_down_scenario(f"parallel_{index}") for index in range(3)
    ]
    serial_problem = _one_block_problem(scenarios)
    parallel_problem = JointBiddingProblem(
        **{
            **serial_problem.__dict__,
            "config": BiddingLPConfig(
                **{
                    **serial_problem.config.__dict__,
                    "scenario_workers": 2,
                }
            ),
        }
    )
    initial = BiddingSolution(
        status="optimal",
        solver="test-seed",
        objective_value=20.0,
        baseline_kw=np.zeros(1),
        up_kw=np.zeros(1),
        down_kw=np.full(1, 20.0),
    )
    serial, serial_summary = solve_joint_hard_bidding_benders(
        serial_problem,
        scenarios,
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=6,
    )
    parallel, parallel_summary = solve_joint_hard_bidding_benders(
        parallel_problem,
        scenarios,
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=6,
    )
    assert parallel_summary["complete"] == serial_summary["complete"]
    assert parallel_summary["cuts"] == serial_summary["cuts"]
    np.testing.assert_allclose(parallel.baseline_kw, serial.baseline_kw)
    np.testing.assert_allclose(parallel.up_kw, serial.up_kw)
    np.testing.assert_allclose(parallel.down_kw, serial.down_kw)


def test_final_benders_recourse_can_be_reused_for_frozen_validation() -> None:
    scenario = _limited_down_scenario("reuse_final_dispatch")
    problem = _one_block_problem([scenario])
    initial = BiddingSolution(
        status="optimal",
        solver="test-seed",
        objective_value=20.0,
        baseline_kw=np.zeros(1),
        up_kw=np.zeros(1),
        down_kw=np.full(1, 20.0),
    )
    final_recourse: list[dict] = []

    solution, summary = solve_joint_hard_bidding_benders(
        problem,
        [scenario],
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=6,
        final_recourse_out=final_recourse,
    )

    assert summary["complete"] is True
    assert len(final_recourse) == 1
    fixed_problem = replace(
        problem,
        fixed_baseline=solution.baseline_kw.copy(),
        fixed_up=solution.up_kw.copy(),
        fixed_down=solution.down_kw.copy(),
    )
    reused = _solve_fixed_bid_scenarios_decomposed(
        fixed_problem,
        precomputed_task_results=final_recourse,
    )
    independently_solved = _solve_fixed_bid_scenarios_decomposed(fixed_problem)

    assert reused.success and independently_solved.success
    assert reused.metadata["precomputed_recourse_reused"] is True
    assert independently_solved.metadata["precomputed_recourse_reused"] is False
    assert validate_joint_solution(fixed_problem, reused)["all_scenarios_ok"]
    assert validate_joint_solution(
        fixed_problem, independently_solved
    )["all_scenarios_ok"]
    np.testing.assert_allclose(
        reused.scenario_power_kw[scenario.name],
        independently_solved.scenario_power_kw[scenario.name],
        atol=1e-9,
    )


def test_colgen_reuses_locally_feasible_vehicle_columns() -> None:
    scenario = _limited_down_scenario("reuse_columns")
    problem = _one_block_problem([scenario])
    _target, _tolerance, lower, upper = fixed_bid_tracking_bands(
        problem.config,
        np.zeros(1),
        np.zeros(1),
        np.full(1, 10.0),
        scenario.up_signal,
        scenario.down_signal,
        apply_transition_band=False,
    )
    feasible, rounds, info = certify_colgen_feasibility(
        scenario.evs,
        lower,
        upper,
        steps=problem.config.steps,
        dt=problem.config.dt_hours,
        return_column_pool=True,
    )
    column_pool = info["column_pool"]

    reused_feasible, reused_rounds, reused_info = certify_colgen_feasibility(
        scenario.evs,
        lower,
        upper,
        steps=problem.config.steps,
        dt=problem.config.dt_hours,
        initial_columns=column_pool,
        return_dispatch=True,
    )

    assert feasible is True and reused_feasible is True
    assert reused_rounds <= rounds
    assert "scenario_power_kw" in reused_info


def test_worker_column_cache_compaction_keeps_quiet_and_newest_paths() -> None:
    columns = [[np.full(4, float(index)) for index in range(20)]]
    compacted, stats = benders_module._compact_worker_column_pool(columns, 5)

    assert len(compacted[0]) == 5
    np.testing.assert_array_equal(compacted[0][0], columns[0][0])
    np.testing.assert_array_equal(compacted[0][-1], columns[0][-1])
    assert stats == {
        "before": 20,
        "after": 5,
        "dropped": 15,
        "limit_per_ev": 5,
    }


def test_small_fleet_can_use_direct_sparse_primary_oracle() -> None:
    scenario = _limited_down_scenario("hybrid_direct")
    problem = _one_block_problem([scenario])
    cfg = BiddingLPConfig(
        **{
            **problem.config.__dict__,
            "direct_oracle_max_evs": 10,
        }
    )
    candidate = np.array([0.0, 0.0, 10.0])
    result = benders_module._scenario_oracle_task((
        0,
        scenario,
        cfg,
        candidate,
        np.array([False]),
        np.array([True]),
        60,
        3,
        False,
    ))

    assert result["feasible"] is True
    # The point of the threshold is that a small fleet does not pay for column
    # generation.  Which of the two exact oracles below it answered is a
    # separate choice, and the label now reports the one that did.
    assert result["oracle_solver"] in {
        "direct_sparse_phase1",
        "prefix_circulation",
    }
    assert result["fallback_solver"] == ""
    assert result["certificate"]["primary_oracle_solver"] == (
        result["oracle_solver"]
    )


def test_the_solver_label_names_the_oracle_that_answered(monkeypatch) -> None:
    import EnvConfig

    scenario = _limited_down_scenario("hybrid_direct")
    problem = _one_block_problem([scenario])
    cfg = BiddingLPConfig(
        **{**problem.config.__dict__, "direct_oracle_max_evs": 10}
    )
    payload = (
        0, scenario, cfg, np.array([0.0, 0.0, 10.0]),
        np.array([False]), np.array([True]), 60, 3, False,
    )
    monkeypatch.setattr(
        EnvConfig, "LOWER_TRAIN_UPPER_BID_CIRCULATION_SCREEN", False
    )
    monkeypatch.setattr(
        EnvConfig, "LOWER_TRAIN_UPPER_BID_CIRCULATION_CUT", False
    )
    without = benders_module._scenario_oracle_task(payload)
    assert without["oracle_solver"] == "direct_sparse_phase1"

    monkeypatch.setattr(
        EnvConfig, "LOWER_TRAIN_UPPER_BID_CIRCULATION_SCREEN", True
    )
    with_screen = benders_module._scenario_oracle_task(payload)
    assert with_screen["oracle_solver"] == "prefix_circulation"
    assert with_screen["feasible"] is without["feasible"]


def test_direct_primary_timeout_is_incomplete_not_physical_failure(
    monkeypatch,
) -> None:
    scenario = _limited_down_scenario("direct_timeout")
    original = _one_block_problem([scenario])
    problem = replace(
        original,
        config=BiddingLPConfig(
            **{
                **original.config.__dict__,
                "direct_oracle_max_evs": 10,
            }
        ),
    )

    def timed_out(payload):
        return {
            "scenario_index": int(payload[0]),
            "feasible": None,
            "colgen_rounds": 0,
            "certificate": {
                "complete": False,
                "timed_out": True,
                "reason": "direct phase-I oracle time limit reached",
            },
            "cut": None,
            "runtime_s": 30.0,
            "oracle_solver": "direct_sparse_phase1",
            "fallback_solver": "direct_sparse_phase1",
        }

    monkeypatch.setattr(
        benders_module, "_scenario_direct_oracle_task", timed_out
    )
    solution, summary = solve_joint_hard_bidding_benders(
        problem,
        [scenario],
        initial_solution=_seed([10.0]),
        minimum_active_width_kw=0.01,
        max_rounds=2,
    )

    assert not solution.success
    assert summary["stop_reason"] == "colgen_oracle_incomplete"
    assert summary["cuts"] == 0
    assert summary["scenario_oracle_timeouts"] == 1
    assert summary["rounds"][0]["infeasible_scenarios"] == 0
    assert summary["rounds"][0]["incomplete_scenarios"] == 1


def test_benders_retries_only_round_limited_oracles(monkeypatch) -> None:
    scenario = _limited_down_scenario("adaptive_retry")
    problem = _one_block_problem([scenario])
    initial = _seed([10.0])
    attempted_rounds: list[int] = []

    def fake_oracle(payload):
        scenario_index = int(payload[0])
        max_rounds = int(payload[6])
        attempted_rounds.append(max_rounds)
        if max_rounds == 60:
            return {
                "scenario_index": scenario_index,
                "feasible": None,
                "colgen_rounds": 60,
                "certificate": {
                    "reason": "round limit reached with improving columns remaining",
                    "rounds": 60,
                },
                "cut": None,
                "runtime_s": 1.0,
            }
        zeros = np.zeros(problem.config.steps)
        return {
            "scenario_index": scenario_index,
            "feasible": True,
            "colgen_rounds": 2,
            "certificate": {
                "complete": True,
                "columns": 3,
                "rounds": 2,
                "scenario_power_kw": zeros,
                "ev_power_kw": [zeros],
                "ev_energy_kwh": [np.zeros(problem.config.steps + 1)],
            },
            "cut": None,
            "runtime_s": 2.0,
        }

    monkeypatch.setattr(benders_module, "_scenario_oracle_task", fake_oracle)
    final_recourse: list[dict] = []
    solution, summary = solve_joint_hard_bidding_benders(
        problem,
        [scenario],
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=1,
        final_recourse_out=final_recourse,
    )

    assert solution.success and summary["complete"] is True
    assert attempted_rounds == [60, 180]
    assert len(final_recourse) == 1
    assert final_recourse[0]["rounds"] == 62


def test_benders_uses_exact_direct_fallback_for_non_round_limit_unknown(
    monkeypatch,
) -> None:
    import EnvConfig

    # This is about what happens when column generation returns "unknown".
    # The circulation would answer first and the fallback would never be
    # reached, so it is switched off to keep the test on its own subject.
    monkeypatch.setattr(
        EnvConfig, "LOWER_TRAIN_UPPER_BID_CIRCULATION_SCREEN", False
    )
    monkeypatch.setattr(
        EnvConfig, "LOWER_TRAIN_UPPER_BID_CIRCULATION_CUT", False
    )
    scenario = _limited_down_scenario("direct_unknown_fallback")
    problem = _one_block_problem([scenario])
    initial = _seed([10.0])
    real_oracle = benders_module._scenario_oracle_task

    def unfinished_oracle(payload):
        if bool(payload[8]):
            return real_oracle(payload)
        return {
            "scenario_index": int(payload[0]),
            "feasible": None,
            "colgen_rounds": 3,
            "certificate": {
                "complete": False,
                "reason": "pricing failed for vehicle 0",
                "rounds": 3,
            },
            "cut": None,
            "runtime_s": 1.0,
        }

    monkeypatch.setattr(
        benders_module, "_scenario_oracle_task", unfinished_oracle
    )
    final_recourse: list[dict] = []
    solution, summary = solve_joint_hard_bidding_benders(
        problem,
        [scenario],
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=1,
        final_recourse_out=final_recourse,
    )

    assert solution.success
    assert summary["complete"] is True
    assert summary["rounds"][0]["feasible_scenarios"] == 1
    assert summary["rounds"][0]["infeasible_scenarios"] == 0
    assert summary["rounds"][0]["incomplete_scenarios"] == 0
    scenario_row = summary["rounds"][0]["scenario_rows"][0]
    assert scenario_row["fallback_solver"] == "direct_sparse_phase1"
    assert scenario_row["column_generation_reason"] == (
        "pricing failed for vehicle 0"
    )
    assert len(final_recourse) == 1


def _seed(down_kw) -> BiddingSolution:
    down = np.asarray(down_kw, dtype=float)
    return BiddingSolution(
        status="optimal",
        solver="test-seed",
        objective_value=float(down.sum()),
        baseline_kw=np.zeros(down.size),
        up_kw=np.zeros(down.size),
        down_kw=down.copy(),
    )


def test_a_free_block_baseline_slides_the_band_without_widening_it():
    """The diagnostic must move each block's window, not stretch it.

    A block whose window sits where the fleet cannot go becomes reachable once
    the window may slide.  A block whose steps demand levels further apart
    than the fleet's own power range stays infeasible, because sliding moves
    every step of that block by the same amount.
    """

    import numpy as np

    from market.physical_lp_bidding.colgen_feasibility import (
        certify_direct_phase_one,
    )
    from market.physical_lp_bidding.data_classes import EVSpec

    steps, blocks = 4, 2
    fleet = [
        EVSpec(
            arrival_t=0,
            departure_t=steps,
            initial_soc=0.50,
            target_soc=0.0,
            capacity_kwh=500.0,
            max_charge_kw=100.0,
            max_discharge_kw=0.0,
            station_id=0,
            ev_id=0,
            target_required=False,
        )
    ]

    def check(lo, hi, free):
        return certify_direct_phase_one(
            fleet,
            np.asarray(lo, dtype=float),
            np.asarray(hi, dtype=float),
            steps=steps,
            dt=0.5,
            free_baseline_blocks=blocks if free else None,
        )[0]

    # Both steps of block 0 sit 300 kW away, past the 100 kW the fleet can
    # draw. Sliding block 0 down brings the whole window into reach.
    off_level = ([300.0, 301.0, 10.0, 10.0], [302.0, 303.0, 12.0, 12.0])
    assert check(*off_level, False) is False
    assert check(*off_level, True) is True

    # Block 0 now asks for 10 kW at one step and 200 kW at the next. The gap
    # is wider than the fleet's whole range, so no single slide covers both.
    off_shape = ([10.0, 200.0, 10.0, 10.0], [12.0, 202.0, 12.0, 12.0])
    assert check(*off_shape, False) is False
    assert check(*off_shape, True) is False

    # A shape the fleet can follow stays feasible either way.
    reachable = ([10.0, 40.0, 10.0, 10.0], [12.0, 42.0, 12.0, 12.0])
    assert check(*reachable, False) is True
    assert check(*reachable, True) is True


def test_a_scenario_with_no_dispatch_is_reported_as_missing_not_as_a_miss():
    """No record is not a failure, and must not be scored as one.

    Substituting a fleet that draws nothing reads as a catastrophic miss --
    most bands exclude zero -- and that reading is about the missing record,
    not about the bid.  It also poisons every average it enters.
    """

    problem = _one_block_problem(
        [_limited_down_scenario("scored"), _limited_down_scenario("missing")]
    )
    config = problem.config
    baseline = np.zeros(config.blocks)
    up = np.zeros(config.blocks)
    down = np.full(config.blocks, 10.0)

    # The scored scenario charges flat out at its award, which is inside the
    # band at every step, and reaches its departure target.
    tracked = np.full(config.steps, 10.0)
    energy = 50.0 + 10.0 * config.dt_hours * np.arange(config.steps + 1)
    solution = BiddingSolution(
        status="optimal",
        solver="test",
        objective_value=0.0,
        baseline_kw=baseline,
        up_kw=up,
        down_kw=down,
        scenario_power_kw={"scored": tracked},
        ev_power_kw={"scored": {"ev": tracked}},
        ev_energy_kwh={"scored": {"ev": energy}},
        metadata={"scenario_name_by_index": ["scored", "missing"]},
    )

    report = validate_joint_solution(problem, solution)
    rows = {str(row["scenario"]): row for row in report["scenario_rows"]}

    missing = rows["missing"]
    assert missing["dispatch_missing"] is True
    assert missing["all_ok"] is False
    # Nothing is claimed about an outcome that was never computed.
    assert missing["global_step_pass_rate"] is None
    assert missing["soc_ok"] is None
    assert missing["min_soc_margin_kwh"] is None

    scored = rows["scored"]
    assert scored["dispatch_missing"] is False
    assert scored["all_ok"] is True
    assert scored["global_step_pass_rate"] == pytest.approx(1.0)

    assert report["scenarios_without_a_dispatch"] == 1
    assert report["scenarios_without_a_dispatch_names"] == ["missing"]
    assert report["all_scenarios_ok"] is False
    # The averages describe the scenario that was actually judged, not a
    # fleet that was never dispatched.
    assert report["min_global_step_pass_rate"] == pytest.approx(1.0)
    assert report["mean_global_step_pass_rate"] == pytest.approx(1.0)


def _two_command_repair_problem():
    """One block, two commands, and a seed bid that misses one of them.

    The solver has to move the baseline for the charge-only command while
    keeping the vehicle-to-grid one.
    """

    config = BiddingLPConfig(
        steps=6,
        blocks=1,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        assessment_band_fraction=0.10,
        time_limit_s=30.0,
    )
    up_signal = np.r_[np.ones(3), np.zeros(3)]
    down_signal = np.r_[np.zeros(3), np.ones(3)]
    easy = ActivationScenario(
        name="easy_v2g",
        up_signal=up_signal,
        down_signal=down_signal,
        evs=[EVSpec(0, 6, 0.50, 0.50, 100.0, 20.0, 10.0, ev_id="v2g")],
    )
    charge_only = ActivationScenario(
        name="charge_only",
        up_signal=up_signal,
        down_signal=down_signal,
        evs=[EVSpec(0, 6, 0.50, 0.50, 100.0, 20.0, 0.0, ev_id="charge")],
    )
    problem = JointBiddingProblem(
        objective_weights=np.ones(1),
        scenarios=[easy],
        baseline_min_kw=0.0,
        baseline_max_kw=10.0,
        u_cap=np.full(1, 10.0),
        d_cap=np.full(1, 10.0),
        global_pass_rate=1.0,
        config=config,
    )
    initial = BiddingSolution(
        status="optimal",
        solver="test-seed",
        objective_value=20.0,
        baseline_kw=np.zeros(1),
        up_kw=np.full(1, 10.0),
        down_kw=np.full(1, 10.0),
    )
    return problem, [easy, charge_only], initial


def test_every_benders_round_checks_every_command() -> None:
    problem, scenarios, initial = _two_command_repair_problem()
    _solution, summary = solve_joint_hard_bidding_benders(
        problem,
        scenarios,
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=8,
    )

    assert summary["complete"] is True
    assert summary["rounds"]
    assert all(
        row["exact_oracle_scenarios"] == len(scenarios)
        for row in summary["rounds"]
    )


def test_the_circulation_supplies_the_cut_by_default() -> None:
    """The same repair, with the Hoffman inequality instead of the LP dual.

    The two cuts describe different things -- one supporting hyperplane of a
    relaxation against the exact feasible region -- so the master can walk a
    different path.  It must still arrive at the same bid.
    """

    problem, scenarios, initial = _two_command_repair_problem()
    solution, summary = solve_joint_hard_bidding_benders(
        problem,
        scenarios,
        initial_solution=initial,
        minimum_active_width_kw=0.01,
        max_rounds=8,
    )

    assert summary["complete"] is True
    assert summary["cut_methods"].get("prefix_circulation_hoffman", 0) >= 1
    assert summary["cut_methods"].get("colgen_phase1", 0) == 0
    assert solution.up_kw[0] > 9.9
    assert solution.down_kw[0] > 9.9

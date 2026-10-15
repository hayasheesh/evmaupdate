"""The PJM performance-score rule for bids (EnvConfig.MARKET_TRACKING_RULE)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from market.bid_participation import step_tolerance_kw, tracking_band_fields
from market.physical_lp_bidding import (
    ActivationScenario,
    BiddingLPConfig,
    BiddingSolution,
    EVSpec,
    JointBiddingProblem,
    solve_joint_hard_bidding_benders,
)
from market.physical_lp_bidding.aggregate_energy import energy_feasible_initial_bid
from market.physical_lp_bidding.colgen_feasibility import affine_tracking_bands
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
from market.pjm_performance import (
    areg_by_step,
    historic_scores,
    interval_scores,
    qualification_summary,
    score_summary,
)

ROOT = Path(__file__).resolve().parents[2]
THETA = 0.75
HALF = 0.5 * (1.0 - THETA)


def _pjm_config(**kw) -> BiddingLPConfig:
    return BiddingLPConfig(
        steps=6,
        blocks=1,
        steps_per_block=6,
        dt_hours=1.0 / 12.0,
        assessment_band_fraction=HALF,
        band_signal_fraction=HALF,
        idle_band_fraction=0.5 * HALF,
        symmetric_award=True,
        apply_transition_band=False,
        **kw,
    )


def test_assessment_ii_band_is_a_flat_tenth_of_the_award():
    regulation = np.array([-30.0, -5.0, 0.0, 8.0, 40.0])
    tol = step_tolerance_kw(regulation, 50.0, 40.0, band_fraction=0.10)
    assert tol == pytest.approx([5.0, 5.0, 9.0, 4.0, 4.0])


def test_the_pjm_band_edge_scores_exactly_the_floor():
    areg = 40.0
    regulation = np.array([-40.0, -30.0, -12.0, 0.0, 5.0, 25.0, 40.0])
    tol = step_tolerance_kw(
        regulation, areg, areg,
        band_fraction=HALF, band_signal_fraction=HALF, idle_band_fraction=0.5 * HALF,
    )
    scores = interval_scores(tol, regulation, np.full(regulation.size, areg), np.ones(regulation.size, bool))
    assert scores == pytest.approx(np.full(regulation.size, THETA))


def test_the_optimizer_band_and_the_judged_band_agree_under_pjm():
    cfg = _pjm_config()
    up_signal = np.array([0.8, 0.3, 0.0, 0.0, 0.0, 0.0])
    down_signal = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 0.0])
    baseline, up, down = np.array([20.0]), np.array([40.0]), np.array([40.0])
    lower, upper, _lo_rows, _hi_rows = affine_tracking_bands(
        cfg, baseline, up, down, up > 0, down > 0, up_signal, down_signal
    )
    _targets, tolerances, band_lower, band_upper = fixed_bid_tracking_bands(
        cfg, baseline, up, down, up_signal, down_signal, apply_transition_band=False
    )
    assert lower == pytest.approx(band_lower)
    assert upper == pytest.approx(band_upper)
    signal = up_signal + down_signal
    expected = np.where(signal > 0, HALF * (1.0 + signal) * 40.0, 0.5 * HALF * 80.0)
    assert tolerances == pytest.approx(expected)


def _two_way_scenario() -> ActivationScenario:
    # Up then down at full scale: the energy comes back by the end of the block.
    return ActivationScenario(
        name="up_then_down",
        up_signal=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        down_signal=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        evs=[EVSpec(0, 6, 0.5, 0.5, 100.0, 10.0, 10.0, ev_id="ev")],
    )


def test_the_seed_offers_one_width_for_both_directions():
    seed = energy_feasible_initial_bid(
        [_two_way_scenario()],
        _pjm_config(),
        baseline_min=0.0,
        baseline_max=0.0,
        up_cap=np.full(1, 10.0),
        down_cap=np.full(1, 6.0),
        minimum_bid_kw=1.0,
    )
    assert seed is not None
    assert seed.up_kw[0] == pytest.approx(seed.down_kw[0])
    assert 1.0 <= seed.up_kw[0] <= 6.0 + 1e-9


def test_the_master_holds_up_equal_to_down():
    scenario = _two_way_scenario()
    problem = JointBiddingProblem(
        objective_weights=np.ones(1),
        scenarios=[scenario],
        baseline_min_kw=0.0,
        baseline_max_kw=0.0,
        u_cap=np.full(1, 10.0),
        d_cap=np.full(1, 6.0),
        global_pass_rate=1.0,
        config=_pjm_config(time_limit_s=30.0),
    )
    seed = BiddingSolution(
        status="optimal", solver="test-seed", objective_value=16.0,
        baseline_kw=np.zeros(1), up_kw=np.full(1, 10.0), down_kw=np.full(1, 6.0),
    )
    solution, summary = solve_joint_hard_bidding_benders(
        problem, [scenario], initial_solution=seed,
        minimum_active_width_kw=1.0, max_rounds=6, allow_widen=True,
    )
    assert summary["complete"] is True
    assert solution.up_kw[0] == pytest.approx(solution.down_kw[0], abs=1e-6)
    assert 1.0 <= solution.up_kw[0] <= 6.0 + 1e-6


def test_a_bid_stored_before_the_rule_reads_as_assessment_ii():
    fields = tracking_band_fields({"assessment_band_fraction": 0.1})
    assert fields == {
        "assessment_band_fraction": 0.1,
        "band_signal_fraction": 0.0,
        "idle_band_fraction": None,
        "symmetric_award": False,
    }


def test_the_score_summary_drops_credit_below_a_quarter():
    scores = np.array([1.0, 0.5, 0.2, np.nan])
    areg = np.array([10.0, 10.0, 20.0, 10.0])
    summary = score_summary(scores, areg)
    assert summary["intervals"] == 3
    assert summary["share_below_payment_floor"] == pytest.approx(1.0 / 3.0)
    assert summary["credit_ratio"] == pytest.approx((10.0 + 5.0) / 40.0)


def test_the_historic_score_counts_only_assigned_intervals():
    scores = np.concatenate([np.full(1200, 0.9), [np.nan] * 50, np.full(1200, 0.3)])
    historic = historic_scores(scores)
    assert historic.size == 2400 - 1200 + 1
    assert historic[0] == pytest.approx(0.9)
    assert historic[-1] == pytest.approx(0.3)
    summary = qualification_summary(scores)
    assert summary["qualified"] is False
    assert qualification_summary(np.full(10, 0.9))["window_full"] is False


def test_areg_follows_the_instructed_direction():
    areg = areg_by_step([30.0], [10.0], [-5.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    assert areg == pytest.approx([30.0, 20.0, 10.0, 20.0, 20.0, 20.0])


def _settings_under(rule: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVMA_")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    if rule is not None:
        env["EVMA_MARKET_TRACKING_RULE"] = rule
    code = (
        "import json, EnvConfig as E\n"
        "from training.lower_bid_training import upper_bid_bank_settings\n"
        "s = upper_bid_bank_settings()\n"
        "print(json.dumps({'settings': {k: s[k] for k in s if 'tracking' in k or 'band' in k or 'symmetric' in k},"
        " 'band': E.LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC, 'signal': E.TRACKING_BAND_SIGNAL_FRACTION,"
        " 'idle': E.TRACKING_IDLE_BAND_FRACTION, 'symmetric': E.SYMMETRIC_AWARD,"
        " 'hours': E.CAPABILITY_DURATION_HOURS, 'market_min': E.MARKET_MINIMUM_BID_QUANTITY_KW,"
        " 'min_bid': E.LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW,"
        " 'min_band': E.LOWER_CONTROLLER_MIN_TRACKING_BAND_KW, 'n': E.NUM_STATIONS,"
        " 'target': E.TARGET_DEPLOYMENT_STATIONS}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
        text=True, encoding="utf-8", timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_pjm_settings_appear_only_under_the_pjm_rule():
    japan = _settings_under(None)
    pjm = _settings_under("pjm")
    assert "market_tracking_rule" not in japan["settings"]
    assert japan["band"] == pytest.approx(0.10)
    assert japan["idle"] == pytest.approx(0.10) and japan["signal"] == 0.0
    assert japan["symmetric"] is False and japan["hours"] == pytest.approx(0.5)
    assert pjm["settings"]["market_tracking_rule"] == "pjm_performance_score"
    assert pjm["settings"]["symmetric_award"] is True
    assert pjm["band"] == pytest.approx(HALF) and pjm["signal"] == pytest.approx(HALF)
    assert pjm["idle"] == pytest.approx(0.5 * HALF)
    assert pjm["hours"] == pytest.approx(5.0 / 60.0)
    assert pjm["market_min"] == pytest.approx(100.0)
    expected = max(100.0 * pjm["n"] / pjm["target"], pjm["min_band"] / HALF)
    assert float(pjm["min_bid"]) == pytest.approx(expected)

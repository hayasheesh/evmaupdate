from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from market.nem_signal_pipeline import (
    AEMO_TIME,
    STEPS_PER_DAY,
    _trading_day,
    build_nem_day_frame,
)


DAY = date(2026, 8, 1)
FIRST = datetime(2026, 8, 1, 4, 5)


def _dispatch(targets, *, availability=100.0, intervention="0"):
    rows = []
    for step in range(STEPS_PER_DAY):
        moment = FIRST + timedelta(minutes=5 * step)
        rows.append({
            "settlementdate": moment.strftime(AEMO_TIME),
            "duid": "TESTB1",
            "intervention": intervention,
            "initialmw": "0",
            "totalcleared": str(float(targets[step])),
            "lowerreg": "0",
            "raisereg": "0",
            "availability": str(float(availability)),
            "energy_storage": "50",
        })
    return pd.DataFrame(rows)


def _plan(levels_by_lead):
    """One row per (block, lead), so the builder has runs to choose between."""

    rows = []
    for block in range(48):
        block_end = datetime(2026, 8, 1, 4, 30) + timedelta(minutes=30 * block)
        for lead_minutes, level in levels_by_lead.items():
            rows.append({
                "seqno": f"lead{lead_minutes}",
                "duid": "TESTB1",
                "intervention": "0",
                "totalcleared": str(float(level)),
                "issued": (
                    block_end - timedelta(minutes=lead_minutes)
                ).strftime(AEMO_TIME),
                "datetime": block_end.strftime(AEMO_TIME),
                "availability": "100",
            })
    return pd.DataFrame(rows)


def test_the_baseline_is_the_last_plan_filed_before_the_gate_closed():
    # Three runs forecast every block: one filed the afternoon before, one an
    # hour out, one after the gate.  Only the middle one may be used.
    dispatch = _dispatch(np.full(STEPS_PER_DAY, 30.0))
    plan = _plan({16 * 60: 0.0, 59: 20.0, 10: 29.0})
    frame = build_nem_day_frame(
        duid="TESTB1", day=DAY, dispatch=dispatch, plan=plan
    )
    assert frame is not None
    assert np.allclose(frame["predispatch_mw"], 20.0)
    # 30 dispatched against a 20 plan is 10 MW of up, a tenth of the band.
    assert np.allclose(frame["up_proxy"], 0.10)
    assert np.allclose(frame["down_proxy"], 0.0)


def test_a_shorter_lead_picks_the_fresher_plan():
    dispatch = _dispatch(np.full(STEPS_PER_DAY, 30.0))
    plan = _plan({16 * 60: 0.0, 59: 20.0, 29: 25.0})
    frame = build_nem_day_frame(
        duid="TESTB1", day=DAY, dispatch=dispatch, plan=plan,
        baseline_lead=timedelta(minutes=25),
    )
    assert frame is not None
    assert np.allclose(frame["predispatch_mw"], 25.0)
    assert np.allclose(frame["up_proxy"], 0.05)


def test_a_negative_target_reads_as_down():
    dispatch = _dispatch(np.full(STEPS_PER_DAY, -40.0))
    frame = build_nem_day_frame(
        duid="TESTB1", day=DAY, dispatch=dispatch, plan=_plan({59: 0.0})
    )
    assert frame is not None
    assert np.allclose(frame["down_proxy"], 0.40)
    assert np.allclose(frame["up_proxy"], 0.0)


def test_a_day_with_no_plan_inside_the_gate_is_dropped():
    # Every run was filed after the gate closed, so there is no baseline the
    # command could be measured against.  Returning a frame built on the
    # post-gate plan would silently understate every command in the day.
    dispatch = _dispatch(np.full(STEPS_PER_DAY, 30.0))
    assert build_nem_day_frame(
        duid="TESTB1", day=DAY, dispatch=dispatch, plan=_plan({10: 25.0})
    ) is None


def test_an_intervention_day_is_dropped():
    # An intervention run publishes a second solution for the same interval;
    # mixing the two would put two targets on one step.
    dispatch = _dispatch(np.full(STEPS_PER_DAY, 30.0), intervention="1")
    assert build_nem_day_frame(
        duid="TESTB1", day=DAY, dispatch=dispatch, plan=_plan({59: 20.0})
    ) is None


def test_the_proxy_is_capped_at_the_offered_band():
    dispatch = _dispatch(np.full(STEPS_PER_DAY, 250.0), availability=100.0)
    frame = build_nem_day_frame(
        duid="TESTB1", day=DAY, dispatch=dispatch, plan=_plan({59: 0.0})
    )
    assert frame is not None
    assert np.allclose(frame["up_proxy"], 1.0)


@pytest.mark.parametrize(
    "moment,expected",
    [
        # The trading day runs 04:05 to 04:00 and is stamped at interval end,
        # so both ends of it belong to the day it opened on.
        (datetime(2026, 8, 1, 4, 5), date(2026, 8, 1)),
        (datetime(2026, 8, 1, 23, 55), date(2026, 8, 1)),
        (datetime(2026, 8, 2, 4, 0), date(2026, 8, 1)),
        (datetime(2026, 8, 2, 4, 5), date(2026, 8, 2)),
    ],
)
def test_the_trading_day_boundary(moment, expected):
    assert _trading_day(moment) == expected

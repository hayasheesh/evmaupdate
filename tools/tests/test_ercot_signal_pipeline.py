from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market.ercot_signal_pipeline import build_ercot_resource_frame
from training.blockwise_bid import _observed_activation_directions_by_block


def _complete_day(day: str = "2026-02-12") -> pd.DataFrame:
    # Each execution precedes its target sampling point, including midnight.
    times = pd.date_range(day, periods=288, freq="5min") - pd.Timedelta(seconds=17)
    base_point = np.full(288, 10.0)
    base_point[3] = 6.0
    base_point[4] = 11.0
    base_point[5] = np.nan
    status = np.full(288, "ONCLR", dtype=object)
    status[5] = "OUTL"
    return pd.DataFrame({
        "SCEDTimestamp": times.astype(str),
        "resourceName": "TEST_ALD1",
        "telResStatus": status,
        "maxPowerConsumption": 12.0,
        "lowPowerConsumption": 2.0,
        "realPowerConsumption": 10.0,
        "basePoint": base_point,
    })










def test_direction_detection_preserves_observed_bidirectional_dispatch() -> None:
    up = np.zeros(288)
    down = np.zeros(288)
    up[12:18] = 0.5
    down[30:36] = 0.4
    observed_up, observed_down = _observed_activation_directions_by_block([{
        "up_proxy": up,
        "down_proxy": down,
    }])

    assert np.flatnonzero(observed_up).tolist() == [2]
    assert np.flatnonzero(observed_down).tolist() == [5]












def test_continuous_online_reference_survives_midnight() -> None:
    rows = pd.concat([_complete_day(), _complete_day("2026-02-13")], ignore_index=True)
    rows["telResStatus"] = "ONCLR"
    rows["basePoint"] = 10 - np.arange(len(rows)) / 1000
    frame, audit = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1")
    assert len(audit) == 1
    second = frame.loc[frame.source_date == "2026-02-13"].iloc[0]
    assert second.reference_base_point_mw == 10
    assert second.signed_activation_raw_up_positive > 0








def test_negative_width_is_not_hidden() -> None:
    rows = _complete_day()
    rows.loc[3, "maxPowerConsumption"] = 1.0
    frame, segments = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1")
    assert "negative_width" in set(frame.quality_reason)
    assert sum(s["negative_width_samples"] for s in segments) == 1


def test_split_crossing_segment_excluded_even_if_width_median_looks_safe() -> None:
    rows = pd.concat([_complete_day(), _complete_day("2026-02-13")], ignore_index=True)
    rows["telResStatus"] = "ONCLR"
    rows["basePoint"] = np.arange(len(rows), dtype=float)
    frame, audit = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1",
                                            train_end=date(2026, 2, 12), validation_end=date(2026, 2, 14))
    assert audit[0]["excluded_reason"] == "split_crossing_segment"
    assert not frame.loc[frame.online.eq(True), "quality_reason"].eq("").any()








def test_repeat_hour_flag_orders_the_two_fall_back_hours() -> None:
    from market.ercot_signal_pipeline import _timestamps
    rows = pd.DataFrame({"SCEDTimestamp": ["2026-11-01 01:05:00"] * 2,
                         "repeatHourFlag": [False, True]})
    utc = _timestamps(rows)
    assert utc.iloc[1] - utc.iloc[0] == pd.Timedelta(hours=1)


def test_midnight_carry_across_split_excludes_whole_source_segment() -> None:
    rows = _complete_day("2026-02-13")
    # The online segment is physically observed only before the boundary but
    # supplies the next day's midnight sample as well.
    rows.loc[1:, "telResStatus"] = "OUTL"
    prior = rows.iloc[[0]].copy()
    prior["SCEDTimestamp"] = "2026-02-12 23:50:00"
    prior["basePoint"] = 9.0
    frame, audit = build_ercot_resource_frame(rows=pd.concat([prior, rows], ignore_index=True),
        resource_name="TEST_ALD1", train_end=date(2026, 2, 12), validation_end=date(2026, 2, 14))
    assert audit[0]["excluded_reason"] == "split_crossing_segment"
    assert not frame.loc[frame.online.eq(True), "quality_reason"].eq("").any()


def test_offline_before_midnight_does_not_falsely_cross_split() -> None:
    rows = _complete_day("2026-02-13")
    rows["telResStatus"] = "OUTL"
    prior = rows.iloc[[0]].copy()
    prior["SCEDTimestamp"] = "2026-02-12 23:59:00"
    prior["telResStatus"] = "ONCLR"
    rows.loc[0, "SCEDTimestamp"] = "2026-02-12 23:59:59"
    _, audit = build_ercot_resource_frame(rows=pd.concat([prior, rows], ignore_index=True),
        resource_name="TEST_ALD1", train_end=date(2026, 2, 12), validation_end=date(2026, 2, 14))
    assert audit[0]["excluded_reason"] == ""




def test_incomplete_library_never_loads(tmp_path: Path) -> None:
    import json
    from market.activation_scenarios import load_proxy_shape_library
    (tmp_path / "metadata.json").write_text(json.dumps({"build_complete": False}))
    with pytest.raises(ValueError, match="Incomplete command library"):
        load_proxy_shape_library(tmp_path)


def _day_rows(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    return frame.loc[frame.source_date == day].reset_index(drop=True)


def test_future_execution_cannot_change_previous_grid_point() -> None:
    rows = _complete_day()
    future = rows.iloc[[1]].copy()
    future["SCEDTimestamp"] = "2026-02-12 00:05:20"
    future["basePoint"] = 3.0
    # The next original execution is earlier than the added one.
    rows.loc[2, "SCEDTimestamp"] = "2026-02-12 00:05:10"
    rows = pd.concat([rows, future], ignore_index=True)
    frame, _ = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1")
    day = _day_rows(frame, "2026-02-12")
    assert day.loc[1, "base_point_mw"] == 10
    assert day.loc[2, "base_point_mw"] == 3
    assert (day.source_sced_timestamp_utc <= day.grid_utc).all()
    assert day.source_age_seconds.between(0, 300, inclusive="left").all()


def test_empty_window_and_open_left_boundary_are_not_forward_filled() -> None:
    rows = _complete_day().drop(index=2)
    rows.loc[1, "SCEDTimestamp"] = "2026-02-12 00:05:00"
    frame, _ = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1")
    point = frame.loc[frame.time_ercot.dt.strftime("%Y-%m-%d %H:%M") == "2026-02-12 00:10"].iloc[0]
    assert point.source_age_seconds == 300
    assert point.quality_reason == "no_fresh_execution"


def test_offline_execution_between_grid_points_starts_new_segment() -> None:
    rows = _complete_day()
    offline = rows.iloc[[2]].copy()
    offline["SCEDTimestamp"] = "2026-02-12 00:06:00"
    offline["telResStatus"] = "OUTL"
    rows.loc[2, "basePoint"] = 7.0
    rows = pd.concat([rows, offline], ignore_index=True)
    frame, _ = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1")
    day = _day_rows(frame, "2026-02-12")
    assert day.loc[1, "segment_id"] != day.loc[2, "segment_id"]


def test_later_posting_correction_wins_but_same_posting_conflict_fails() -> None:
    rows = _complete_day()
    rows["source_posting_date"] = "2026-04-13T12:00:00"
    corrected = rows.iloc[[3]].copy()
    corrected["basePoint"] = 8.0
    corrected["source_posting_date"] = "2026-04-14T12:00:00"
    frame, _ = build_ercot_resource_frame(rows=pd.concat([rows, corrected]), resource_name="TEST_ALD1")
    assert _day_rows(frame, "2026-02-12").loc[3, "base_point_mw"] == 8
    corrected["source_posting_date"] = "2026-04-13T12:00:00"
    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_ercot_resource_frame(rows=pd.concat([rows, corrected]), resource_name="TEST_ALD1")

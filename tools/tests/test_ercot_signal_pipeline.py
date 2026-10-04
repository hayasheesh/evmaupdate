from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market.activation_scenarios import build_activation_scenario_set
from market.ercot_signal_pipeline import (
    SOURCE_TYPE,
    build_ercot_day_frame,
    build_ercot_scenario_library,
    build_ercot_resource_frame,
)
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


def test_ercot_day_maps_base_point_changes_to_relative_level() -> None:
    rows = _complete_day()
    # A later SCED re-run in the same slot supersedes the original solution.
    duplicate = rows.iloc[[3]].copy()
    duplicate["SCEDTimestamp"] = "2026-02-12 00:14:49"
    duplicate["basePoint"] = 8.0
    rows = pd.concat([rows, duplicate], ignore_index=True)

    frame = build_ercot_day_frame(
        resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows
    )

    assert frame is not None
    assert len(frame) == 288
    # Reference is 10 MW. The duplicate makes step 3 an up command of 2 MW
    # over the fixed 10 MW segment-median span; step 4 is a down command of 1 MW.
    assert frame.loc[3, "delta_base_point_mw"] == -2.0
    assert frame.loc[3, "relative_base_point_mw"] == -2.0
    assert frame.loc[3, "up_proxy_raw"] == 0.2
    assert frame.loc[4, "delta_base_point_mw"] == 3.0
    assert frame.loc[4, "relative_base_point_mw"] == 1.0
    assert frame.loc[4, "down_proxy_raw"] == 0.1
    assert frame.loc[5, "up_proxy_raw"] == 0.0
    assert frame.loc[5, "down_proxy_raw"] == 0.0
    assert frame.loc[0, "reference_base_point_mw"] == 10.0
    assert frame.loc[0, "reference_response_width_mw"] == 10.0
    assert set(frame["source_type"]) == {SOURCE_TYPE}


def test_fixed_segment_width_does_not_create_false_command_motion() -> None:
    rows = _complete_day()
    rows.loc[1, "basePoint"] = 8.0
    rows.loc[2, "basePoint"] = 8.0
    rows.loc[1, "maxPowerConsumption"] = 12.0  # instantaneous span 10 MW
    rows.loc[2, "maxPowerConsumption"] = 7.0   # instantaneous span 5 MW

    frame = build_ercot_day_frame(
        resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows
    )

    assert frame is not None
    assert frame.loc[1, "reference_response_width_mw"] == 10.0
    assert frame.loc[2, "reference_response_width_mw"] == 10.0
    assert frame.loc[1, "signed_activation_raw_up_positive"] == 0.2
    assert frame.loc[2, "signed_activation_raw_up_positive"] == 0.2


def test_raw_activation_is_retained_and_clipped_only_by_scenario_loader(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "dispatch.csv"
    output_dir = tmp_path / "processed"
    rows = _complete_day()
    rows.loc[3, "basePoint"] = -5.0  # 15 MW below the 10 MW reference.
    rows.to_csv(input_csv, index=False)

    build_ercot_scenario_library(
        input_csv=input_csv,
        output_dir=output_dir,
        resource_names=["TEST_ALD1"],
    )
    generated = pd.read_csv(next(output_dir.glob("ercot_sced_*.csv")))
    scenarios = build_activation_scenario_set(
        service_date="2026-09-22",
        n_scenarios=1,
        seed=1,
        proxy_shape_dir=output_dir,
    )

    assert generated.loc[3, "up_proxy_raw"] == 1.5
    assert bool(generated.loc[3, "simulation_clip_required"])
    assert scenarios[0].up_proxy[3] == 1.0


def test_library_output_loads_through_shared_activation_interface(tmp_path: Path) -> None:
    input_csv = tmp_path / "dispatch.csv"
    output_dir = tmp_path / "processed"
    rows = _complete_day()
    incomplete = _complete_day("2026-02-13").iloc[:-1]
    pd.concat([rows, incomplete], ignore_index=True).to_csv(input_csv, index=False)

    metadata = build_ercot_scenario_library(
        input_csv=input_csv,
        output_dir=output_dir,
        resource_names=["TEST_ALD1"],
    )
    scenarios = build_activation_scenario_set(
        service_date="2026-09-22",
        n_scenarios=1,
        seed=1,
        proxy_shape_dir=output_dir,
    )

    assert len(metadata["written_files"]) == 1
    assert metadata["resources"][0]["days"] == 1
    assert metadata["offset_required"] is False
    assert len(scenarios) == 1
    assert scenarios[0].source_bmu == "TEST_ALD1"
    assert scenarios[0].source_date == "2026-02-12"
    assert scenarios[0].source.startswith(f"{SOURCE_TYPE}:")


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


def test_online_missing_base_point_rejects_day() -> None:
    rows = _complete_day()
    rows.loc[5, "telResStatus"] = "ONL"
    assert build_ercot_day_frame(
        resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows
    ) is None


def test_missing_status_rejects_day_instead_of_becoming_offline() -> None:
    rows = _complete_day()
    rows.loc[5, "telResStatus"] = np.nan
    assert build_ercot_day_frame(
        resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows
    ) is None


def test_all_offline_day_is_not_a_command_scenario() -> None:
    rows = _complete_day()
    rows["telResStatus"] = "OUTL"
    rows["basePoint"] = np.nan
    assert build_ercot_day_frame(
        resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows
    ) is None


def test_future_execution_cannot_change_previous_grid_point() -> None:
    rows = _complete_day()
    future = rows.iloc[[1]].copy()
    future["SCEDTimestamp"] = "2026-02-12 00:05:20"
    future["basePoint"] = 3.0
    # The next original execution is earlier than the added one.
    rows.loc[2, "SCEDTimestamp"] = "2026-02-12 00:05:10"
    rows = pd.concat([rows, future], ignore_index=True)
    frame = build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows)
    assert frame.loc[1, "base_point_mw"] == 10
    assert frame.loc[2, "base_point_mw"] == 3
    assert (frame.source_sced_timestamp_utc <= frame.grid_utc).all()
    assert frame.source_age_seconds.between(0, 300, inclusive="left").all()


def test_empty_window_and_open_left_boundary_are_not_forward_filled() -> None:
    rows = _complete_day().drop(index=2)
    rows.loc[1, "SCEDTimestamp"] = "2026-02-12 00:05:00"
    frame, _ = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1")
    point = frame.loc[frame.time_ercot.dt.strftime("%Y-%m-%d %H:%M") == "2026-02-12 00:10"].iloc[0]
    assert point.source_age_seconds == 300
    assert point.quality_reason == "no_fresh_execution"
    assert build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows) is None


def test_continuous_online_reference_survives_midnight() -> None:
    rows = pd.concat([_complete_day(), _complete_day("2026-02-13")], ignore_index=True)
    rows["telResStatus"] = "ONCLR"
    rows["basePoint"] = 10 - np.arange(len(rows)) / 1000
    frame, audit = build_ercot_resource_frame(rows=rows, resource_name="TEST_ALD1")
    assert len(audit) == 1
    second = frame.loc[frame.source_date == "2026-02-13"].iloc[0]
    assert second.reference_base_point_mw == 10
    assert second.signed_activation_raw_up_positive > 0


def test_offline_execution_between_grid_points_starts_new_segment() -> None:
    rows = _complete_day()
    offline = rows.iloc[[2]].copy()
    offline["SCEDTimestamp"] = "2026-02-12 00:06:00"
    offline["telResStatus"] = "OUTL"
    rows.loc[2, "basePoint"] = 7.0
    rows = pd.concat([rows, offline], ignore_index=True)
    frame = build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows)
    assert frame.loc[2, "reference_base_point_mw"] == 7
    assert frame.loc[2, "relative_base_point_mw"] == 0
    assert frame.loc[1, "segment_id"] != frame.loc[2, "segment_id"]


@pytest.mark.parametrize("kind", ["ESR", "GEN"])
def test_generation_and_esr_use_injection_positive_sign(kind: str) -> None:
    rows = _complete_day().rename(columns={"maxPowerConsumption": "HSL", "lowPowerConsumption": "LSL"})
    rows["telResStatus"] = "ON"
    rows["basePoint"] = -2.0
    rows.loc[3, "basePoint"] = 2.0
    rows["HSL"], rows["LSL"] = 5.0, -5.0
    frame = build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows, resource_kind=kind)
    assert frame.loc[3, "up_proxy_raw"] == 0.4
    assert frame.loc[3, "reference_response_width_mw"] == 10


def test_zero_width_sample_does_not_delete_valid_bp() -> None:
    rows = _complete_day()
    rows.loc[3, "maxPowerConsumption"] = 2.0
    frame = build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows)
    assert frame.loc[3, "up_proxy_raw"] == 0.4
    assert frame.loc[3, "reference_response_width_mw"] == 10
    rows["maxPowerConsumption"] = 2.0
    assert build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=rows) is None


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


def test_loader_respects_declared_periods_and_never_reranks(tmp_path: Path) -> None:
    rows = pd.concat([_complete_day(f"2026-02-{d}") for d in (12, 13, 14)], ignore_index=True)
    # Explicit shutdown at the end of each day, so segments don't cross splits.
    rows.loc[rows.index % 288 == 287, "telResStatus"] = "OUTL"
    rows.loc[rows.index % 288 == 0, "telResStatus"] = "OUTL"
    source = tmp_path / "input.csv"
    rows.to_csv(source, index=False)
    output = tmp_path / "library"
    metadata = build_ercot_scenario_library(input_csv=source, output_dir=output,
                                           train_end=date(2026, 2, 12), validation_end=date(2026, 2, 13))
    assert len(metadata["written_files"]) == 3
    for partition, day in [("forecast", 12), ("feedback", 13), ("holdout", 14), ("test", 14)]:
        scenarios = build_activation_scenario_set(n_scenarios=2, proxy_shape_dir=output, scenario_partition=partition)
        assert {s.source_date for s in scenarios} == {f"2026-02-{day}"}
    with pytest.raises(ValueError, match="explicit scenario_partition"):
        build_activation_scenario_set(proxy_shape_dir=output)
    with pytest.raises(FileExistsError):
        build_ercot_scenario_library(input_csv=source, output_dir=output)


def test_later_posting_correction_wins_but_same_posting_conflict_fails() -> None:
    rows = _complete_day()
    rows["source_posting_date"] = "2026-04-13T12:00:00"
    corrected = rows.iloc[[3]].copy()
    corrected["basePoint"] = 8.0
    corrected["source_posting_date"] = "2026-04-14T12:00:00"
    frame = build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 2, 12), rows=pd.concat([rows, corrected]))
    assert frame.loc[3, "base_point_mw"] == 8
    corrected["source_posting_date"] = "2026-04-13T12:00:00"
    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_ercot_resource_frame(rows=pd.concat([rows, corrected]), resource_name="TEST_ALD1")


def test_dst_short_day_is_rejected_without_interpolation() -> None:
    rows = _complete_day("2026-03-08")
    rows["SCEDTimestamp"] = pd.date_range("2026-03-08", periods=288, freq="5min", tz="America/Chicago").astype(str)
    assert build_ercot_day_frame(resource_name="TEST_ALD1", day=date(2026, 3, 8), rows=rows) is None


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


def test_no_positive_width_and_nan_are_audited_not_exported(tmp_path: Path) -> None:
    rows = _complete_day()
    rows["maxPowerConsumption"] = 2.0
    source = tmp_path / "input.csv"
    rows.to_csv(source, index=False)
    metadata = build_ercot_scenario_library(input_csv=source, output_dir=tmp_path / "result")
    assert metadata["written_files"] == []
    assert any("invalid_segment_reference_or_width" in item["reasons"] for item in metadata["days"])


def test_incomplete_library_never_loads(tmp_path: Path) -> None:
    import json
    from market.activation_scenarios import load_proxy_shape_library
    (tmp_path / "metadata.json").write_text(json.dumps({"build_complete": False}))
    with pytest.raises(ValueError, match="Incomplete command library"):
        load_proxy_shape_library(tmp_path)

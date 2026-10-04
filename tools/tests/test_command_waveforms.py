from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market.command_waveforms import (
    STEPS_PER_DAY,
    build_command_waveform_day,
    select_end_stamped_calendar_day,
    waveform_to_activation_proxy,
)


def _source(target: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "step": np.arange(STEPS_PER_DAY),
        "time_local": [f"t{i}" for i in range(STEPS_PER_DAY)],
        "time_utc": [f"z{i}" for i in range(STEPS_PER_DAY)],
        "target": target,
        "changing_forecast": np.linspace(-1000.0, 1000.0, STEPS_PER_DAY),
        "availability": np.linspace(1.0, 500.0, STEPS_PER_DAY),
    })


def test_aemo_waveform_keeps_raw_target_and_ignores_forecast_and_availability():
    target = np.linspace(-2.0, 7.0, STEPS_PER_DAY)
    source = _source(target)
    result = build_command_waveform_day(
        source,
        market="AEMO NEM",
        resource_kind="BESS",
        resource_name="TESTB1",
        source_date="2026-07-22",
        step_column="step",
        target_column="target",
        direction_sign=1.0,
        local_time_column="time_local",
        utc_time_column="time_utc",
        availability_column="availability",
    )
    assert np.array_equal(result.target_mw.to_numpy(), target)
    assert np.allclose(result.relative_up_positive_mw, target - target[0])
    assert np.allclose(result.delta_up_positive_mw.iloc[1:], np.diff(target))
    assert np.isnan(result.delta_up_positive_mw.iloc[0])
    assert result.reference_target_mw.nunique() == 1
    assert result.relative_up_positive_mw.iloc[0] == 0


def test_clr_waveform_uses_sign_only_and_does_not_scale_or_clip():
    target = np.linspace(10.0, -5.0, STEPS_PER_DAY)
    result = build_command_waveform_day(
        _source(target),
        market="ERCOT",
        resource_kind="CLR",
        resource_name="TEST_ALD1",
        source_date="2026-07-22",
        step_column="step",
        target_column="target",
        direction_sign=-1.0,
        local_time_column="time_local",
    )
    assert np.array_equal(result.target_mw.to_numpy(), target)
    assert np.allclose(result.relative_up_positive_mw, -(target - target[0]))
    assert result.relative_up_positive_mw.max() == pytest.approx(15.0)
    assert result.target_mw.min() == pytest.approx(-5.0)


def test_fixed_affine_proxy_preserves_bess_shape_without_clipping():
    target = np.array([5.0, 7.0, 3.0] + [5.0] * (STEPS_PER_DAY - 3))
    waveform = build_command_waveform_day(
        _source(target), market="AEMO", resource_kind="BESS", resource_name="B1",
        source_date="2026-07-22", step_column="step", target_column="target",
        direction_sign=1.0, local_time_column="time_local",
    )
    proxy = waveform_to_activation_proxy(
        waveform, reference_mode="first", source_type="test",
        scenario_partition="train", segment_id="B1:2026-07-22",
    )
    assert proxy.normalization_scale_mw.iloc[0] == pytest.approx(2.0)
    assert proxy.signed_activation_up_positive_raw.iloc[:3].tolist() == [0.0, 1.0, -1.0]
    assert proxy.up_proxy_raw.iloc[:3].tolist() == [0.0, 1.0, 0.0]
    assert proxy.down_proxy_raw.iloc[:3].tolist() == [0.0, 0.0, 1.0]
    assert np.allclose(
        proxy.signed_activation_up_positive_raw * proxy.normalization_scale_mw,
        proxy.target_mw - proxy.reference_target_mw,
    )


def test_wdru_proxy_uses_physical_zero_not_first_sample():
    target = np.array([2.0, 0.0, 4.0] + [0.0] * (STEPS_PER_DAY - 3))
    waveform = build_command_waveform_day(
        _source(target), market="AEMO", resource_kind="WDRU", resource_name="DR1",
        source_date="2026-07-22", step_column="step", target_column="target",
        direction_sign=1.0, local_time_column="time_local",
    )
    proxy = waveform_to_activation_proxy(
        waveform, reference_mode="zero", source_type="test",
        scenario_partition="validation", segment_id="DR1:2026-07-22",
    )
    assert proxy.reference_target_mw.iloc[0] == 0.0
    assert proxy.up_proxy_raw.iloc[:3].tolist() == [0.5, 0.0, 1.0]
    assert proxy.down_proxy_raw.max() == 0.0


def test_plan_deviation_uses_the_unit_width_and_keeps_values_beyond_it():
    """A command amount is the deviation itself, scaled by one fixed unit width."""

    deviation = np.array([0.0, 50.0, -150.0] + [0.0] * (STEPS_PER_DAY - 3))
    waveform = build_command_waveform_day(
        _source(deviation), market="AEMO", resource_kind="BESS_PLAN_DEVIATION",
        resource_name="B1", source_date="2026-07-22", step_column="step",
        target_column="target", direction_sign=1.0, local_time_column="time_local",
    )
    proxy = waveform_to_activation_proxy(
        waveform, reference_mode="zero", source_type="test",
        scenario_partition="train", segment_id="B1:2026-07-22", scale_mw=100.0,
    )
    assert proxy.normalization_scale_mw.iloc[0] == 100.0
    assert proxy.signed_activation_up_positive_raw.iloc[:3].tolist() == [0.0, 0.5, -1.5]
    assert proxy.down_proxy_raw.iloc[2] == 1.5


def test_a_day_without_deviation_is_a_valid_no_command_day():
    waveform = build_command_waveform_day(
        _source(np.zeros(STEPS_PER_DAY)), market="AEMO",
        resource_kind="BESS_PLAN_DEVIATION", resource_name="B1",
        source_date="2026-07-22", step_column="step", target_column="target",
        direction_sign=1.0, local_time_column="time_local",
    )
    proxy = waveform_to_activation_proxy(
        waveform, reference_mode="zero", source_type="test",
        scenario_partition="train", segment_id="B1:2026-07-22", scale_mw=100.0,
    )
    assert proxy.signed_activation_up_positive_raw.abs().max() == 0.0


def test_flat_waveform_cannot_enter_activation_bank():
    waveform = build_command_waveform_day(
        _source(np.ones(STEPS_PER_DAY)), market="AEMO", resource_kind="BESS",
        resource_name="B1", source_date="2026-07-22", step_column="step",
        target_column="target", direction_sign=1.0, local_time_column="time_local",
    )
    with pytest.raises(ValueError, match="Flat target"):
        waveform_to_activation_proxy(
            waveform, reference_mode="first", source_type="test",
            scenario_partition="train", segment_id="B1:2026-07-22",
        )


def test_incomplete_or_nonfinite_waveform_is_rejected():
    source = _source(np.arange(STEPS_PER_DAY, dtype=float))
    with pytest.raises(ValueError, match="Expected 288"):
        build_command_waveform_day(
            source.iloc[:-1], market="AEMO", resource_kind="BESS", resource_name="X",
            source_date="2026-07-22", step_column="step", target_column="target",
            direction_sign=1.0, local_time_column="time_local",
        )
    source.loc[10, "target"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_command_waveform_day(
            source, market="AEMO", resource_kind="BESS", resource_name="X",
            source_date="2026-07-22", step_column="step", target_column="target",
            direction_sign=1.0, local_time_column="time_local",
        )


def test_offline_steps_are_not_silently_interpreted_as_zero():
    source = _source(np.arange(STEPS_PER_DAY, dtype=float))
    source["online"] = True
    source.loc[100, "online"] = False
    with pytest.raises(ValueError, match="offline or unknown-status"):
        build_command_waveform_day(
            source, market="ERCOT", resource_kind="CLR", resource_name="X",
            source_date="2026-07-22", step_column="step", target_column="target",
            direction_sign=-1.0, local_time_column="time_local", online_column="online",
        )


def test_calendar_day_stitches_interval_end_samples_across_archive_partitions():
    start = pd.Timestamp("2026-07-22")
    timestamps = pd.date_range(
        start=start - pd.Timedelta(days=1) + pd.Timedelta(hours=4, minutes=5),
        periods=576,
        freq="5min",
    )
    source = pd.DataFrame({
        "step": np.arange(len(timestamps)),
        "time_nem": timestamps.astype(str),
        "source_archive_date": np.where(
            timestamps < start + pd.Timedelta(hours=4, minutes=5),
            "2026-07-21",
            "2026-07-22",
        ),
        "target": np.arange(len(timestamps), dtype=float),
    })

    day = select_end_stamped_calendar_day(
        source, timestamp_column="time_nem", day="2026-07-22"
    )

    assert len(day) == STEPS_PER_DAY
    assert day.step.tolist() == list(range(STEPS_PER_DAY))
    assert day.time_nem.iloc[0] == "2026-07-22 00:05:00"
    assert day.time_nem.iloc[-1] == "2026-07-23 00:00:00"
    assert set(day.source_archive_date) == {"2026-07-21", "2026-07-22"}


def test_calendar_day_rejects_missing_or_duplicate_samples():
    start = pd.Timestamp("2026-07-22")
    timestamps = pd.date_range(
        start=start + pd.Timedelta(minutes=5), periods=288, freq="5min"
    )
    source = pd.DataFrame({"time_nem": timestamps.astype(str)})

    with pytest.raises(ValueError, match="Expected complete interval-end samples"):
        select_end_stamped_calendar_day(
            source.iloc[:-1], timestamp_column="time_nem", day="2026-07-22"
        )

    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate samples"):
        select_end_stamped_calendar_day(
            duplicated, timestamp_column="time_nem", day="2026-07-22"
        )




def test_every_month_is_split_60_20_20_in_consecutive_blocks():
    from datetime import date, timedelta

    from market.command_waveforms import month_balanced_partitions

    days = [date(2025, 12, 1) + timedelta(days=i) for i in range(31 + 31 + 14)]
    for given in (days, [d.isoformat() for d in days]):
        partitions = month_balanced_partitions(given)
        for month, counts in (("2025-12", (18, 6, 7)), ("2026-01", (18, 6, 7)), ("2026-02", (8, 3, 3))):
            labels = [partitions[d] for d in given if str(d).startswith(month)]
            assert labels == ["train"] * counts[0] + ["validation"] * counts[1] + ["test"] * counts[2]

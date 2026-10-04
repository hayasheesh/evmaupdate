"""Lossless, offset-independent extraction of market dispatch target waveforms.

The historical activation-proxy builders remain available for reproducing
simulation runs. This module is the separate, source-faithful path for studying
command shapes: it keeps each market's native target in MW and derives only a
constant-offset-relative curve and first difference. It does not consult a
forecast baseline, divide by availability or resource width, clip, or
interpolate.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


STEPS_PER_DAY = 288


def month_balanced_partitions(days) -> dict:
    """Split the days of each calendar month into consecutive 60/20/20 blocks.

    Every partition then holds the same months, so a command set whose
    direction changes with the season has the same season mix in train,
    validation and test. Days are ``date`` objects or ISO strings; the keys of
    the result are the given values.
    """
    months: dict[str, list] = {}
    for day in sorted(set(days), key=str):
        months.setdefault(str(day)[:7], []).append(day)
    partitions = {}
    for ordered in months.values():
        train_end = int(np.floor(len(ordered) * 0.60))
        validation_end = int(np.floor(len(ordered) * 0.80))
        for index, day in enumerate(ordered):
            partitions[day] = (
                "train" if index < train_end else "validation" if index < validation_end else "test"
            )
    return partitions


def waveform_to_activation_proxy(
    waveform: pd.DataFrame,
    *,
    reference_mode: str,
    source_type: str,
    scenario_partition: str,
    segment_id: str,
    scale_mw: float | None = None,
) -> pd.DataFrame:
    """Convert one native-MW waveform into the bidder's dimensionless shape.

    One constant reference and one constant scale are used for all 288 points,
    preserving turns, plateaus, durations, and directional asymmetry. No
    time-varying availability division, clipping, interpolation, or fill is
    performed. ``zero`` takes 0 MW as the reference, which is an idle BESS, an
    un-curtailed WDRU, or no deviation from plan. ``first`` subtracts the
    scenario's first target.

    Without ``scale_mw`` the scale is the scenario's largest deviation, so the
    shape spans exactly [-1, 1]. With ``scale_mw`` the given unit width is used
    as is: a quiet scenario stays small, values beyond 1 are kept unclipped for
    the input boundary to clip, and an all-zero scenario is a valid no-command
    day.
    """
    required = {
        "step", "target_mw", "direction_sign", "source_date",
        "resource_name", "source_time_local",
    }
    missing = required.difference(waveform.columns)
    if missing:
        raise ValueError(f"Missing waveform columns for activation conversion: {sorted(missing)}")
    frame = waveform.sort_values("step", kind="stable").reset_index(drop=True).copy()
    if len(frame) != STEPS_PER_DAY or not np.array_equal(
        pd.to_numeric(frame["step"], errors="coerce").to_numpy(float),
        np.arange(STEPS_PER_DAY, dtype=float),
    ):
        raise ValueError("Activation waveform must contain unique steps 0..287")
    target = pd.to_numeric(frame["target_mw"], errors="coerce").to_numpy(float)
    if not np.isfinite(target).all():
        raise ValueError("Activation target contains missing or non-finite values")
    signs = pd.to_numeric(frame["direction_sign"], errors="coerce").to_numpy(float)
    if not np.isfinite(signs).all() or not np.all(signs == signs[0]) or signs[0] not in (-1.0, 1.0):
        raise ValueError("Waveform must have one finite direction_sign of +1 or -1")

    mode = str(reference_mode).strip().lower()
    if mode == "first":
        reference = float(target[0])
    elif mode == "zero":
        reference = 0.0
    else:
        raise ValueError("reference_mode must be 'first' or 'zero'")
    relative = float(signs[0]) * (target - reference)
    if scale_mw is None:
        scale = float(np.max(np.abs(relative)))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("Flat target has no command shape and cannot enter the bank")
    else:
        scale = float(scale_mw)
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("scale_mw must be a positive finite width")
    signed = relative / scale
    if scale_mw is None and np.max(np.abs(signed)) > 1.0 + 1e-12:
        raise RuntimeError("Activation normalization escaped its fixed scale")

    result = frame.copy()
    result["reference_target_mw"] = reference
    result["relative_up_positive_mw"] = relative
    result["normalization_scale_mw"] = scale
    result["signed_activation_up_positive_raw"] = signed
    result["up_proxy_raw"] = np.maximum(signed, 0.0)
    result["down_proxy_raw"] = np.maximum(-signed, 0.0)
    result["source_type"] = str(source_type)
    result["source_bmu"] = frame["resource_name"].astype(str)
    result["scenario_partition"] = str(scenario_partition)
    result["segment_id"] = str(segment_id)
    result["reference_mode"] = mode
    return result


def select_end_stamped_calendar_day(
    source: pd.DataFrame,
    *,
    timestamp_column: str,
    day: date | str,
    step_column: str = "step",
) -> pd.DataFrame:
    """Select a complete local calendar day of interval-end 5-minute samples.

    For AEMO NEM data, a calendar day is represented by the timestamps
    00:05 through 00:00 on the following date (24:00). Input archive files may
    be partitioned at another boundary, so rows are selected by timestamp
    rather than by their archive/source-date label.
    """
    if timestamp_column not in source:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    timestamps = pd.to_datetime(source[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Timestamp contains missing or invalid values")
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Calendar-day timestamps must be local and timezone-naive")

    start = pd.Timestamp(day)
    if start != start.normalize():
        raise ValueError("day must be a calendar date without a time")
    expected = pd.date_range(
        start=start + pd.Timedelta(minutes=5),
        periods=STEPS_PER_DAY,
        freq="5min",
    )
    selected_mask = timestamps.isin(expected)
    selected_timestamps = pd.DatetimeIndex(timestamps.loc[selected_mask])
    if selected_timestamps.has_duplicates:
        raise ValueError(f"Duplicate samples in calendar day {start.date()}")
    if not selected_timestamps.sort_values().equals(expected):
        raise ValueError(
            f"Expected complete interval-end samples 00:05..24:00 for "
            f"{start.date()}, found {len(selected_timestamps)}"
        )

    selected = source.loc[selected_mask].copy()
    selected["_calendar_timestamp"] = selected_timestamps.to_numpy()
    selected = selected.sort_values("_calendar_timestamp", kind="stable").reset_index(drop=True)
    selected[step_column] = np.arange(STEPS_PER_DAY, dtype=int)
    return selected.drop(columns="_calendar_timestamp")


def aggregate_calendar_day_targets(
    source: pd.DataFrame,
    *,
    timestamp_column: str,
    resource_column: str,
    target_column: str,
    day: date | str,
) -> pd.DataFrame:
    """Sum a complete, fixed resource panel over a local calendar day.

    Every resource must have exactly one finite target at each interval-end
    timestamp from 00:05 through the next day's 00:00. Missing resources or
    duplicate resource/timestamp pairs are rejected rather than zero-filled.
    """
    required = {timestamp_column, resource_column, target_column}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing aggregate columns: {sorted(missing)}")

    timestamps = pd.to_datetime(source[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Timestamp contains missing or invalid values")
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Calendar-day timestamps must be local and timezone-naive")
    start = pd.Timestamp(day)
    if start != start.normalize():
        raise ValueError("day must be a calendar date without a time")
    expected = pd.date_range(
        start=start + pd.Timedelta(minutes=5),
        periods=STEPS_PER_DAY,
        freq="5min",
    )

    selected_mask = timestamps.isin(expected)
    selected = source.loc[selected_mask, [resource_column, target_column]].copy()
    selected["_calendar_timestamp"] = timestamps.loc[selected_mask].to_numpy()
    if selected[resource_column].isna().any():
        raise ValueError("Resource identifiers contain missing values")
    selected["_resource"] = selected[resource_column].astype(str)
    if selected["_resource"].eq("").any():
        raise ValueError("Resource identifiers contain missing values")
    selected["_target"] = pd.to_numeric(selected[target_column], errors="coerce")
    if not np.isfinite(selected["_target"].to_numpy(float)).all():
        raise ValueError("Target contains missing or non-finite values")
    if selected.duplicated(["_calendar_timestamp", "_resource"]).any():
        raise ValueError("Duplicate resource/timestamp pair in calendar day")

    resources = set(selected["_resource"])
    if not resources:
        raise ValueError(f"No resource targets for calendar day {start.date()}")
    resource_sets = selected.groupby("_calendar_timestamp", sort=True)["_resource"].agg(set)
    if not pd.DatetimeIndex(resource_sets.index).equals(expected) or any(
        resource_set != resources for resource_set in resource_sets
    ):
        raise ValueError(
            f"Incomplete resource panel for interval-end calendar day {start.date()}"
        )

    grouped = selected.groupby("_calendar_timestamp", sort=True)
    summed = grouped["_target"].sum()
    active = grouped["_target"].apply(lambda values: int(values.ne(0).sum()))
    return pd.DataFrame({
        "calendar_step": np.arange(STEPS_PER_DAY, dtype=int),
        "calendar_time": summed.index.astype(str),
        "aggregate_target_mw": summed.to_numpy(float),
        "source_resource_count": len(resources),
        "active_resource_count": active.to_numpy(int),
    })


def build_command_waveform_day(
    source: pd.DataFrame,
    *,
    market: str,
    resource_kind: str,
    resource_name: str,
    source_date: date | str,
    step_column: str,
    target_column: str,
    direction_sign: float,
    local_time_column: str,
    utc_time_column: str | None = None,
    execution_time_column: str | None = None,
    status_column: str | None = None,
    online_column: str | None = None,
    availability_column: str | None = None,
) -> pd.DataFrame:
    """Return one complete 288-sample source-target waveform in native MW.

    ``target_mw`` is copied without rescaling. ``relative_up_positive_mw``
    subtracts one constant reference (the first sample of this scenario) and
    applies only the requested sign convention. ``delta_up_positive_mw`` is
    the first difference, with the undefined first value left missing.

    ``direction_sign`` is +1 when increasing the native target means more
    upward support and -1 when decreasing the native target means more upward
    support. The original target is always retained so callers can choose a
    different reference, sign, or bid capacity later.
    """
    if not np.isfinite(direction_sign) or direction_sign not in (-1.0, 1.0):
        raise ValueError("direction_sign must be +1 or -1")
    required = {step_column, target_column, local_time_column}
    for optional in (utc_time_column, execution_time_column, status_column,
                     online_column, availability_column):
        if optional:
            required.add(optional)
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing waveform columns: {sorted(missing)}")

    frame = source.sort_values(step_column, kind="stable").reset_index(drop=True)
    if len(frame) != STEPS_PER_DAY:
        raise ValueError(f"Expected {STEPS_PER_DAY} samples, found {len(frame)}")
    steps = pd.to_numeric(frame[step_column], errors="coerce").to_numpy(float)
    if not np.array_equal(steps, np.arange(STEPS_PER_DAY, dtype=float)):
        raise ValueError("Expected unique, contiguous step values 0..287")
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    if not np.isfinite(target).all():
        raise ValueError("Target contains missing or non-finite values")
    if frame[local_time_column].isna().any():
        raise ValueError("Timestamp contains missing values")
    if online_column:
        online = frame[online_column]
        if online.dtype == bool:
            all_online = bool(online.all())
        else:
            all_online = online.astype(str).str.strip().str.upper().isin(
                {"TRUE", "1", "ON", "ONL", "ONCLR", "ONREG", "ONRR", "ONRRREG"}
            ).all()
        if not all_online:
            raise ValueError("Scenario contains offline or unknown-status samples")

    relative = direction_sign * (target - target[0])
    delta = np.full(STEPS_PER_DAY, np.nan, dtype=float)
    delta[1:] = direction_sign * np.diff(target)
    dates = [str(source_date)] * STEPS_PER_DAY
    result = pd.DataFrame({
        "market": market,
        "resource_kind": resource_kind,
        "resource_name": resource_name,
        "source_date": dates,
        "step": steps.astype(int),
        "source_time_local": frame[local_time_column].astype(str),
        "target_mw": target,
        "reference_target_mw": float(target[0]),
        "relative_up_positive_mw": relative,
        "delta_up_positive_mw": delta,
        "direction_sign": float(direction_sign),
        "source_target_column": target_column,
    })
    if utc_time_column:
        result["source_time_utc"] = frame[utc_time_column].astype(str)
    if execution_time_column:
        result["source_execution_time"] = frame[execution_time_column].astype(str)
    if status_column:
        result["resource_status"] = frame[status_column].astype(str)
    if availability_column:
        result["availability_mw"] = pd.to_numeric(
            frame[availability_column], errors="coerce"
        ).to_numpy(float)
    return result

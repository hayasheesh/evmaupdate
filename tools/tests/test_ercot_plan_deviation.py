import numpy as np
import pandas as pd
import pytest

from market.ercot_plan_deviation import (
    day_activation,
    load_cop_planned_soc,
    planned_output_mw,
)


def _plan(values, start="2026-03-10 00:00"):
    index = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(np.asarray(values, dtype=float), index=index)


def _day(base_point):
    local = pd.date_range("2026-03-10 00:00", periods=288, freq="5min", tz="America/Chicago")
    return pd.DataFrame({
        "time_ercot": local,
        "grid_utc": local.tz_convert("UTC"),
        "source_date": "2026-03-10",
        "quality_reason": "",
        "online": True,
        "base_point_mw": np.asarray(base_point, dtype=float),
    })


def test_planned_output_is_the_soc_drop_over_the_hour():
    soc = _plan([10.0, 15.0, 15.0, 5.0])
    hours = pd.date_range("2026-03-10 00:00", periods=3, freq="h")
    # Charging 5 MWh, holding, then discharging 10 MWh.
    assert planned_output_mw(soc, hours).tolist() == [-5.0, 0.0, 10.0]


def test_conflicting_cop_rows_are_left_out(tmp_path):
    path = tmp_path / "cop.csv"
    pd.DataFrame({
        "Delivery Date": ["03/10/2026"] * 4,
        "Resource Name": ["A_ESR", "A_ESR", "A_ESR", "B_ESR"],
        "Hour Ending": ["01:00", "01:00", "02:00", "01:00"],
        "Hour Beginning Planned SOC": [5.0, 6.0, 7.0, 1.0],
    }).to_csv(path, index=False)
    plans = load_cop_planned_soc([path])
    assert pd.Timestamp("2026-03-10 00:00") not in plans["A_ESR"].index
    assert plans["A_ESR"][pd.Timestamp("2026-03-10 01:00")] == 7.0
    assert plans["B_ESR"][pd.Timestamp("2026-03-10 00:00")] == 1.0


def test_command_is_base_point_minus_plan_over_the_unit_width():
    # Plan: discharge 10 MW every hour; the day ends at the next day's 00:00 SOC.
    soc = _plan([250.0 - 10.0 * h for h in range(25)])
    base_point = np.full(288, 10.0)
    base_point[:12] = 20.0
    activation, reason = day_activation(
        _day(base_point), soc, resource_name="A_ESR", width_mw=20.0, partition="train",
    )
    assert reason == ""
    assert activation["plan_mw"].iloc[0] == pytest.approx(10.0)
    assert activation["signed_activation_up_positive_raw"].iloc[:12].tolist() == [0.5] * 12
    assert activation["signed_activation_up_positive_raw"].iloc[12:].abs().max() == 0.0


def test_a_day_without_the_next_morning_plan_is_left_out():
    soc = _plan([100.0] * 24)
    activation, reason = day_activation(
        _day(np.zeros(288)), soc, resource_name="A_ESR", width_mw=20.0, partition="train",
    )
    assert activation is None
    assert reason == "missing_cop_plan"

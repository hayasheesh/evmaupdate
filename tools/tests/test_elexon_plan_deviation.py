from datetime import date, datetime, timezone

import numpy as np
import pytest

from market.elexon_plan_deviation import (
    acceptance_in_force,
    day_activation,
    storage_units,
    to_segments,
)


def _row(start, end, level_from, level_to, accepted=None, number=1, so=False):
    row = {"timeFrom": start, "timeTo": end, "levelFrom": level_from, "levelTo": level_to}
    if accepted is not None:
        row.update(acceptanceTime=accepted, acceptanceNumber=number, soFlag=so)
    return row


FPN_ALL_DAY = [_row("2026-01-10T00:00:00Z", "2026-01-11T00:00:00Z", 10.0, 10.0)]


def test_an_acceptance_counts_only_once_issued_and_the_latest_wins():
    boa = to_segments([
        _row("2026-01-10T01:00:00Z", "2026-01-10T03:00:00Z", -20.0, -20.0, "2026-01-10T00:50:00Z", 1),
        _row("2026-01-10T02:00:00Z", "2026-01-10T03:00:00Z", 5.0, 5.0, "2026-01-10T02:30:00Z", 2),
    ], boa=True)
    at = lambda hh, mm: datetime(2026, 1, 10, hh, mm, tzinfo=timezone.utc)
    assert acceptance_in_force(boa, at(0, 55)) is None
    assert acceptance_in_force(boa, at(2, 10)).acceptance_number == 1
    assert acceptance_in_force(boa, at(2, 35)).acceptance_number == 2


def test_command_is_the_instructed_level_minus_fpn_over_the_width():
    boa = to_segments([
        _row("2026-01-10T01:00:00Z", "2026-01-10T02:00:00Z", -20.0, -20.0, "2026-01-10T00:50:00Z"),
    ], boa=True)
    activation, reason = day_activation(
        unit="T_TESTB-1", day=date(2026, 1, 10), boa=boa, fpn=to_segments(FPN_ALL_DAY, boa=False),
        width_mw=50.0, partition="train",
    )
    assert reason == ""
    a = activation["signed_activation_up_positive_raw"].to_numpy()
    assert a[12:24] == pytest.approx(np.full(12, -0.6))
    assert np.abs(np.delete(a, np.arange(12, 24))).max() == 0.0
    assert activation["fpn_mw"].iloc[0] == 10.0


def test_a_dst_change_day_is_left_out():
    activation, reason = day_activation(
        unit="T_TESTB-1", day=date(2026, 3, 29), boa=[], fpn=to_segments(FPN_ALL_DAY, boa=False),
        width_mw=50.0, partition="train",
    )
    assert activation is None and reason == "non_288_slot_day"


def test_a_day_without_full_fpn_is_left_out():
    fpn = to_segments([_row("2026-01-10T00:00:00Z", "2026-01-10T12:00:00Z", 10.0, 10.0)], boa=False)
    activation, reason = day_activation(
        unit="T_TESTB-1", day=date(2026, 1, 10), boa=[], fpn=fpn, width_mw=50.0, partition="train",
    )
    assert activation is None and reason == "missing_fpn"


def test_storage_units_are_two_way_units_of_comparable_size():
    reference = [
        {"elexonBmUnit": "T_BATT-1", "fuelType": "OTHER", "generationCapacity": "50", "demandCapacity": "-57"},
        {"elexonBmUnit": "E_BESS-1", "fuelType": None, "generationCapacity": "20", "demandCapacity": "-20"},
        {"elexonBmUnit": "T_PUMP-1", "fuelType": "PS", "generationCapacity": "400", "demandCapacity": "-400"},
        {"elexonBmUnit": "T_CCGT-1", "fuelType": "OTHER", "generationCapacity": "470", "demandCapacity": "-7"},
    ]
    assert storage_units(reference) == {"T_BATT-1": 57.0, "E_BESS-1": 20.0}


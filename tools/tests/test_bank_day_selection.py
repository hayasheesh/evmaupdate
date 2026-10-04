from __future__ import annotations

from datetime import date, timedelta

import pytest

from training.run_after_day_ahead_bid import (
    stratified_bank_day_selection,
    weekday_class_for_date,
)


def _no_holiday(_day) -> bool:
    return False


def _payloads(start: str, days: int) -> list[dict]:
    first = date.fromisoformat(start)
    return [
        {"date": (first + timedelta(days=i)).isoformat(), "series": None}
        for i in range(days)
    ]


def _covered_cells(dates: list[str], is_holiday=_no_holiday) -> set[tuple[str, int]]:
    return {
        (
            weekday_class_for_date(d, is_holiday=is_holiday),
            date.fromisoformat(d).month,
        )
        for d in dates
    }


def test_weekday_class_covers_saturday_sunday_and_holidays() -> None:
    assert weekday_class_for_date("2024-08-05", is_holiday=_no_holiday) == "weekday"
    assert weekday_class_for_date("2024-08-03", is_holiday=_no_holiday) == "saturday"
    assert weekday_class_for_date("2024-08-04", is_holiday=_no_holiday) == "sunday_holiday"
    # A holiday joins the sunday_holiday class regardless of its weekday.
    assert (
        weekday_class_for_date("2024-08-05", is_holiday=lambda _d: True)
        == "sunday_holiday"
    )
    # Default calendar is the service market's (Japan): 4 Nov 2024 is a
    # substitute holiday on a Monday, US Thanksgiving is a working day.
    assert weekday_class_for_date("2024-11-04") == "sunday_holiday"
    assert weekday_class_for_date("2024-11-28") == "weekday"


def test_stratified_selection_is_deterministic_and_disjoint() -> None:
    pool = _payloads("2024-08-01", 153)  # Aug 1 .. Dec 31

    train_a, test_a, info_a = stratified_bank_day_selection(
        pool, train_count=25, test_count=5, is_holiday=_no_holiday
    )
    train_b, test_b, info_b = stratified_bank_day_selection(
        pool, train_count=25, test_count=5, is_holiday=_no_holiday
    )

    assert [p["date"] for p in train_a] == [p["date"] for p in train_b]
    assert [p["date"] for p in test_a] == [p["date"] for p in test_b]
    assert info_a["train_dates"] == info_b["train_dates"]
    assert info_a["test_dates"] == info_b["test_dates"]
    assert len(train_a) == 25
    assert len(test_a) == 5
    train_dates = {p["date"] for p in train_a}
    test_dates = {p["date"] for p in test_a}
    assert not train_dates & test_dates
    pool_dates = {p["date"] for p in pool}
    assert train_dates <= pool_dates
    assert test_dates <= pool_dates




def test_stratified_selection_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="ISO dates"):
        stratified_bank_day_selection(
            [{"date": "not-a-date", "series": None}],
            train_count=1,
            test_count=0,
            is_holiday=_no_holiday,
        )
    with pytest.raises(RuntimeError, match="needs 30 dates"):
        stratified_bank_day_selection(
            _payloads("2024-08-01", 10),
            train_count=25,
            test_count=5,
            is_holiday=_no_holiday,
        )

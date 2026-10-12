"""Public-holiday calendars and the weekday / holiday day class.

EV sessions are grouped into two day classes, ``weekday`` (Monday to Friday and
not a day off) and ``holiday`` (Saturday, Sunday or a day off). A day off is a
public holiday or a weekday inside a period that most workplaces of that
country take off by custom (``customary_off_period``):
  JP: 29 Dec - 3 Jan (the closure in the Act on Holidays of Administrative
      Organs, 行政機関の休日に関する法律, Art. 1(1)(iii)), Golden Week
      (29 Apr - 6 May) and Obon (13 - 16 Aug);
  US: the Friday after Thanksgiving and 24 - 31 Dec;
  NO: Monday to Wednesday of Holy Week, the fellesferie (ISO weeks 28 - 30)
      and 24 - 31 Dec.
Only the Japanese year-end closure has a legal basis; the other periods follow
custom.
Each source dataset is classified with its own country's calendar; the service
day of the simulated market is classified with the Japanese calendar.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

WEEKDAY = "weekday"
HOLIDAY = "holiday"
DAY_CLASSES = (WEEKDAY, HOLIDAY)


def as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def us_federal_holidays(year: int) -> set[date]:
    """US federal holidays as observed (5 U.S.C. 6103); Juneteenth from 2021."""
    days = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _observed(date(year, 11, 11)),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2021:
        days.add(_observed(date(year, 6, 19)))
    return days


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def norway_public_holidays(year: int) -> set[date]:
    """Norwegian public holidays (lov om helligdager og helligdagsfred, and 1 and 17 May)."""
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        easter - timedelta(days=3),   # Maundy Thursday
        easter - timedelta(days=2),   # Good Friday
        easter,
        easter + timedelta(days=1),   # Easter Monday
        date(year, 5, 1),
        date(year, 5, 17),
        easter + timedelta(days=39),  # Ascension Day
        easter + timedelta(days=49),  # Whit Sunday
        easter + timedelta(days=50),  # Whit Monday
        date(year, 12, 25),
        date(year, 12, 26),
    }


# Japanese national holidays including substitute holidays, as published by the
# Cabinet Office (国民の祝日について). Only the years the service calendar uses.
_JAPAN_HOLIDAYS = {
    2023: (
        "2023-01-01", "2023-01-02", "2023-01-09", "2023-02-11", "2023-02-23",
        "2023-03-21", "2023-04-29", "2023-05-03", "2023-05-04", "2023-05-05",
        "2023-07-17", "2023-08-11", "2023-09-18", "2023-09-23", "2023-10-09",
        "2023-11-03", "2023-11-23",
    ),
    2024: (
        "2024-01-01", "2024-01-08", "2024-02-11", "2024-02-12", "2024-02-23",
        "2024-03-20", "2024-04-29", "2024-05-03", "2024-05-04", "2024-05-05",
        "2024-05-06", "2024-07-15", "2024-08-11", "2024-08-12", "2024-09-16",
        "2024-09-22", "2024-09-23", "2024-10-14", "2024-11-03", "2024-11-04",
        "2024-11-23",
    ),
    2025: (
        "2025-01-01", "2025-01-13", "2025-02-11", "2025-02-23", "2025-02-24",
        "2025-03-20", "2025-04-29", "2025-05-03", "2025-05-04", "2025-05-05",
        "2025-05-06", "2025-07-21", "2025-08-11", "2025-09-15", "2025-09-23",
        "2025-10-13", "2025-11-03", "2025-11-23", "2025-11-24",
    ),
}


def japan_public_holidays(year: int) -> set[date]:
    try:
        return {as_date(day) for day in _JAPAN_HOLIDAYS[int(year)]}
    except KeyError as exc:
        raise ValueError(f"no Japanese holiday calendar for {year}") from exc


_CALENDARS = {
    "US": us_federal_holidays,
    "NO": norway_public_holidays,
    "JP": japan_public_holidays,
}


def is_public_holiday(day, country: str) -> bool:
    d = as_date(day)
    return d in _CALENDARS[str(country)](d.year)


def customary_off_period(day, country: str) -> bool:
    """Inside a period most workplaces of ``country`` take off by custom."""
    d = as_date(day)
    country = str(country)
    if country == "JP":
        return (
            (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3)
            or (d.month == 4 and d.day >= 29) or (d.month == 5 and d.day <= 6)
            or (d.month == 8 and 13 <= d.day <= 16)
        )
    if country == "US":
        first = date(d.year, 11, 1)
        thanksgiving = first + timedelta(days=(3 - first.weekday()) % 7 + 21)
        return d == thanksgiving + timedelta(days=1) or (d.month == 12 and d.day >= 24)
    if country == "NO":
        easter = _easter_sunday(d.year)
        return (
            easter - timedelta(days=6) <= d <= easter - timedelta(days=4)
            or 28 <= d.isocalendar()[1] <= 30
            or (d.month == 12 and d.day >= 24)
        )
    return False


def is_day_off(day, country: str) -> bool:
    """A public holiday, or a day of a customary off period."""
    d = as_date(day)
    return is_public_holiday(d, country) or customary_off_period(d, country)


def day_class(day, country: str) -> str:
    """``weekday`` or ``holiday`` for one calendar date in ``country``."""
    d = as_date(day)
    if d.weekday() >= 5 or is_day_off(d, country):
        return HOLIDAY
    return WEEKDAY

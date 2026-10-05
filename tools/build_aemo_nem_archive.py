"""Build the legacy NEM schedule-deviation proxy library from the archive.

For source-faithful raw target waveform exports, use
``market.command_waveforms`` instead.

The daily reports reach back sixty days, which is one season and sixty
independent days.  The commands are partitioned by calendar day because
commands from the same day share the system conditions that produced them, so
sixty days is the sample size that matters.

The archive reaches 2009.  Its cost is the baseline: it keeps one pre-dispatch
run per period, issued about 29 minutes out, where the rolling runs reach the
hour that matches Japan's gate closure.  Measured over ten days the two agree
to within a tenth on every median statistic, so the archive is used for volume
and the sixty-day pool remains as the check that the lead did not decide
anything.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.nem_signal_pipeline import (
    ARCHIVE_BASELINE_LEAD,
    build_nem_scenario_library,
    download_nem_month,
)


DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "aemo" / "nem" / "archive_cache"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "aemo" / "nem" / "processed_5min" / "archive"
)


def _month(value: str) -> tuple[int, int]:
    try:
        year, month = value.split("-")
        return int(year), int(month)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("month must be YYYY-MM") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-month", type=_month, default=(2025, 8))
    parser.add_argument("--to-month", type=_month, default=(2026, 7),
                        help="inclusive")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-availability-mw", type=float, default=20.0)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def _months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    out, (year, month) = [], start
    while (year, month) <= end:
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def main() -> int:
    args = parse_args()
    days: list[date] = []
    for year, month in _months(args.from_month, args.to_month):
        started = time.perf_counter()
        fetched = download_nem_month(
            year, month, cache_dir=args.cache_dir, force=bool(args.force_download)
        )
        days.extend(fetched)
        print(
            f"[NEM] {year}-{month:02d}: {len(fetched)} days cached "
            f"in {time.perf_counter() - started:.0f}s",
            flush=True,
        )

    metadata = build_nem_scenario_library(
        days=sorted(set(days)),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        minimum_availability_mw=args.min_availability_mw,
        baseline_lead=ARCHIVE_BASELINE_LEAD,
        download=False,
    )
    print(
        f"[NEM] files={len(metadata['written_files'])} "
        f"units={len(metadata['units'])} days={len(set(days))} "
        f"skipped={len(metadata['days_skipped'])} "
        f"output={Path(args.output_dir).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the legacy AEMO schedule-deviation activation-proxy library."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.nem_signal_pipeline import build_nem_scenario_library


DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "aemo" / "nem" / "cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "aemo" / "nem" / "processed_5min" / "empirical"


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Next_Day_PreDispatch is kept for about sixty days, which is the binding
    # limit; Next_Day_Dispatch goes back over a year.
    parser.add_argument("--from-date", type=_date, default=date(2026, 7, 12))
    parser.add_argument("--to-date", type=_date, default=date(2026, 9, 10),
                        help="exclusive end date")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-availability-mw", type=float, default=20.0)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    days = [
        args.from_date + timedelta(days=offset)
        for offset in range((args.to_date - args.from_date).days)
    ]
    metadata = build_nem_scenario_library(
        days=days,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        minimum_availability_mw=args.min_availability_mw,
        force_download=bool(args.force_download),
    )
    for unit in metadata["units"]:
        print(
            f"[NEM] {unit['duid']}: {unit['days']} days  "
            f"active={unit['acceptance_frequency']:.1%}  "
            f"max_spell={unit['max_nonzero_spell_minutes']:.0f}min  "
            f"avail={unit['max_availability_mw']:.0f}MW",
            flush=True,
        )
    print(
        f"[NEM] files={len(metadata['written_files'])} "
        f"units={len(metadata['units'])} "
        f"skipped_days={len(metadata['days_skipped'])} "
        f"output={Path(args.output_dir).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the legacy fixed-width ERCOT activation-proxy library.

For source-faithful raw Base Point waveform exports, use
``tools.build_market_command_waveforms`` instead.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.ercot_signal_pipeline import build_ercot_scenario_library


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "ercot_local_downloader"
    / "ercot_ader_sced_output"
    / "ader_sced_dispatch.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ercot" / "sced" / "processed_5min" / "rtc_b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-csv", type=Path, action="append")
    inputs.add_argument("--input-dir", type=Path, help="Downloader output with manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resource",
        action="append",
        dest="resources",
        default=None,
        help="Optional resource restriction; default includes every downloaded resource",
    )
    parser.add_argument("--resource-kind", choices=("CLR", "ESR", "GEN"), default="CLR",
                        help="Kind for legacy CSVs without resource_kind")
    parser.add_argument("--train-end", type=date.fromisoformat)
    parser.add_argument("--validation-end", type=date.fromisoformat)
    parser.add_argument("--audit-only", action="store_true", help="Write QA metadata only, not scenario CSVs")
    parser.add_argument("--min-change-mw", type=float, default=1e-6,
                        help="Minimum within-segment BP change for an active day; MW, not bid capacity")
    args = parser.parse_args()
    if bool(args.train_end) != bool(args.validation_end):
        parser.error("Specify both --train-end and --validation-end")
    if not args.audit_only and not args.train_end:
        parser.error("Specify split dates, or use --audit-only to inspect segment lengths first")
    return args


def main() -> int:
    args = parse_args()
    inputs = args.input_csv or [DEFAULT_INPUT]
    if args.input_dir:
        manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("download_complete"):
            raise ValueError("Download not complete; resume the downloader before building a library")
        inputs = [args.input_dir / "documents" / item["file"]
                  for document in manifest["documents"].values()
                  for item in document["files"] if item["rows"]]
    metadata = build_ercot_scenario_library(
        input_csv=inputs,
        output_dir=args.output_dir,
        resource_names=args.resources,
        resource_kind=args.resource_kind,
        train_end=args.train_end,
        validation_end=args.validation_end,
        audit_only=args.audit_only,
        min_change_mw=args.min_change_mw,
    )
    for resource in metadata["resources"]:
        print(
            f"[ERCOT] {resource['resource_name']}: {resource['days']} active days  "
            f"level-active={resource['activation_frequency']:.1%}  "
            f"up={resource['up_frequency']:.1%}  "
            f"down={resource['down_frequency']:.1%}  "
            f"raw>|1|={resource['raw_above_unit_frequency']:.2%}  "
            f"raw-peak={resource['maximum_raw_activation_fraction']:.1%}",
            flush=True,
        )
    print(
        f"[ERCOT] files={len(metadata['written_files'])} "
        f"output={Path(args.output_dir).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

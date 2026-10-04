"""Build the configured blockwise bid banks for lower MARL training.

This command deliberately stops after the day-ahead bid banks are complete.
It does not start lower-controller training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BUILD_ONE_SPLIT = PROJECT_ROOT / "tools" / "build_training_bid_bank.py"


def _run_split(
    *,
    split: str,
    days: int,
    paired_train_days: int,
    paired_test_days: int,
    train_split_count: int,
    workers: int,
    output_dir: str | None,
    overwrite: bool,
) -> None:
    command = [
        sys.executable,
        str(BUILD_ONE_SPLIT),
        "--split",
        split,
        "--days",
        str(int(days)),
        "--train-split-count",
        str(int(train_split_count)),
        "--paired-train-days",
        str(int(paired_train_days)),
        "--paired-test-days",
        str(int(paired_test_days)),
        "--workers",
        str(int(workers)),
    ]
    if output_dir:
        command.extend(["--output-dir", str(Path(output_dir).resolve())])
    if overwrite:
        command.append("--overwrite")
    print(
        f"[all-bids] building split={split} days={days} workers={workers}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"upper-bid bank failed for split={split} with exit code "
            f"{completed.returncode}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split-count", type=int, default=25)
    parser.add_argument("--train-days", type=int, default=25)
    parser.add_argument("--test-days", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-output-dir", default=None)
    parser.add_argument("--test-output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--activation-signal-set",
        default=None,
        help="EnvConfig signal-set name, e.g. aemo_bess_dispatch.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate three disjoint 128-command pools and exit without solving bids.",
    )
    args = parser.parse_args()

    if int(args.train_days) <= 0 or int(args.test_days) <= 0:
        raise ValueError("--train-days and --test-days must both be positive")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    if args.activation_signal_set:
        os.environ["EVMA_ACTIVATION_SIGNAL_SET"] = str(args.activation_signal_set)

    # Import only after the optional signal-set override is installed.  The
    # check uses the same loader and uniqueness rule as the actual bidder.
    from EnvConfig import (
        LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS,
        LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,
    )
    from market.activation_scenarios import build_activation_scenario_set

    required = int(LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS)
    pools = {}
    for partition in ("forecast", "feedback", "holdout"):
        scenarios = build_activation_scenario_set(
            service_date="bid-bank-preflight",
            n_scenarios=required,
            seed=73000,
            proxy_shape_dir=LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,
            scenario_partition=partition,
            require_unique=True,
        )
        pools[partition] = {
            (str(row.source_date), str(row.source_bmu)) for row in scenarios
        }
        if len(pools[partition]) != required:
            raise RuntimeError(
                f"{partition} preflight selected {len(pools[partition])} distinct "
                f"commands; expected {required}"
            )
    for left, right in (
        ("forecast", "feedback"),
        ("forecast", "holdout"),
        ("feedback", "holdout"),
    ):
        overlap = pools[left].intersection(pools[right])
        if overlap:
            raise RuntimeError(
                f"Command partitions {left}/{right} overlap: {sorted(overlap)[:3]}"
            )
    print(
        f"[all-bids] preflight passed source={LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR} "
        f"distinct={required}x3",
        flush=True,
    )
    if args.preflight_only:
        print("[all-bids] preflight-only: no upper-bid bank was executed", flush=True)
        return 0

    _run_split(
        split="train",
        days=int(args.train_days),
        paired_train_days=int(args.train_days),
        paired_test_days=int(args.test_days),
        train_split_count=int(args.train_split_count),
        workers=int(args.workers),
        output_dir=args.train_output_dir,
        overwrite=bool(args.overwrite),
    )
    _run_split(
        split="test",
        days=int(args.test_days),
        paired_train_days=int(args.train_days),
        paired_test_days=int(args.test_days),
        train_split_count=int(args.train_split_count),
        workers=int(args.workers),
        output_dir=args.test_output_dir,
        overwrite=bool(args.overwrite),
    )
    print("[all-bids] configured train and validation bid banks are complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

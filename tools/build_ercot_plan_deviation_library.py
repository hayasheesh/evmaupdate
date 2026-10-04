"""Build the ERCOT ESR command library: SCED Base Point minus the COP plan.

Inputs are the ESR files kept by ``download_ercot_sced_waveforms.py`` and the
COP snapshots kept by ``download_ercot_cop_snapshots.py``. Calendar days in
ERCOT local time are split within each month 60/20/20. Each resource uses one
fixed width, its largest one-sided capability max(HSL, -LSL) while online.
Nothing is clipped; the bidder's input boundary clips to [0, 1].
This prepares inputs only; it never solves or writes an upper-bid bank.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.command_waveforms import month_balanced_partitions  # noqa: E402
from market.ercot_plan_deviation import (  # noqa: E402
    SOURCE_TYPE, day_activation, load_cop_planned_soc, unit_width_mw,
)
from market.ercot_signal_pipeline import _resource_inputs, build_ercot_resource_frame  # noqa: E402

DOWNLOADS = PROJECT_ROOT / "ercot_local_downloader"
DEFAULT_SCED_DIR = DOWNLOADS / "ercot_sced_rtc_output" / "documents"
DEFAULT_COP_DIR = DOWNLOADS / "ercot_cop_snapshot_output" / "documents"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ercot" / "sced" / "command_libraries" / "esr_plan_deviation_calendar_day"


def esr_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=200_000,
                                 usecols=lambda c: re.sub(r"[^a-z0-9]", "", c.lower()) == "resourcename"):
            names.update(chunk.iloc[:, 0].dropna().astype(str))
    return names


def build(*, sced_paths: list[Path], cop_paths: list[Path], output_dir: Path,
          overwrite: bool, required_unique_per_partition: int) -> dict:
    if not sced_paths or not cop_paths:
        raise ValueError("Both SCED ESR files and COP snapshot files are required")
    names = esr_names(sced_paths)
    plans = load_cop_planned_soc(cop_paths, names)
    plan_days = sorted({str(d.date()) for series in plans.values() for d in series.index})
    partitions = month_balanced_partitions(plan_days)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reasons: Counter = Counter()
    counts = {name: 0 for name in ("train", "validation", "test")}
    widths: dict[str, float] = {}
    beyond_width_steps = 0
    written = 0
    for _kind, name, rows in _resource_inputs(sced_paths, None, "ESR"):
        if name not in plans:
            reasons["no_cop_plan_for_resource"] += 1
            continue
        frame, _segments = build_ercot_resource_frame(rows=rows, resource_name=str(name), resource_kind="ESR")
        width = unit_width_mw(rows)
        widths[str(name)] = width
        for day, day_frame in frame.groupby("source_date", sort=True):
            if day not in partitions:
                reasons["day_outside_cop_range"] += 1
                continue
            activation, reason = day_activation(
                day_frame, plans[name], resource_name=str(name),
                width_mw=width, partition=partitions[day],
            )
            if activation is None:
                reasons[reason] += 1
                continue
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))
            activation.to_csv(output_dir / f"ercot_esr_plan_{safe}_{day}.csv", index=False, float_format="%.12g")
            beyond_width_steps += int(np.sum(np.abs(activation["signed_activation_up_positive_raw"]) > 1.0))
            counts[partitions[day]] += 1
            written += 1
    bank_ready = all(counts[p] >= required_unique_per_partition for p in counts)
    metadata = {
        "build_complete": True,
        "bank_ready": bool(bank_ready),
        "regime": SOURCE_TYPE,
        "source_kind": "ERCOT ESR SCED Base Point minus the QSE's COP plan one hour before each hour",
        "plan": "NP1-301 COP Adjustment Period Snapshot; planned output = Hour Beginning Planned SOC(h) - SOC(h+1)",
        "sampling": "latest SCED execution in (t-5min, t]; stale points, DST days and offline steps excluded",
        "calendar_window": "America/Chicago local day, 288 five-minute points",
        "reference": "zero deviation from the COP plan",
        "normalization": "one fixed width per resource: largest max(HSL, -LSL) while online",
        "clipping": False,
        "interpolation_or_fill": False,
        "partition_policy": "whole calendar dates; within each calendar month consecutive 60/20/20 blocks",
        "partition_counts": counts,
        "written_file_count": written,
        "excluded_resource_days": dict(reasons),
        "steps_beyond_unit_width": beyond_width_steps,
        "unit_width_mw": widths,
        "required_unique_scenarios_per_partition": int(required_unique_per_partition),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sced-dir", type=Path, default=DEFAULT_SCED_DIR)
    parser.add_argument("--cop-dir", type=Path, default=DEFAULT_COP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--required-unique-per-partition", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sced_paths = sorted(args.sced_dir.glob("*_ESR_*.csv")) or sorted(args.sced_dir.glob("*ESR*.csv"))
    cop_paths = sorted(args.cop_dir.glob("*.csv"))
    metadata = build(
        sced_paths=sced_paths, cop_paths=cop_paths, output_dir=args.output_dir,
        overwrite=args.overwrite, required_unique_per_partition=args.required_unique_per_partition,
    )
    print(f"[ercot-plan-library] output={args.output_dir.resolve()}")
    print(f"[ercot-plan-library] partition_counts={metadata['partition_counts']}")
    print(f"[ercot-plan-library] excluded={metadata['excluded_resource_days']}")
    print(f"[ercot-plan-library] bank_ready={metadata['bank_ready']}")
    print("[ercot-plan-library] upper-bid bank was not executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build bidder-ready AEMO command-shape libraries from native targets.

Calendar days are split within each month 60/20/20, so every partition holds
the same months. This prepares inputs only; it never solves or writes an
upper-bid bank.

``plan_deviation`` is the command library. In Secondary Reserve 2 the operator
sends a command amount that is added to the resource's own submitted baseline.
The AEMO counterpart of that amount is a storage unit's dispatch target minus
the pre-dispatch output AEMO projected for it from its bids (AEMO has no
participant-submitted MW plan), so the library keeps exactly that difference in MW
and divides it by one fixed width per unit (the unit's largest availability in
the source data). Nothing time-varying divides it and nothing clips it; the
bidder's input boundary clips to [0, 1].

``bess`` keeps the targets themselves with 0 MW as the reference: an idle
battery. A battery's own charge/discharge plan is then part of the waveform, so
it is an explanation set, not a command library.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
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

from market.command_waveforms import (
    build_command_waveform_day,
    month_balanced_partitions,
    select_end_stamped_calendar_day,
    waveform_to_activation_proxy,
)


DEFAULT_BESS_DIR = PROJECT_ROOT / "data" / "aemo" / "nem" / "processed_5min" / "dense8"
DEFAULT_PLAN_DIR = PROJECT_ROOT / "data" / "aemo" / "nem" / "processed_5min" / "archive"
DEFAULT_PLAN_OUTPUT = (
    PROJECT_ROOT / "data" / "aemo" / "nem" / "command_libraries"
    / "plan_deviation_calendar_day"
)
DEFAULT_BESS_OUTPUT = (
    PROJECT_ROOT / "data" / "aemo" / "nem" / "command_libraries"
    / "bess_dispatch_calendar_day"
)
FILE_PATTERN = re.compile(r"^nem_(?P<resource>.+)_(?P<day>\d{4}-\d{2}-\d{2})\.csv$")


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
        if existing:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_metadata(output_dir: Path, metadata: dict) -> None:
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _bess_index(source_dir: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for path in sorted(source_dir.glob("*.csv")):
        match = FILE_PATTERN.match(path.name)
        if match:
            result[(match.group("resource"), match.group("day"))] = path
    return result


def _load_bess_calendar_day(
    index: dict[tuple[str, str], Path], resource: str, calendar_day: str
) -> pd.DataFrame:
    current = date.fromisoformat(calendar_day)
    parts = []
    for partition_day in (
        (current - timedelta(days=1)).isoformat(), current.isoformat()
    ):
        path = index[(resource, partition_day)]
        part = pd.read_csv(path)
        if "source_date" not in part or not part["source_date"].astype(str).eq(partition_day).all():
            raise ValueError(f"Unexpected archive partition label: {path}")
        if "source_bmu" not in part or not part["source_bmu"].astype(str).eq(resource).all():
            raise ValueError(f"Unexpected BESS resource label: {path}")
        parts.append(part)
    selected = select_end_stamped_calendar_day(
        pd.concat(parts, ignore_index=True), timestamp_column="time_nem",
        day=calendar_day, step_column="step",
    )
    return build_command_waveform_day(
        selected, market="AEMO NEM", resource_kind="BESS",
        resource_name=resource, source_date=calendar_day, step_column="step",
        target_column="dispatch_target_mw", direction_sign=1.0,
        local_time_column="time_nem", utc_time_column="time_utc",
    )


def _load_plan_calendar_day(
    index: dict[tuple[str, str], Path], resource: str, calendar_day: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the deviation waveform and the plan/target rows it came from."""
    current = date.fromisoformat(calendar_day)
    parts = []
    for partition_day in (
        (current - timedelta(days=1)).isoformat(), current.isoformat()
    ):
        path = index[(resource, partition_day)]
        part = pd.read_csv(path)
        if not part["source_date"].astype(str).eq(partition_day).all():
            raise ValueError(f"Unexpected archive partition label: {path}")
        if not part["source_bmu"].astype(str).eq(resource).all():
            raise ValueError(f"Unexpected resource label: {path}")
        parts.append(part)
    selected = select_end_stamped_calendar_day(
        pd.concat(parts, ignore_index=True), timestamp_column="time_nem",
        day=calendar_day, step_column="step",
    )
    deviation = (
        pd.to_numeric(selected["dispatch_target_mw"], errors="coerce")
        - pd.to_numeric(selected["predispatch_mw"], errors="coerce")
    )
    if not np.allclose(deviation, selected["delta_mw"], rtol=0.0, atol=1e-6):
        raise ValueError(
            f"delta_mw is not dispatch target minus plan for {resource} {calendar_day}"
        )
    waveform = build_command_waveform_day(
        selected, market="AEMO NEM", resource_kind="BESS_PLAN_DEVIATION",
        resource_name=resource, source_date=calendar_day, step_column="step",
        target_column="delta_mw", direction_sign=1.0,
        local_time_column="time_nem", utc_time_column="time_utc",
    )
    return waveform, selected.sort_values("step", kind="stable").reset_index(drop=True)


def build_plan_deviation_library(
    *, source_dir: Path, output_dir: Path, overwrite: bool = False,
    required_unique_per_partition: int = 128,
) -> dict:
    source_metadata = json.loads(
        (source_dir / "metadata.json").read_text(encoding="utf-8")
    )
    widths = {
        str(unit["duid"]): float(unit["max_availability_mw"])
        for unit in source_metadata["units"]
    }
    index = _bess_index(source_dir)
    candidates = []
    for resource, day in sorted(index):
        previous = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        if (resource, previous) in index:
            candidates.append((resource, day))
    if not candidates:
        raise RuntimeError(f"No stitchable calendar days found in {source_dir}")
    partitions = month_balanced_partitions([day for _, day in candidates])
    _prepare_output(output_dir, overwrite)
    metadata = {
        "build_complete": False,
        "bank_ready": False,
        "regime": "aemo_storage_dispatch_minus_own_plan_calendar_day_unit_width",
        "source_kind": "AEMO storage TOTALCLEARED minus AEMO's PREDISPATCH projection for the unit",
        "source_directory": str(source_dir.resolve()),
        "plan": source_metadata.get("source", ""),
        "calendar_window": "AEST interval ends 00:05 through next-day 00:00",
        "reference": "zero deviation from the PREDISPATCH projection",
        "normalization": "one fixed width per unit: its largest AVAILABILITY in the source data",
        "clipping": False,
        "interpolation_or_fill": False,
        "partition_policy": "whole calendar dates; within each calendar month consecutive 60/20/20 blocks",
        "required_unique_scenarios_per_partition": int(required_unique_per_partition),
        "unit_width_mw": widths,
    }
    _write_metadata(output_dir, metadata)
    written: list[str] = []
    rejected: list[dict[str, str]] = []
    counts = {name: 0 for name in ("train", "validation", "test")}
    date_ranges: dict[str, list[str]] = {name: [] for name in counts}
    beyond_width_steps = 0
    for resource, day in candidates:
        partition = partitions[day]
        width = widths.get(resource, 0.0)
        try:
            if not np.isfinite(width) or width <= 0.0:
                raise ValueError(f"no positive unit width for {resource}")
            waveform, source_rows = _load_plan_calendar_day(index, resource, day)
            activation = waveform_to_activation_proxy(
                waveform, reference_mode="zero",
                source_type="aemo_storage_plan_deviation_shape",
                scenario_partition=partition,
                segment_id=f"aemo_plan:{resource}:{day}",
                scale_mw=width,
            )
        except (KeyError, OSError, ValueError) as exc:
            rejected.append({"resource": resource, "date": day, "reason": str(exc)})
            continue
        activation["plan_mw"] = source_rows["predispatch_mw"].to_numpy(float)
        activation["plan_issued_local"] = (
            source_rows["predispatch_issued"].astype(str).to_numpy()
        )
        activation["dispatch_target_mw"] = source_rows["dispatch_target_mw"].to_numpy(float)
        activation["availability_mw"] = source_rows["availability_mw"].to_numpy(float)
        beyond_width_steps += int(
            np.sum(np.abs(activation["signed_activation_up_positive_raw"]) > 1.0)
        )
        filename = f"aemo_plan_{resource}_{day}.csv"
        activation.to_csv(output_dir / filename, index=False, float_format="%.12g")
        written.append(filename)
        counts[partition] += 1
        date_ranges[partition].append(day)

    bank_ready = all(
        counts[name] >= int(required_unique_per_partition)
        for name in ("train", "validation", "test")
    )
    metadata.update({
        "build_complete": True,
        "bank_ready": bool(bank_ready),
        "resources": sorted({resource for resource, _ in candidates}),
        "written_file_count": len(written),
        "rejected_file_count": len(rejected),
        "partition_counts": counts,
        "partition_date_ranges": {
            name: ([min(days), max(days)] if days else [])
            for name, days in date_ranges.items()
        },
        "steps_beyond_unit_width": int(beyond_width_steps),
        "rejected": rejected,
    })
    _write_metadata(output_dir, metadata)
    if not bank_ready:
        raise RuntimeError(
            "Library was built but is not bank-ready: each partition "
            f"needs {required_unique_per_partition} unique scenarios; found {counts}"
        )
    return metadata


def build_bess_library(
    *, source_dir: Path, output_dir: Path, overwrite: bool = False,
    required_unique_per_partition: int = 128,
) -> dict:
    index = _bess_index(source_dir)
    resources = sorted({resource for resource, _ in index})
    candidates = []
    for resource in resources:
        resource_days = {day for found_resource, day in index if found_resource == resource}
        for day in sorted(resource_days):
            previous = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
            if (resource, previous) in index:
                candidates.append((resource, day))
    if not candidates:
        raise RuntimeError(f"No stitchable BESS calendar days found in {source_dir}")
    partitions = month_balanced_partitions([day for _, day in candidates])
    _prepare_output(output_dir, overwrite)
    metadata = {
        "build_complete": False,
        "bank_ready": False,
        "regime": "aemo_bess_zero_based_target_calendar_day_fixed_scale",
        "source_kind": "AEMO BESS TOTALCLEARED",
        "source_directory": str(source_dir.resolve()),
        "calendar_window": "AEST interval ends 00:05 through next-day 00:00",
        "reference": "physical zero (idle battery)",
        "normalization": "one max(abs(target)) scale per scenario",
        "clipping": False,
        "interpolation_or_fill": False,
        "partition_policy": "whole calendar dates; within each calendar month consecutive 60/20/20 blocks",
        "required_unique_scenarios_per_partition": int(required_unique_per_partition),
    }
    _write_metadata(output_dir, metadata)
    written: list[str] = []
    rejected: list[dict[str, str]] = []
    counts = {name: 0 for name in ("train", "validation", "test")}
    date_ranges: dict[str, list[str]] = {name: [] for name in counts}
    for resource, day in candidates:
        partition = partitions[day]
        try:
            waveform = _load_bess_calendar_day(index, resource, day)
            activation = waveform_to_activation_proxy(
                waveform, reference_mode="zero",
                source_type="aemo_bess_dispatch_target_shape",
                scenario_partition=partition,
                segment_id=f"aemo_bess:{resource}:{day}",
            )
        except (KeyError, OSError, ValueError) as exc:
            rejected.append({"resource": resource, "date": day, "reason": str(exc)})
            continue
        filename = f"aemo_bess_{resource}_{day}.csv"
        activation.to_csv(output_dir / filename, index=False, float_format="%.12g")
        written.append(filename)
        counts[partition] += 1
        date_ranges[partition].append(day)

    bank_ready = all(
        counts[name] >= int(required_unique_per_partition)
        for name in ("train", "validation", "test")
    )
    metadata.update({
        "build_complete": True,
        "bank_ready": bool(bank_ready),
        "resources": resources,
        "written_file_count": len(written),
        "rejected_file_count": len(rejected),
        "partition_counts": counts,
        "partition_date_ranges": {
            name: ([min(days), max(days)] if days else [])
            for name, days in date_ranges.items()
        },
        "rejected": rejected,
    })
    _write_metadata(output_dir, metadata)
    if not bank_ready:
        raise RuntimeError(
            "Library was built but is not bank-ready: each partition "
            f"needs {required_unique_per_partition} unique scenarios; found {counts}"
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("plan_deviation", "bess"), default="plan_deviation"
    )
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--bess-dir", type=Path, default=DEFAULT_BESS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--required-unique-per-partition", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.required_unique_per_partition <= 0:
        raise ValueError("--required-unique-per-partition must be positive")
    output_dir = args.output_dir or {
        "plan_deviation": DEFAULT_PLAN_OUTPUT,
        "bess": DEFAULT_BESS_OUTPUT,
    }[args.source]
    if args.source == "plan_deviation":
        metadata = build_plan_deviation_library(
            source_dir=args.plan_dir, output_dir=output_dir,
            overwrite=args.overwrite,
            required_unique_per_partition=args.required_unique_per_partition,
        )
    else:
        metadata = build_bess_library(
            source_dir=args.bess_dir, output_dir=output_dir,
            overwrite=args.overwrite,
            required_unique_per_partition=args.required_unique_per_partition,
        )
    print(f"[aemo-command-library] output={output_dir.resolve()}")
    print(f"[aemo-command-library] partition_counts={metadata['partition_counts']}")
    print(f"[aemo-command-library] bank_ready={metadata['bank_ready']}")
    print("[aemo-command-library] upper-bid bank was not executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rebuild lossless raw-target waveforms and a like-window comparison figure."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.command_waveforms import (
    build_command_waveform_day,
    select_end_stamped_calendar_day,
)


DEFAULT_AEMO_DIR = PROJECT_ROOT / "data" / "aemo" / "nem" / "processed_5min" / "archive"
DEFAULT_ERCOT_CSV = (
    PROJECT_ROOT / "execute_results" / "ercot_waveforms_ar_ald1_20260923"
    / "AR_ALD1_three_actual_days.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "execute_results" / "market_command_waveforms_raw_v2_20260923"
DEFAULT_DAYS = ("2026-07-22", "2026-07-23", "2026-07-24")


def _aemo_day(directory: Path, day: str, duid: str) -> pd.DataFrame:
    calendar_day = date.fromisoformat(day)
    partitions = (calendar_day - timedelta(days=1), calendar_day)
    parts = []
    for partition_day in partitions:
        path = directory / f"nem_{duid}_{partition_day.isoformat()}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"AEMO calendar day {day} needs adjacent archive segment: {path}"
            )
        part = pd.read_csv(path)
        partition_label = partition_day.isoformat()
        if "source_date" not in part or not part["source_date"].astype(str).eq(
            partition_label
        ).all():
            raise ValueError(f"AEMO archive segment has unexpected source_date: {path}")
        if "source_bmu" not in part or not part["source_bmu"].astype(str).eq(duid).all():
            raise ValueError(f"AEMO source does not contain only {duid}: {path}")
        part["source_archive_date"] = part["source_date"].astype(str)
        parts.append(part)

    source = select_end_stamped_calendar_day(
        pd.concat(parts, ignore_index=True),
        timestamp_column="time_nem",
        day=day,
        step_column="step",
    )
    result = build_command_waveform_day(
        source,
        market="AEMO NEM",
        resource_kind="BESS",
        resource_name=duid,
        source_date=day,
        step_column="step",
        target_column="dispatch_target_mw",
        direction_sign=1.0,
        local_time_column="time_nem",
        utc_time_column="time_utc",
        availability_column="availability_mw",
    )
    result["source_archive_date"] = source["source_archive_date"].to_numpy()
    return result


def _ercot_days(path: Path, days: list[str], resource: str) -> list[pd.DataFrame]:
    source = pd.read_csv(path)
    if "source_type" not in source or not source["source_type"].astype(str).str.startswith(
        "ercot_sced_relative_fixed_width_causal_v2"
    ).all():
        raise ValueError("ERCOT input must be the causal NP3 SCED sample, not the legacy ADER bank")
    if not source["source_bmu"].astype(str).eq(resource).all():
        raise ValueError(f"ERCOT sample does not contain only {resource}")
    frames = []
    for day in days:
        daily = source.loc[source["source_date"].astype(str) == day].copy()
        if "step" not in daily:
            order_column = "grid_utc" if "grid_utc" in daily else "time_ercot"
            daily = daily.sort_values(order_column, kind="stable").reset_index(drop=True)
            daily["step"] = range(len(daily))
        frames.append(build_command_waveform_day(
            daily,
            market="ERCOT",
            resource_kind="CLR",
            resource_name=resource,
            source_date=day,
            step_column="step",
            target_column="base_point_mw",
            direction_sign=-1.0,
            local_time_column="time_ercot",
            utc_time_column="grid_utc",
            execution_time_column="source_sced_timestamp_utc",
            status_column="tel_res_status",
            online_column="online",
            availability_column="response_span_mw",
        ))
    return frames


def _plot(waveforms: pd.DataFrame, aemo_resource: str, ercot_resource: str,
          days: list[str], path: Path) -> None:
    colors = {day: color for day, color in zip(days, ("#2563eb", "#d97706", "#16845b"))}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    specs = [
        ("AEMO NEM", aemo_resource, 0, "AEST calendar day: TOTALCLEARED stamped 00:05–24:00"),
        ("ERCOT", ercot_resource, 1, "ERCOT local day: SCED Base Point sampled 00:00–23:55"),
    ]
    for market, resource, col, note in specs:
        market_rows = waveforms.loc[waveforms.market == market]
        target_ax, relative_ax = axes[0, col], axes[1, col]
        for day in days:
            daily = market_rows.loc[market_rows.source_date == day].sort_values("step")
            interval_end_offset = 1 if market == "AEMO NEM" else 0
            x_hours = (daily.step.to_numpy(float) + interval_end_offset) * 5.0 / 60.0
            color = colors[day]
            target_ax.plot(x_hours, daily.target_mw.to_numpy(float), color=color,
                           linestyle="none", marker=".", markersize=2.0, label=day)
            relative_ax.plot(x_hours, daily.relative_up_positive_mw.to_numpy(float), color=color,
                             linestyle="none", marker=".", markersize=2.0, label=day)
        target_ax.set_title(f"{market} — {resource}: source target")
        relative_ax.set_title(f"{market} — {resource}: fixed-offset-relative shape")
        target_ax.set_ylabel("Native target (MW)")
        relative_ax.set_ylabel("Change from first target, up-positive (MW)")
        target_ax.set_xlabel("Local clock time (hours)")
        relative_ax.set_xlabel("Local clock time (hours)")
        target_ax.grid(True, alpha=0.22)
        relative_ax.grid(True, alpha=0.22)
        relative_ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.65)
        target_ax.legend(title="Source date", fontsize=9, frameon=False)
        target_ax.text(0.01, 0.02, note, transform=target_ax.transAxes,
                       fontsize=8, va="bottom", color="#444444")
        target_ax.set_xlim(0, 24)
        relative_ax.set_xlim(0, 24)
    fig.suptitle("Raw market targets and fixed-offset-relative samples (no scaling or clipping)",
                 fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aemo-dir", type=Path, default=DEFAULT_AEMO_DIR)
    parser.add_argument("--aemo-resource", default="BBATTERY1")
    parser.add_argument("--ercot-csv", type=Path, default=DEFAULT_ERCOT_CSV)
    parser.add_argument("--ercot-resource", default="AR_ALD1")
    parser.add_argument("--days", nargs="+", default=list(DEFAULT_DAYS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    days = [date.fromisoformat(value).isoformat() for value in args.days]
    outputs = []
    for day in days:
        outputs.append(_aemo_day(args.aemo_dir, day, args.aemo_resource))
    outputs.extend(_ercot_days(args.ercot_csv, days, args.ercot_resource))
    waveforms = pd.concat(outputs, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "raw_command_waveforms.csv"
    figure_path = args.output_dir / "market_command_waveforms.png"
    metadata_path = args.output_dir / "metadata.json"
    if csv_path.exists() or figure_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output in {args.output_dir}")
    waveforms.to_csv(csv_path, index=False, float_format="%.12g")
    _plot(waveforms, args.aemo_resource, args.ercot_resource, days, figure_path)
    metadata = {
        "method": "raw_dispatch_target_plus_constant_offset_relative_view",
        "scenario_reference": "first target in each complete local calendar-day window",
        "transformations": {
            "raw_target_mw": "copied from market target without modification",
            "relative_up_positive_mw": "direction_sign * (target_mw - first target_mw)",
            "delta_up_positive_mw": "first difference in native MW; first value is missing",
            "predispatch_subtraction": False,
            "availability_or_width_division": False,
            "clipping": False,
            "interpolation": False,
        },
        "sign_convention": {
            "AEMO BESS": "positive means increasing net injection",
            "ERCOT CLR": "positive means reduced MW consumption",
        },
        "source_time_conventions": {
            "AEMO NEM": "calendar date, 00:05 through next-date 00:00 (24:00), interval-end TOTALCLEARED; stitched across 04:05 archive partitions",
            "ERCOT": "causally sampled SCED Base Point on a five-minute local grid",
        },
        "days": days,
        "aemo": {
            "resource": args.aemo_resource,
            "input_directory": str(args.aemo_dir.resolve()),
            "source_column": "dispatch_target_mw",
        },
        "ercot": {
            "resource": args.ercot_resource,
            "input_csv": str(args.ercot_csv.resolve()),
            "source_column": "base_point_mw",
            "resource_kind": "CLR",
        },
        "outputs": [csv_path.name, figure_path.name],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[waveforms] samples={len(waveforms)} csv={csv_path.resolve()}")
    print(f"[waveforms] figure={figure_path.resolve()}")
    print(f"[waveforms] metadata={metadata_path.resolve()}")
    print("[waveforms] native targets preserved; no predispatch subtraction, scaling, clipping, or interpolation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot AEMO BESS and WDRU targets on matched interval-end calendar days."""

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
    aggregate_calendar_day_targets,
    build_command_waveform_day,
)
from tools.build_market_command_waveforms import _aemo_day


DEFAULT_BESS_DIR = (
    PROJECT_ROOT / "data" / "aemo" / "nem" / "processed_5min" / "dense8"
)
DEFAULT_WDRU_CSV = (
    PROJECT_ROOT / "data" / "aemo" / "nem" / "wdru_dispatch" / "2025Q2"
    / "wdru_q2_2025_event_day_targets.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "execute_results" / "aemo_bess_wdru_calendar_day_20260923"
DEFAULT_BESS_DAY = "2025-12-18"
DEFAULT_WDRU_DAY = "2025-06-12"
DEFAULT_BESS_RESOURCES = (
    "BBATTERY1", "BRNDBES1", "HPR1", "MREHA1",
    "ULPBESS1", "VBB1", "WDBESS1", "WTAHB1",
)


def _wdr_day(path: Path, day: str) -> pd.DataFrame:
    source = pd.read_csv(path)
    aggregate = aggregate_calendar_day_targets(
        source,
        timestamp_column="settlementdate_aest",
        resource_column="duid",
        target_column="totalcleared_mw",
        day=day,
    )
    result = build_command_waveform_day(
        aggregate,
        market="AEMO NEM",
        resource_kind="WDRU",
        resource_name="WDRU aggregate",
        source_date=day,
        step_column="calendar_step",
        target_column="aggregate_target_mw",
        direction_sign=1.0,
        local_time_column="calendar_time",
    )
    result["source_resource_count"] = aggregate["source_resource_count"].to_numpy(int)
    result["active_resource_count"] = aggregate["active_resource_count"].to_numpy(int)
    return result


def _plot(waveforms: pd.DataFrame, bess_day: str, wdr_day: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    bess = waveforms.loc[
        (waveforms.resource_kind == "BESS") & (waveforms.source_date == bess_day)
    ]
    wdr = waveforms.loc[
        (waveforms.resource_kind == "WDRU") & (waveforms.source_date == wdr_day)
    ].sort_values("step")
    palette = plt.get_cmap("tab10")

    target_ax, relative_ax = axes[0, 0], axes[1, 0]
    for index, (resource, daily) in enumerate(bess.groupby("resource_name", sort=True)):
        daily = daily.sort_values("step")
        x_hours = (daily.step.to_numpy(float) + 1.0) * 5.0 / 60.0
        color = palette(index % 10)
        target_ax.plot(x_hours, daily.target_mw, color=color, linewidth=1.0, label=resource)
        relative_ax.plot(
            x_hours, daily.relative_up_positive_mw, color=color, linewidth=1.0, label=resource
        )
    target_ax.set_title(f"BESS — {bess_day}: resource targets")
    relative_ax.set_title(f"BESS — {bess_day}: change from 00:05 target")
    target_ax.text(
        0.01, 0.02, "AEST calendar day; interval-ending samples 00:05–24:00",
        transform=target_ax.transAxes, fontsize=8, va="bottom", color="#444444",
    )
    target_ax.legend(title="DUID", ncol=2, fontsize=8, frameon=False)
    relative_ax.legend(title="DUID", ncol=2, fontsize=8, frameon=False)

    target_ax, relative_ax = axes[0, 1], axes[1, 1]
    x_hours = (wdr.step.to_numpy(float) + 1.0) * 5.0 / 60.0
    target_ax.plot(x_hours, wdr.target_mw, color="#d97706", linewidth=1.4)
    relative_ax.plot(x_hours, wdr.relative_up_positive_mw, color="#d97706", linewidth=1.4)
    target_ax.set_title(f"WDRU — {wdr_day}: aggregate reduction target")
    relative_ax.set_title(f"WDRU — {wdr_day}: change from 00:05 target")
    target_ax.text(
        0.01, 0.02,
        f"AEST calendar day; {int(wdr.source_resource_count.iloc[0])} complete candidate resources",
        transform=target_ax.transAxes, fontsize=8, va="bottom", color="#444444",
    )

    for row_index, row in enumerate(axes):
        for ax in row:
            ylabel = "Native target (MW)" if row_index == 0 else "Change from first target (MW)"
            ax.set_ylabel(ylabel)
            ax.set_xlabel("AEST clock time (hours)")
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 4))
            ax.grid(True, alpha=0.22)
    for ax in axes[1]:
        ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.65)

    fig.suptitle("AEMO BESS and WDRU targets on matched calendar-day windows", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bess-dir", type=Path, default=DEFAULT_BESS_DIR)
    parser.add_argument("--wdr-csv", type=Path, default=DEFAULT_WDRU_CSV)
    parser.add_argument("--bess-day", default=DEFAULT_BESS_DAY)
    parser.add_argument("--wdr-day", default=DEFAULT_WDRU_DAY)
    parser.add_argument("--bess-resources", nargs="+", default=list(DEFAULT_BESS_RESOURCES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bess_day = date.fromisoformat(args.bess_day).isoformat()
    wdr_day = date.fromisoformat(args.wdr_day).isoformat()
    bess_frames = [
        _aemo_day(args.bess_dir, bess_day, resource)
        for resource in args.bess_resources
    ]
    wdr = _wdr_day(args.wdr_csv, wdr_day)
    waveforms = pd.concat([*bess_frames, wdr], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "aemo_bess_wdru_calendar_waveforms.csv"
    figure_path = args.output_dir / "aemo_bess_wdru_calendar_waveforms.png"
    metadata_path = args.output_dir / "metadata.json"
    if csv_path.exists() or figure_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output in {args.output_dir}")
    waveforms.to_csv(csv_path, index=False, float_format="%.12g")
    _plot(waveforms, bess_day, wdr_day, figure_path)

    metadata = {
        "method": "native_interval_end_targets_and_constant_offset_relative_curves",
        "calendar_window": "AEST 00:05 on source date through 00:00 on the next date (24:00), 288 interval-ending samples",
        "transformations": {
            "BESS target": "raw dispatch_target_mw retained in native MW",
            "WDRU target": "sum totalcleared_mw across a complete, fixed candidate-resource panel at each timestamp",
            "relative_curve": "direction_sign * (target_mw - first target_mw)",
            "scaling": False,
            "clipping": False,
            "interpolation_or_zero_fill": False,
        },
        "bess": {
            "source_directory": str(args.bess_dir.resolve()),
            "calendar_date": bess_day,
            "resources": args.bess_resources,
            "input_archive_segments": [
                (date.fromisoformat(bess_day) - timedelta(days=1)).isoformat(),
                bess_day,
            ],
        },
        "wdru": {
            "source_csv": str(args.wdr_csv.resolve()),
            "calendar_date": wdr_day,
            "candidate_resources": int(wdr.source_resource_count.iloc[0]),
            "maximum_active_resources": int(wdr.active_resource_count.max()),
        },
        "outputs": [csv_path.name, figure_path.name],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[waveforms] samples={len(waveforms)} csv={csv_path.resolve()}")
    print(f"[waveforms] figure={figure_path.resolve()}")
    print(f"[waveforms] metadata={metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

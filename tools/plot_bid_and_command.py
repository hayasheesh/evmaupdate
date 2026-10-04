"""保存済みの固定入札と未見指令を重ねて描画する。環境や最適化は実行しない。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from market.bid_participation import participation_by_block
from market.physical_lp_bidding.data_classes import BiddingLPConfig
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
from training.bid_bank import load_fixed_bid
from training.lower_bid_training import _bid_to_target_tol_from_activation

BLUE = "#3779ad"
COMMAND = "#cb4b28"
UP = "#28796c"
DOWN = "#ab761b"


def step_values(values):
    values = np.asarray(values)
    return np.r_[values, values[-1]]


def load_day(bank: Path, entry: dict, command_index: int, split: str) -> dict:
    path = bank / entry["bid_path"]
    fixed = load_fixed_bid(path)
    command = fixed["activation_scenario_payload"][command_index]
    identity = lambda row: (str(row["source_date"]), str(row["source_bmu"]))
    assert identity(command) not in {
        identity(row) for row in fixed["design_activation_scenario_payload"]
    }
    baseline = np.asarray(fixed["baseline_plan"], dtype=float)
    up = np.asarray(fixed["up_plan"], dtype=float)
    down = np.asarray(fixed["down_plan"], dtype=float)
    target, _, _ = _bid_to_target_tol_from_activation(
        baseline, up, down, command["up_proxy"], command["down_proxy"],
        band_fraction=float(fixed["assessment_band_fraction"]),
    )
    cfg = BiddingLPConfig(
        assessment_band_fraction=float(fixed["assessment_band_fraction"]),
        apply_transition_band=bool(fixed["apply_transition_band"]),
    )
    check_target, _, _, _ = fixed_bid_tracking_bands(
        cfg, baseline, up, down,
        np.asarray(command["up_proxy"]), np.asarray(command["down_proxy"]),
        apply_transition_band=cfg.apply_transition_band,
    )
    np.testing.assert_allclose(target, check_target, atol=1e-3, rtol=1e-6)
    assert baseline.shape == up.shape == down.shape == (48,)
    assert target.shape == (288,)
    active = np.repeat(participation_by_block(up, down), 6)
    center = np.repeat(baseline, 6)
    lower = center - np.repeat(up, 6)
    upper = center + np.repeat(down, 6)
    assert np.all(target[active] >= lower[active] - 1e-3)
    assert np.all(target[active] <= upper[active] + 1e-3)
    return {
        "date": entry["service_date"], "split": split,
        "bid_path": str(path.resolve()),
        "bid_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "command_index": command_index, "command_count": len(fixed["activation_scenario_payload"]),
        "source_date": command["source_date"], "source_resource": command["source_bmu"],
        "source": command["source"], "design_count": len(fixed["design_activation_scenario_payload"]),
        "baseline": baseline, "up": up, "down": down,
        "center": center, "lower": lower, "upper": upper,
        "target": target, "active": active,
    }


def shade_inactive(ax, day):
    block_active = day["active"].reshape(48, 6).any(axis=1)
    for block in np.flatnonzero(~block_active):
        ax.axvspan(block / 2, (block + 1) / 2, color="#eef0f2", zorder=0)


def draw_power(ax, day, *, legend=False):
    x = np.arange(289) / 12
    active = step_values(day["active"])
    lower = np.where(active, step_values(day["lower"]), np.nan)
    upper = np.where(active, step_values(day["upper"]), np.nan)
    center = np.where(active, step_values(day["center"]), np.nan)
    target = np.where(active, step_values(day["target"]), np.nan)
    shade_inactive(ax, day)
    ax.fill_between(x, lower, upper, step="post", color=BLUE, alpha=.16,
                    label="入札から定まる指令の範囲", zorder=1)
    ax.step(x, lower, where="post", color=BLUE, linewidth=.7, alpha=.7)
    ax.step(x, upper, where="post", color=BLUE, linewidth=.7, alpha=.7)
    ax.step(x, center, where="post", color=BLUE, linestyle="--", linewidth=1.4,
            label="基準出力", zorder=2)
    ax.step(x, target, where="post", color=COMMAND, linewidth=1.7,
            label="当日の指令", zorder=3)
    ax.axhline(0, color="#87909a", linewidth=.7, zorder=0)
    ax.grid(axis="y", color="#dfe3e7", linewidth=.65)
    ax.set_xlim(0, 24)
    ax.set_ylabel("電力 [kW]")
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=10)


def clock_axis(ax):
    ticks = np.arange(0, 25, 3)
    ax.set_xticks(ticks, [f"{hour:02d}:00" for hour in ticks])
    ax.set_xlabel("対象日の時刻")


def draw_detail(day, path: Path, *, svg=False):
    fig, (power, widths) = plt.subplots(
        2, 1, figsize=(13, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.subplots_adjust(left=.085, right=.975, top=.84, bottom=.15, hspace=.12)
    fig.suptitle(f"入札と当日の指令  |  {day['date']}", fontsize=20, x=.085, ha="left", y=.965)
    fig.text(.085, .895,
             f"256指令・3EVケースの入札  /  未見指令 {day['command_index'] + 1}/{day['command_count']}"
             f"  /  履歴波形: {day['source_date']}  {day['source_resource']}",
             fontsize=11, color="#535d67")
    draw_power(power, day)
    power.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3,
                 frameon=False, fontsize=10, borderaxespad=0)
    shade_inactive(widths, day)
    edges = np.arange(49) / 2
    widths.step(edges, step_values(day["up"]), where="post", color=UP, linewidth=1.5,
                label="上げの入札幅")
    widths.step(edges, step_values(day["down"]), where="post", color=DOWN, linewidth=1.5,
                label="下げの入札幅")
    widths.set_ylabel("入札幅 [kW]")
    widths.set_ylim(bottom=0)
    widths.grid(axis="y", color="#dfe3e7", linewidth=.65)
    widths.spines[["top", "right"]].set_visible(False)
    widths.legend(loc="upper left", ncol=2, frameon=False, fontsize=10)
    clock_axis(widths)
    fig.text(.085, .045,
             "電力の正は充電、負は放電。灰色は入札なし。\n"
             "当日の指令は、保存済みAEMO履歴波形をこの入札量に換算した5分値。",
             fontsize=10, color="#535d67")
    fig.savefig(path, dpi=170, facecolor="white")
    if svg:
        fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--command-index", type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont("C:/Windows/Fonts/meiryo.ttc")
    plt.rcParams.update({"font.family": "Meiryo", "font.size": 11, "axes.unicode_minus": False})
    days = []
    for split, name in (
        ("train", "train_25_minmedmax_3of128ev_256cmd_all_commands_aemo_plan_deviation"),
        ("validation", "validation_5_minmedmax_3of128ev_256cmd_all_commands_aemo_plan_deviation"),
    ):
        bank = ROOT / "execute_results" / "bid_banks" / name
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            day = load_day(bank, entry, args.command_index, split)
            prefix = f"{split}_{day['date']}"
            draw_detail(day, output / f"{prefix}.png", svg=(day["date"] == "2024-04-02"))
            with (output / f"{prefix}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_hours", "tracking_enabled", "baseline_kw", "bid_lower_kw", "bid_upper_kw", "command_kw"])
                for i in range(288):
                    writer.writerow([i / 12, int(day["active"][i]), day["center"][i], day["lower"][i], day["upper"][i], day["target"][i]])
            days.append(day)
    validation = [day for day in days if day["split"] == "validation"]
    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.085, hspace=.42)
    fig.suptitle("検証用5日  |  入札範囲と当日の指令", x=.09, ha="left", fontsize=20)
    for index, (ax, day) in enumerate(zip(axes, validation)):
        draw_power(ax, day)
        ax.set_title(f"{day['date']}  /  履歴: {day['source_date']}  {day['source_resource']}", loc="left", fontsize=11)
    clock_axis(axes[-1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.98, .948),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.09, .026, "各日とも保存済み未見指令の先頭1本を表示。正は充電、負は放電。灰色は入札なし。", fontsize=10, color="#535d67")
    fig.savefig(output / "validation_5days.png", dpi=160, facecolor="white")
    plt.close(fig)
    metadata = [{key: value for key, value in day.items() if not isinstance(value, np.ndarray)} for day in days]
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "all_30days.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.suffix in {".png", ".svg", ".csv", ".json"}:
                archive.write(path, path.name)
    print(f"saved {len(days)} daily graphs, validation overview and ZIP to {output}")


if __name__ == "__main__":
    main()

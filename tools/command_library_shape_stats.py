"""Shape statistics of the plan-deviation command libraries, same definitions for all.

a = signed_activation_up_positive_raw (up positive, unclipped), 288 five-minute steps.
A step is active when |a| > 0.01. Up to --sample files per library, fixed seed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = {
    "aemo_plan_deviation": ROOT / "data/aemo/nem/command_libraries/plan_deviation_calendar_day",
    "ercot_plan_deviation": ROOT / "data/ercot/sced/command_libraries/esr_plan_deviation_calendar_day",
    "elexon_plan_deviation": ROOT / "data/elexon/command_libraries/battery_plan_deviation_calendar_day",
}
ACTIVE = 0.01


def longest(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def shape(a: np.ndarray) -> dict:
    up, down = a > ACTIVE, a < -ACTIVE
    active = up | down
    sign = np.sign(np.where(active, a, 0.0))
    nonzero = sign[sign != 0]
    return {
        "active_frac": float(active.mean()),
        "longest_one_way": max(longest(up), longest(down)),
        "reversals": int(np.sum(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0,
        "mean_abs": float(np.abs(a).mean()),
        "max_abs_cum_band_h": float(np.max(np.abs(np.cumsum(a) / 12.0))),
        "up_share_of_active": float(up.sum() / active.sum()) if active.any() else np.nan,
        "beyond_width_frac": float(np.mean(np.abs(a) > 1.0)),
        "idle_day": bool(not active.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    result = {}
    rows_by_library = {}
    for name, directory in LIBRARIES.items():
        with os.scandir(directory) as it:
            files = sorted(e.path for e in it if e.name.endswith(".csv"))
        chosen = [files[i] for i in sorted(rng.choice(len(files), min(args.sample, len(files)), replace=False))]
        rows = []
        for path in chosen:
            frame = pd.read_csv(path)
            row = shape(frame["signed_activation_up_positive_raw"].to_numpy(float))
            row["resource"] = str(frame["resource_name"].iloc[0])
            row["partition"] = str(frame["scenario_partition"].iloc[0])
            if "so_flag_in_force" in frame:
                row["so_flag_share_of_instructed"] = (
                    float(frame["so_flag_in_force"].sum() / frame["acceptance_in_force"].sum())
                    if frame["acceptance_in_force"].any() else np.nan
                )
            rows.append(row)
        df = pd.DataFrame(rows)
        rows_by_library[name] = df
        numeric = ["active_frac", "longest_one_way", "reversals", "mean_abs", "max_abs_cum_band_h",
                   "up_share_of_active", "beyond_width_frac"]
        result[name] = {
            "files_total": len(files),
            "files_sampled": len(chosen),
            "idle_day_share": float(df["idle_day"].mean()),
            "median": df[numeric].median().round(3).to_dict(),
            "p90": df[numeric].quantile(0.9).round(3).to_dict(),
            "by_partition_median": df.groupby("partition")[["active_frac", "mean_abs", "max_abs_cum_band_h", "up_share_of_active"]]
            .median().round(3).to_dict(orient="index"),
        }
    gb = rows_by_library["elexon_plan_deviation"]
    result["elexon_plan_deviation"]["by_unit_median"] = (
        gb.groupby("resource")[["active_frac", "mean_abs", "max_abs_cum_band_h", "up_share_of_active", "so_flag_share_of_instructed"]]
        .median().round(3).to_dict(orient="index")
    )
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

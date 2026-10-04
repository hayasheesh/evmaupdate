"""Rebuild only the missing train-bank date, preserving completed dates."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_BANK = ROOT / "execute_results/bid_banks/train_25_20station_256cmd_3ev_aemo_plan_deviation"
FAILED_INDEX = 9
FAILED_DATE = "2024-08-10"


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)

    from environment.arrival_context import ArrivalScenarioSampler
    from environment.readcsv import load_multiple_demand_files_with_labels
    from tools.build_training_bid_bank import _build_one_day, select_payloads_for_bank
    from training.bid_bank import manifest_settings_match, merge_training_bid_bank
    from training.lower_bid_training import upper_bid_bank_settings

    manifest = json.loads((TRAIN_BANK / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") or manifest.get("completed_days") != 24:
        raise RuntimeError("Expected the original 24/25 train bank")
    all_data = load_multiple_demand_files_with_labels(train_split=25)
    payloads, selection = select_payloads_for_bank(
        all_data,
        split="train",
        selected_days=25,
        paired_train_days=25,
        paired_test_days=5,
    )
    selected_dates = [str(payload.get("date")) for payload in payloads]
    if len(payloads) != 25 or selected_dates[FAILED_INDEX] != FAILED_DATE:
        raise RuntimeError("The original train date selection changed")
    settings = {
        "split": "train",
        "train_split_count": 25,
        "base_forecast_seed": 73000,
        "day_selection_mode": str(selection["mode"]),
        "selected_dates": selected_dates,
        "arrival_model": ArrivalScenarioSampler().settings_signature(),
        **upper_bid_bank_settings(),
        "scenario_workers_per_day": 5,
    }
    if not manifest_settings_match(manifest, settings):
        raise RuntimeError("The existing train bank uses different bid inputs")
    if int(manifest.get("base_forecast_seed", -1)) != 73000:
        raise RuntimeError("The original train forecast seed changed")
    existing_indices = {int(entry["index"]) for entry in manifest["entries"]}
    if existing_indices != set(range(25)) - {FAILED_INDEX}:
        raise RuntimeError("Unexpected completed train dates")

    payload = dict(payloads[FAILED_INDEX])
    payload["_bid_bank_index"] = FAILED_INDEX
    print(f"[recovery] solve only index={FAILED_INDEX} day={FAILED_DATE}", flush=True)
    entry = _build_one_day(
        payload,
        str(TRAIN_BANK),
        73000,
        settings,
        False,
        5,
    )
    if int(entry["index"]) != FAILED_INDEX or entry["service_date"] != FAILED_DATE:
        raise RuntimeError("The rebuilt date did not match the missing date")

    merged = merge_training_bid_bank(
        TRAIN_BANK,
        expected_days=25,
        base_forecast_seed=73000,
        settings=settings,
    )
    if not merged["complete"] or merged["completed_days"] != 25:
        raise RuntimeError("The merged train bank is not complete")
    print("[recovery] train bank complete=25/25", flush=True)


if __name__ == "__main__":
    main()

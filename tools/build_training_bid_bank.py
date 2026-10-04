"""Precompute fixed upper bids for lower-controller train or test dates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def select_payloads_for_bank(
    all_data: dict,
    *,
    split: str,
    selected_days: int,
    paired_train_days: int,
    paired_test_days: int,
) -> tuple[list[dict], dict]:
    """Use the same day selection contract as the training driver."""

    from training.run_after_day_ahead_bid import stratified_bank_day_selection

    pool = list(all_data.get("train") or []) + list(all_data.get("test") or [])
    train_payloads, test_payloads, info = stratified_bank_day_selection(
        pool,
        train_count=int(paired_train_days),
        test_count=int(paired_test_days),
    )
    return (train_payloads if split == "train" else test_payloads), info


def _with_fixed_ev_all_command_bid(
    *,
    scenario_workers: int = 1,
):
    """Bind worker count to the one canonical bank-bid builder."""

    def build(series, service_date, *, arrival_scenario, forecast_seed):
        from training.lower_bid_training import build_fixed_upper_bid_for_day

        return build_fixed_upper_bid_for_day(
            series,
            service_date,
            arrival_scenario=arrival_scenario,
            forecast_seed=forecast_seed,
            scenario_workers=max(int(scenario_workers), 1),
        )

    return build


def _build_one_day(
    payload,
    output_dir,
    base_seed,
    settings,
    overwrite,
    scenario_workers,
):
    from environment.arrival_context import ArrivalScenarioSampler
    from training.bid_bank import build_training_bid_bank
    from training.lower_bid_training import (
        build_fixed_upper_bid_training_episode,
        set_upper_bid_progress_log,
    )
    from training.run_after_day_ahead_bid import write_bid_artifacts

    manifest = build_training_bid_bank(
        [payload],
        output_dir,
        base_forecast_seed=int(base_seed),
        build_fixed_bid=_with_fixed_ev_all_command_bid(
            scenario_workers=int(scenario_workers)
        ),
        build_episode=build_fixed_upper_bid_training_episode,
        arrival_sampler=ArrivalScenarioSampler(),
        set_progress_log=set_upper_bid_progress_log,
        artifact_writer=write_bid_artifacts,
        settings=settings,
        overwrite=bool(overwrite),
        manifest_name=f"manifest_worker_{int(payload['_bid_bank_index']):03d}.json",
    )
    return manifest["entries"][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split-count", type=int, default=25)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of dates from the selected split (default: all train, 5 test).",
    )
    parser.add_argument(
        "--paired-train-days",
        type=int,
        default=None,
        help="Train count used by the shared stratified train/test selection.",
    )
    parser.add_argument(
        "--paired-test-days",
        type=int,
        default=None,
        help="Test count used by the shared stratified train/test selection.",
    )
    parser.add_argument(
        "--forecast-seed",
        type=int,
        default=None,
        help="Base upper-forecast seed (default: 73000 train, 1073003 test).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Independent day-level bid workers.",
    )
    parser.add_argument(
        "--scenario-workers",
        type=int,
        default=8,
        help="Per-day workers for the three EV-count cases x all-command recourse checks.",
    )
    args = parser.parse_args()

    selected_days = (
        int(args.days)
        if args.days is not None
        else (int(args.train_split_count) if args.split == "train" else 5)
    )
    if selected_days <= 0:
        raise ValueError("--days must be positive")
    base_forecast_seed = (
        int(args.forecast_seed)
        if args.forecast_seed is not None
        else 73000 + (1_000_003 if args.split == "test" else 0)
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from environment.arrival_context import ArrivalScenarioSampler
    from EnvConfig import (
        ACTIVATION_SIGNAL_SET,
        LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS,
        LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT,
        LOWER_TRAIN_UPPER_BID_TEST_BANK_COUNT,
    )
    from training.lower_bid_training import ev_population_signature, upper_bid_bank_settings
    from training.bid_bank import (
        build_training_bid_bank,
        manifest_settings_match,
        merge_training_bid_bank,
    )

    if args.output_dir is None:
        split_label = "validation" if args.split == "test" else "train"
        signal_suffix = (
            "" if ACTIVATION_SIGNAL_SET == "nem" else f"_{ACTIVATION_SIGNAL_SET}"
        )
        args.output_dir = str(
            PROJECT_ROOT
            / "execute_results"
            / "bid_banks"
            / (
                f"{split_label}_{selected_days}_{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT}ev_"
                f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS}cmd_all_commands"
                f"{signal_suffix}"
            )
        )

    from training.run_after_day_ahead_bid import service_day_payloads

    all_data = service_day_payloads()
    paired_train_days = int(
        args.paired_train_days
        if args.paired_train_days is not None
        else (selected_days if args.split == "train" else args.train_split_count)
    )
    paired_test_days = int(
        args.paired_test_days
        if args.paired_test_days is not None
        else (selected_days if args.split == "test" else LOWER_TRAIN_UPPER_BID_TEST_BANK_COUNT)
    )
    payloads, day_selection_info = select_payloads_for_bank(
        all_data,
        split=str(args.split),
        selected_days=int(selected_days),
        paired_train_days=paired_train_days,
        paired_test_days=paired_test_days,
    )
    if len(payloads) != selected_days:
        raise RuntimeError(
            f"expected {selected_days} {args.split} dates, found {len(payloads)}"
        )

    arrival_sampler = ArrivalScenarioSampler()
    settings = {
        "split": str(args.split),
        "train_split_count": int(args.train_split_count),
        "base_forecast_seed": int(base_forecast_seed),
        "day_selection_mode": str(day_selection_info["mode"]),
        "selected_dates": [str(payload.get("date")) for payload in payloads],
        "arrival_model": arrival_sampler.settings_signature(),
        "ev_population": ev_population_signature(),
        **upper_bid_bank_settings(),
        "scenario_workers_per_day": max(int(args.scenario_workers), 1),
    }
    if args.split == "test":
        settings["selected_days"] = int(selected_days)
    manifest_path = Path(args.output_dir).resolve() / "manifest.json"
    if manifest_path.exists() and not bool(args.overwrite):
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            arrival_matches = manifest_settings_match(existing_manifest, settings)
        except (OSError, ValueError, TypeError):
            arrival_matches = False
        if not arrival_matches:
            args.overwrite = True
            print(
                "[bid-bank] existing bank uses different upper-bid inputs; "
                "all dates will be rebuilt",
                flush=True,
            )
    indexed_payloads = []
    for day_index, payload in enumerate(payloads):
        indexed = dict(payload)
        indexed["_bid_bank_index"] = int(day_index)
        indexed_payloads.append(indexed)

    workers = max(int(args.workers), 1)
    if workers == 1:
        from training.lower_bid_training import (
            build_fixed_upper_bid_training_episode,
            set_upper_bid_progress_log,
        )
        from training.run_after_day_ahead_bid import write_bid_artifacts

        manifest = build_training_bid_bank(
            indexed_payloads,
            args.output_dir,
            base_forecast_seed=int(base_forecast_seed),
            build_fixed_bid=_with_fixed_ev_all_command_bid(
                scenario_workers=max(int(args.scenario_workers), 1)
            ),
            build_episode=build_fixed_upper_bid_training_episode,
            arrival_sampler=arrival_sampler,
            set_progress_log=set_upper_bid_progress_log,
            artifact_writer=write_bid_artifacts,
            settings=settings,
            overwrite=bool(args.overwrite),
        )
    else:
        failures = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_payload = {
                pool.submit(
                    _build_one_day,
                    payload,
                    str(Path(args.output_dir).resolve()),
                    int(base_forecast_seed),
                    settings,
                    bool(args.overwrite),
                    max(int(args.scenario_workers), 1),
                ): payload
                for payload in indexed_payloads
            }
            for future in as_completed(future_to_payload):
                payload = future_to_payload[future]
                try:
                    entry = future.result()
                    print(
                        f"[bid-bank] completed index={entry['index']} "
                        f"day={entry['service_date']}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append((payload.get("date"), exc))
                    print(
                        f"[bid-bank] failed day={payload.get('date')}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        manifest = merge_training_bid_bank(
            args.output_dir,
            expected_days=len(indexed_payloads),
            base_forecast_seed=int(base_forecast_seed),
            settings=settings,
        )
        if failures:
            raise RuntimeError(
                f"{len(failures)} bid-bank worker(s) failed; rerun resumes completed dates"
            )
    print(
        f"[bid-bank] complete={manifest['complete']} "
        f"days={manifest['completed_days']}/{manifest['expected_days']} "
        f"dir={Path(args.output_dir).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

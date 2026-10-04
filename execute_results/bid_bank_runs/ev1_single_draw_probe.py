from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\admin\Desktop\EVMALOCALUPDATE")
OUTPUT_DIR = PROJECT_ROOT / "execute_results" / "bid_banks" / "probe_aemo_bess_single_ev_2024-04-02_20260924_guarded"
SERVICE_DATE = "2024-04-02"
FORECAST_SEED = 73000
SCENARIO_WORKERS = 24

def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    os.environ["EVMA_ACTIVATION_SIGNAL_SET"] = "aemo_bess_dispatch"
    os.environ["EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES"] = "1"
    os.environ["EVMA_BID_SOLVE_CACHE"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("EVMA_ACTIVATION_SCENARIO_DIR", None)

    from EnvConfig import (
        ACTIVATION_SIGNAL_SET,
        LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES,
    )
    from environment.readcsv import load_multiple_demand_files_with_labels
    from environment.arrival_context import ArrivalScenarioSampler
    from tools.build_training_bid_bank import select_payloads_for_bank, _build_one_day
    from training.lower_bid_training import upper_bid_bank_settings
    from training.bid_bank import merge_training_bid_bank

    if ACTIVATION_SIGNAL_SET != "aemo_bess_dispatch":
        raise RuntimeError(f"wrong activation signal set: {ACTIVATION_SIGNAL_SET}")
    if LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES != 1:
        raise RuntimeError(f"expected one EV draw, got {LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty diagnostic output: {OUTPUT_DIR}")

    print(
        f"[probe] signal={ACTIVATION_SIGNAL_SET} EV-cases=1 draws=1 commands=128 "
        f"workers={SCENARIO_WORKERS} date={SERVICE_DATE} seed={FORECAST_SEED}",
        flush=True,
    )
    started = time.perf_counter()
    all_data = load_multiple_demand_files_with_labels(train_split=25)
    payloads, day_selection_info = select_payloads_for_bank(
        all_data,
        split="train",
        selected_days=25,
        paired_train_days=25,
        paired_test_days=5,
    )
    payload = next(
        (dict(item) for item in payloads if str(item.get("date")) == SERVICE_DATE),
        None,
    )
    if payload is None:
        raise RuntimeError(f"Service date {SERVICE_DATE} was not found in the train bank")
    payload["_bid_bank_index"] = 0
    arrival_sampler = ArrivalScenarioSampler()
    settings = {
        "split": "train",
        "diagnostic_one_day": True,
        "diagnostic_ev_case_count": 1,
        "train_split_count": 25,
        "base_forecast_seed": FORECAST_SEED,
        "day_selection_mode": str(day_selection_info["mode"]),
        "selected_dates": [SERVICE_DATE],
        "arrival_model": arrival_sampler.settings_signature(),
        **upper_bid_bank_settings(),
        "scenario_workers_per_day": SCENARIO_WORKERS,
    }
    entry = _build_one_day(
        payload,
        str(OUTPUT_DIR),
        FORECAST_SEED,
        settings,
        False,
        SCENARIO_WORKERS,
    )
    manifest = merge_training_bid_bank(
        OUTPUT_DIR,
        expected_days=1,
        base_forecast_seed=FORECAST_SEED,
        settings=settings,
    )
    elapsed = time.perf_counter() - started
    print(
        f"[probe] complete={manifest['complete']} days={manifest['completed_days']}/1 "
        f"date={entry['service_date']} elapsed_s={elapsed:.1f} dir={OUTPUT_DIR}",
        flush=True,
    )

if __name__ == "__main__":
    main()

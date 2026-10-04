"""Re-solve one bid-bank day with the current bid code: same date, seeds, EV bank and commands.

    python one_day.py <stations> <signal_set> <train|test> <YYYY-MM-DD> <out_dir> <scenario_workers> [min_band=KW] [step_weight=W] [growth=G]

The day is picked from the same stratified selection the bank uses (train 25 /
test 5) and keeps its bank index, so its forecast seed, EV candidates and
design commands are the ones the bank solved. ``min_band`` overrides the
minimum tracking band (minimum bid = 10 x band), ``step_weight`` the
baseline step weight and ``growth`` the EV arrival growth multiplier; without
them the EnvConfig values are used.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    stations, signal_set, split, day, out_dir, scenario_workers = sys.argv[1:7]
    options = dict(arg.split("=", 1) for arg in sys.argv[7:])
    min_band = options.get("min_band")
    step_weight = options.get("step_weight")
    growth = options.get("growth")
    multiprocessing.set_start_method("spawn", force=True)
    for key in list(os.environ):
        if key.startswith("EVMA_"):
            del os.environ[key]
    os.environ.update({
        "EVMA_NUM_STATIONS": str(stations),
        "EVMA_MARL_ALGORITHM": "hybrid",
        "EVMA_ACTIVATION_SIGNAL_SET": signal_set,
        "EVMA_LOWER_TRAIN_BUILD_BID_BANK": "0",
        "EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS": "0",
        "EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS": "128",
        "EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES": "128",
    })
    if min_band is not None:
        os.environ["EVMA_LOWER_CONTROLLER_MIN_TRACKING_BAND_KW"] = str(min_band)
    if step_weight is not None:
        os.environ["EVMA_LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT"] = str(step_weight)
    if growth is not None:
        os.environ["EVMA_FUTURE_EV_ARRIVAL_GROWTH"] = str(growth)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    import torch
    torch.set_num_threads(1)
    import tools.build_training_bid_bank as bank_builder
    from environment.arrival_context import ArrivalScenarioSampler
    from EnvConfig import FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER, LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT
    from training.blockwise_bid import minimum_bid_quantity_kw
    from training.lower_bid_training import upper_bid_bank_settings
    from training.run_after_day_ahead_bid import service_day_payloads

    days = 25 if split == "train" else 5
    payloads, info = bank_builder.select_payloads_for_bank(
        service_day_payloads(), split=split, selected_days=days,
        paired_train_days=25, paired_test_days=5,
    )
    dates = [str(p.get("date"))[:10] for p in payloads]
    index = dates.index(day)
    payload = dict(payloads[index])
    payload["_bid_bank_index"] = index
    base_seed = 73000 + (1_000_003 if split == "test" else 0)
    settings = {
        "split": split,
        "train_split_count": 25,
        "base_forecast_seed": base_seed,
        "day_selection_mode": str(info["mode"]),
        "selected_dates": dates,
        "arrival_model": ArrivalScenarioSampler().settings_signature(),
        **upper_bid_bank_settings(),
        "scenario_workers_per_day": int(scenario_workers),
    }
    print(json.dumps({
        "stations": int(stations), "signal_set": signal_set, "split": split, "date": day,
        "bank_index": index, "minimum_bid_kw": minimum_bid_quantity_kw(),
        "baseline_step_weight": float(LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT),
        "arrival_growth": float(FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER),
    }), flush=True)
    started = time.time()
    entry = bank_builder._build_one_day(
        payload, str(Path(out_dir).resolve()), base_seed, settings, True, int(scenario_workers)
    )
    print(json.dumps({"entry": entry, "wall_s": time.time() - started}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

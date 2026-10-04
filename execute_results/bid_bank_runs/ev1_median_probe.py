from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\admin\Desktop\EVMALOCALUPDATE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ["EVMA_ACTIVATION_SIGNAL_SET"] = "aemo_bess_dispatch"
os.environ["EVMA_BID_SOLVE_CACHE"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.pop("EVMA_ACTIVATION_SCENARIO_DIR", None)

from training import lower_bid_training as lbt

def select_median_ev_case(candidates):
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidate EV realizations")
    ranked = sorted(range(len(candidates)), key=lambda i: (len(candidates[i]), i))
    rank = (len(ranked) - 1) // 2
    candidate_index = ranked[rank]
    evs = candidates[candidate_index]
    return [evs], [{
        "label": "median",
        "candidate_index": int(candidate_index),
        "rank_zero_based": int(rank),
        "candidate_count": int(len(candidates)),
        "ev_count": int(len(evs)),
    }]

lbt._select_ev_scenarios_by_count = select_median_ev_case
_original_bank_settings = lbt.upper_bid_bank_settings
def one_ev_bank_settings():
    settings = _original_bank_settings()
    settings.update({
        "fixed_ev_scenarios": 1,
        "bank_bid_contract": "fixed_median_ev_all_commands_k0",
        "ev_scenario_selection": "lower_median_by_session_count",
    })
    return settings
lbt.upper_bid_bank_settings = one_ev_bank_settings

from environment.readcsv import load_multiple_demand_files_with_labels
from environment.arrival_context import ArrivalScenarioSampler
from tools.build_training_bid_bank import select_payloads_for_bank, _build_one_day
from training.bid_bank import merge_training_bid_bank

FORECAST_SEED = 73000
SERVICE_DATE = "2024-04-02"
SCENARIO_WORKERS = 24
OUTPUT_DIR = PROJECT_ROOT / "execute_results" / "bid_banks" / "probe_aemo_bess_1ev_median_20260924"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[probe] signal=aemo_bess_dispatch EV-cases=1 selection=median commands=128 workers={SCENARIO_WORKERS} date={SERVICE_DATE}", flush=True)
started = time.perf_counter()
all_data = load_multiple_demand_files_with_labels(train_split=25)
payloads, day_selection_info = select_payloads_for_bank(
    all_data,
    split="train",
    selected_days=25,
    paired_train_days=25,
    paired_test_days=5,
)
payload = next((dict(item) for item in payloads if str(item.get("date")) == SERVICE_DATE), None)
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
    **one_ev_bank_settings(),
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

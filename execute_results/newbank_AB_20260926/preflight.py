"""Check the AB-on-the-new-bank launch before the trainer starts.

usage: preflight.py [--list] [--per-ev] [--potential] [--rule] [--soc-floor]   (run with the launcher's environment variables set)

--rule: STATION_RULE_ALLOCATION = True (execute_results/rule_alloc_20260929).
--soc-floor: STATION_SOC_FLOOR = True (with --rule).

--per-ev: the per-EV local critic run (execute_results/perev_20260927); the
only setting allowed to differ from AB besides the bank is
LOCAL_CRITIC_PER_EV = True. --potential: LOCAL_REWARD_MODE = "potential"
(execute_results/perev_potential_20260928).

1. The new AEMO plan-deviation train and validation banks must match the
   current code's upper-bid settings exactly, the check the trainer itself
   makes, so the run needs neither EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS nor a
   rebuild.
2. Every Config/EnvConfig constant is compared with what the AB run
   (prod_f500_dense8_AB_7station_20260922_022453) recorded. Differences that
   come from the bank change are in BANK_KEYS; any other difference exits 2.
   Keys that did not exist when AB ran are listed for review by hand.
With --list every difference is printed and nothing fails.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = r"C:\Users\admin\Desktop\EVMALOCALUPDATE"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import Config  # noqa: E402
import EnvConfig  # noqa: E402
from environment.arrival_context import ArrivalScenarioSampler  # noqa: E402
from training.bid_bank import manifest_settings_match  # noqa: E402
from training.lower_bid_training import upper_bid_bank_settings  # noqa: E402

AB = os.path.join(ROOT, "archive", "prod_f500_dense8_AB_7station_20260922_022453", "resume", "latest.json")
BANKS = {
    "train": os.path.join(ROOT, "execute_results", "bid_banks",
                          "train_25_minmedmax_3of128ev_128cmd_all_commands_aemo_plan_deviation"),
    "test": os.path.join(ROOT, "execute_results", "bid_banks",
                         "validation_5_minmedmax_3of128ev_128cmd_all_commands_aemo_plan_deviation"),
}
# Inputs that name or describe the command library and the bank.
BANK_KEYS = {
    "ACTIVATION_SIGNAL_SET",
    "ACTIVATION_SCENARIO_DIR",
    "LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW",
    "LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR",
    "LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS",
    "LOWER_TRAIN_UPPER_BID_BANK_DIR",
    "LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR",
    "LOWER_TRAIN_UPPER_BID_MODEL_NAME",
}
# The AB rewards and learner, as AB ran them. A key that did not exist when AB
# ran must still hold the value that reproduces AB.
AB_DEFAULTS = {
    "USE_CENTRAL_EV_RESIDUAL_ALLOCATOR": False,
    "TRAIN_USE_RESIDUAL_BESS": False,
    "LOWER_TRAIN_ACCEPT_BANK_AS_IS": False,
    "LOCAL_SHAPING_REDUCTION": "mean",
    "LOCAL_SURPLUS_SHAPING_COEF": 0.0,
    "LOCAL_URGENCY_BASIS": "time",
    "LOCAL_DEPARTURE_REWARD_MODE": "step",
    "TRAIN_FORCE_CHARGING": False,
    "LOCAL_FORCED_PENALTY_PER_POINT": 0.0,
    "LOCAL_REWARD_SCALE": 1.0,
    "LOCAL_CRITIC_PER_EV": "--per-ev" in sys.argv[1:],
    "LOCAL_REWARD_MODE": "potential" if "--potential" in sys.argv[1:] else "legacy",
    "STATION_RULE_ALLOCATION": "--rule" in sys.argv[1:],
    "STATION_SOC_FLOOR": "--soc-floor" in sys.argv[1:],
}
list_only = "--list" in sys.argv[1:]
failures: list[str] = []

common = {"arrival_model": ArrivalScenarioSampler().settings_signature(), **upper_bid_bank_settings()}
for split, path in BANKS.items():
    payload = json.load(open(os.path.join(path, "manifest.json"), encoding="utf-8"))
    recorded = payload.get("settings") or {}
    differing = sorted(k for k, v in common.items() if recorded.get(k) != v)
    ok = bool(payload.get("complete")) and manifest_settings_match(payload, {**common, **{
        k: recorded.get(k) for k in ("split", "train_split_count", "base_forecast_seed",
                                     "day_selection_mode", "selected_dates", "selected_days")
        if k in recorded}})
    print(f"[bank {split}] {os.path.basename(path)} complete={payload.get('complete')} "
          f"matches current code={ok} differing={differing}")
    if not ok:
        failures.append(f"{split} bank does not match the current code: {differing}")

# Rollout commands: every partition except train, the bid design partition.
from market.activation_scenarios import (  # noqa: E402
    LOWER_CONTROL_POOL, build_activation_scenario_set, load_proxy_shape_library, lower_control_pool_label,
)
source = EnvConfig.LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR
library = load_proxy_shape_library(source)
partition_of = dict(zip(library["path"], library["declared_partition"]))
drawn = [s.source for seed in range(200) for s in build_activation_scenario_set(
    service_date="2024-04-02", n_scenarios=1, seed=seed, proxy_shape_dir=source,
    scenario_partition=LOWER_CONTROL_POOL)]
by_name = {p.split("\\")[-1]: part for p, part in partition_of.items()}
counts: dict[str, int] = {}
for src in drawn:
    part = by_name.get(src.split(":", 1)[1], "?")
    counts[part] = counts.get(part, 0) + 1
print(f"[commands] pool={lower_control_pool_label(source)} 200 draws by partition={counts}")
if lower_control_pool_label(source) != "validation+test" or set(counts) - {"validation", "test"}:
    failures.append(f"rollout commands leave the validation+test pool: {counts}")

for key, want in AB_DEFAULTS.items():
    module = Config if hasattr(Config, key) else EnvConfig
    if getattr(module, key, want) != want:
        failures.append(f"{key} = {getattr(module, key)!r}, AB needs {want!r}")

ab = json.load(open(AB, encoding="utf-8"))["context"]["runtime"]
new_keys: list[str] = []
for section, module in (("Config", Config), ("EnvConfig", EnvConfig)):
    for key in sorted(k for k in dir(module) if k.isupper()):
        value = getattr(module, key)
        if not isinstance(value, (bool, int, float, str, type(None))):
            continue
        if key not in ab[section]:
            new_keys.append(f"{section}.{key} = {value!r}")
            continue
        if value == ab[section][key]:
            continue
        line = f"{section}.{key}: AB {ab[section][key]!r} -> {value!r}"
        if key in BANK_KEYS:
            print("[bank change]", line)
        else:
            print("[DIFFERS]", line)
            failures.append(line)
for line in new_keys:
    print("[new key]", line)
for line in failures:
    print("[FAIL]", line)
if failures and not list_only:
    sys.exit(2)
print("[preflight] done" + (" (list only)" if list_only else ": bank matches, only bank inputs differ from AB"))

"""Compare the settings this process would train with against the AB run's record.

usage: preflight.py [socshape|lax|force]   (run with the launcher's environment variables set)

socshape (default): the summed and surplus SoC shaping. lax: that plus the
slack-weighted urgency and smooth departure reward. force: AB's rewards with
the execution-time departure force floor applied during training and a penalty
per SoC point it adds.

Prints every Config/EnvConfig constant whose value differs from what the AB run
(prod_f500_dense8_AB_7station_20260922_022453) recorded, and exits 2 when a
difference is not in EXPECTED. Keys that did not exist when AB ran are listed
separately; they are reviewed by hand once, since a new key can still change
behaviour.
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

AB = os.path.join(ROOT, "archive", "prod_f500_dense8_AB_7station_20260922_022453", "resume", "latest.json")
variant = (sys.argv[1:] or ["socshape"])[0]
SHAPING = {
    "LOCAL_DEFICIT_SHAPING_CLIP": 1.0,
    "LOCAL_SHAPING_REDUCTION": "sum",
    "LOCAL_SURPLUS_SHAPING_COEF": 0.25,
} if variant in ("socshape", "lax") else {
    "LOCAL_SHAPING_REDUCTION": "mean",
    "LOCAL_SURPLUS_SHAPING_COEF": 0.0,
}
EXPECTED = {
    **SHAPING,
    "LOWER_TRAIN_ACCEPT_BANK_AS_IS": True,
    # AB recorded the relative spelling; the path is the same directory.
    "LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR": None,
    # Defaults for solving new bids and naming the one-day runner's model. A
    # pretrain on an existing bank reads the bank's stored commands, and the
    # launcher passes --bank-dir, --test-bank-dir and --model-name itself.
    "LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS": None,
    "LOWER_TRAIN_UPPER_BID_BANK_DIR": None,
    "LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR": None,
    "LOWER_TRAIN_UPPER_BID_MODEL_NAME": None,
}

if variant == "force":
    EXPECTED.update({
        "TRAIN_FORCE_CHARGING": True,
        "TRAIN_FORCE_SLACK_KWH": 0.1,
        "LOCAL_FORCED_PENALTY_PER_POINT": 0.05,
    })
if variant == "lax":
    EXPECTED.update({
        "LOCAL_URGENCY_BASIS": "laxity",
        "LOCAL_DEFICIT_SHAPING_URGENCY_GAIN": 3.0,
        "LOCAL_DEPARTURE_REWARD_MODE": "smooth",
        "LOCAL_DEPARTURE_SMOOTH_LINEAR": 6.0,
        "LOCAL_DEPARTURE_SMOOTH_QUADRATIC": 12.0,
    })

recorded = json.load(open(AB, encoding="utf-8"))["context"]["runtime"]
unexpected, new_keys = [], []
for section, module in (("Config", Config), ("EnvConfig", EnvConfig)):
    ab = recorded[section]
    for key in sorted(k for k in dir(module) if k.isupper()):
        value = getattr(module, key)
        if not isinstance(value, (bool, int, float, str, type(None))):
            continue
        if key not in ab:
            if key in EXPECTED and EXPECTED[key] is not None and value != EXPECTED[key]:
                unexpected.append(f"{section}.{key} = {value!r}, expected {EXPECTED[key]!r}")
            else:
                new_keys.append(f"{section}.{key} = {value!r}")
            continue
        if value == ab[key]:
            continue
        if key in EXPECTED:
            if EXPECTED[key] is None or value == EXPECTED[key]:
                print(f"[expected] {section}.{key}: AB {ab[key]!r} -> {value!r}")
                continue
        unexpected.append(f"{section}.{key}: AB {ab[key]!r} -> {value!r}")
for line in new_keys:
    print("[new key]", line)
for line in unexpected:
    print("[DIFFERS]", line)
if unexpected:
    sys.exit(2)
print("[preflight] every recorded AB setting matches except the expected ones")

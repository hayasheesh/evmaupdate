"""Leaving the fleet residual, the bid context and the pass band out of the observation.

The flags are read when observation_config is imported, so each case runs in a
fresh interpreter.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_PROBE = r"""
import json
import numpy as np
import torch
from environment.EVEnv import EVEnv
from environment.normalize import AG_REQUEST_SCALE, normalize_observation
from environment.observation_config import (
    GLOBAL_TAIL_FEATURE_NAMES, LOCAL_TAIL_FEATURE_NAMES, ev_block_dim, local_obs_dim,
)
from tools.evaluator import set_env_seed

set_env_seed(11)
env = EVEnv()
band = np.full(288, 30.0, dtype=np.float32)
env.reset(net_demand_series=np.full(288, 50.0, dtype=np.float32), tol_narrow_series=band,
          tracking_enabled_series=np.ones(288, bool), service_date="2024-04-02")
raw = env.begin_step()
obs = normalize_observation(raw)
tail = obs[0, ev_block_dim(env.max_ev_per_station):]
names = list(LOCAL_TAIL_FEATURE_NAMES)
print(json.dumps({
    "local": names, "global": list(GLOBAL_TAIL_FEATURE_NAMES),
    "obs_dim": int(raw.shape[1]), "expected_dim": int(local_obs_dim(env.max_ev_per_station)),
    "tracking": float(tail[names.index("market_tracking_enabled")]),
    "step": float(tail[names.index("step")]),
    "band": float(tail[names.index("tracking_band_kw")]) if "tracking_band_kw" in names else None,
    "band_expected": 30.0 / AG_REQUEST_SCALE,
}))
"""


def _probe(**flags):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVMA_")}
    env.update({"EVMA_NUM_STATIONS": "7", **flags})
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_every_tail_feature_is_observed_by_default():
    d = _probe()
    assert {"fleet_residual_prev", "instruction_scale_kw", "tracking_band_kw"} <= set(d["local"])
    assert any(n.startswith("bid_") for n in d["local"])
    assert d["obs_dim"] == d["expected_dim"]
    assert abs(d["band"] - d["band_expected"]) < 1e-6


def test_the_three_groups_leave_every_network_and_the_tail_stays_aligned():
    d = _probe(EVMA_OBSERVE_FLEET_RESIDUAL="0", EVMA_OBSERVE_BID_CONTEXT="0", EVMA_OBSERVE_TRACKING_BAND="0")
    assert d["local"] == ["demand_0", "market_tracking_enabled", "step"]
    assert d["global"] == ["total_power", "step", "demand_0", "market_tracking_enabled"]
    assert d["obs_dim"] == d["expected_dim"]
    # Each remaining column is normalized as itself: the tracking flag reads 1 and
    # the first step reads 1/288, which would not hold if the columns had shifted.
    assert d["tracking"] == 1.0
    assert abs(d["step"] - 1.0 / 288) < 1e-6

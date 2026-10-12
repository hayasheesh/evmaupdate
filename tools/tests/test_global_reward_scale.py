"""EVMA_GLOBAL_REWARD_SCALE sets the multiplier on the tracking reward in the global critics' TD target.

Config reads the variable when it is imported, so each case runs in a fresh interpreter.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _scales(value=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVMA_")}
    if value is not None:
        env["EVMA_GLOBAL_REWARD_SCALE"] = value
    code = ("import json, Config as C; import training.Agent.maddpg as M; "
            "print(json.dumps([C.GLOBAL_REWARD_SCALE, M.GLOBAL_REWARD_SCALE, C.LOCAL_REWARD_SCALE]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_default_keeps_the_global_reward_unscaled():
    assert _scales() == [1.0, 1.0, 1.0]


def test_the_variable_reaches_the_agent_module_and_leaves_the_local_scale_alone():
    assert _scales("0.125") == [0.125, 0.125, 1.0]

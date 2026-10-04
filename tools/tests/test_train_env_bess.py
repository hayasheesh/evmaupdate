"""The battery is not part of what the lower controller learns.

It is an independent actuator at the point of common coupling: it never rewrites
an EV action, the tracking reward is the raw actor deviation, and the departure
reward reads the raw actor SoC trajectory. So on a pretrain or a fine-tune it is
arithmetic whose only consumer is a reported column, and training now runs with
it off.

That is only safe while the claim holds, which is what this pins. If a future
change lets the battery reach an EV action or an EV-side reward, the run stops
being reproducible across the flag and these fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STEPS = 40


def _rollout(use_bess: bool):
    from Config import EPISODE_STEPS, NUM_EVS, NUM_STATIONS
    from environment.EVEnv import EVEnv
    from tools.evaluator import set_env_seed

    set_env_seed(4242)
    env = EVEnv(num_stations=int(NUM_STATIONS), num_evs=int(NUM_EVS),
                episode_steps=int(EPISODE_STEPS))
    env.use_residual_bess = bool(use_bess)
    t = np.linspace(0.0, 4.0 * np.pi, int(EPISODE_STEPS))
    env.reset(
        net_demand_series=(300.0 * np.sin(t)).astype(np.float32),
        tol_narrow_series=np.full(int(EPISODE_STEPS), 60.0, dtype=np.float32),
    )
    rng = np.random.default_rng(7)
    global_rewards, local_rewards = [], []
    for _ in range(STEPS):
        env.begin_step()
        action = rng.uniform(
            -1.0, 1.0, (env.num_stations, env.max_ev_per_station)
        ).astype(np.float32)
        _obs, local, glob, _done, _info = env.apply_action(torch.as_tensor(action))
        global_rewards.append(float(glob))
        local_rewards.append(np.asarray(local, dtype=np.float64).reshape(-1).copy())
    return {
        "global": global_rewards,
        "local": np.stack(local_rewards),
        "soc": env.soc.detach().cpu().numpy().copy(),
        "metrics": env.get_metrics(),
    }


@pytest.fixture(scope="module")
def paired():
    return _rollout(True), _rollout(False)


def test_battery_does_not_touch_the_tracking_reward(paired):
    on, off = paired
    assert on["global"] == off["global"]


def test_battery_does_not_touch_the_local_rewards(paired):
    on, off = paired
    assert np.array_equal(on["local"], off["local"])


def test_battery_does_not_touch_the_ev_state(paired):
    """If it ever rewrote an EV action this is where it would show."""

    on, off = paired
    assert np.array_equal(on["soc"], off["soc"])


@pytest.mark.parametrize("key", [
    "tracking_success_rate",
    "raw_actor_mae_kw",
    "soc_miss_rate",
    "avg_soc_deficit",
])
def test_marl_side_metrics_are_identical(paired, key):
    on, off = paired
    assert on["metrics"].get(key) == off["metrics"].get(key)


def test_only_the_battery_column_moves(paired):
    """The flag has to do something, or the test above proves nothing."""

    on, off = paired
    assert on["metrics"]["bess_enabled"] is True
    assert off["metrics"]["bess_enabled"] is False
    # With no battery the post-battery figure is just the pre-battery one.
    assert off["metrics"]["post_bess_mae_kw"] == pytest.approx(
        off["metrics"]["pre_bess_mae_kw"]
    )
    assert on["metrics"]["post_bess_mae_kw"] != off["metrics"]["post_bess_mae_kw"]


def test_training_default_is_off_while_the_environment_default_is_on():
    from EnvConfig import TRAIN_USE_RESIDUAL_BESS, USE_RESIDUAL_BESS

    # The final precision evaluation builds its own environment and must keep
    # the battery; only training and the scoring inside training drop it.
    assert bool(USE_RESIDUAL_BESS) is True
    assert bool(TRAIN_USE_RESIDUAL_BESS) is False

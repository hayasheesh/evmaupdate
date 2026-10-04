from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from training.Agent.replay_buffer import ReplayBuffer
from training.training_resume import (
    ResumeStateError,
    capture_rng_state,
    clear_stop_request,
    load_training_resume,
    read_resume_manifest,
    request_stop,
    restore_rng_state,
    save_training_resume,
    stop_requested,
)


class _FakeResumeAgent:
    def __init__(self) -> None:
        self.value = torch.tensor([1.0, 2.0])

    def training_resume_state_dict(self):
        return {"format_version": 1, "value": self.value.clone()}


def _transition(i: int, *, n_agents: int = 2, s_dim: int = 3, max_evs: int = 2):
    base = float(i)
    return (
        torch.full((n_agents, s_dim), base),
        torch.full((n_agents, s_dim), base + 0.5),
        torch.full((n_agents, max_evs), base + 1.0),
        torch.full((n_agents,), base + 2.0),
        torch.tensor(base + 3.0),
        torch.zeros(n_agents),
        torch.full((n_agents,), base + 4.0),
        torch.full((n_agents, max_evs), base + 5.0),
    )


def _cache(buffer: ReplayBuffer, i: int) -> None:
    s, s2, a, rl, rg, done, station_p, ev_delta = _transition(i)
    buffer.cache(
        s,
        s2,
        a,
        rl,
        rg,
        done,
        actual_station_powers=station_p,
        actual_ev_soc_changes=ev_delta,
    )


def test_rng_state_roundtrip_reproduces_all_streams() -> None:
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20)

    state = capture_rng_state()
    expected = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
        torch.rand(4, device="cuda") if torch.cuda.is_available() else None,
    )
    restore_rng_state(state)
    actual = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
        torch.rand(4, device="cuda") if torch.cuda.is_available() else None,
    )

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    if expected[3] is not None:
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def test_uniform_replay_exact_roundtrip_preserves_layout_and_next_sample() -> None:
    original = ReplayBuffer(cap=5)
    for i in range(7):
        _cache(original, i)
    assert original.size == 5
    assert original.ptr == 2

    state = original.training_resume_state_dict()
    restored = ReplayBuffer(cap=5)
    restored.load_training_resume_state_dict(state)

    assert restored.size == original.size
    assert restored.ptr == original.ptr
    for name in ReplayBuffer._RESUME_TENSOR_NAMES:
        torch.testing.assert_close(
            getattr(restored, name), getattr(original, name), rtol=0, atol=0
        )

    rng = capture_rng_state()
    expected = original.sample(4)
    restore_rng_state(rng)
    actual = restored.sample(4)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    _cache(original, 99)
    _cache(restored, 99)
    assert restored.ptr == original.ptr
    for name in ReplayBuffer._RESUME_TENSOR_NAMES:
        torch.testing.assert_close(
            getattr(restored, name), getattr(original, name), rtol=0, atol=0
        )


def test_maddpg_resume_restores_target_networks_optimizer_phase_and_replay() -> None:
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim
    from training.Agent.maddpg import MADDPG

    state_dim = local_obs_dim(MAX_EV_PER_STATION)
    original = MADDPG(
        s_dim=state_dim,
        max_evs_per_station=MAX_EV_PER_STATION,
        n_agent=1,
        batch=2,
        num_episodes=10,
    )
    with torch.no_grad():
        for parameter in original.t_actors[0].parameters():
            parameter.add_(0.125)
    original.opt_a[0].zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in original.actors[0].parameters()).backward()
    original.opt_a[0].step()
    original.current_episode = 7
    original.update_step = 19
    original.warmup_steps = 0
    original.epsilon = 0.23
    original.ou_noise_scale = 0.41
    for i in range(4):
        s, s2, a, rl, rg, done, station_p, ev_delta = _transition(
            i, n_agents=1, s_dim=state_dim, max_evs=MAX_EV_PER_STATION
        )
        original.buf.cache(
            s,
            s2,
            a,
            rl,
            rg,
            done,
            actual_station_powers=station_p,
            actual_ev_soc_changes=ev_delta,
        )

    state = original.training_resume_state_dict()
    restored = MADDPG(
        s_dim=state_dim,
        max_evs_per_station=MAX_EV_PER_STATION,
        n_agent=1,
        batch=2,
        num_episodes=10,
    )
    info = restored.load_training_resume_state_dict(state)

    assert info == {
        "current_episode": 7,
        "update_step": 19,
        "replay_size": 4,
        "replay_ptr": 4,
    }
    assert restored.epsilon == pytest.approx(0.23)
    assert restored.ou_noise_scale == pytest.approx(0.41)
    assert restored.opt_a[0].state_dict()["state"]
    for expected, actual in zip(
        original.t_actors[0].parameters(), restored.t_actors[0].parameters()
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in ReplayBuffer._RESUME_TENSOR_NAMES:
        expected = getattr(original.buf, name)[: original.buf.size]
        actual = getattr(restored.buf, name)[: restored.buf.size]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    rng = capture_rng_state()
    original.update()
    restore_rng_state(rng)
    restored.update()
    assert restored.update_step == original.update_step == 20
    module_pairs = [
        *zip(original.actors, restored.actors),
        *zip(original.t_actors, restored.t_actors),
        *zip(original.critics, restored.critics),
        *zip(original.t_critics, restored.t_critics),
        (original.global_critic1, restored.global_critic1),
        (original.t_global_critic1, restored.t_global_critic1),
    ]
    for expected_module, actual_module in module_pairs:
        for expected, actual in zip(
            expected_module.parameters(), actual_module.parameters()
        ):
            # CUDA GEMM kernels are not promised bitwise identity across two
            # sequential executions, but restoring the same state/RNG keeps
            # the next optimizer step within floating-point roundoff.
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)


def test_atomic_resume_manifest_roundtrip_and_context_guard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = _FakeResumeAgent()
    context = {"kind": "unit-test", "bank": {"sha256": "abc"}}
    manifest = save_training_resume(
        run_dir=run_dir,
        agent=agent,
        completed_training_episode=12,
        completed_environment_episodes=15,
        all_rewards=list(range(12)),
        all_local_rewards=[0.1] * 12,
        all_global_rewards=[0.2] * 12,
        performance_metrics={"soc_miss_count": [1.0] * 12},
        all_episode_data={},
        context=context,
        reason="test",
    )

    assert manifest["exact"] is True
    assert manifest["completed_training_episode"] == 12
    assert read_resume_manifest(run_dir)["state_sha256"] == manifest["state_sha256"]
    payload, loaded_manifest = load_training_resume(
        run_dir, expected_context=context, map_location="cpu"
    )
    assert payload["completed_environment_episodes"] == 15
    assert payload["histories"]["all_rewards"] == list(range(12))
    torch.testing.assert_close(payload["agent"]["value"], agent.value)
    assert loaded_manifest == manifest

    with pytest.raises(ResumeStateError, match="context differs"):
        load_training_resume(
            run_dir,
            expected_context={"kind": "unit-test", "bank": {"sha256": "changed"}},
        )


def test_stop_request_is_explicit_and_clearable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = request_stop(run_dir)
    assert marker.is_file()
    assert stop_requested(run_dir)
    clear_stop_request(run_dir)
    assert not stop_requested(run_dir)

"""The attention-critic MADDPG (MAAC's critic) and MAPPO used in the algorithm comparison."""

from __future__ import annotations

import pytest
import torch

from training.training_resume import capture_rng_state, restore_rng_state

N_AGENTS = 3


def _kwargs():
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim

    return dict(s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
                n_agent=N_AGENTS, batch=8)


def _maac():
    from training.Agent.maac import MAACMADDPG

    agent = MAACMADDPG(**_kwargs())
    agent.warmup_steps = 0
    return agent


def _mappo():
    from training.Agent.mappo import MAPPO

    agent = MAPPO(**_kwargs())
    agent.warmup_steps = 0
    return agent


def _obs(agent, generator):
    from environment.observation_config import EV_FEAT_DIM

    s = torch.rand((N_AGENTS, agent.s_dim), generator=generator)
    ev = s[:, :agent.max_evs * EV_FEAT_DIM].view(N_AGENTS, agent.max_evs, EV_FEAT_DIM)
    ev[..., 0] = (torch.rand((N_AGENTS, agent.max_evs), generator=generator) > 0.3).float()
    return s


def _fill(agent, transitions: int = 16, seed: int = 3) -> None:
    generator = torch.Generator().manual_seed(seed)
    for _ in range(transitions):
        s, s2 = _obs(agent, generator), _obs(agent, generator)
        a = torch.rand((N_AGENTS, agent.a_dim), generator=generator) * 2.0 - 1.0
        agent.buf.cache(
            s, s2, a,
            torch.rand((N_AGENTS,), generator=generator),
            torch.rand((), generator=generator),
            torch.zeros(N_AGENTS),
            actual_station_powers=torch.rand((N_AGENTS,), generator=generator),
            actual_ev_soc_changes=a.clone(),
        )


def _snapshot(modules):
    return [p.detach().clone() for m in modules for p in m.parameters()]


# --- attention critic -----------------------------------------------------------

def test_attention_critic_gives_every_station_a_value_that_depends_on_the_others():
    from training.Agent.maac import AttentionCritic

    torch.manual_seed(0)
    critic = AttentionCritic(obs_dim=5, act_dim=2, n_agent=N_AGENTS, hid=8, heads=2)
    obs, act = torch.rand((4, N_AGENTS, 5)), torch.rand((4, N_AGENTS, 2))
    q = critic(obs, act)
    assert q.shape == (4, N_AGENTS)
    # Station 0's value moves with station 1's action (through attention) ...
    moved = act.clone()
    moved[:, 1] += 1.0
    assert not torch.allclose(critic(obs, moved)[:, 0], q[:, 0])


def test_maac_has_one_shared_critic_and_one_step_moves_it_and_the_actors():
    agent = _maac()
    assert len(agent.critics) == 1 and len(agent.opt_c) == 1
    _fill(agent)
    before = {"critics": _snapshot(agent.critics), "actors": _snapshot(agent.actors)}
    agent.update()
    assert agent.update_step == 1
    for name, modules in (("critics", agent.critics), ("actors", agent.actors)):
        after = _snapshot(modules)
        assert all(torch.isfinite(p).all() for p in after), name
        assert any(not torch.equal(a, b) for a, b in zip(before[name], after)), name


def test_maac_resume_restores_the_learner_exactly():
    original = _maac()
    _fill(original)
    original.update()
    restored = _maac()
    assert restored.load_training_resume_state_dict(original.training_resume_state_dict())["update_step"] == 1
    rng = capture_rng_state()
    original.update()
    restore_rng_state(rng)
    restored.update()
    for expected_module, actual_module in (*zip(original.actors, restored.actors),
                                           *zip(original.critics, restored.critics)):
        for expected, actual in zip(expected_module.parameters(), actual_module.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


# --- MAPPO ----------------------------------------------------------------------

def _episode(agent, steps=6, seed=5):
    generator = torch.Generator().manual_seed(seed)
    agent.episode_start()
    for t in range(steps):
        s = _obs(agent, generator)
        a = agent.act(s, noise=True)
        s2 = _obs(agent, generator)
        done = torch.full((N_AGENTS,), float(t == steps - 1))
        agent.cache_experience(s, s2, a, torch.rand((N_AGENTS,), generator=generator),
                               torch.rand((), generator=generator), done,
                               actual_station_powers=torch.rand((N_AGENTS,), generator=generator))
    agent.episode_end()


def test_mappo_acts_in_range_and_keeps_empty_slots_inert():
    agent = _mappo()
    generator = torch.Generator().manual_seed(1)
    s = _obs(agent, generator)
    for noise in (True, False):
        a = agent.act(s, noise=noise)
        assert a.shape == (N_AGENTS, agent.a_dim)
        assert torch.all(a.abs() <= 1.0)
        empty = s[:, :agent.max_evs * agent.actors[0].ev_feat_dim].view(
            N_AGENTS, agent.max_evs, -1)[..., 0].to(a.device) <= 0.5
        assert torch.all(a[empty] == 0.0)


def test_mappo_evaluation_uses_the_mean_which_is_the_station_actor_output():
    agent = _mappo()
    agent.set_test_mode(True)
    generator = torch.Generator().manual_seed(2)
    s = _obs(agent, generator).to(next(agent.actors[0].parameters()).device)
    a = agent.act(s)
    expected = torch.stack([agent.actors[i](s[i:i + 1]).squeeze(0) for i in range(N_AGENTS)])
    torch.testing.assert_close(a, expected)
    assert agent._pending is None


def test_mappo_learns_once_per_episode_from_that_episode():
    agent = _mappo()
    before = {"actors": _snapshot(agent.actors), "critics": _snapshot(agent.critics)}
    _episode(agent)
    assert agent.update_step == 1
    assert agent._rollout["reward"] == [] and agent._pending is None
    for name, modules in (("actors", agent.actors), ("critics", agent.critics)):
        after = _snapshot(modules)
        assert all(torch.isfinite(p).all() for p in after), name
        assert any(not torch.equal(a, b) for a, b in zip(before[name], after)), name


def test_mappo_does_not_learn_from_warmup_episodes():
    agent = _mappo()
    agent.warmup_steps = 10_000
    before = _snapshot(agent.actors)
    _episode(agent)
    assert agent.update_step == 0
    assert all(torch.equal(a, b) for a, b in zip(before, _snapshot(agent.actors)))


def test_mappo_skipped_steps_keep_actions_and_rewards_aligned():
    agent = _mappo()
    generator = torch.Generator().manual_seed(7)
    agent.episode_start()
    agent.act(_obs(agent, generator), noise=True)  # this step's transition is skipped
    s = _obs(agent, generator)
    a = agent.act(s, noise=True)
    agent.cache_experience(s, _obs(agent, generator), a, torch.rand((N_AGENTS,)), torch.rand(()),
                           torch.zeros(N_AGENTS), actual_station_powers=torch.rand((N_AGENTS,)))
    assert len(agent._rollout["obs"]) == len(agent._rollout["reward"]) == 1
    torch.testing.assert_close(agent._rollout["obs"][0], torch.as_tensor(s, device=agent._rollout["obs"][0].device))


def test_mappo_state_is_saved_between_episodes_only():
    agent = _mappo()
    _episode(agent)
    state = agent.training_resume_state_dict()
    restored = _mappo()
    assert restored.load_training_resume_state_dict(state)["update_step"] == 1
    for expected, actual in zip(_snapshot(agent.actors), _snapshot(restored.actors)):
        torch.testing.assert_close(actual, expected)
    generator = torch.Generator().manual_seed(9)
    s = _obs(agent, generator)
    agent.cache_experience(s, s, agent.act(s, noise=True), torch.rand((N_AGENTS,)), torch.rand(()),
                           torch.zeros(N_AGENTS), actual_station_powers=torch.rand((N_AGENTS,)))
    with pytest.raises(RuntimeError):
        agent.training_resume_state_dict()


def test_the_switch_builds_the_comparison_learners(monkeypatch):
    import Config
    from training.Agent import comparison_learners
    from training.Agent.maac import MAACMADDPG
    from training.Agent.mappo import MAPPO

    kwargs = {**_kwargs(), "gamma": 0.985, "lr_global_c": 3e-6}
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "maac")
    assert type(comparison_learners.build_learner(**kwargs)) is MAACMADDPG
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "mappo")
    assert type(comparison_learners.build_learner(**kwargs)) is MAPPO

"""MADDPG as in Lowe et al. (2017): structure, one learning step, and resume."""

from __future__ import annotations

import pytest
import torch

from training.training_resume import capture_rng_state, restore_rng_state

N_AGENTS = 2


def _agent():
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim
    from training.Agent.standard_maddpg import StandardMADDPG

    agent = StandardMADDPG(
        s_dim=local_obs_dim(MAX_EV_PER_STATION),
        max_evs_per_station=MAX_EV_PER_STATION,
        n_agent=N_AGENTS,
        batch=8,
    )
    agent.warmup_steps = 0
    return agent


def _fill(agent, transitions: int = 16, seed: int = 3) -> None:
    generator = torch.Generator().manual_seed(seed)
    for _ in range(transitions):
        s = torch.rand((N_AGENTS, agent.s_dim), generator=generator)
        s2 = torch.rand((N_AGENTS, agent.s_dim), generator=generator)
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


def test_each_critic_sees_every_station_observation_and_action():
    agent = _agent()
    first = agent.critics[0].net[0]
    assert first.in_features == N_AGENTS * agent.s_dim + N_AGENTS * agent.a_dim
    assert len(agent.critics) == N_AGENTS
    assert agent.critics2 == [] and agent.global_critic1 is None
    assert agent.policy_delay == 1


def test_exploration_is_gaussian_noise_only():
    agent = _agent()
    agent.episode_start()
    assert agent.epsilon == 0.0


def test_one_step_moves_critics_actors_and_targets():
    agent = _agent()
    _fill(agent)
    before = {
        "critics": _snapshot(agent.critics),
        "actors": _snapshot(agent.actors),
        "t_critics": _snapshot(agent.t_critics),
        "t_actors": _snapshot(agent.t_actors),
    }
    agent.update()
    assert agent.update_step == 1
    for name, modules in (("critics", agent.critics), ("actors", agent.actors),
                          ("t_critics", agent.t_critics), ("t_actors", agent.t_actors)):
        after = _snapshot(modules)
        assert all(torch.isfinite(p).all() for p in after), name
        assert any(not torch.equal(a, b) for a, b in zip(before[name], after)), name


def test_resume_restores_the_learner_exactly():
    original = _agent()
    _fill(original)
    original.update()
    state = original.training_resume_state_dict()

    restored = _agent()
    info = restored.load_training_resume_state_dict(state)
    assert info["update_step"] == 1

    rng = capture_rng_state()
    original.update()
    restore_rng_state(rng)
    restored.update()
    # On CUDA two learners restored from one state and stepped with one RNG
    # state already differ by up to about 1e-6 after one step: Adam scales the
    # roundoff in near-zero critic gradients up to the learning rate (3e-6).
    # The tolerance sits just above that noise; a state that was not restored
    # leaves freshly initialized weights, which differ by orders of magnitude more.
    for expected_module, actual_module in (*zip(original.actors, restored.actors),
                                           *zip(original.critics, restored.critics)):
        for expected, actual in zip(expected_module.parameters(), actual_module.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


def test_a_research_resume_state_is_refused():
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim
    from training.Agent.maddpg import MADDPG

    research = MADDPG(
        s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
        n_agent=N_AGENTS, batch=8,
    )
    with pytest.raises(ValueError, match="compatibility mismatch"):
        _agent().load_training_resume_state_dict(research.training_resume_state_dict())


def test_the_switch_picks_the_learner(monkeypatch):
    import Config
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim
    from training.Agent.maddpg import MADDPG
    from training.Agent.standard_maddpg import StandardMADDPG, build_marl_agent

    kwargs = dict(s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
                  n_agent=N_AGENTS, batch=8, gamma=0.985, lr_global_c=3e-6)
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "maddpg")
    assert type(build_marl_agent(**kwargs)) is StandardMADDPG
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "hybrid")
    assert type(build_marl_agent(**kwargs)) is MADDPG
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "qmix")
    with pytest.raises(ValueError):
        build_marl_agent(**kwargs)

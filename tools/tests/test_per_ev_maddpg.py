"""MADDPG with one actor per EV slot: what each slot sees, one learning step, resume, and the switch."""

from __future__ import annotations

import pytest
import torch

from training.training_resume import capture_rng_state, restore_rng_state

N_AGENTS = 2


def _agent():
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim
    from training.Agent.per_ev_maddpg import PerEVMADDPG

    agent = PerEVMADDPG(
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


def _present_obs(agent, batch=4, seed=0):
    from environment.observation_config import EV_FEAT_DIM

    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((batch, agent.s_dim), generator=generator)
    ev = x[:, :agent.max_evs * EV_FEAT_DIM].view(batch, agent.max_evs, EV_FEAT_DIM)
    ev[..., 0] = 1.0
    return x.to(next(agent.actors[0].parameters()).device)


def test_a_slot_action_depends_only_on_its_own_ev_and_the_tail():
    from environment.observation_config import EV_FEAT_DIM, LOCAL_TAIL_DIM

    agent = _agent()
    actor = agent.actors[0]
    for p in actor.parameters():
        torch.nn.init.normal_(p, std=0.5)
    x = _present_obs(agent)
    base = actor(x)
    other = x.clone()
    other[:, EV_FEAT_DIM + 1:2 * EV_FEAT_DIM] += 1.0  # slot 1's features, presence kept
    moved = actor(other)
    assert torch.equal(moved[:, 0], base[:, 0])
    assert not torch.equal(moved[:, 1], base[:, 1])
    tail = x.clone()
    start = agent.max_evs * EV_FEAT_DIM
    tail[:, start:start + LOCAL_TAIL_DIM] += 1.0
    assert not torch.equal(actor(tail)[:, 0], base[:, 0])


def test_slots_have_their_own_parameters():
    agent = _agent()
    actor = agent.actors[0]
    assert actor.w1.shape[0] == agent.max_evs
    assert not torch.equal(actor.w1[0], actor.w1[1])


def test_empty_slots_are_inert():
    agent = _agent()
    x = _present_obs(agent)
    x[:, 0] = 0.0  # slot 0 empty
    assert torch.all(agent.actors[0](x)[:, 0] == 0.0)


def test_one_step_moves_critics_actors_and_targets():
    agent = _agent()
    _fill(agent)
    before = {"critics": _snapshot(agent.critics), "actors": _snapshot(agent.actors),
              "t_actors": _snapshot(agent.t_actors)}
    agent.update()
    assert agent.update_step == 1
    for name, modules in (("critics", agent.critics), ("actors", agent.actors), ("t_actors", agent.t_actors)):
        after = _snapshot(modules)
        assert all(torch.isfinite(p).all() for p in after), name
        assert any(not torch.equal(a, b) for a, b in zip(before[name], after)), name


def test_resume_restores_the_learner_exactly():
    original = _agent()
    _fill(original)
    original.update()
    state = original.training_resume_state_dict()
    restored = _agent()
    assert restored.load_training_resume_state_dict(state)["update_step"] == 1
    rng = capture_rng_state()
    original.update()
    restore_rng_state(rng)
    restored.update()
    for expected_module, actual_module in (*zip(original.actors, restored.actors),
                                           *zip(original.critics, restored.critics)):
        for expected, actual in zip(expected_module.parameters(), actual_module.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


def test_a_station_actor_resume_state_is_refused():
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim
    from training.Agent.standard_maddpg import StandardMADDPG

    station = StandardMADDPG(s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
                             n_agent=N_AGENTS, batch=8)
    with pytest.raises(ValueError, match="compatibility mismatch"):
        _agent().load_training_resume_state_dict(station.training_resume_state_dict())


def test_the_switch_picks_the_learner(monkeypatch):
    import Config
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim
    from training.Agent import comparison_learners, standard_maddpg
    from training.Agent.per_ev_maddpg import PerEVMADDPG
    from training.Agent.standard_maddpg import StandardMADDPG

    kwargs = dict(s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
                  n_agent=N_AGENTS, batch=8, gamma=0.985, lr_global_c=3e-6)
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "maddpg_per_ev")
    assert type(comparison_learners.build_learner(**kwargs)) is PerEVMADDPG
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "maddpg")
    assert type(comparison_learners.build_learner(**kwargs)) is StandardMADDPG
    monkeypatch.setattr(standard_maddpg, "build_marl_agent", standard_maddpg.build_marl_agent)
    comparison_learners.install()
    monkeypatch.setattr(Config, "MARL_ALGORITHM", "maddpg_per_ev")
    assert type(standard_maddpg.build_marl_agent(**kwargs)) is PerEVMADDPG

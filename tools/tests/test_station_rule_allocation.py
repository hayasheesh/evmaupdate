"""EVMA_STATION_RULE_ALLOCATION: the actor sets each station's total, a laxity rule splits it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from environment.station_allocation import allocate_by_laxity


def _case(generator, n=8, batch=64):
    lo = -torch.rand((batch, n), generator=generator) * 20
    hi = torch.rand((batch, n), generator=generator) * 20
    laxity = torch.randn((batch, n), generator=generator) * 30
    present = torch.rand((batch, n), generator=generator) < 0.7
    lo, hi = lo * present, hi * present
    total = (torch.rand(batch, generator=generator) * 2 - 1) * 90
    return total, lo, hi, laxity, present


def test_the_split_keeps_the_total_the_bounds_and_one_direction():
    total, lo, hi, laxity, present = _case(torch.Generator().manual_seed(0))
    split = allocate_by_laxity(total, lo, hi, laxity, present)
    expected = torch.minimum(torch.maximum(total, lo.sum(-1)), hi.sum(-1))
    torch.testing.assert_close(split.sum(-1), expected, atol=1e-4, rtol=1e-5)
    assert torch.all(split >= lo - 1e-5) and torch.all(split <= hi + 1e-5)
    assert torch.all(split[~present] == 0)
    assert not torch.any((split > 1e-6).any(-1) & (split < -1e-6).any(-1))


def test_charge_goes_to_the_least_slack_and_discharge_comes_from_the_most():
    lo = torch.tensor([[-10.0, -10.0, -10.0]])
    hi = torch.tensor([[10.0, 10.0, 10.0]])
    laxity = torch.tensor([[50.0, 2.0, 20.0]])
    present = torch.ones((1, 3), dtype=torch.bool)
    charged = allocate_by_laxity(torch.tensor([15.0]), lo, hi, laxity, present)
    assert charged.tolist() == [[0.0, 10.0, 5.0]]
    discharged = allocate_by_laxity(torch.tensor([-15.0]), lo, hi, laxity, present)
    assert discharged.tolist() == [[-10.0, 0.0, -5.0]]


def test_the_total_reaches_the_marginal_ev_by_gradient():
    lo = torch.tensor([[-10.0, -10.0, -10.0]])
    hi = torch.tensor([[10.0, 10.0, 10.0]])
    laxity = torch.tensor([[50.0, 2.0, 20.0]])
    total = torch.tensor([15.0], requires_grad=True)
    split = allocate_by_laxity(total, lo, hi, laxity, torch.ones((1, 3), dtype=torch.bool))
    weights = torch.tensor([[1.0, 2.0, 3.0]])
    (split * weights).sum().backward()
    # EV 1 is full, EV 2 takes the last kilowatt: d/dtotal = its weight.
    assert float(total.grad) == 3.0


def _agent(monkeypatch, n_agent=3, batch=8):
    import training.Agent.maddpg as m
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim

    monkeypatch.setattr(m, "STATION_RULE_ALLOCATION", True)
    torch.manual_seed(0)
    return m, m.MADDPG(s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
                       n_agent=n_agent, batch=batch)


def _states(agent, generator, batch):
    from environment.observation_config import EV_FEAT_DIM, get_ev_feature_names

    names = get_ev_feature_names()
    s = torch.rand((batch, agent.n, agent.s_dim), generator=generator)
    block = s[:, :, :agent.max_evs * EV_FEAT_DIM].view(batch, agent.n, agent.max_evs, EV_FEAT_DIM)
    block[..., names.index("presence")] = (torch.rand((batch, agent.n, agent.max_evs), generator=generator) < 0.6).float()
    block[..., names.index("needed_soc")] = torch.rand((batch, agent.n, agent.max_evs), generator=generator) * 1.2 - 0.4
    return s.to(agent.active_slot_mask.device)


def test_the_agent_acts_with_the_split_and_the_learner_rebuilds_the_same(monkeypatch):
    m, agent = _agent(monkeypatch)
    assert agent.actors[0].station_total
    assert agent.ou_noise.max_evs == 1
    generator = torch.Generator().manual_seed(1)
    for _ in range(5):
        state = _states(agent, generator, 1)[0]
        with torch.no_grad():
            station_actions = torch.stack([
                agent.actors[i](state[i:i + 1]).squeeze(0) for i in range(agent.n)
            ], dim=0)
        assert station_actions.shape == (agent.n, 1)
        acted = agent.act(state, noise=False)
        ev_block = state[:, :agent.max_evs * m.EV_FEAT_DIM].reshape(agent.n, agent.max_evs, m.EV_FEAT_DIM)
        socs, capacity, max_power = agent._extract_ev_physics(ev_block)
        present = ev_block[..., 0] > 0.5
        kw = acted * max_power
        # One direction per station, empty slots at zero.
        assert not torch.any((kw > 1e-4).any(-1) & (kw < -1e-4).any(-1))
        assert torch.all(acted[~present] == 0)
        # The learner converts the same one-scalar actions to the same EV split.
        rebuilt, _ = agent._apply_soc_constraint(station_actions.unsqueeze(0), socs.unsqueeze(0), (~present).unsqueeze(0),
                                                  capacity_kwh=capacity.unsqueeze(0), max_power_kw=max_power.unsqueeze(0),
                                                  ev_block=ev_block.unsqueeze(0))
        torch.testing.assert_close(rebuilt.squeeze(0), kw, atol=1e-3, rtol=1e-4)


def test_one_station_action_sets_the_total_before_the_rule_split(monkeypatch):
    m, agent = _agent(monkeypatch, n_agent=1)
    n = agent.max_evs
    ev_block = torch.zeros((1, 1, n, m.EV_FEAT_DIM))
    ev_block[..., :2, 0] = 1.0
    ev_block[..., :2, m.REMAINING_FEATURE_IDX] = 0.5
    ev_block[..., 0, m.NEEDED_FEATURE_IDX] = -0.2
    ev_block[..., 1, m.NEEDED_FEATURE_IDX] = 0.2
    soc = torch.zeros((1, 1, n))
    soc[..., 0] = 100.0
    soc[..., 1] = 50.0
    capacity = torch.full_like(soc, 60.0)
    max_power = torch.zeros_like(soc)
    max_power[..., :2] = 10.0
    absent = ev_block[..., 0] <= 0.5
    scalar = torch.tensor([[[0.5]]], requires_grad=True)
    split, total = agent._apply_soc_constraint(
        scalar, soc, absent, capacity_kwh=capacity,
        max_power_kw=max_power, ev_block=ev_block,
    )
    torch.testing.assert_close(total, torch.tensor([[5.0]]), atol=1e-4, rtol=0)
    torch.testing.assert_close(split[0, 0, :2], torch.tensor([0.0, 5.0]), atol=1e-4, rtol=0)
    total.sum().backward()
    torch.testing.assert_close(scalar.grad, torch.tensor([[[10.0]]]), atol=1e-4, rtol=0)
    with pytest.raises(ValueError, match="one scalar"):
        agent._apply_soc_constraint(
            torch.zeros((1, 1, n)), soc, absent,
            capacity_kwh=capacity, max_power_kw=max_power, ev_block=ev_block,
        )


def test_an_update_runs_with_the_split(monkeypatch):
    m, agent = _agent(monkeypatch, batch=8)
    agent.warmup_steps = 0
    generator = torch.Generator().manual_seed(2)
    states = _states(agent, generator, 24)
    for k in range(23):
        a = agent.act(states[k], noise=True)
        agent.buf.cache(states[k], states[k + 1], a, torch.rand(agent.n, generator=generator), torch.rand((), generator=generator),
                        torch.zeros(agent.n), actual_station_powers=torch.rand(agent.n, generator=generator),
                        actual_ev_soc_changes=a.clone())
    before = [p.detach().clone() for p in agent.actors[0].parameters()]
    for _ in range(4):
        agent.update()
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.actors[0].parameters()))
    assert np.isfinite(agent.critic_losses).all()


def test_standard_maddpg_uses_the_same_scalar_action(monkeypatch):
    import training.Agent.maddpg as m
    import training.Agent.standard_maddpg as standard
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim

    monkeypatch.setattr(m, "STATION_RULE_ALLOCATION", True)
    monkeypatch.setattr(standard, "STATION_RULE_ALLOCATION", True)
    agent = standard.StandardMADDPG(
        s_dim=local_obs_dim(MAX_EV_PER_STATION),
        max_evs_per_station=MAX_EV_PER_STATION, n_agent=3, batch=8,
    )
    agent.warmup_steps = 0
    generator = torch.Generator().manual_seed(3)
    states = _states(agent, generator, 24)
    assert agent.actors[0](states[0, 0]).shape == (1,)
    for k in range(23):
        action = agent.act(states[k], noise=False)
        agent.buf.cache(
            states[k], states[k + 1], action,
            torch.rand(agent.n, generator=generator),
            torch.rand((), generator=generator), torch.zeros(agent.n),
            actual_station_powers=torch.rand(agent.n, generator=generator),
            actual_ev_soc_changes=action.clone(),
        )
    agent.update()
    assert np.isfinite(agent.critic_losses).all()

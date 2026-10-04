"""EVMA_STATION_SOC_FLOOR: the station rule keeps SoC; only the global critic trains the actors."""

from __future__ import annotations

import numpy as np
import torch


def _agent(monkeypatch, n_agent, batch=8):
    import training.Agent.maddpg as m
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim

    monkeypatch.setattr(m, "STATION_RULE_ALLOCATION", True)
    monkeypatch.setattr(m, "STATION_SOC_FLOOR", True)
    torch.manual_seed(0)
    return m, m.MADDPG(s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
                       n_agent=n_agent, batch=batch)


def test_the_floor_is_charged_whatever_the_actor_says(monkeypatch):
    m, agent = _agent(monkeypatch, n_agent=1)
    n = agent.max_evs
    ev_block = torch.zeros((1, 1, n, m.EV_FEAT_DIM))
    ev_block[..., :3, 0] = 1.0
    # EV 0 needs 30 points in 3 steps (must charge); EV 1 is above target; EV 2 has time.
    ev_block[..., 0, m.REMAINING_FEATURE_IDX] = 3 / 288
    ev_block[..., 0, m.NEEDED_FEATURE_IDX] = 0.30
    ev_block[..., 1, m.REMAINING_FEATURE_IDX] = 20 / 288
    ev_block[..., 1, m.NEEDED_FEATURE_IDX] = -0.20
    ev_block[..., 2, m.REMAINING_FEATURE_IDX] = 200 / 288
    ev_block[..., 2, m.NEEDED_FEATURE_IDX] = 0.30
    soc = torch.zeros((1, 1, n))
    soc[..., :3] = torch.tensor([40.0, 90.0, 40.0])
    capacity = torch.full_like(soc, 60.0)
    max_power = torch.zeros_like(soc)
    max_power[..., :3] = 11.0
    absent = ev_block[..., 0] <= 0.5
    floor = agent._soc_floor_kw(ev_block, capacity, max_power)[0, 0, 0]
    assert float(floor) > 0
    totals = []
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        split, total = agent._apply_soc_constraint(torch.tensor([[[a]]]), soc, absent, capacity_kwh=capacity,
                                                   max_power_kw=max_power, ev_block=ev_block)
        assert float(split[0, 0, 0]) >= float(floor) - 1e-4
        totals.append(float(total))
    assert totals == sorted(totals)
    split0, total0 = agent._apply_soc_constraint(torch.tensor([[[0.0]]]), soc, absent, capacity_kwh=capacity,
                                                 max_power_kw=max_power, ev_block=ev_block)
    torch.testing.assert_close(total0, floor.reshape(1, 1), atol=1e-4, rtol=0)


def test_acting_leaves_nothing_for_the_evaluation_force_charging(monkeypatch):
    import environment.EVEnv as evenv_module
    from Config import EPISODE_STEPS, NUM_EVS, NUM_STATIONS
    from environment.normalize import normalize_observation
    from tools.evaluator import set_env_seed
    from training.system_controller import apply_force_charging

    m, agent = _agent(monkeypatch, n_agent=int(NUM_STATIONS))
    set_env_seed(4242)
    env = evenv_module.EVEnv(num_stations=int(NUM_STATIONS), num_evs=int(NUM_EVS), episode_steps=int(EPISODE_STEPS))
    t = np.linspace(0.0, 4.0 * np.pi, int(EPISODE_STEPS))
    env.reset(net_demand_series=(300.0 * np.sin(t)).astype(np.float32),
              tol_narrow_series=np.full(int(EPISODE_STEPS), 60.0, dtype=np.float32))
    raised = 0
    for _ in range(200):
        obs = normalize_observation(env.begin_step())
        acted = agent.act(obs, env=env, noise=True)
        forced, _, _ = apply_force_charging(acted, env, slack_kwh=0.1)
        # Compare what would be executed: the force rule may still ask a full
        # battery for its 0.1 kWh slack, which the SoC limit then drops.
        headroom = torch.zeros_like(acted)
        for st in range(env.num_stations):
            order = env._get_sorted_active_evs(st)
            k = order.numel()
            if k:
                headroom[st, :k] = ((100.0 - env.soc[st, order]) * env.ev_kwh_per_soc_pct[st, order]
                                    / (5.0 / 60.0) / env.ev_max_power_kw[st, order]).clamp(max=1.0)
        executed_gap = torch.minimum(forced, headroom) - torch.minimum(acted, headroom)
        raised += int((executed_gap > 1e-3).sum())
        env.apply_action(acted, build_info=False, return_observation=False)
    metrics = env.get_metrics()
    assert raised == 0
    assert metrics["departing_evs"] > 0


def test_only_the_actors_learn_and_only_from_the_global_critic(monkeypatch):
    m, agent = _agent(monkeypatch, n_agent=3)
    agent.warmup_steps = 0
    generator = torch.Generator().manual_seed(2)
    from environment.observation_config import get_ev_feature_names
    names = get_ev_feature_names()
    s = torch.rand((24, agent.n, agent.s_dim), generator=generator)
    block = s[:, :, :agent.max_evs * m.EV_FEAT_DIM].view(24, agent.n, agent.max_evs, m.EV_FEAT_DIM)
    block[..., names.index("presence")] = (torch.rand((24, agent.n, agent.max_evs), generator=generator) < 0.6).float()
    block[..., names.index("needed_soc")] = torch.rand((24, agent.n, agent.max_evs), generator=generator) * 1.2 - 0.4
    s = s.to(agent.active_slot_mask.device)
    for k in range(23):
        a = agent.act(s[k], noise=True)
        agent.buf.cache(s[k], s[k + 1], a, torch.rand(agent.n, generator=generator), torch.rand((), generator=generator),
                        torch.zeros(agent.n), actual_station_powers=torch.rand(agent.n, generator=generator),
                        actual_ev_soc_changes=a.clone())
    local_before = [p.detach().clone() for p in agent.critics[0].parameters()]
    actor_before = [p.detach().clone() for p in agent.actors[0].parameters()]
    for _ in range(4):
        agent.update()
    assert all(torch.equal(b, p) for b, p in zip(local_before, agent.critics[0].parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(actor_before, agent.actors[0].parameters()))
    assert agent.actor_source_local_norms_before_clip[0] == 0.0

"""Per-EV local critics: each EV's action is credited with that EV's outcome.

The environment splits the station's local reward over its EVs exactly, the
replay finds the same EV in the next observation by id, the critic's station
value is the sum of per-EV values whose action gradients do not cross EVs, and
an update runs end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

STEPS = 120


def test_shaping_shares_sum_to_the_station_value():
    from environment.EVEnv import soc_progress_shaping

    generator = torch.Generator().manual_seed(0)
    for reduction in ("mean", "sum"):
        for clip in (0.08, 1e-4):
            prev = torch.rand(7, generator=generator) * 80
            new = prev + torch.rand(7, generator=generator) * 5 - 1
            target = torch.rand(7, generator=generator) * 100
            remaining = torch.randint(1, 200, (7,), generator=generator).float()
            station = soc_progress_shaping(prev, new, target, remaining, clip=clip, reduction=reduction)
            again, shares = soc_progress_shaping(prev, new, target, remaining, clip=clip, reduction=reduction,
                                                 return_per_ev=True)
            torch.testing.assert_close(again, station)
            torch.testing.assert_close(shares.sum(), station, atol=1e-7, rtol=1e-5)


def test_the_next_slot_follows_the_ev_id_not_the_physical_slot():
    from training.Agent.maddpg import MADDPG

    # Station 0: EV 5 leaves, EV 9 moves from slot 2 to slot 0, EV 7 from 1 to 2,
    # new EV 11 arrives. Station 1: slot 0's EV 0 stays (id 0 is a real id).
    ids_s = torch.tensor([[5, 7, 9, -1], [0, -1, -1, -1]])
    ids_s2 = torch.tensor([[9, 11, 7, -1], [0, 4, -1, -1]])
    got = MADDPG.next_slot_from_ids(ids_s, ids_s2)
    assert got.tolist() == [[-1, 2, 0, -1], [0, -1, -1, -1]]


def test_station_value_is_a_sum_without_cross_ev_action_gradients():
    from training.Agent.critic import LocalPerEvCritic
    from environment.observation_config import EV_FEAT_DIM, LOCAL_TAIL_DIM

    torch.manual_seed(0)
    n = 5
    critic = LocalPerEvCritic(EV_FEAT_DIM, n, hid=64)
    s = torch.rand(3, n * EV_FEAT_DIM + LOCAL_TAIL_DIM)
    presence = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    s[:, 0:n * EV_FEAT_DIM:EV_FEAT_DIM] = presence
    a = (torch.rand(3, n) * 2 - 1).requires_grad_(True)
    per_ev = critic.per_ev(s, a)
    torch.testing.assert_close(critic(s, a), per_ev.sum(dim=1, keepdim=True))
    assert torch.all(per_ev[presence == 0] == 0)
    for j in range(n):
        grad = torch.autograd.grad(per_ev[:, j].sum(), a, retain_graph=True)[0]
        others = torch.ones(n, dtype=torch.bool)
        others[j] = False
        assert torch.all(grad[:, others] == 0), f"Q_{j} depends on another EV's action"


def _rollout(monkeypatch):
    import environment.EVEnv as evenv_module
    from Config import EPISODE_STEPS, NUM_EVS, NUM_STATIONS
    from tools.evaluator import set_env_seed

    monkeypatch.setattr(evenv_module, "LOCAL_CRITIC_PER_EV", True)
    set_env_seed(4242)
    env = evenv_module.EVEnv(num_stations=int(NUM_STATIONS), num_evs=int(NUM_EVS),
                             episode_steps=int(EPISODE_STEPS))
    t = np.linspace(0.0, 4.0 * np.pi, int(EPISODE_STEPS))
    # Three EVs per station at 00:00, so every station's critics see EVs from
    # the first step (several stations are empty overnight otherwise).
    env.reset(net_demand_series=(300.0 * np.sin(t)).astype(np.float32),
              tol_narrow_series=np.full(int(EPISODE_STEPS), 60.0, dtype=np.float32),
              initial_evs_by_station=np.full(int(NUM_STATIONS), 3))
    return env


def test_per_ev_rewards_sum_to_the_station_reward_and_departures_land_on_their_slot(monkeypatch):
    env = _rollout(monkeypatch)
    rng = np.random.default_rng(7)
    departures = 0
    for _ in range(STEPS):
        env.begin_step()
        ids_before = env.slot_ev_ids()
        action = torch.as_tensor(rng.uniform(-1, 1, (env.num_stations, env.max_ev_per_station)), dtype=torch.float32)
        _obs, local, _g, _d, _info = env.apply_action(action, build_info=False, return_observation=False)
        ev = env.last_ev_local_rewards
        torch.testing.assert_close(ev.sum(dim=1), torch.as_tensor(local, dtype=torch.float32, device=ev.device),
                                   atol=1e-5, rtol=1e-5)
        assert torch.all(ev[ids_before < 0] == 0)
        # A departure reward is at least 1.5 or at most -0.5; shaping stays within the 0.08 clip.
        gone = (ids_before >= 0) & ~torch.isin(ids_before, env.slot_ev_ids())
        departures += int(gone.sum())
        assert torch.all(ev[gone].abs() >= 0.4)
        assert torch.all(ev[(ids_before >= 0) & ~gone].abs() <= 0.08 + 1e-6)
    assert departures > 0


@pytest.mark.parametrize("station_rule", [False, True])
def test_a_per_ev_update_runs_on_stored_environment_transitions(monkeypatch, station_rule):
    import training.Agent.maddpg as maddpg_module
    from Config import MAX_EV_PER_STATION
    from environment.normalize import normalize_observation

    env = _rollout(monkeypatch)
    monkeypatch.setattr(maddpg_module, "LOCAL_CRITIC_PER_EV", True)
    monkeypatch.setattr(maddpg_module, "STATION_RULE_ALLOCATION", station_rule)
    torch.manual_seed(0)
    agent = maddpg_module.MADDPG(s_dim=env._get_obs().shape[1], max_evs_per_station=MAX_EV_PER_STATION,
                                 n_agent=env.num_stations, batch=16)
    agent.warmup_steps = 0
    obs = normalize_observation(env.begin_step())
    for _ in range(40):
        act = agent.act(obs, env=env, noise=True)
        _, rl, rg, done, info = env.apply_action(act, build_info=False, return_observation=False)
        nxt = normalize_observation(env.begin_step())
        agent.cache_experience(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(nxt, dtype=torch.float32),
                               act, rl, torch.tensor(rg), torch.as_tensor(done, dtype=torch.float32),
                               actual_station_powers=info['raw_actor_station_powers'],
                               actual_ev_power_kw=info['raw_actor_ev_power_kw'])
        obs = nxt
    before = [p.detach().clone() for p in agent.critics[0].parameters()]
    for _ in range(4):
        agent.update()
    after = list(agent.critics[0].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert all(np.isfinite(agent.critic_losses))
    r_ev, next_slot = agent.buf.per_ev_batch()
    assert r_ev.shape == next_slot.shape == (16, env.num_stations, MAX_EV_PER_STATION)

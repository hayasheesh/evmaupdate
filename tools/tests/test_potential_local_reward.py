"""EVMA_LOCAL_REWARD_MODE=potential: urgency-weighted shortfall reward per EV per step."""

from __future__ import annotations

import numpy as np
import torch

COEF, GAIN, WINDOW, GAMMA = 1.5, 3.0, 48, 0.985


def _rewards(prev, new, target, steps_left):
    from environment.EVEnv import soc_potential_rewards

    return soc_potential_rewards(prev, new, target, steps_left, coef=COEF, urgency_gain=GAIN, window=WINDOW,
                                 gamma=GAMMA)


def test_an_evs_discounted_rewards_add_up_to_arrival_minus_departure_shortfall():
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        stay = int(torch.randint(1, 150, (1,), generator=generator))
        soc = float(torch.rand(1, generator=generator)) * 60
        target = torch.tensor([60.0 + float(torch.rand(1, generator=generator)) * 40])
        u0 = 1 + GAIN * min(max((WINDOW - stay) / WINDOW, 0.0), 1.0)
        arrival = COEF * u0 * max(float(target) - soc, 0.0) / 100
        total = 0.0
        for t in range(stay):
            new = soc + float(torch.rand(1, generator=generator)) * 4 - 1.5
            r = _rewards(torch.tensor([soc]), torch.tensor([new]), target, torch.tensor([float(stay - t)]))
            total += GAMMA ** t * float(r)
            soc = new
        departure = COEF * (1 + GAIN) * max(float(target) - soc, 0.0) / 100
        np.testing.assert_allclose(total, arrival - GAMMA ** stay * departure, rtol=1e-5, atol=1e-6)


def test_closing_the_shortfall_pays_more_near_departure_and_waiting_costs_inside_the_window():
    target = torch.tensor([80.0])
    early = _rewards(torch.tensor([50.0]), torch.tensor([52.0]), target, torch.tensor([100.0]))
    late = _rewards(torch.tensor([50.0]), torch.tensor([52.0]), target, torch.tensor([10.0]))
    assert float(late) > float(early) > 0
    waiting = _rewards(torch.tensor([50.0]), torch.tensor([50.0]), target, torch.tensor([10.0]))
    assert float(waiting) < 0
    at_target = _rewards(torch.tensor([85.0]), torch.tensor([80.0]), target, torch.tensor([5.0]))
    assert float(at_target) == 0.0


def test_the_environment_pays_it_per_ev_with_only_a_miss_penalty_at_departure(monkeypatch):
    import environment.EVEnv as evenv_module
    from Config import EPISODE_STEPS, NUM_EVS, NUM_STATIONS, GAMMA as CONFIG_GAMMA
    from tools.evaluator import set_env_seed

    assert CONFIG_GAMMA == evenv_module.LOCAL_POTENTIAL_GAMMA
    monkeypatch.setattr(evenv_module, "LOCAL_CRITIC_PER_EV", True)
    monkeypatch.setattr(evenv_module, "LOCAL_REWARD_MODE", "potential")
    set_env_seed(4242)
    env = evenv_module.EVEnv(num_stations=int(NUM_STATIONS), num_evs=int(NUM_EVS), episode_steps=int(EPISODE_STEPS))
    t = np.linspace(0.0, 4.0 * np.pi, int(EPISODE_STEPS))
    env.reset(net_demand_series=(300.0 * np.sin(t)).astype(np.float32),
              tol_narrow_series=np.full(int(EPISODE_STEPS), 60.0, dtype=np.float32))
    rng = np.random.default_rng(7)
    hits = misses = 0
    for _ in range(160):
        env.begin_step()
        ids = env.slot_ev_ids()
        before = []
        for st in range(env.num_stations):
            order = env._get_sorted_active_evs(st)
            before.append((env.soc[st, order].clone(), env.target[st, order].clone(),
                           torch.clamp(env._remaining_action_steps(env.depart[st, order]), min=1.0),
                           env.ev_max_power_kw[st, order] * env.ev_soc_step_per_kw[st, order]))
        action = torch.as_tensor(rng.uniform(-1, 1, (env.num_stations, env.max_ev_per_station)), dtype=torch.float32)
        _o, local, _g, _d, _i = env.apply_action(action, build_info=False, return_observation=False)
        ev = env.last_ev_local_rewards
        torch.testing.assert_close(ev.sum(dim=1), torch.as_tensor(local, dtype=torch.float32, device=ev.device),
                                   atol=1e-5, rtol=1e-5)
        after = env.slot_ev_ids()
        for st, (soc, target, steps, soc_per_full_step) in enumerate(before):
            k = soc.numel()
            if k == 0:
                continue
            new = torch.clamp(soc + action[st, :k].to(soc.device) * soc_per_full_step, 0.0, 100.0)
            expected = evenv_module.soc_potential_rewards(soc, new, target, steps)
            # No hit bonus at departure: a departing EV gets its last potential
            # term, and 0.5 less if it leaves below target.
            leaving = ~torch.isin(ids[st, :k], after[st])
            missed = leaving & (new < target)
            expected = expected - 0.5 * missed.float()
            torch.testing.assert_close(ev[st, :k], expected, atol=1e-4, rtol=1e-4)
            hits += int((leaving & ~missed).sum())
            misses += int(missed.sum())
    assert hits > 0 and misses > 0

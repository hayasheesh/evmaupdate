"""The training-time force floor is the execution-time force-charging rule."""

from __future__ import annotations

import types

import torch

from environment.force_floor import NO_FLOOR, force_floor_fraction

STEP_H = 5.0 / 60.0


class _Station:
    """The EV state apply_force_charging and EVEnv.apply_force_floor read."""

    def __init__(self, generator, n=6, stations=2):
        self.num_stations = stations
        self.step_count = int(torch.randint(1, 200, (1,), generator=generator))
        self.ev_mask = torch.ones((stations, n), dtype=torch.bool)
        self.soc = torch.rand((stations, n), generator=generator) * 100
        self.target = torch.rand((stations, n), generator=generator) * 100
        self.depart = self.step_count + torch.randint(0, 30, (stations, n), generator=generator)
        self.ev_capacity_kwh = torch.tensor([50.0, 60.0, 75.0, 90.0, 100.0, 120.0])[
            torch.randint(0, 6, (stations, n), generator=generator)]
        self.ev_max_power_kw = torch.tensor([11.0, 19.2, 27.5, 50.0])[
            torch.randint(0, 4, (stations, n), generator=generator)]
        self.ev_kwh_per_soc_pct = self.ev_capacity_kwh / 100.0
        self.ev_soc_step_per_kw = STEP_H * 100.0 / self.ev_capacity_kwh
        self.metrics = {}

    def _sort_active_evs(self, station, active):
        return active[torch.argsort(self.depart[station, active])]

    def _get_sorted_active_evs(self, station):
        return self._sort_active_evs(station, torch.nonzero(self.ev_mask[station]).squeeze(-1))

    def _remaining_action_steps(self, depart):
        return torch.clamp(depart.float() - float(self.step_count) + 1.0, min=0.0)


def test_the_floor_is_the_execution_force_rule():
    from training.system_controller import apply_force_charging

    generator = torch.Generator().manual_seed(0)
    for _ in range(30):
        env = _Station(generator)
        actions = torch.rand((env.num_stations, 6), generator=generator) * 2 - 1
        forced, _, _ = apply_force_charging(actions, env, slack_kwh=0.1)
        for st in range(env.num_stations):
            idx = env._get_sorted_active_evs(st)
            floor = force_floor_fraction(env.target[st, idx] - env.soc[st, idx],
                                         env._remaining_action_steps(env.depart[st, idx]),
                                         env.ev_capacity_kwh[st, idx], env.ev_max_power_kw[st, idx],
                                         STEP_H, slack_kwh=0.1)
            torch.testing.assert_close(forced[st], torch.maximum(actions[st], floor))


def test_the_environment_raises_actions_to_the_floor_and_counts_the_points():
    import environment.EVEnv as ev

    env = _Station(torch.Generator().manual_seed(1))
    env._train_forced_kwh = None
    actions = -torch.ones((env.num_stations, 6))
    raised, points = ev.EVEnv.apply_force_floor(env, actions)
    for st in range(env.num_stations):
        idx = env._get_sorted_active_evs(st)
        floor = force_floor_fraction(env.target[st, idx] - env.soc[st, idx],
                                     env._remaining_action_steps(env.depart[st, idx]),
                                     env.ev_capacity_kwh[st, idx], env.ev_max_power_kw[st, idx],
                                     STEP_H, slack_kwh=0.1)
        torch.testing.assert_close(raised[st], torch.maximum(actions[st], floor))
        added = torch.maximum(actions[st], floor) - actions[st]
        expected = (added * env.ev_max_power_kw[st, idx] * env.ev_soc_step_per_kw[st, idx]).sum()
        torch.testing.assert_close(points[st], expected)


def test_no_floor_while_the_target_stays_reachable():
    floor = force_floor_fraction(torch.tensor([10.0]), torch.tensor([50.0]), torch.tensor([60.0]),
                                 torch.tensor([11.0]), STEP_H, slack_kwh=0.1)
    assert float(floor) == NO_FLOOR


def test_the_learner_reads_the_same_floor_from_the_observation(monkeypatch):
    import training.Agent.maddpg as m
    from environment.observation_config import get_ev_feature_names
    from EnvConfig import EV_CAPACITY_OBS_SCALE_KWH, EV_CHARGER_POWER_OBS_SCALE_KW, EPISODE_STEPS

    monkeypatch.setattr(m, "TRAIN_FORCE_CHARGING", True)
    names = get_ev_feature_names()
    need = torch.tensor([[30.0, 5.0, 60.0]])
    remaining = torch.tensor([[4.0, 40.0, 12.0]])
    capacity = torch.tensor([[60.0, 75.0, 100.0]])
    power = torch.tensor([[11.0, 19.2, 50.0]])
    block = torch.zeros((1, 3, len(names)))
    block[..., names.index("presence")] = 1.0
    block[..., names.index("remaining_time")] = remaining / EPISODE_STEPS
    block[..., names.index("needed_soc")] = need / 100.0
    block[..., names.index("battery_capacity_kwh")] = capacity / EV_CAPACITY_OBS_SCALE_KWH
    block[..., names.index("max_power_kw")] = power / EV_CHARGER_POWER_OBS_SCALE_KW
    agent = types.SimpleNamespace()
    got = m.MADDPG._force_floor_kw(agent, block, capacity, power)
    expected = force_floor_fraction(need, remaining, capacity, power, STEP_H, slack_kwh=0.1) * power
    torch.testing.assert_close(got, expected)


def test_the_learner_lifts_proposed_actions_and_keeps_their_gradient():
    import training.Agent.maddpg as m

    agent = types.SimpleNamespace()
    actions = torch.tensor([[[-10.0, 5.0]]], requires_grad=True)
    floor = torch.tensor([[[2.0, -11.0]]])
    out, _ = m.MADDPG._apply_soc_constraint(
        agent, actions, torch.tensor([[[50.0, 50.0]]]), use_ste=True,
        capacity_kwh=torch.tensor([[[60.0, 60.0]]]), max_power_kw=torch.tensor([[[11.0, 11.0]]]),
        floor_kw=floor,
    )
    torch.testing.assert_close(out.detach(), torch.tensor([[[2.0, 5.0]]]))
    out.sum().backward()
    torch.testing.assert_close(actions.grad, torch.ones_like(actions))

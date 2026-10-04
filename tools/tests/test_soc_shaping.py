"""The per-step SoC shaping reward of one station."""

from __future__ import annotations

import torch

from environment.EVEnv import soc_progress_shaping


def _legacy(prev, new, target, remaining, coef=0.5, clip=0.08, gain=1.0, steps=48):
    deficit_prev = torch.clamp(target - prev, min=0.0) / 100.0
    deficit_curr = torch.clamp(target - new, min=0.0) / 100.0
    window = max(float(steps), 1.0)
    urgency = 1.0 + gain * torch.clamp((window - remaining) / window, 0.0, 1.0)
    progress = ((deficit_prev - deficit_curr) * urgency).mean()
    return torch.clamp(coef * progress, -clip, clip)


def test_the_defaults_reproduce_the_previous_reward():
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        n = int(torch.randint(1, 12, (1,), generator=generator))
        prev = torch.rand(n, generator=generator) * 100
        new = torch.clamp(prev + (torch.rand(n, generator=generator) - 0.5) * 6, 0, 100)
        target = torch.rand(n, generator=generator) * 100
        remaining = torch.randint(1, 200, (n,), generator=generator).float()
        got = soc_progress_shaping(prev, new, target, remaining, reduction="mean", surplus_coef=0.0, clip=0.08)
        torch.testing.assert_close(got, _legacy(prev, new, target, remaining))


def test_summing_keeps_one_ev_from_being_diluted_by_idle_neighbours():
    target = torch.tensor([50.0])
    alone = soc_progress_shaping(torch.tensor([30.0]), torch.tensor([31.5]), target,
                                 torch.tensor([100.0]), reduction="sum", surplus_coef=0.0, clip=1.0)
    prev = torch.tensor([30.0, 40.0, 40.0, 40.0, 40.0])
    new = torch.tensor([31.5, 40.0, 40.0, 40.0, 40.0])
    crowd_target = torch.full((5,), 50.0)
    remaining = torch.full((5,), 100.0)
    summed = soc_progress_shaping(prev, new, crowd_target, remaining, reduction="sum", surplus_coef=0.0, clip=1.0)
    averaged = soc_progress_shaping(prev, new, crowd_target, remaining, reduction="mean", surplus_coef=0.0, clip=1.0)
    torch.testing.assert_close(summed, alone)
    torch.testing.assert_close(averaged, alone / 5)


def test_the_surplus_term_prefers_taking_energy_from_satisfied_evs():
    remaining = torch.tensor([100.0])
    target = torch.tensor([50.0])

    def shaping(prev, new, surplus_coef=0.25):
        return float(soc_progress_shaping(torch.tensor([prev]), torch.tensor([new]), target, remaining,
                                          reduction="sum", surplus_coef=surplus_coef, clip=1.0))

    assert shaping(70.0, 71.5) < 0          # charging an EV already above target costs
    assert shaping(70.0, 68.5) > 0          # discharging it toward target pays
    assert shaping(30.0, 28.5) < 0          # discharging a short EV costs
    assert shaping(70.0, 68.5) > shaping(30.0, 28.5)
    assert shaping(70.0, 71.5, surplus_coef=0.0) == 0.0   # without the term the reward is blind above target
    assert abs(shaping(70.0, 71.5)) == abs(shaping(70.0, 68.5))


def test_the_total_is_clipped():
    prev = torch.zeros(20)
    new = torch.full((20,), 3.0)
    got = soc_progress_shaping(prev, new, torch.full((20,), 80.0), torch.full((20,), 10.0),
                               reduction="sum", surplus_coef=0.0, clip=0.08)
    torch.testing.assert_close(got, torch.tensor(0.08))


def test_laxity_weighs_a_tight_ev_above_a_slack_one_leaving_at_the_same_time():
    from environment.EVEnv import soc_progress_shaping

    target = torch.tensor([60.0])
    remaining = torch.tensor([30.0])
    rate = torch.tensor([2.0])      # SoC points per step at full power

    def discharge_cost(soc, basis):
        return float(soc_progress_shaping(torch.tensor([soc]), torch.tensor([soc - 1.0]), target, remaining,
                                          full_power_soc_per_step=rate, urgency_basis=basis,
                                          urgency_gain=3.0, reduction="sum", surplus_coef=0.0, clip=1.0))

    # 58%: 1 step of charging needed, 29 steps of slack. 10%: 25 needed, 5 of slack.
    assert discharge_cost(10.0, "laxity") < discharge_cost(58.0, "laxity") < 0
    torch.testing.assert_close(torch.tensor(discharge_cost(10.0, "time")), torch.tensor(discharge_cost(58.0, "time")))


def test_an_ev_that_can_no_longer_make_it_gets_the_full_weight():
    from environment.EVEnv import soc_progress_shaping

    cost = soc_progress_shaping(torch.tensor([10.0]), torch.tensor([9.0]), torch.tensor([90.0]), torch.tensor([5.0]),
                                full_power_soc_per_step=torch.tensor([2.0]), urgency_basis="laxity",
                                urgency_gain=3.0, reduction="sum", surplus_coef=0.0, clip=1.0)
    torch.testing.assert_close(cost, torch.tensor(-0.5 * 0.01 * 4.0))


def test_the_step_departure_reward_is_the_previous_one():
    from environment.EVEnv import departure_rewards

    final = torch.tensor([50.0, 49.0, 30.0, 0.0])
    target = torch.full((4,), 50.0)
    expected = torch.tensor([1.5, -(0.5 + 0.05 * 1), -(0.5 + 0.05 * 20), -(0.5 + 0.05 * 50)])
    torch.testing.assert_close(departure_rewards(final, target, mode="step"), expected)


def test_the_smooth_departure_reward_is_continuous_and_steeper_for_deep_shortfalls():
    from environment.EVEnv import departure_rewards

    target = torch.full((5,), 50.0)
    final = torch.tensor([50.0, 49.9, 40.0, 30.0, 10.0])
    r = departure_rewards(final, target, mode="smooth", smooth_linear=6.0, smooth_quadratic=12.0)
    torch.testing.assert_close(r[0], torch.tensor(1.5))
    assert abs(float(r[1]) - 1.5) < 0.01                      # no drop at the target
    drops = [float(r[i] - r[i + 1]) for i in (2, 3)]           # 10 points deeper each time
    assert drops[1] > drops[0] > 0                             # each further 10 points costs more

"""The per-step SoC shaping reward of one station."""

from __future__ import annotations

import torch

from environment.EVEnv import departure_rewards, soc_progress_shaping


def _reference(prev, new, target, remaining, coef=0.5, clip=0.08, gain=1.0, steps=48):
    deficit_prev = torch.clamp(target - prev, min=0.0) / 100.0
    deficit_curr = torch.clamp(target - new, min=0.0) / 100.0
    window = max(float(steps), 1.0)
    urgency = 1.0 + gain * torch.clamp((window - remaining) / window, 0.0, 1.0)
    progress = ((deficit_prev - deficit_curr) * urgency).mean()
    return torch.clamp(coef * progress, -clip, clip)


def test_the_shaping_is_the_urgency_weighted_mean_deficit_fall():
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        n = int(torch.randint(1, 12, (1,), generator=generator))
        prev = torch.rand(n, generator=generator) * 100
        new = torch.clamp(prev + (torch.rand(n, generator=generator) - 0.5) * 6, 0, 100)
        target = torch.rand(n, generator=generator) * 100
        remaining = torch.randint(1, 200, (n,), generator=generator).float()
        got = soc_progress_shaping(prev, new, target, remaining, clip=0.08)
        torch.testing.assert_close(got, _reference(prev, new, target, remaining))


def test_one_ev_is_averaged_over_the_station():
    target = torch.tensor([50.0])
    alone = soc_progress_shaping(torch.tensor([30.0]), torch.tensor([31.5]), target,
                                 torch.tensor([100.0]), clip=1.0)
    prev = torch.tensor([30.0, 40.0, 40.0, 40.0, 40.0])
    new = torch.tensor([31.5, 40.0, 40.0, 40.0, 40.0])
    averaged = soc_progress_shaping(prev, new, torch.full((5,), 50.0), torch.full((5,), 100.0), clip=1.0)
    torch.testing.assert_close(averaged, alone / 5)


def test_the_reward_is_blind_above_target():
    got = soc_progress_shaping(torch.tensor([70.0]), torch.tensor([71.5]), torch.tensor([50.0]),
                               torch.tensor([100.0]), clip=1.0)
    assert float(got) == 0.0


def test_the_total_is_clipped():
    prev = torch.zeros(20)
    new = torch.full((20,), 30.0)
    got = soc_progress_shaping(prev, new, torch.full((20,), 80.0), torch.full((20,), 10.0), clip=0.08)
    torch.testing.assert_close(got, torch.tensor(0.08))


def test_the_departure_reward():
    final = torch.tensor([50.0, 49.0, 30.0, 0.0])
    target = torch.full((4,), 50.0)
    expected = torch.tensor([1.5, -(0.5 + 0.05 * 1), -(0.5 + 0.05 * 20), -(0.5 + 0.05 * 50)])
    torch.testing.assert_close(departure_rewards(final, target), expected)

"""LOCAL_REWARD_SCALE multiplies the station reward in the local critics' TD target."""

from __future__ import annotations

import torch
import torch.nn.functional as F

N_AGENTS = 2


def _local_targets(monkeypatch, scale: float) -> list[torch.Tensor]:
    import training.Agent.maddpg as maddpg_module
    from Config import MAX_EV_PER_STATION
    from environment.observation_config import local_obs_dim

    torch.manual_seed(0)
    agent = maddpg_module.MADDPG(
        s_dim=local_obs_dim(MAX_EV_PER_STATION), max_evs_per_station=MAX_EV_PER_STATION,
        n_agent=N_AGENTS, batch=8,
    )
    agent.warmup_steps = 0
    agent.gamma = 0.0
    generator = torch.Generator().manual_seed(3)
    for _ in range(16):
        a = torch.rand((N_AGENTS, agent.a_dim), generator=generator) * 2.0 - 1.0
        agent.buf.cache(
            torch.rand((N_AGENTS, agent.s_dim), generator=generator),
            torch.rand((N_AGENTS, agent.s_dim), generator=generator),
            a,
            torch.rand((N_AGENTS,), generator=generator),
            torch.rand((), generator=generator),
            torch.zeros(N_AGENTS),
            actual_station_powers=torch.rand((N_AGENTS,), generator=generator),
            actual_ev_soc_changes=a.clone(),
        )
    captured: list[torch.Tensor] = []
    original = F.smooth_l1_loss

    def recording(input, target, *args, **kwargs):
        captured.append(target.detach().clone())
        return original(input, target, *args, **kwargs)

    monkeypatch.setattr(maddpg_module, "LOCAL_REWARD_SCALE", scale)
    monkeypatch.setattr(maddpg_module.F, "smooth_l1_loss", recording)
    torch.manual_seed(1)
    agent.update()
    # The local critics are updated first: two twin losses per station.
    return captured[:2 * N_AGENTS]


def test_the_local_td_target_scales_with_the_reward_multiplier(monkeypatch):
    base = _local_targets(monkeypatch, 1.0)
    tripled = _local_targets(monkeypatch, 3.0)
    for expected, actual in zip(base, tripled):
        torch.testing.assert_close(actual, 3.0 * expected)
    assert any(t.abs().max() > 0 for t in base)

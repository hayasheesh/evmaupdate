"""Gradient through the rating and SoC bounds on the actor's update path."""

from __future__ import annotations

import pytest
import torch

CAPACITY_KWH = 60.0
RATED_KW = 50.0


def _step(monkeypatch, mode, proposal_kw, soc, want_more_power):
    """Clip one EV's proposal on the actor path; return (executed kW, d loss / d proposal)."""
    import training.Agent.maddpg as M

    monkeypatch.setattr(M, "ACTOR_BOUND_GRADIENT", mode)
    proposal = torch.tensor([[[proposal_kw]]], requires_grad=True)
    soc_step_per_kw = torch.tensor([[[M.POWER_TO_ENERGY * 100.0 / CAPACITY_KWH]]])
    executed, station = M.MADDPG._apply_soc_constraint(
        None, proposal, torch.tensor([[[soc]]]), actor_path=True,
        max_power_kw=torch.tensor([[[RATED_KW]]]),
        soc_step_per_kw=soc_step_per_kw, kw_per_soc_step=1.0 / soc_step_per_kw,
    )
    loss = -station.sum() if want_more_power else station.sum()
    loss.backward()
    return float(executed.detach()), float(proposal.grad)


@pytest.mark.parametrize("mode", ["inward", "straight_through"])
@pytest.mark.parametrize("proposal_kw, soc, executed_kw", [
    (30.0, 100.0, 0.0),     # full EV cannot take charge
    (-30.0, 0.0, 0.0),      # empty EV cannot give any
    (20.0, 50.0, 20.0),     # inside both bounds
    (80.0, 50.0, RATED_KW),  # above the rating
])
def test_the_executed_power_does_not_depend_on_the_gradient_mode(monkeypatch, mode, proposal_kw, soc, executed_kw):
    executed, _ = _step(monkeypatch, mode, proposal_kw, soc, want_more_power=True)
    assert executed == pytest.approx(executed_kw)


def test_a_full_ev_gets_no_gradient_asking_for_more_charge(monkeypatch):
    _, grad = _step(monkeypatch, "inward", 30.0, 100.0, want_more_power=True)
    assert grad == 0.0


def test_a_full_ev_still_gets_the_gradient_asking_for_less_charge(monkeypatch):
    _, grad = _step(monkeypatch, "inward", 30.0, 100.0, want_more_power=False)
    assert grad > 0.0   # a descent step lowers the proposal toward what the EV can take


def test_an_empty_ev_gets_no_gradient_asking_for_more_discharge(monkeypatch):
    _, grad = _step(monkeypatch, "inward", -30.0, 0.0, want_more_power=False)
    assert grad == 0.0


def test_an_empty_ev_still_gets_the_gradient_asking_for_less_discharge(monkeypatch):
    _, grad = _step(monkeypatch, "inward", -30.0, 0.0, want_more_power=True)
    assert grad < 0.0


def test_straight_through_pushes_a_full_ev_to_charge_more(monkeypatch):
    _, grad = _step(monkeypatch, "straight_through", 30.0, 100.0, want_more_power=True)
    assert grad < 0.0


@pytest.mark.parametrize("want_more_power", [True, False])
def test_inside_the_bounds_both_modes_give_the_plain_gradient(monkeypatch, want_more_power):
    _, inward = _step(monkeypatch, "inward", 20.0, 50.0, want_more_power)
    _, through = _step(monkeypatch, "straight_through", 20.0, 50.0, want_more_power)
    assert inward == pytest.approx(through)
    assert abs(inward) == pytest.approx(1.0)


def test_a_fixed_rating_is_rebuilt_for_present_evs_only() -> None:
    import numpy as np
    from environment.EVEnv import EVEnv
    from environment.normalize import normalize_observation
    from environment.observation_config import EV_FEAT_DIM, EV_FEATURE_NAMES, OBSERVE_EV_MAX_POWER
    from tools.evaluator import set_env_seed
    from training.system_controller import build_agent

    if OBSERVE_EV_MAX_POWER:
        pytest.skip("ratings differ between EVs; the rating is observed")
    assert "max_power_kw" not in EV_FEATURE_NAMES
    set_env_seed(7)
    env = EVEnv()
    env.reset(net_demand_series=np.zeros(288, dtype=np.float32), service_date="2024-04-02")
    obs = normalize_observation(env.begin_step())
    agent = build_agent(env)
    block = obs[:, : env.max_ev_per_station * EV_FEAT_DIM].reshape(env.num_stations, env.max_ev_per_station, EV_FEAT_DIM)
    _soc, _cap, max_power = agent._extract_ev_physics(block.to(agent.device) if hasattr(agent, "device") else block)
    expected = torch.zeros_like(max_power)
    for st in range(env.num_stations):
        order = env._get_sorted_active_evs(st)
        expected[st, : order.numel()] = env.ev_max_power_kw[st, order].to(expected.device)
    assert torch.equal(max_power.cpu(), expected.cpu())

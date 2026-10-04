from __future__ import annotations

import pytest
import numpy as np

import training.evaluate_controller_precision as precision_eval


class _FakeAgent:
    def __init__(self) -> None:
        self.test_mode = False
        self.epsilon = 0.37
        self.ou_noise_scale = 0.61

    def set_test_mode(self, mode: bool) -> None:
        self.test_mode = bool(mode)
        if mode:
            self.epsilon = 0.0
            self.ou_noise_scale = 0.0


def test_controller_precision_restores_training_and_exploration(monkeypatch) -> None:
    agent = _FakeAgent()

    def fake_impl(inner_agent, _fixed_bid, **kwargs):
        assert inner_agent.test_mode is True
        return {"ok": True}

    monkeypatch.setattr(precision_eval, "_evaluate_controller_precision_impl", fake_impl)

    assert precision_eval.evaluate_controller_precision(agent, {}) == {"ok": True}
    assert agent.test_mode is False
    assert agent.epsilon == pytest.approx(0.37)
    assert agent.ou_noise_scale == pytest.approx(0.61)

def test_controller_precision_restores_mode_after_error(monkeypatch) -> None:
    agent = _FakeAgent()

    def failing_impl(_agent, _fixed_bid, **_kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(precision_eval, "_evaluate_controller_precision_impl", failing_impl)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        precision_eval.evaluate_controller_precision(agent, {})

    assert agent.test_mode is False
    assert agent.epsilon == pytest.approx(0.37)
    assert agent.ou_noise_scale == pytest.approx(0.61)


def test_primary_evaluation_pipelines_do_not_mix_controllers() -> None:
    marl = precision_eval.evaluation_pipeline_settings("marl_force")
    assert marl == {
        "force_layer": True,
        "central_allocator": False,
        "residual_bess": False,
        "response_source": "ev",
        "use_actor": True,
    }

    central = precision_eval.evaluation_pipeline_settings("rule_based_central")
    assert central == {
        "force_layer": True,
        "central_allocator": True,
        "residual_bess": False,
        "response_source": "ev",
        "use_actor": False,
    }

    with pytest.raises(ValueError, match="unknown evaluation_pipeline"):
        precision_eval.evaluation_pipeline_settings("marl_plus_everything")


def test_directional_block_tracking_separates_up_only_failure() -> None:
    metrics = precision_eval.directional_block_tracking(
        dispatch_kw_per_step=np.asarray([-100.0, -100.0, 100.0, 100.0, 0.0, 0.0]),
        response_kw_per_step=np.asarray([0.0, 0.0, 100.0, 100.0, 0.0, 0.0]),
        baseline_kw=0.0,
        tolerance_kw_per_step=np.full(6, 10.0),
    )

    assert metrics["up_active_steps"] == 2
    assert metrics["down_active_steps"] == 2
    assert metrics["idle_active_steps"] == 2
    assert metrics["up_pass_II"] is False
    assert metrics["down_pass_II"] is True
    assert metrics["idle_pass_II"] is True
    assert metrics["up_stay_rate"] == pytest.approx(0.0)
    assert metrics["down_stay_rate"] == pytest.approx(1.0)


def test_direction_step_metrics_separates_up_only_failure() -> None:
    # Two 30-min blocks (6 steps each). Block 0 is participating (up-only
    # award) and misses tracking during every up-active step but tracks the
    # idle steps perfectly. Block 1 is participating (down-only award) and
    # tracks every down-active step perfectly. If block failures were not
    # attributed per direction, up and down rates would come out identical
    # (the historical bug); this asserts they diverge.
    steps_per_block = 6
    dispatch = np.asarray(
        [-100.0, -100.0, -100.0, -100.0, 0.0, 0.0] +   # block 0: up-active x4, idle x2
        [100.0, 100.0, 100.0, 100.0, 0.0, 0.0],          # block 1: down-active x4, idle x2
        dtype=float,
    )
    response = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] +                 # block 0: response stuck at baseline -> up misses
        [100.0, 100.0, 100.0, 100.0, 0.0, 0.0],          # block 1: response tracks perfectly
        dtype=float,
    )
    baseline = np.zeros(12, dtype=float)
    tolerance = np.full(12, 10.0)
    up_plan = np.asarray([50.0, 0.0])
    down_plan = np.asarray([0.0, 50.0])
    participation = precision_eval.participation_by_step(
        up_plan, down_plan, steps_per_block=steps_per_block, steps=12,
    )

    metrics = precision_eval.direction_step_metrics(
        dispatch,
        response,
        baseline,
        tolerance,
        participation,
        steps_per_block=steps_per_block,
    )

    assert metrics["up_active_steps"] == 4
    assert metrics["down_active_steps"] == 4
    assert metrics["idle_active_steps"] == 4
    assert metrics["up_step_pass_rate"] < metrics["down_step_pass_rate"]
    assert metrics["up_step_pass_rate"] == pytest.approx(0.0)
    assert metrics["down_step_pass_rate"] == pytest.approx(1.0)
    assert metrics["idle_step_pass_rate"] == pytest.approx(1.0)
    assert metrics["up_step_failed_blocks"] == [0]
    assert metrics["down_step_failed_blocks"] == []
    assert metrics["idle_step_failed_blocks"] == []

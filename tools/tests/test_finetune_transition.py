"""What happens in the first moments of a fine-tune.

Two things went wrong at the hand-off from pretrain to a single operating day,
and neither shows up in an end-of-run comparison.

The warmup gate counted the pretrain's own transitions, so updates began on the
new day's first step.  The offline and online regions are sampled separately, so
the batch drawn there was half pretrain and half one transition repeated two
hundred and fifty-six times.

The exploration schedule was scaled but not shortened, and a warm start does not
restore the episode counter, so it restarted from the top.  The slot-level
random override came back at 30% against the 0.5% the pretrain had reached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.Agent.replay_buffer import ReplayBuffer
from training.agent_checkpoint import apply_finetune_exploration_schedule

N_AGENTS = 3
MAX_EVS = 4


def _cache_one(buf: ReplayBuffer, value: float) -> None:
    s = torch.full((N_AGENTS, 6), value, dtype=torch.float32)
    a = torch.full((N_AGENTS, MAX_EVS), value, dtype=torch.float32)
    buf.cache(
        s,
        s.clone(),
        a,
        torch.full((N_AGENTS,), value, dtype=torch.float32),
        torch.tensor([value], dtype=torch.float32),
        torch.zeros(N_AGENTS, dtype=torch.float32),
        actual_station_powers=torch.zeros(N_AGENTS, dtype=torch.float32),
        actual_ev_soc_changes=a.clone(),
    )


def test_pending_size_is_the_whole_buffer_without_an_offline_region():
    buf = ReplayBuffer(cap=64)
    for _ in range(10):
        _cache_one(buf, 1.0)
    assert buf.offline_size == 0
    assert buf.pending_size == buf.size == 10


def test_pending_size_ignores_the_inherited_transitions():
    buf = ReplayBuffer(cap=256)
    for _ in range(40):
        _cache_one(buf, 1.0)
    assert buf.mark_offline_region(ratio_initial=0.5, ratio_final=0.25, decay_steps=100)

    # The gate the pretrain's transitions used to clear on their own.
    assert buf.pending_size == 0
    for expected in (1, 2, 3):
        _cache_one(buf, 0.0)
        assert buf.pending_size == expected
    assert buf.size == 43, "the inherited transitions are still there"


def test_first_online_step_would_repeat_one_transition():
    """Why the gate matters, stated as the thing it prevents."""

    buf = ReplayBuffer(cap=256)
    for _ in range(40):
        _cache_one(buf, 1.0)
    buf.mark_offline_region(ratio_initial=0.5, ratio_final=0.5, decay_steps=0)
    _cache_one(buf, 0.0)

    idxs = buf._mixed_start_indices(32, 1)
    online = idxs[idxs >= buf.offline_size]
    assert online.numel() == 16, "half the batch is drawn from the online region"
    assert torch.all(online == buf.offline_size), (
        "every online draw is the same single transition"
    )


def _pretrain_agent():
    class Agent:
        epsilon_start_episode = 1
        epsilon_end_episode = 1000
        epsilon_initial = 1.0
        epsilon_final = 0.005
        epsilon = 0.005
        ou_noise_scale_initial = 1.0
        ou_noise_scale_final = 0.2
        ou_noise_scale = 0.2854

    return Agent()


def _epsilon_at(agent, episode: int) -> float:
    from training.Agent.noise import linear_epsilon_decay

    return linear_epsilon_decay(
        episode,
        agent.epsilon_start_episode,
        agent.epsilon_end_episode,
        agent.epsilon_initial,
        agent.epsilon_final,
    )


def test_old_scaling_left_the_finetune_exploring_to_the_end():
    """What the replaced behaviour did, so the number is on the record.

    Scaling every endpoint by the fine-tune factor and keeping the pretrain's
    schedule length. The helper that did this is gone; the arithmetic is not,
    and it is the reason the fine-tune schedule is now its own.
    """

    agent = _pretrain_agent()
    agent.epsilon_initial *= 0.3       # 1.0   -> 0.3
    agent.epsilon_final *= 0.3         # 0.005 -> 0.0015
    # The length is untouched, and the warm start does not restore the episode
    # counter, so a 300-episode fine-tune walks only the first 30% of it.
    assert _epsilon_at(agent, 1) == pytest.approx(0.300, abs=1e-3)
    assert _epsilon_at(agent, 300) == pytest.approx(0.211, abs=1e-3)


def test_finetune_schedule_decays_inside_its_own_budget():
    agent = _pretrain_agent()
    applied = apply_finetune_exploration_schedule(agent, noise_scale=0.3, episodes=300)

    assert applied["epsilon_end_episode"] == 300
    assert _epsilon_at(agent, 1) == pytest.approx(0.05)
    assert _epsilon_at(agent, 300) == pytest.approx(0.005)
    # Far below the 21% the scaled pretrain schedule still had at this point.
    assert _epsilon_at(agent, 300) < 0.01


def test_finetune_schedule_still_scales_the_ou_noise():
    agent = _pretrain_agent()
    apply_finetune_exploration_schedule(agent, noise_scale=0.3, episodes=300)

    # Unchanged on purpose: 0.3 lands within a few percent of the 0.2854 the
    # pretrain finished at, so the OU half was never the problem.
    assert agent.ou_noise_scale_initial == pytest.approx(0.3)
    assert agent.ou_noise_scale_final == pytest.approx(0.06)


def test_finetune_schedule_rejects_a_negative_scale():
    with pytest.raises(ValueError):
        apply_finetune_exploration_schedule(_pretrain_agent(), noise_scale=-1.0, episodes=300)


# --- Model selection ---------------------------------------------------------
#
# A fine-tune that ends below its own warm start used to become the day's
# controller anyway: the final weights were evaluated and kept whatever they
# were. Selection makes "no adaptation" a reachable answer.


class _SelectAgent:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, int]] = []

    def load_checkpoint(self, path, episode, map_location=None):
        self.loaded.append((str(path), int(episode)))
        return {"path": str(path), "episode": int(episode),
                "actors": True, "critics": True, "optimizers": True}


def _finetune_tree(root: Path, episodes) -> None:
    for ep in episodes:
        d = root / "results" / f"TEST{ep}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            (d / f"actor_{i}_ep{ep}.pth").write_bytes(b"stub")
        (d / f"agent_state_ep{ep}.pth").write_bytes(b"stub")


def _select(tmp_path, scores, **kwargs):
    """Run selection where each candidate's tracking rate is dictated by tag."""

    from training.run_after_day_ahead_bid import select_finetuned_model

    warm = tmp_path / "warm" / "results" / "TEST900"
    warm.mkdir(parents=True, exist_ok=True)
    ft = tmp_path / "ft"
    _finetune_tree(ft, (50, 100, 150))

    agent = _SelectAgent()
    seen: list[dict] = []

    def fake_evaluate(a, fixed_bid, *, n_seeds, base_seed, out_dir,
                      evaluation_pipeline="system"):
        tag = Path(out_dir).name
        seen.append({"tag": tag, "base_seed": int(base_seed),
                     "pipeline": str(evaluation_pipeline),
                     "activation_scenarios": int(
                         fixed_bid.get("activation_scenarios", 0)
                     ),
                     "payload_size": len(
                         fixed_bid.get("activation_scenario_payload") or []
                     )})
        return {"global_tracking_rate": scores.get(tag, 0.0), "soc_hit_rate": 0.5}

    scenario_count = int(kwargs.get("scenario_count", 0))
    fixed_bid = {"service_date": "2024-12-02"}
    if scenario_count:
        fixed_bid.update({
            "activation_scenarios": scenario_count,
            "activation_scenario_payload": [
                {"scenario_id": idx} for idx in range(scenario_count)
            ],
        })
    result = select_finetuned_model(
        agent,
        fixed_bid,
        evaluate_fn=fake_evaluate,
        finetune_dir=ft,
        warmstart_dir=warm,
        warmstart_episode=900,
        out_dir=tmp_path / "selection",
        n_seeds=2,
        base_seed=610_000,
        max_candidates=kwargs.get("max_candidates", 6),
        max_activation_scenarios=kwargs.get("max_activation_scenarios"),
    )
    return result, agent, seen


def test_selection_keeps_the_warm_start_when_no_candidate_beats_it(tmp_path):
    scores = {"warmstart": 0.90, "ft_ep50": 0.80, "ft_ep100": 0.85, "ft_ep150": 0.70}
    result, agent, _ = _select(tmp_path, scores)

    assert result["adopted"] == "warmstart"
    assert result["adopted_episode"] == 900
    # The warm start is what is left loaded, not the fine-tune's last weights.
    assert agent.loaded[-1][1] == 900


def test_selection_can_adopt_an_early_checkpoint(tmp_path):
    """A run that peaks early and then degrades is the case this exists for."""

    scores = {"warmstart": 0.80, "ft_ep50": 0.95, "ft_ep100": 0.85, "ft_ep150": 0.60}
    result, agent, _ = _select(tmp_path, scores)

    assert result["adopted"] == "ft_ep50"
    assert agent.loaded[-1][1] == 50


def test_selection_uses_seeds_the_final_evaluation_does_not(tmp_path):
    _, _, seen = _select(tmp_path, {"warmstart": 0.9})
    assert {call["base_seed"] for call in seen} == {610_000}, (
        "selection must not score on the seeds that get reported"
    )


def test_selection_checks_the_layer_under_the_correction(tmp_path):
    """Central allocator and battery can carry a worse controller."""

    scores = {
        "warmstart": 0.80, "ft_ep50": 0.70, "ft_ep100": 0.95, "ft_ep150": 0.60,
        # The pre-correction view, where the winner is the worse controller.
        "warmstart_marl_only": 0.60, "ft_ep100_marl_only": 0.40,
    }
    result, _, seen = _select(tmp_path, scores)

    assert result["adopted"] == "ft_ep100"
    assert any(call["pipeline"] == "marl_force" for call in seen)
    assert result["marl_only_tracking_before"] == pytest.approx(0.60)
    assert result["marl_only_tracking_after"] == pytest.approx(0.40)
    assert result["correction_masked_a_regression"] is True


def test_selection_spreads_candidates_over_the_run(tmp_path):
    result, _, _ = _select(tmp_path, {"warmstart": 0.9}, max_candidates=2)
    tags = [c["tag"] for c in result["candidates"]]
    # Endpoints kept: an early peak stays reachable when the budget is small.
    assert tags == ["warmstart", "ft_ep50", "ft_ep150"]


def test_selection_uses_only_the_configured_holdout_subset(tmp_path):
    result, _, seen = _select(
        tmp_path,
        {"warmstart": 0.9},
        scenario_count=512,
        max_activation_scenarios=96,
    )

    assert result["selection_activation_scenarios"] == 96
    assert {call["activation_scenarios"] for call in seen} == {96}
    assert {call["payload_size"] for call in seen} == {96}

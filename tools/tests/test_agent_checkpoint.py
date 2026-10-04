from __future__ import annotations

import json

import pytest

from training.agent_checkpoint import (
    find_latest_checkpoint,
    initialize_agent_from_checkpoint,
)
from training.run_after_day_ahead_bid import run_finetune_stage


class _FakeAgent:
    def __init__(self) -> None:
        self.load_checkpoint_calls: list[tuple[str, int]] = []
        self.epsilon_initial = 1.0
        self.epsilon_final = 0.05
        self.epsilon = 1.0
        self.ou_noise_scale_initial = 0.4
        self.ou_noise_scale_final = 0.02
        self.ou_noise_scale = 0.4

    def load_checkpoint(self, path, episode, map_location=None):
        self.load_checkpoint_calls.append((str(path), int(episode)))
        return {
            "path": str(path),
            "episode": int(episode),
            "actors": True,
            "critics": True,
            "optimizers": True,
        }


def _make_checkpoint_tree(root, episodes=(50, 100)) -> dict[int, str]:
    dirs = {}
    for ep in episodes:
        test_dir = root / "results" / f"TEST{ep}"
        test_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            (test_dir / f"actor_{i}_ep{ep}.pth").write_bytes(b"stub")
        dirs[ep] = str(test_dir)
    return dirs


def test_find_latest_checkpoint_prefers_newest_episode(tmp_path) -> None:
    dirs = _make_checkpoint_tree(tmp_path, episodes=(50, 100))

    path, episode = find_latest_checkpoint(tmp_path)
    assert episode == 100
    assert path == dirs[100]

    # A checkpoint directory itself also resolves.
    direct_path, direct_episode = find_latest_checkpoint(dirs[50])
    assert direct_episode == 50
    assert direct_path == dirs[50]

    with pytest.raises(FileNotFoundError, match="actor_"):
        find_latest_checkpoint(tmp_path / "empty")


def test_initialize_agent_from_checkpoint_uses_bundle_loader(tmp_path) -> None:
    dirs = _make_checkpoint_tree(tmp_path, episodes=(70,))
    agent = _FakeAgent()

    info = initialize_agent_from_checkpoint(agent, tmp_path)

    assert agent.load_checkpoint_calls == [(dirs[70], 70)]
    assert info["episode"] == 70
    assert info["critics"] is True


class _FakeBank:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_date(self, service_date):
        self.loaded.append(str(service_date))
        return {"service_date": str(service_date)}


def test_run_finetune_stage_trains_and_settles_each_day(tmp_path) -> None:
    work_dir = tmp_path / "run"
    dirs = _make_checkpoint_tree(work_dir, episodes=(50, 100))
    payloads = [
        {"date": "2024-08-26", "series": None},
        {"date": "2024-09-15", "series": None},
    ]
    bank = _FakeBank()
    preparer = object()
    train_calls: list[dict] = []
    eval_calls: list[dict] = []
    agents: list[_FakeAgent] = []

    def fake_train(**kwargs):
        train_calls.append(kwargs)
        agent = _FakeAgent()
        agents.append(agent)
        return agent, [], {}, [], kwargs["working_dir"]

    def fake_evaluate(agent, fixed_bid, *, n_seeds, base_seed, out_dir,
                      evaluation_pipeline="system"):
        eval_calls.append({
            "date": fixed_bid["service_date"],
            "n_seeds": int(n_seeds),
            "base_seed": int(base_seed),
            "out_dir": str(out_dir),
            "pipeline": str(evaluation_pipeline),
        })
        return {"mean_offered_capacity_kw": 1234.5}

    rows = run_finetune_stage(
        work_dir=work_dir,
        test_bid_bank=bank,
        test_bank_payloads=payloads,
        arrival_sampler=None,
        episode_preparer=preparer,
        episodes=7,
        noise_scale=0.3,
        eval_seeds=2,
        model_name="unit_model",
        select_model=False,
        train_fn=fake_train,
        evaluate_fn=fake_evaluate,
    )

    assert [row["date"] for row in rows] == ["2024-08-26", "2024-09-15"]
    assert all("error" not in row for row in rows)

    # Fine-tune training: warm start from the newest checkpoint, unaugmented
    # test-bank preparer, one day per run, low-noise exploration.
    assert len(train_calls) == 2
    for call, payload in zip(train_calls, payloads):
        assert call["initial_agent_checkpoint"] == dirs[100]
        assert call["exploration_noise_scale"] == pytest.approx(0.3)
        assert call["episode_preparer"] is preparer
        assert call["test_episode_preparer"] is preparer
        assert call["demand_data_override"] == [payload]
        assert call["num_episodes"] == 7

    # Realized settlement: finetuned then zeroshot, identical per-day seed,
    # day seeds follow the base + 1009 * day_index idiom.
    assert len(eval_calls) == 4
    day0, day1 = eval_calls[:2], eval_calls[2:]
    assert day0[0]["out_dir"].endswith("finetuned")
    assert day0[1]["out_dir"].endswith("zeroshot")
    assert day0[0]["base_seed"] == day0[1]["base_seed"] == 910_000
    assert day1[0]["base_seed"] == day1[1]["base_seed"] == 910_000 + 1009

    # The zero-shot pass reloads the untouched warm-start weights in place.
    for agent in agents:
        assert agent.load_checkpoint_calls == [(dirs[100], 100)]

    for payload in payloads:
        day_summary = (
            work_dir / "finetune" / payload["date"] / "finetune_day_summary.json"
        )
        assert day_summary.is_file()
        summary = json.loads(day_summary.read_text(encoding="utf-8"))
        assert summary["warmstart_checkpoint_episode"] == 100
        assert summary["finetuned"]["mean_offered_capacity_kw"] == pytest.approx(1234.5)
    assert (work_dir / "finetune" / "finetune_stage_summary.json").is_file()

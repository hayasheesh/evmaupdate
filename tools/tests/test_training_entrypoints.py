from __future__ import annotations

from pathlib import Path

import pytest

import EnvConfig
from legacy.finetune import fine_tune
import pre_train


def test_pretrain_entrypoint_accepts_no_arguments() -> None:
    args = pre_train.build_parser().parse_args([])

    assert args.episodes > 0
    assert args.train_days > 0


def test_finetune_entrypoint_accepts_no_arguments() -> None:
    args = fine_tune.build_parser().parse_args([])

    assert args.day == fine_tune.DEFAULT_SERVICE_DAY
    if fine_tune.DEFAULT_WARMSTART is None:
        assert args.warmstart is None
    else:
        assert Path(args.warmstart).resolve() == fine_tune.DEFAULT_WARMSTART.resolve()
        assert fine_tune.DEFAULT_WARMSTART.exists()
    assert args.episodes > 0
    assert args.command_scenarios == EnvConfig.FINETUNE_ACTIVATION_SCENARIOS == 256
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS == 128
    assert args.eval_command_scenarios >= 512
    assert 0 < args.selection_command_scenarios < args.eval_command_scenarios
    assert args.graph_interval == 50

def test_finetune_rejects_a_day_seen_in_pretrain_or_validation(tmp_path) -> None:
    run = tmp_path / "run"
    results = run / "results"
    results.mkdir(parents=True)
    (results / "bid_bank_manifest.json").write_text(
        '{"settings":{"selected_dates":["2024-01-01"]}}', encoding="utf-8"
    )
    (results / "bid_bank_test_manifest.json").write_text(
        '{"entries":[{"service_date":"2024-01-02"}]}', encoding="utf-8"
    )
    checkpoint = results / "TEST100"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="held-out service day"):
        fine_tune.require_unseen_finetune_day("2024-01-01", checkpoint)
    with pytest.raises(ValueError, match="held-out service day"):
        fine_tune.require_unseen_finetune_day("2024-01-02", checkpoint)

    fine_tune.require_unseen_finetune_day("2024-01-03", checkpoint)

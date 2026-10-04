"""The lower controller's rollout commands come from outside the bid design pool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market.activation_scenarios import (
    LOWER_CONTROL_POOL,
    build_activation_scenario_set,
)


def _write_library(root: Path, partitions: dict[str, str]) -> Path:
    root.mkdir()
    for index, (day, partition) in enumerate(sorted(partitions.items())):
        frame = pd.DataFrame({
            "up_proxy_raw": np.full(288, 0.1 * (index + 1)),
            "down_proxy_raw": np.zeros(288),
            "source_type": "test_5min",
            "source_date": day,
            "source_bmu": "UNIT",
            "scenario_partition": partition,
            "segment_id": f"UNIT:{day}",
        })
        frame.to_csv(root / f"day_{day}.csv", index=False)
    return root


def test_a_partitioned_library_draws_every_pool_but_train(tmp_path):
    days = {"2026-01-01": "train", "2026-01-02": "train", "2026-01-03": "validation",
            "2026-01-04": "test"}
    library = _write_library(tmp_path / "declared", days)
    drawn = {s.source_date for seed in range(20) for s in build_activation_scenario_set(
        service_date="2024-04-02", n_scenarios=1, seed=seed, proxy_shape_dir=library,
        scenario_partition=LOWER_CONTROL_POOL)}
    assert drawn == {"2026-01-03", "2026-01-04"}


def test_excluded_sources_are_never_drawn(tmp_path):
    days = {f"2026-01-0{d}": "test" for d in range(1, 6)}
    library = _write_library(tmp_path / "declared", days)
    excluded = {"test_5min:day_2026-01-01.csv", "test_5min:day_2026-01-04.csv"}
    drawn = {s.source for seed in range(20) for s in build_activation_scenario_set(
        service_date="2024-04-02", n_scenarios=3, seed=seed, proxy_shape_dir=library,
        scenario_partition="holdout", require_unique=True, exclude_sources=excluded)}
    assert drawn == {f"test_5min:day_2026-01-0{d}.csv" for d in (2, 3, 5)}


def test_the_pretrain_draws_are_rebuilt_from_their_seeds(tmp_path, monkeypatch):
    import training.lower_bid_training as lower

    days = {f"2026-02-{d:02d}": ("train" if d % 3 == 0 else "validation" if d % 3 == 1 else "test")
            for d in range(1, 28)}
    library = _write_library(tmp_path / "declared", days)
    monkeypatch.setattr(lower, "LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR", str(library))
    train_entries = [{"service_date": f"2024-04-0{d}", "forecast_seed": 73000 + d} for d in range(1, 4)]
    test_entries = [{"service_date": "2024-05-01", "forecast_seed": 1073003},
                    {"service_date": "2024-05-02", "forecast_seed": 1073004}]
    # training.train: episode 0 on entry 0, then episode e on entry (e - 1) mod n;
    # every interim test: index i on test entry i mod n. Training never draws
    # an interim-test command.
    interim = lower.interim_test_command_sources(test_entries, interim_test_episodes=3, library_dir=library)
    train_drawn = {lower.sample_random_historical_activation(
        train_entries[max(e - 1, 0) % 3], e, stream="train", exclude_sources=interim)["source"]
        for e in range(0, 9)}
    validation_drawn = {lower.sample_random_historical_activation(
        test_entries[i % 2], i, stream="validation")["source"] for i in range(3)}
    assert validation_drawn == interim
    assert not train_drawn & interim
    rebuilt = lower.lower_commands_seen_in_pretrain(
        train_entries, test_entries, environment_episodes=8, interim_test_episodes=3, library_dir=library)
    assert rebuilt == train_drawn | validation_drawn


def test_a_library_without_partitions_is_refused(tmp_path):
    library = _write_library(tmp_path / "undeclared", {f"2026-01-0{d}": "test" for d in range(1, 3)})
    for path in library.glob("*.csv"):
        pd.read_csv(path).drop(columns=["scenario_partition"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="scenario_partition"):
        build_activation_scenario_set(service_date="2024-04-02", n_scenarios=1, seed=0,
                                      proxy_shape_dir=library, scenario_partition=LOWER_CONTROL_POOL)

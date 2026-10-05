from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market.activation_scenarios import (
    activation_library_signature,
    build_activation_scenario_set,
)


def _write_command(path: Path, phase: float, partition: str = "test") -> None:
    x = np.linspace(0.0, 4.0 * np.pi, 288, endpoint=False)
    values = np.sin(x + phase) + 0.15 * np.cos(3.0 * x - phase)
    pd.DataFrame({
        "step": np.arange(288),
        "up_proxy_raw": np.maximum(-1.2 * values, 0.0),
        "down_proxy_raw": np.maximum(1.2 * values, 0.0),
        "source_type": "test_5min",
        "scenario_partition": partition,
    }).to_csv(path, index=False)


def test_scenarios_are_seeded_direct_file_samples(tmp_path: Path) -> None:
    for idx in range(6):
        _write_command(tmp_path / f"day_2024-01-{idx + 1:02d}.csv", phase=idx * 0.4)

    first = build_activation_scenario_set(
        service_date="2024-08-26",
        n_scenarios=4,
        seed=8123,
        proxy_shape_dir=tmp_path,
        scenario_partition="holdout",
    )
    repeated = build_activation_scenario_set(
        service_date="2024-08-26",
        n_scenarios=4,
        seed=8123,
        proxy_shape_dir=tmp_path,
        scenario_partition="holdout",
    )

    assert [scenario.source for scenario in first] == [
        scenario.source for scenario in repeated
    ]
    assert len({scenario.source for scenario in first}) == 4
    assert all(scenario.name.startswith("test_5min_") for scenario in first)

    for scenario in first:
        filename = scenario.source.split(":", 1)[1]
        raw = pd.read_csv(tmp_path / filename)
        # The loader clips to [0, 1] at the simulator-input boundary only.
        assert np.allclose(scenario.up_proxy, np.clip(raw["up_proxy_raw"], 0.0, 1.0))
        assert np.allclose(scenario.down_proxy, np.clip(raw["down_proxy_raw"], 0.0, 1.0))
        assert not np.any(
            (scenario.up_proxy > 0.0) & (scenario.down_proxy > 0.0)
        )


def test_activation_library_signature_changes_with_command_content(tmp_path: Path) -> None:
    path = tmp_path / "boa_E_TEST-1_2024-08-03.csv"
    pd.DataFrame({
        "step": np.arange(288),
        "up_proxy_raw": np.zeros(288),
        "down_proxy_raw": np.zeros(288),
    }).to_csv(path, index=False)
    first = activation_library_signature(tmp_path)

    frame = pd.read_csv(path)
    frame.loc[0, "up_proxy_raw"] = 0.25
    frame.to_csv(path, index=False)
    second = activation_library_signature(tmp_path)

    assert first["activation_library_file_count"] == 1
    assert first["activation_library_sha256"] != second["activation_library_sha256"]


def test_unique_requirement_rejects_recycling_command_files(tmp_path: Path) -> None:
    for idx in range(3):
        _write_command(tmp_path / f"day_2024-01-{idx + 1:02d}.csv", phase=idx)
    with pytest.raises(RuntimeError, match="3 unique files"):
        build_activation_scenario_set(
            n_scenarios=4,
            proxy_shape_dir=tmp_path,
            scenario_partition="holdout",
            require_unique=True,
        )


def test_activation_library_signature_is_reused_while_the_library_is_unchanged(tmp_path: Path) -> None:
    from market import activation_scenarios as module

    pd.DataFrame({"step": np.arange(288), "up_proxy_raw": np.zeros(288), "down_proxy_raw": np.zeros(288)}).to_csv(
        tmp_path / "boa_E_TEST-1_2024-08-03.csv", index=False
    )
    first = activation_library_signature(tmp_path)
    cached = len(module._LIBRARY_SIGNATURE_CACHE)
    second = activation_library_signature(tmp_path)
    assert second == first
    assert len(module._LIBRARY_SIGNATURE_CACHE) == cached

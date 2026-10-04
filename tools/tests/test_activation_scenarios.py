from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environment.readcsv import _load_and_normalize_demand_file
from market.activation_scenarios import (
    activation_library_signature,
    build_activation_scenario_set,
)
from market.bid_env import BASE_DOWN_KW, BASE_UP_KW


def _write_command(path: Path, phase: float) -> None:
    x = np.linspace(0.0, 4.0 * np.pi, 288, endpoint=False)
    values = np.sin(x + phase) + 0.15 * np.cos(3.0 * x - phase)
    pd.DataFrame({
        "step": np.arange(288),
        "demand_adjustment": values,
    }).to_csv(path, index=False)


def test_local_5min_scenarios_are_seeded_direct_file_samples(tmp_path: Path) -> None:
    for idx in range(6):
        _write_command(tmp_path / f"day_2024-01-{idx + 1:02d}.csv", phase=idx * 0.4)

    first = build_activation_scenario_set(
        service_date="2024-08-26",
        n_scenarios=4,
        seed=8123,
        proxy_shape_dir=tmp_path,
    )
    repeated = build_activation_scenario_set(
        service_date="2024-08-26",
        n_scenarios=4,
        seed=8123,
        proxy_shape_dir=tmp_path,
    )

    assert [scenario.source for scenario in first] == [
        scenario.source for scenario in repeated
    ]
    assert len({scenario.source for scenario in first}) == 4
    assert all(scenario.name.startswith("local_5min_") for scenario in first)

    for scenario in first:
        filename = scenario.source.split(":", 1)[1]
        normalized = _load_and_normalize_demand_file(str(tmp_path / filename))
        expected_up = np.clip(-normalized / BASE_UP_KW, 0.0, 1.0)
        expected_down = np.clip(normalized / BASE_DOWN_KW, 0.0, 1.0)
        assert np.allclose(scenario.up_proxy, expected_up)
        assert np.allclose(scenario.down_proxy, expected_down)
        assert not np.any(
            (scenario.up_proxy > 0.0) & (scenario.down_proxy > 0.0)
        )


def test_activation_library_signature_changes_with_command_content(tmp_path: Path) -> None:
    path = tmp_path / "boa_E_TEST-1_2024-08-03.csv"
    pd.DataFrame({
        "step": np.arange(288),
        "up_proxy": np.zeros(288),
        "down_proxy": np.zeros(288),
    }).to_csv(path, index=False)
    first = activation_library_signature(tmp_path)

    frame = pd.read_csv(path)
    frame.loc[0, "up_proxy"] = 0.25
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
            require_unique=True,
        )

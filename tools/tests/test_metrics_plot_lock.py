"""The test-metrics CSV is written even while its PNG is held open elsewhere."""

from __future__ import annotations

import time

import matplotlib.figure


def _history() -> dict:
    return {
        "soc_miss_count": [10.0, 8.0],
        "avg_soc_deficit": [2.0, 1.5],
        "surplus_absorption_rate": [60.0, 70.0],
        "supply_cooperation_rate": [50.0, 55.0],
        "departing_evs": [405, 405],
        "departing_evs_soc_met": [365, 373],
        "surplus_steps": [645, 645],
        "surplus_within_narrow": [400, 450],
        "shortage_steps": [231, 231],
        "shortage_within_narrow": [150, 160],
    }


def test_the_csv_holds_every_test_when_the_png_cannot_be_written(tmp_path, monkeypatch):
    from tools.Utils import plot_performance_metrics

    calls = []

    def held(self, path, *args, **kwargs):
        calls.append(path)
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", held)
    plot_performance_metrics(_history(), str(tmp_path / "TEST40"),
                             title_prefix="Test Results", x_values=[20, 40])

    rows = (tmp_path / "test_performance_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 2
    assert len(calls) == 2 * 5  # the figure and its legend, five attempts each


def test_a_brief_hold_is_retried_until_the_png_is_written(tmp_path, monkeypatch):
    from tools.Utils import plot_performance_metrics

    original = matplotlib.figure.Figure.savefig
    failures = {"left": 2}

    def briefly_held(self, path, *args, **kwargs):
        if failures["left"] > 0:
            failures["left"] -= 1
            raise OSError(22, "Invalid argument")
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", briefly_held)
    plot_performance_metrics(_history(), str(tmp_path), title_prefix="Test Results", x_values=[20, 40])

    assert (tmp_path / "test_performance_metrics.png").stat().st_size > 0

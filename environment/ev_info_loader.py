"""
ev_info_loader.py
=================
Loader for the plug-in SoC distribution.

`plugin_soc_inl_ev_project.csv` holds the EV Project's shares of charging events
by SoC at the start of charging (Nissan Leaf, Oct-Dec 2012, vehicle telematics),
in 10-point bins, for home and for away-from-home charging locations. Where the
numbers come from: data/input_EVinfo/plugin_soc_inl_ev_project.README.txt.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PluginSocBins:
    """Plug-in SoC bins [low, high) in percent and the cumulative share of each location."""

    low: np.ndarray
    high: np.ndarray
    home_cdf: np.ndarray
    away_cdf: np.ndarray

    def draw(self, rng, home: bool) -> float:
        """One plug-in SoC: a bin by its share, then uniform inside the bin."""
        cdf = self.home_cdf if home else self.away_cdf
        idx = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
        return float(self.low[idx] + (self.high[idx] - self.low[idx]) * rng.random())


def load_plugin_soc_bins(path: str) -> PluginSocBins:
    return _cached_load_plugin_soc_bins(str(Path(path).resolve()))


def _cdf(shares: np.ndarray, path: str, column: str) -> np.ndarray:
    shares = np.clip(np.asarray(shares, dtype=float), 0.0, None)
    if float(shares.sum()) <= 0.0:
        raise ValueError(f"plug-in SoC distribution has zero total weight in {column}: {path}")
    cdf = np.cumsum(shares) / shares.sum()
    cdf.setflags(write=False)
    return cdf


@lru_cache(maxsize=8)
def _cached_load_plugin_soc_bins(path: str) -> PluginSocBins:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"plug-in SoC distribution has no rows: {path}")
    low = np.asarray([float(r["bin_low_pct"]) for r in rows])
    high = np.asarray([float(r["bin_high_pct"]) for r in rows])
    if np.any(high <= low) or np.any(low < 0.0) or np.any(high > 100.0):
        raise ValueError(f"plug-in SoC bins must satisfy 0 <= low < high <= 100: {path}")
    low.setflags(write=False)
    high.setflags(write=False)
    return PluginSocBins(
        low=low,
        high=high,
        home_cdf=_cdf([float(r["home_pct"]) for r in rows], path, "home_pct"),
        away_cdf=_cdf([float(r["away_pct"]) for r in rows], path, "away_pct"),
    )

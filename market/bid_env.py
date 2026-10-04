"""Shared block geometry and baseline-aware bid-to-command conversion.

The research bid is ``(B_b, U_b, D_b)`` for each 30-minute block.  This module
contains no bidder, settlement, or monetary objective; it only maps the stored
directional quantities onto the five-minute command trace used by controllers.
"""

from __future__ import annotations

import numpy as np

from EnvConfig import (
    DEMAND_TARGET_MAX_KW,
    DEMAND_TARGET_MIN_KW,
    EPISODE_STEPS,
)


STEPS_PER_BLOCK = 6
N_BLOCKS = EPISODE_STEPS // STEPS_PER_BLOCK

BASE_AWARDED_KW = 1500.0
BASE_TOL_KW = 150.0
STAY_THRESHOLD = 0.90
SUSTAIN_HOURS = 0.5
BASE_UP_KW = float(-DEMAND_TARGET_MIN_KW)
BASE_DOWN_KW = float(DEMAND_TARGET_MAX_KW)


def dispatch_from_two_sided_bid(
    base_demand_segment: np.ndarray,
    awarded_up_kw: float,
    awarded_down_kw: float,
) -> np.ndarray:
    """Map a normalized proxy into regulation around a separate baseline."""

    base = np.asarray(base_demand_segment, dtype=np.float32)
    up_proxy = np.clip(-base / max(BASE_UP_KW, 1e-6), 0.0, 1.0)
    down_proxy = np.clip(base / max(BASE_DOWN_KW, 1e-6), 0.0, 1.0)
    dispatch = (
        float(awarded_down_kw) * down_proxy
        - float(awarded_up_kw) * up_proxy
    )
    return dispatch.astype(np.float32)


def baseline_aware_target_series(
    base_demand_segment: np.ndarray,
    baseline_kw: float,
    awarded_up_kw: float,
    awarded_down_kw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total-power target and regulation delta for one bid."""

    regulation = dispatch_from_two_sided_bid(
        base_demand_segment,
        awarded_up_kw,
        awarded_down_kw,
    )
    target = float(baseline_kw) + regulation
    return target.astype(np.float32), regulation.astype(np.float32)

"""Shared helpers for awarded-block participation semantics."""

from __future__ import annotations

import numpy as np


def participation_by_block(up_kw, down_kw, *, eps: float = 1e-6) -> np.ndarray:
    """Return blocks in which at least one reserve direction is awarded."""

    up = np.asarray(up_kw, dtype=float).reshape(-1)
    down = np.asarray(down_kw, dtype=float).reshape(-1)
    if up.shape != down.shape:
        raise ValueError(f"up/down shape mismatch: {up.shape} != {down.shape}")
    return (up > float(eps)) | (down > float(eps))


def participation_by_step(
    up_kw,
    down_kw,
    *,
    steps_per_block: int,
    steps: int | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Expand awarded-block participation to the control-step resolution."""

    block_mask = participation_by_block(up_kw, down_kw, eps=eps)
    step_mask = np.repeat(block_mask, int(steps_per_block))
    if steps is not None:
        step_mask = step_mask[: int(steps)]
        if step_mask.size < int(steps):
            step_mask = np.pad(step_mask, (0, int(steps) - step_mask.size))
    return np.asarray(step_mask, dtype=bool)


def zero_instruction_tolerance_by_block(
    up_kw,
    down_kw,
    *,
    band_fraction: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return the assessment-II band for an awarded block at zero instruction.

    The band is ``band_fraction * (U + D)``: the block is held to a precision
    set by the whole range it contracted to move over, not by one direction.

    The rule does not settle this.  It fixes the band at ten percent of "the
    30-minute block's award" and, with only an up product in the market today,
    never has to say which award is meant when both directions are held.  Two
    other readings exist.  Applying the up and the down assessment to the same
    baseline error independently gives ``min(U, D)``, which is the tightest;
    it also means a small down award cuts the block's idle tolerance by an
    order of magnitude, so offering down at all is penalised.  ``max(U, D)``
    reads the pair as one resource and asks the precision the larger award
    would have asked alone.  The sum is the reading in which the tolerance
    tracks the flexibility the block sold, and it is the only one of the three
    that is linear in ``(U, D)``.
    """

    up = np.asarray(up_kw, dtype=float).reshape(-1)
    down = np.asarray(down_kw, dtype=float).reshape(-1)
    if up.shape != down.shape:
        raise ValueError(f"up/down shape mismatch: {up.shape} != {down.shape}")
    awarded = np.where(up > float(eps), up, 0.0)
    awarded = awarded + np.where(down > float(eps), down, 0.0)
    return max(float(band_fraction), 0.0) * awarded


def masked_pass_rate(passed, enabled) -> float:
    """Return a pass rate over assessed points; an empty set passes vacuously."""

    passed_arr = np.asarray(passed, dtype=bool).reshape(-1)
    enabled_arr = np.asarray(enabled, dtype=bool).reshape(-1)
    if passed_arr.shape != enabled_arr.shape:
        raise ValueError(
            f"passed/enabled shape mismatch: {passed_arr.shape} != {enabled_arr.shape}"
        )
    return float(np.mean(passed_arr[enabled_arr])) if np.any(enabled_arr) else 1.0

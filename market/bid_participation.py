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


def instruction_band_coefficient(
    signal_fraction,
    *,
    band_fraction: float,
    band_signal_fraction: float = 0.0,
):
    """Band per kW of award for an instruction of ``signal_fraction`` of it.

    Assessment II is a flat 10 % (``band_signal_fraction`` 0). Under the PJM
    performance-score rule the band grows with the instruction, because the
    score divides the error by 0.5 * |signal| + 0.5 * AReg
    (EnvConfig.MARKET_TRACKING_RULE).
    """

    return (
        max(float(band_fraction), 0.0)
        + max(float(band_signal_fraction), 0.0) * np.abs(np.asarray(signal_fraction, dtype=float))
    )


def resolved_idle_band_fraction(band_fraction: float, idle_band_fraction=None) -> float:
    """The idle band fraction, which is the band fraction unless set apart."""

    return float(band_fraction if idle_band_fraction is None else idle_band_fraction)


def tracking_band_fields(fixed_bid: dict) -> dict:
    """The BiddingLPConfig band fields a stored bid was built with.

    Bids built before the PJM rule store only ``assessment_band_fraction``;
    they were all built under Assessment II.
    """

    frac = fixed_bid.get("assessment_band_fraction")
    idle = fixed_bid.get("idle_band_fraction")
    return {
        "assessment_band_fraction": 0.10 if frac is None else float(frac),
        "band_signal_fraction": float(fixed_bid.get("band_signal_fraction") or 0.0),
        "idle_band_fraction": None if idle is None else float(idle),
        "symmetric_award": bool(fixed_bid.get("symmetric_award", False)),
    }


def step_tolerance_kw(
    regulation_kw,
    up_kw: float,
    down_kw: float,
    *,
    band_fraction: float,
    band_signal_fraction: float = 0.0,
    idle_band_fraction=None,
    idle_eps: float = 1e-6,
) -> np.ndarray:
    """Per-step band (kW) for the steps of one awarded block.

    ``regulation_kw`` is the instruction in kW, negative for up (less
    consumption) and positive for down. An up step is held to
    instruction_band_coefficient(|r| / U) * U, a down step likewise with D, and
    a step without an instruction to the idle fraction times (U + D)
    (zero_instruction_tolerance_by_block).
    """

    regulation = np.asarray(regulation_kw, dtype=float)
    up = max(float(up_kw), 0.0)
    down = max(float(down_kw), 0.0)
    up_tol = instruction_band_coefficient(
        np.abs(regulation) / max(up, 1e-12),
        band_fraction=band_fraction,
        band_signal_fraction=band_signal_fraction,
    ) * up
    down_tol = instruction_band_coefficient(
        np.abs(regulation) / max(down, 1e-12),
        band_fraction=band_fraction,
        band_signal_fraction=band_signal_fraction,
    ) * down
    idle_tol = float(zero_instruction_tolerance_by_block(
        [up], [down], band_fraction=resolved_idle_band_fraction(band_fraction, idle_band_fraction),
    )[0])
    return np.where(
        regulation < -idle_eps,
        up_tol,
        np.where(regulation > idle_eps, down_tol, idle_tol),
    )


def masked_pass_rate(passed, enabled) -> float:
    """Return a pass rate over assessed points; an empty set passes vacuously."""

    passed_arr = np.asarray(passed, dtype=bool).reshape(-1)
    enabled_arr = np.asarray(enabled, dtype=bool).reshape(-1)
    if passed_arr.shape != enabled_arr.shape:
        raise ValueError(
            f"passed/enabled shape mismatch: {passed_arr.shape} != {enabled_arr.shape}"
        )
    return float(np.mean(passed_arr[enabled_arr])) if np.any(enabled_arr) else 1.0

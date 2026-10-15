"""PJM Regulation performance score, at this model's 5-minute resolution.

PJM Manual 12 Rev.57 4.4.5 and OA Schedule 1 3.2.2(j) score each 5-minute
interval from its 10-second samples k:

    Error_k = |Response_k - Signal_k| / (0.5 * mean |Signal| + 0.5 * AReg)
    Performance Score = 1 - (1 / n) * sum_k Error_k

The model has one sample per interval (n = 1), so the score of a step is
1 - |error| / (0.5 * |instruction| + 0.5 * AReg). PJM's 10-second latency
allowance is shorter than a step and has no counterpart here. The documents
do not clip the score, so a score below zero is kept.

An interval scoring below 25 % earns no regulation credit (Manual 28 Rev.104
4.2.1, OA 3.2.2(j)). A resource whose historic score falls below 40 % is
disqualified (Manual 12 4.4.4); the historic score is the rolling average over
the last 100 clock hours (OA 3.2.2(j)), the last 200 half-hour intervals the
resource operated (Manual 12 4.4.4).
"""

from __future__ import annotations

import numpy as np

PAYMENT_FLOOR = 0.25
QUALIFICATION_FLOOR = 0.40
HISTORIC_WINDOW_HOURS = 100.0
STEP_HOURS = 5.0 / 60.0


def areg_by_step(up_kw, down_kw, regulation_kw, *, steps_per_block: int = 6) -> np.ndarray:
    """AReg of each step from a per-block bid.

    A bid under the PJM rule has up = down, which is AReg. For a bid with two
    quantities the award of the instructed direction is taken, and their mean
    at a step without an instruction.
    """

    up = np.repeat(np.asarray(up_kw, dtype=float).reshape(-1), int(steps_per_block))
    down = np.repeat(np.asarray(down_kw, dtype=float).reshape(-1), int(steps_per_block))
    regulation = np.asarray(regulation_kw, dtype=float).reshape(-1)[: up.size]
    return np.where(
        regulation < 0.0, up, np.where(regulation > 0.0, down, 0.5 * (up + down))
    )


def interval_scores(residual_kw, regulation_kw, areg_kw, assessed) -> np.ndarray:
    """Score of each 5-minute step; NaN where it is not assessed or AReg is zero.

    ``residual_kw`` is response minus instruction, ``regulation_kw`` the
    instruction measured from the baseline.
    """

    residual = np.abs(np.asarray(residual_kw, dtype=float).reshape(-1))
    signal = np.abs(np.asarray(regulation_kw, dtype=float).reshape(-1))
    areg = np.asarray(areg_kw, dtype=float).reshape(-1)
    mask = np.asarray(assessed, dtype=bool).reshape(-1) & (areg > 0.0)
    scores = np.full(residual.shape, np.nan)
    normalizer = 0.5 * signal[mask] + 0.5 * areg[mask]
    scores[mask] = 1.0 - residual[mask] / normalizer
    return scores


def score_summary(scores, areg_kw) -> dict:
    """Interval-level figures for one run of steps.

    ``credit_ratio`` is the regulation credit as a share of what a score of 1
    in every interval would earn: each interval earns AReg x score, nothing
    below PAYMENT_FLOOR (prices and the mileage ratio left out).
    """

    scores = np.asarray(scores, dtype=float).reshape(-1)
    areg = np.asarray(areg_kw, dtype=float).reshape(-1)
    assessed = np.isfinite(scores)
    if not np.any(assessed):
        return {"intervals": 0, "mean_score": float("nan"),
                "share_below_payment_floor": float("nan"), "credit_ratio": float("nan")}
    picked = scores[assessed]
    weight = areg[assessed]
    paid = np.where(picked >= PAYMENT_FLOOR, picked, 0.0)
    return {
        "intervals": int(assessed.sum()),
        "mean_score": float(np.mean(picked)),
        "min_score": float(np.min(picked)),
        "share_below_payment_floor": float(np.mean(picked < PAYMENT_FLOOR)),
        "credit_ratio": float(np.sum(paid * weight) / max(np.sum(weight), 1e-12)),
    }


def historic_scores(scores, *, window_hours: float = HISTORIC_WINDOW_HOURS,
                    step_hours: float = STEP_HOURS) -> np.ndarray:
    """Rolling average of the last ``window_hours`` of assessed intervals.

    ``scores`` is the chain of interval scores in time order (NaN where the
    resource was not assigned, which does not count towards the window). The
    result has one value per assessed interval once the window is full, and
    is empty before that.
    """

    picked = np.asarray(scores, dtype=float).reshape(-1)
    picked = picked[np.isfinite(picked)]
    window = int(round(float(window_hours) / float(step_hours)))
    if window <= 0 or picked.size < window:
        return np.zeros(0)
    cumulative = np.concatenate([[0.0], np.cumsum(picked)])
    return (cumulative[window:] - cumulative[:-window]) / window


def qualification_summary(scores, *, window_hours: float = HISTORIC_WINDOW_HOURS,
                          step_hours: float = STEP_HOURS) -> dict:
    """Lowest historic score of a chain and whether it stays at QUALIFICATION_FLOOR."""

    historic = historic_scores(scores, window_hours=window_hours, step_hours=step_hours)
    if not historic.size:
        return {"window_full": False, "min_historic_score": float("nan"), "qualified": None}
    low = float(np.min(historic))
    return {"window_full": True, "min_historic_score": low,
            "qualified": bool(low >= QUALIFICATION_FLOOR)}

"""Probabilistic forecasting metrics.

The existing training scripts report MAE and a hand-rolled "coverage" number.
MAE is the wrong headline metric for a *quantile* model, and the coverage check
is computed inconsistently between the p10/p50/p90 heads. This module provides
the metrics that actually evaluate a probabilistic forecast:

- ``pinball_loss``      — the proper scoring rule for a single quantile.
- ``calibration``       — does the q-quantile actually sit above q% of outcomes?
- ``interval_coverage`` — does the [p_low, p_high] band contain the truth as often
                          as advertised?
- ``mae`` / ``rmse``    — kept for the point (p50) forecast and baselines.

All functions take 1-D array-likes and ignore NaNs pairwise so a partially
populated backtest fold still scores cleanly.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _clean(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Align arrays and drop any row where *any* of them is NaN."""
    arrays = [np.asarray(a, dtype=float).ravel() for a in arrays]
    n = arrays[0].size
    if any(a.size != n for a in arrays):
        raise ValueError("metric inputs must all have the same length")
    mask = np.ones(n, dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return tuple(a[mask] for a in arrays)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Mean pinball (quantile) loss for one quantile level.

    For quantile ``q`` the loss asymmetrically penalises under- vs over-prediction:
        q * (y - ŷ)      when y >= ŷ
        (1 - q) * (ŷ - y) when y <  ŷ
    Lower is better. Minimised in expectation by the true q-quantile, which is
    exactly why it is the correct objective *and* metric for these models.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def calibration(y_true, y_pred, quantile: float) -> float:
    """Empirical fraction of outcomes at or below the predicted quantile.

    A well-calibrated q-quantile forecast satisfies ``calibration ≈ q``. E.g. the
    p90 head should land above the realized spread ~90% of the time. The gap
    ``calibration - quantile`` is the calibration error.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true <= y_pred))


def interval_coverage(y_true, y_low, y_high) -> float:
    """Fraction of outcomes inside the [y_low, y_high] band (PICP)."""
    y_true, y_low, y_high = _clean(y_true, y_low, y_high)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true >= y_low) & (y_true <= y_high)))


def interval_width(y_low, y_high) -> float:
    """Mean width of the prediction interval (sharpness — lower is better,
    but only meaningful once coverage is on target)."""
    y_low, y_high = _clean(y_low, y_high)
    if y_low.size == 0:
        return float("nan")
    return float(np.mean(y_high - y_low))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def quantile_scorecard(
    y_true,
    predictions: Mapping[float, np.ndarray],
) -> dict[str, float]:
    """Aggregate scorecard for a multi-quantile forecast.

    ``predictions`` maps quantile level -> predicted array, e.g.
    ``{0.1: p10, 0.5: p50, 0.9: p90}``. Returns mean pinball across quantiles
    (the single headline number), per-quantile calibration error, point-forecast
    error on the median, and central-interval coverage/width using the extreme
    quantiles supplied.
    """
    quantiles = sorted(predictions)
    if not quantiles:
        raise ValueError("predictions is empty")

    pinballs = {q: pinball_loss(y_true, predictions[q], q) for q in quantiles}
    calib_err = {q: calibration(y_true, predictions[q], q) - q for q in quantiles}

    score: dict[str, float] = {
        "mean_pinball": float(np.nanmean(list(pinballs.values()))),
    }
    for q in quantiles:
        score[f"pinball_p{int(round(q * 100))}"] = pinballs[q]
        score[f"calib_err_p{int(round(q * 100))}"] = calib_err[q]

    # Point forecast quality on the median if present, else the nearest quantile.
    median_q = min(quantiles, key=lambda q: abs(q - 0.5))
    score["mae_p50"] = mae(y_true, predictions[median_q])
    score["rmse_p50"] = rmse(y_true, predictions[median_q])

    # Central interval from the outermost quantiles provided.
    lo, hi = quantiles[0], quantiles[-1]
    if hi > lo:
        score["interval_coverage"] = interval_coverage(
            y_true, predictions[lo], predictions[hi]
        )
        score["nominal_coverage"] = hi - lo
        score["interval_width"] = interval_width(predictions[lo], predictions[hi])

    return score

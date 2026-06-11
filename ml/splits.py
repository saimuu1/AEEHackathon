"""Leakage-free temporal splitting for backtests.

Why this module exists
-----------------------
``scripts/train_spread_forecaster.py`` builds (current, horizon, future) pairs
*per site*, concatenates them, and then slices ``X[:0.75]`` calling it a
"time-based split". It is not. Because the rows are grouped by site before the
slice, the test set is simply the last sites in iteration order, and overlapping
``(t, t+h)`` windows put the same future timestamps in both train and test. The
reported MAE is therefore optimistic.

The fix is to split on the **global timeline**, train only on the past, test only
on the future, and insert an *embargo* gap so that a training row's forecast
target cannot fall inside the test window (and vice-versa). The embargo must be
at least the maximum forecast horizon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One backtest fold, expressed as boolean masks over the original frame."""

    index: int
    train: np.ndarray  # bool mask
    test: np.ndarray   # bool mask
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_origin_folds(
    timestamps: pd.Series,
    n_folds: int = 5,
    horizon: int = 72,
    min_train_frac: float = 0.4,
    expanding: bool = True,
) -> list[Fold]:
    """Expanding- (or sliding-) window folds over a shared global timeline.

    Parameters
    ----------
    timestamps : per-row timestamps (any order; site is irrelevant here because
        we split purely on wall-clock time, which keeps every site's train rows
        strictly before its test rows).
    n_folds : number of successive test windows.
    horizon : maximum forecast horizon in hours. Used as the embargo gap: rows
        within ``horizon`` hours *before* a test window are dropped from train so
        their targets cannot leak into the test period.
    min_train_frac : fraction of the timeline reserved for the first train window
        before the first test fold begins.
    expanding : if True each fold trains on all history up to its embargo; if
        False the train window slides with a fixed length.

    Returns
    -------
    list[Fold] ordered from earliest to latest test window.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    if ts.isna().any():
        raise ValueError("timestamps contain NaT")
    if not 0 < n_folds:
        raise ValueError("n_folds must be positive")

    t_min, t_max = ts.min(), ts.max()
    embargo = pd.Timedelta(hours=horizon)

    first_test_start = t_min + (t_max - t_min) * min_train_frac
    # The testable span is divided evenly into n_folds contiguous windows.
    test_span = (t_max - first_test_start) / n_folds
    if test_span <= pd.Timedelta(0):
        raise ValueError("not enough history for the requested folds/horizon")

    folds: list[Fold] = []
    for i in range(n_folds):
        test_start = first_test_start + test_span * i
        test_end = first_test_start + test_span * (i + 1)
        train_end = test_start - embargo

        test_mask = ((ts >= test_start) & (ts < test_end)).to_numpy()
        if expanding:
            train_mask = (ts < train_end).to_numpy()
        else:
            window_start = train_end - (first_test_start - t_min)
            train_mask = ((ts >= window_start) & (ts < train_end)).to_numpy()

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        folds.append(
            Fold(
                index=i,
                train=train_mask,
                test=test_mask,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    if not folds:
        raise ValueError("no valid folds produced; relax horizon/min_train_frac")
    return folds

"""Leakage-free backtest harness.

Ties the pieces together: build (origin -> target) pairs for several horizons,
walk forward through rolling-origin folds, refit each forecaster on each fold's
*past*, score it on the fold's *future*, and aggregate into one scorecard plus an
economic scorecard. Any object with ``fit(pairs)`` / ``predict(pairs)`` plugs in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml import metrics
from ml.baselines import DEFAULT_QUANTILES
from ml.dataset import build_multi_horizon_targets
from ml.economics import dispatch_decisions, economic_value
from ml.splits import rolling_origin_folds


@dataclass
class BacktestResult:
    name: str
    overall: dict = field(default_factory=dict)
    economics: dict = field(default_factory=dict)
    per_horizon: dict = field(default_factory=dict)  # horizon -> metric dict


def _fold_pair_masks(pairs: pd.DataFrame, fold) -> tuple[np.ndarray, np.ndarray]:
    ts = pairs["ts"]
    train = (ts < fold.train_end).to_numpy()
    test = ((ts >= fold.test_start) & (ts < fold.test_end)).to_numpy()
    return train, test


def run_backtest(
    df: pd.DataFrame,
    forecaster_factory,
    name: str,
    horizons=(1, 6, 24, 72),
    n_folds: int = 4,
    quantiles=DEFAULT_QUANTILES,
    mw: float = 100.0,
) -> BacktestResult:
    """Backtest one forecaster across folds and horizons.

    ``forecaster_factory`` is a zero-arg callable returning a fresh forecaster, so
    every fold trains from scratch on its own history (no state bleeds across folds).
    """
    horizons = list(horizons)
    pairs = build_multi_horizon_targets(df, horizons)
    folds = rolling_origin_folds(
        df["ts"], n_folds=n_folds, horizon=max(horizons)
    )

    # Accumulate out-of-sample predictions so the scorecard is computed once over
    # all held-out rows (rather than averaging per-fold averages of unequal size).
    y_all: list[np.ndarray] = []
    pred_all: dict[float, list[np.ndarray]] = {q: [] for q in quantiles}
    horizon_all: list[np.ndarray] = []

    for fold in folds:
        tr, te = _fold_pair_masks(pairs, fold)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        train_pairs = pairs[tr]
        test_pairs = pairs[te]

        model = forecaster_factory()
        model.fit(train_pairs, quantiles)
        preds = model.predict(test_pairs, quantiles)

        y_all.append(test_pairs["target_spread"].to_numpy(dtype=float))
        horizon_all.append(test_pairs["horizon"].to_numpy())
        for q in quantiles:
            pred_all[q].append(np.asarray(preds[q], dtype=float))

    y = np.concatenate(y_all)
    horizon_arr = np.concatenate(horizon_all)
    preds = {q: np.concatenate(pred_all[q]) for q in quantiles}

    result = BacktestResult(name=name)
    result.overall = metrics.quantile_scorecard(y, preds)

    # Economics use the median (p50) as the dispatch signal.
    median_q = min(quantiles, key=lambda q: abs(q - 0.5))
    decisions = dispatch_decisions(preds[median_q])
    result.economics = economic_value(y, decisions, mw=mw)

    for h in horizons:
        m = horizon_arr == h
        if m.sum() == 0:
            continue
        result.per_horizon[h] = metrics.quantile_scorecard(
            y[m], {q: preds[q][m] for q in quantiles}
        )
        result.per_horizon[h]["capture_rate"] = economic_value(
            y[m], dispatch_decisions(preds[median_q][m]), mw=mw
        ).get("capture_rate", float("nan"))

    return result

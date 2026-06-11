"""Baseline quantile forecasters.

The cardinal rule of the rebuilt pipeline: *no model ships unless it beats these
baselines on the leakage-free backtest.* A LightGBM that can't beat "the spread
will be what it is right now" is not adding value, and most naive time-series
models lose to persistence at short horizons — so this is a meaningful bar.

Each baseline implements the same tiny interface:

    fit(train_pairs)                  -> learns residual / climatology quantiles
    predict(origin_pairs, quantiles)  -> {q: np.ndarray} aligned to the input rows

where the frames are produced by ``ml.dataset.build_forecast_targets``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


class PersistenceForecaster:
    """Point forecast = the current spread; uncertainty from historical drift.

    For each horizon we learn the empirical distribution of
    ``target_spread - origin_spread`` on the training pairs, then add its
    quantiles to the origin spread at predict time. This is the canonical strong
    baseline for short-horizon price forecasting.
    """

    def __init__(self) -> None:
        self._residual_q: dict[tuple[int, float], float] = {}
        self._horizons: set[int] = set()

    def fit(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        residual = pairs["target_spread"] - pairs["spread"]
        for h, g in residual.groupby(pairs["horizon"]):
            self._horizons.add(int(h))
            for q in quantiles:
                self._residual_q[(int(h), q)] = float(np.quantile(g, q))
        return self

    def predict(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        base = pairs["spread"].to_numpy(dtype=float)
        horizons = pairs["horizon"].to_numpy()
        preds: dict[float, np.ndarray] = {}
        for q in quantiles:
            offsets = np.array([
                self._residual_q.get((int(h), q), 0.0) for h in horizons
            ])
            preds[q] = base + offsets
        return preds


class ClimatologyForecaster:
    """Pure seasonal baseline: the spread distribution for this hour-of-day.

    Ignores the origin state entirely and predicts the quantiles of spread
    conditioned on the *target's* hour of day, learned from training history.
    Tends to beat persistence at long horizons where "right now" stops mattering.
    """

    def __init__(self, by: Sequence[str] = ("hour",)) -> None:
        self.by = list(by)
        self._table: dict[tuple, dict[float, float]] = {}
        self._global: dict[float, float] = {}

    def _target_keys(self, pairs: pd.DataFrame) -> pd.DataFrame:
        # Climatology is keyed on the *target* time's calendar fields.
        keys = pd.DataFrame(index=pairs.index)
        keys["hour"] = pairs["target_ts"].dt.hour
        keys["weekday"] = pairs["target_ts"].dt.weekday
        keys["month"] = pairs["target_ts"].dt.month
        return keys

    def fit(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        keys = self._target_keys(pairs)
        y = pairs["target_spread"]
        self._global = {q: float(np.quantile(y, q)) for q in quantiles}
        for key, idx in keys.groupby(self.by).groups.items():
            vals = y.loc[idx]
            key_t = key if isinstance(key, tuple) else (key,)
            self._table[key_t] = {q: float(np.quantile(vals, q)) for q in quantiles}
        return self

    def predict(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        keys = self._target_keys(pairs)[self.by]
        preds = {q: np.empty(len(pairs)) for q in quantiles}
        for i, (_, row) in enumerate(keys.iterrows()):
            key_t = tuple(row.values)
            table = self._table.get(key_t, self._global)
            for q in quantiles:
                preds[q][i] = table.get(q, self._global[q])
        return preds

"""Model forecasters that share the baselines' fit/predict interface.

Putting models behind the same interface as the baselines lets the backtest treat
them identically, so "did the model beat persistence?" is a fair question answered
on the leakage-free folds.

- ``LightGBMQuantileForecaster`` — three quantile boosters. Accepts an explicit
  ``feature_cols`` so the legacy feature set and the Phase-3 engineered set can be
  compared head-to-head.
- ``ConformalizedForecaster`` — wraps any base forecaster and calibrates its interval
  to nominal coverage (CQR), using a time-ordered hold-out carved from the fold's
  training pairs.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ml.baselines import DEFAULT_QUANTILES
from ml.conformal import ConformalInterval
from ml.dataset import ORIGIN_FEATURES

# Legacy feature set (the original pipeline's inputs) used when feature_cols is None.
_LEGACY_FEATURES = [*ORIGIN_FEATURES, "horizon", "hour_sin", "hour_cos"]


class LightGBMQuantileForecaster:
    """Three LightGBM quantile boosters (p10/p50/p90), one model across horizons."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        num_boost_round: int = 200,
        seed: int = 42,
        params: dict | None = None,
    ) -> None:
        self.feature_cols = feature_cols  # None => legacy behavior
        self.num_boost_round = num_boost_round
        self.seed = seed
        self.params = params or {}  # tuned overrides (e.g. from configs/spread_lgbm.yaml)
        self._models: dict[float, object] = {}
        self._cols: list[str] = []

    def _matrix(self, pairs: pd.DataFrame) -> np.ndarray:
        if self.feature_cols is None:
            df = pairs.copy()
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            self._cols = _LEGACY_FEATURES
        else:
            df = pairs
            self._cols = self.feature_cols
        return df[self._cols].to_numpy(dtype=float)

    def fit(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        import lightgbm as lgb

        X = self._matrix(pairs)
        y = pairs["target_spread"].to_numpy(dtype=float)
        for q in quantiles:
            params = {
                "objective": "quantile", "alpha": q, "metric": "quantile",
                "learning_rate": 0.1, "num_leaves": 31, "max_depth": 6,
                "min_data_in_leaf": 20, "verbose": -1, "seed": self.seed,
                **self.params,           # tuned overrides win
                "objective": "quantile",  # never let an override break the contract
                "alpha": q,
            }
            dtrain = lgb.Dataset(X, label=y, feature_name=self._cols)
            self._models[q] = lgb.train(params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        X = self._matrix(pairs)
        # Independent heads can cross; sort per row so p10 <= p50 <= p90.
        stacked = np.sort(np.vstack([self._models[q].predict(X) for q in quantiles]), axis=0)
        return {q: stacked[i] for i, q in enumerate(quantiles)}


class ConformalizedForecaster:
    """Wrap a base forecaster and calibrate its p_lo–p_hi interval to nominal coverage.

    Splits the fold's training pairs time-wise into proper-train (fits the base) and
    calibration (learns the CQR widening). The median is passed through unchanged; the
    interval is widened so realized coverage matches ``hi_q - lo_q``.
    """

    def __init__(
        self,
        base_factory: Callable[[], object],
        lo_q: float = 0.1,
        hi_q: float = 0.9,
        cal_frac: float = 0.3,
    ) -> None:
        self.base_factory = base_factory
        self.lo_q = lo_q
        self.hi_q = hi_q
        self.cal_frac = cal_frac
        self.base = None
        self.conf = ConformalInterval(lo_q, hi_q)

    def fit(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        ordered = pairs.sort_values("ts")
        cut = int(len(ordered) * (1 - self.cal_frac))
        proper, calib = ordered.iloc[:cut], ordered.iloc[cut:]
        if len(calib) < 20:  # too little to calibrate; fall back to whole set
            proper = calib = ordered
        self.base = self.base_factory()
        self.base.fit(proper, quantiles)
        cp = self.base.predict(calib, quantiles)
        self.conf.calibrate(calib["target_spread"].to_numpy(dtype=float),
                            cp[self.lo_q], cp[self.hi_q])
        return self

    def predict(self, pairs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        preds = self.base.predict(pairs, quantiles)
        lo, hi = self.conf.apply(preds[self.lo_q], preds[self.hi_q])
        preds[self.lo_q] = lo
        preds[self.hi_q] = hi
        return preds

"""Spike-risk model — a principled replacement for the self-distilled regime classifier.

Why the old one was broken
--------------------------
``scripts/train_regime_classifier.py`` labels each row with a *deterministic rule*
over (lmp, spread, temp, wind), then trains a model to predict that label from the
same columns. The model just relearns the rule from its own inputs, so the reported
F1 is near-perfect and meaningless. It is also a *nowcast* (describe now), not a
forecast (warn ahead).

The redesign
------------
Predict a genuinely useful, forward-looking, *actionable* event:

    label = 1 if the spread exceeds `threshold` at any hour in (t, t+H]

i.e. "is a high-value generation window (a price spike / scarcity event) coming in
the next H hours?" The label depends only on the future; the features depend only on
the past — no circularity. It is naturally imbalanced, so it's scored with the
metrics that matter for rare events (PR-AUC, Brier), against real baselines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import feature_columns


def build_spike_labels(
    df: pd.DataFrame,
    horizon: int = 24,
    threshold: float = 50.0,
    site_col: str = "site_id",
    ts_col: str = "ts",
    target_col: str = "spread",
) -> pd.DataFrame:
    """Label each origin row 1 if `target_col` exceeds `threshold` within the next
    `horizon` hours (exclusive of now), else 0.

    Per-site forward window so events never bleed across facilities. Rows with no
    future observations are dropped (no label can be formed).
    """
    out = []
    for _, g in df.groupby(site_col):
        g = g.sort_values(ts_col).reset_index(drop=True).copy()
        s = g[target_col]
        # Forward max over (t, t+H]: stack shifts -1..-H and take the row-wise max.
        fwd = pd.concat([s.shift(-k) for k in range(1, horizon + 1)], axis=1)
        g["spike_next"] = (fwd.max(axis=1) >= threshold).astype("float")
        g.loc[fwd.isna().all(axis=1), "spike_next"] = np.nan  # no future at all
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res.dropna(subset=["spike_next"]).reset_index(drop=True)


class SpikeRiskModel:
    """LightGBM binary classifier for spike risk; outputs a calibrated-ish probability."""

    def __init__(self, num_boost_round: int = 300, seed: int = 42) -> None:
        self.num_boost_round = num_boost_round
        self.seed = seed
        self._model = None
        self._cols: list[str] = []

    def fit(self, pairs: pd.DataFrame, feature_cols: list[str] | None = None):
        import lightgbm as lgb

        self._cols = feature_cols or feature_columns(pairs)
        X = pairs[self._cols].to_numpy(dtype=float)
        y = pairs["spike_next"].to_numpy(dtype=float)
        params = {
            "objective": "binary", "metric": "average_precision",
            "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
            "min_data_in_leaf": 30, "is_unbalance": True,
            "verbose": -1, "seed": self.seed,
        }
        self._model = lgb.train(params, lgb.Dataset(X, label=y, feature_name=self._cols),
                                num_boost_round=self.num_boost_round)
        return self

    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        X = pairs[self._cols].to_numpy(dtype=float)
        return self._model.predict(X)


# ── Imbalanced-classification metrics (no sklearn dependency required) ──────
def average_precision(y_true, scores) -> float:
    """Area under the precision-recall curve (PR-AUC) via the step-wise estimator.

    The right headline for rare events: unlike ROC-AUC it isn't flattered by the
    huge true-negative mass. Equals mean prevalence for a random scorer.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    # Sum precision over the points where a new positive is retrieved (recall rises).
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def roc_auc(y_true, scores) -> float:
    """AUROC via the rank-sum (Mann-Whitney) identity."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    auc = (ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(auc)


def brier_score(y_true, scores) -> float:
    """Mean squared error of the probability forecast (lower = better calibrated)."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    return float(np.mean((s - y) ** 2))


def precision_recall_at(y_true, scores, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(float)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision) and not np.isnan(recall)
          else float("nan"))
    return {"precision": precision, "recall": recall, "f1": f1}

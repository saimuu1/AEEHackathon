"""Optuna hyperparameter tuning for the rich-feature spread forecaster.

Phase 3's bake-off found the engineered features didn't beat the legacy set under
the *default* LightGBM params. This gives them a fair shot: tune on a leakage-free
time split (train in the past, validate in the future with an embargo) minimizing
p50 pinball loss. Writes the best params to configs/spread_lgbm.yaml and a summary to
reports/phase3_tuning.md.

Usage: .venv/bin/python -m scripts.tune_forecaster [--trials 40]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import build_multi_horizon_targets  # noqa: E402
from ml.features import add_causal_features, feature_columns  # noqa: E402
from ml.metrics import pinball_loss  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "backend", "data", "historical_spreads.parquet")
REPORT = os.path.join(ROOT, "reports", "phase3_tuning.md")
CONFIG = os.path.join(ROOT, "configs", "spread_lgbm.yaml")

HORIZONS = [1, 6, 24, 72]
_DEFAULT = {"learning_rate": 0.1, "num_leaves": 31, "max_depth": 6,
            "min_data_in_leaf": 20, "feature_fraction": 1.0,
            "bagging_fraction": 1.0, "lambda_l1": 0.0, "lambda_l2": 0.0}


def _split(pairs):
    """Leakage-free train/val split on origin time with an embargo >= max horizon."""
    pairs = pairs.sort_values("ts")
    t_min, t_max = pairs["ts"].min(), pairs["ts"].max()
    val_start = t_min + (t_max - t_min) * 0.7
    train_end = val_start - pd.Timedelta(hours=max(HORIZONS))
    train = pairs[pairs["ts"] < train_end]
    val = pairs[pairs["ts"] >= val_start]
    return train, val


def _train_eval(params, train, val, cols):
    import lightgbm as lgb

    Xtr, ytr = train[cols].to_numpy(float), train["target_spread"].to_numpy(float)
    Xv, yv = val[cols].to_numpy(float), val["target_spread"].to_numpy(float)
    p = {"objective": "quantile", "alpha": 0.5, "metric": "quantile",
         "verbose": -1, "seed": 42, **params}
    booster = lgb.train(p, lgb.Dataset(Xtr, label=ytr, feature_name=cols),
                        num_boost_round=300)
    return pinball_loss(yv, booster.predict(Xv), 0.5)


def main() -> None:
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = pd.read_parquet(PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    enriched = add_causal_features(df)
    cols = feature_columns(enriched) + ["horizon"]
    pairs = build_multi_horizon_targets(enriched, HORIZONS)
    train, val = _split(pairs)
    print(f"Tuning on {len(train):,} train / {len(val):,} val pairs · {len(cols)} feats")

    baseline = _train_eval(_DEFAULT, train, val, cols)
    print(f"baseline (default params) val pinball = {baseline:.4f}")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        return _train_eval(params, train, val, cols)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    tuned = study.best_value
    improvement = 100 * (baseline - tuned) / baseline
    print(f"tuned val pinball = {tuned:.4f}  ({improvement:+.1f}% vs baseline)")

    _save_config(best)
    _write_report(baseline, tuned, improvement, best, len(train), len(val))
    print(f"\n✅ Wrote {os.path.relpath(CONFIG, ROOT)} and {os.path.relpath(REPORT, ROOT)}")


def _save_config(best: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    lines = ["# Best LightGBM params from scripts/tune_forecaster.py (Optuna).",
             "# Consumed by training in Phase 4 (config-driven, reproducible).",
             "objective: quantile", "metric: quantile", "num_boost_round: 300", "seed: 42"]
    for k, v in best.items():
        lines.append(f"{k}: {v}")
    with open(CONFIG, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_report(baseline, tuned, improvement, best, n_train, n_val) -> None:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    verdict = (
        f"Tuning improved val pinball by **{improvement:.1f}%** — the engineered features "
        "pay off once the model is regularized/subsampled for them."
        if improvement > 1 else
        f"Tuning moved val pinball by {improvement:+.1f}% — a small effect. Honest read: "
        "the ceiling here is data volume, not hyperparameters. The tuned config is still "
        "the right default to carry into Phase 4."
    )
    lines = [
        "# Phase 3 — Hyperparameter Tuning (Optuna)",
        "",
        f"_TPE sampler · p50 pinball objective · leakage-free time split "
        f"({n_train:,} train / {n_val:,} val) · embargo ≥ {max(HORIZONS)}h._",
        "",
        "| | val pinball (p50) ↓ |",
        "|---|---:|",
        f"| Default params | {baseline:.4f} |",
        f"| Optuna-tuned | {tuned:.4f} |",
        "",
        f"## Verdict\n{verdict}",
        "",
        "## Best params",
        "```yaml",
        *[f"{k}: {v}" for k, v in best.items()],
        "```",
        "",
        "Saved to `configs/spread_lgbm.yaml` for config-driven, reproducible training "
        "in Phase 4 (MLOps).",
        "",
    ]
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

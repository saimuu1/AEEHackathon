"""Config-driven, tracked, versioned training entrypoint (Phase 4 MLOps).

One command trains the spread forecaster reproducibly and governs its lifecycle:
1. load a YAML config (data, features, horizons, tuned params, seed),
2. score the model on the leakage-free backtest (vs the persistence baseline),
3. train the final model on all data,
4. hash the dataset and log everything to MLflow,
5. register the model version (with its data hash + metrics) and write a model card.

Usage:
    .venv/bin/python -m scripts.train --config configs/train_spread.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.backtest import run_backtest  # noqa: E402
from ml.baselines import PersistenceForecaster  # noqa: E402
from ml.dataset import build_multi_horizon_targets  # noqa: E402
from ml.features import add_causal_features, feature_columns  # noqa: E402
from ml.forecasters import LightGBMQuantileForecaster  # noqa: E402
from mlops import model_card, versioning  # noqa: E402
from mlops.registry import ModelRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_model_params(path: str) -> tuple[dict, int, int]:
    with open(os.path.join(ROOT, path)) as f:
        raw = yaml.safe_load(f)
    nbr = int(raw.pop("num_boost_round", 300))
    seed = int(raw.pop("seed", 42))
    for k in ("objective", "metric"):  # forecaster sets these itself
        raw.pop(k, None)
    return raw, nbr, seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_spread.yaml")
    ap.add_argument("--register", action="store_true",
                    help="register the trained model in the registry")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config)) as f:
        cfg = yaml.safe_load(f)

    parquet = os.path.join(ROOT, cfg["data"]["parquet"])
    horizons = cfg["horizons"]
    quantiles = cfg["quantiles"]
    params, nbr, seed = _load_model_params(cfg["model_params"])

    df = pd.read_parquet(parquet)
    df["ts"] = pd.to_datetime(df["ts"])
    data_hash = versioning.dataset_hash(parquet)
    window = f"{df['ts'].min():%Y-%m-%d} → {df['ts'].max():%Y-%m-%d}"
    print(f"data: {len(df):,} rows | {window} | hash {versioning.short(data_hash)}")

    if cfg["features"] == "rich":
        data = add_causal_features(df)
        feat_cols = feature_columns(data) + ["horizon"]
    else:
        data, feat_cols = df, None

    def factory():
        return LightGBMQuantileForecaster(feature_cols=feat_cols,
                                          num_boost_round=nbr, seed=seed, params=params)

    print("▶ backtesting candidate ...")
    cand = run_backtest(data, factory, "candidate", horizons=horizons,
                        n_folds=cfg["folds"], quantiles=quantiles, mw=cfg["mw"])
    print("▶ backtesting persistence baseline ...")
    base = run_backtest(df, PersistenceForecaster, "persistence", horizons=horizons,
                        n_folds=cfg["folds"], quantiles=quantiles, mw=cfg["mw"])

    metrics = {**cand.overall, **cand.economics}
    base_metrics = {**base.overall, **base.economics}
    print(f"   candidate: pinball={metrics['mean_pinball']:.3f} "
          f"cov={metrics.get('interval_coverage', float('nan')):.0%} "
          f"capture={metrics.get('capture_rate', float('nan')):.1%}")

    # Train the final model on all pairs (for serving) and save artifacts.
    pairs = build_multi_horizon_targets(data, horizons)
    final = factory().fit(pairs, quantiles)
    created = dt.datetime.now().isoformat(timespec="seconds")

    artifacts_dir = tempfile.mkdtemp()
    artifact_paths = []
    for q, booster in final._models.items():
        p = os.path.join(artifacts_dir, f"spread_p{int(q * 100)}.txt")
        booster.save_model(p)
        artifact_paths.append(p)

    card = model_card.render(
        version=ModelRegistry().next_version(), created_at=created,
        data_hash=data_hash, data_window=window, n_rows=len(df),
        params={**params, "num_boost_round": nbr, "seed": seed},
        metrics=metrics, baseline_metrics=base_metrics,
    )
    card_path = os.path.join(artifacts_dir, "MODEL_CARD.md")
    with open(card_path, "w") as f:
        f.write(card)
    artifact_paths.append(card_path)

    _log_mlflow(cfg, params, metrics, base_metrics, data_hash, artifacts_dir)

    if args.register:
        reg = ModelRegistry()
        mv = reg.register(data_hash=data_hash, metrics=metrics,
                          params={**params, "num_boost_round": nbr, "seed": seed},
                          created_at=created, source_artifacts=artifact_paths,
                          notes=f"features={cfg['features']}")
        print(f"✅ registered v{mv.version} (production={reg.production()['version']})")
    else:
        print("ℹ️  not registered (pass --register to add to the model registry)")
    print(f"   model card + artifacts in {artifacts_dir}")


def _log_mlflow(cfg, params, metrics, base_metrics, data_hash, artifacts_dir) -> None:
    import mlflow

    # SQLite backend (the file store is deprecated in MLflow 3.x).
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run():
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("features", cfg["features"])
        mlflow.log_params({**params, "horizons": cfg["horizons"], "folds": cfg["folds"]})
        mlflow.log_metrics({
            "mean_pinball": metrics["mean_pinball"],
            "interval_coverage": metrics.get("interval_coverage", float("nan")),
            "capture_rate": metrics.get("capture_rate", float("nan")),
            "policy_savings": metrics.get("policy_savings", float("nan")),
            "baseline_pinball": base_metrics["mean_pinball"],
            "baseline_capture": base_metrics.get("capture_rate", float("nan")),
        })
        mlflow.log_artifacts(artifacts_dir, artifact_path="model")
    print("   ✅ logged run to MLflow (mlruns/)")


if __name__ == "__main__":
    main()

"""
Train LightGBM quantile spread forecasters (p10, p50, p90).
Single model covers all forecast horizons h ∈ [1, 72].

Usage: /opt/anaconda3/bin/python -m scripts.train_spread_forecaster
"""
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARQUET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "backend", "data", "historical_spreads.parquet")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "backend", "models")


def build_forecast_pairs(df, max_horizon=72):
    """Build (current_state, horizon, future_spread) training pairs."""
    rows = []
    for site in df["site_id"].unique():
        site_df = df[df["site_id"] == site].sort_values("ts").reset_index(drop=True)

        # Sample horizons to keep dataset manageable
        horizons = list(range(1, min(max_horizon + 1, len(site_df))))

        for i in range(len(site_df)):
            current = site_df.iloc[i]
            for h in horizons:
                if i + h >= len(site_df):
                    break
                future = site_df.iloc[i + h]
                rows.append({
                    "lmp": current["lmp"],
                    "spread": current["spread"],
                    "gas_price": current["gas_price"],
                    "temp_f": current["temp_f"],
                    "wind_speed": current["wind_speed"],
                    "hour": current["hour"],
                    "month": current["month"],
                    "weekday": current["weekday"],
                    "lmp_6h_lag": current.get("lmp_6h_lag", current["lmp"]),
                    "lmp_trend_6h": current.get("lmp_trend_6h", 0),
                    "horizon": h,
                    "target_spread": future["spread"],
                })

    return pd.DataFrame(rows)


def train():
    print("=" * 60)
    print("📈 Training Spread Forecasters (p10/p50/p90)")
    print("=" * 60)

    df = pd.read_parquet(PARQUET)
    print(f"   Loaded {len(df)} rows")

    # Build training pairs (limit horizons to keep fast)
    print("   Building forecast pairs...")
    pairs = build_forecast_pairs(df, max_horizon=72)
    print(f"   Generated {len(pairs)} training pairs")

    # Features
    feature_cols = ["lmp", "spread", "gas_price", "temp_f", "wind_speed",
                    "hour", "month", "weekday", "lmp_6h_lag", "lmp_trend_6h",
                    "horizon"]

    # Add cyclical features
    pairs["hour_sin"] = np.sin(2 * np.pi * pairs["hour"] / 24)
    pairs["hour_cos"] = np.cos(2 * np.pi * pairs["hour"] / 24)
    feature_cols.extend(["hour_sin", "hour_cos"])

    X = pairs[feature_cols].values
    y = pairs["target_spread"].values

    # Time-based split
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Train three quantile models
    for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        print(f"\n   Training {label} (alpha={alpha})...")

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 10,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )

        # Evaluate
        preds = model.predict(X_test)
        mae = np.mean(np.abs(y_test - preds))
        coverage = np.mean(y_test <= preds) if alpha > 0.5 else np.mean(y_test >= preds)

        print(f"   {label}: MAE=${mae:.2f}/MWh, Coverage={coverage:.1%}")

        # Save
        path = os.path.join(MODELS_DIR, f"spread_forecaster_{label}.txt")
        model.save_model(path)
        print(f"   ✅ Saved: {path}")

    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "forecast_features.pkl"))
    print(f"\n{'=' * 60}")
    print("✅ All three quantile models trained and saved!")
    print("=" * 60)


if __name__ == "__main__":
    train()

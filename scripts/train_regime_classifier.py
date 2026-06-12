"""
Train LightGBM regime classifier on historical spread data.
Self-distills labels from rule-based classifier, then trains ML model.

Usage: /opt/anaconda3/bin/python -m scripts.train_regime_classifier
"""
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARQUET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "backend", "data", "historical_spreads.parquet")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "backend", "models")


def classify_regime_rules(row):
    """Rule-based regime classification (teacher for self-distillation)."""
    lmp = row["lmp"]
    spread = row["spread"]
    temp = row.get("temp_f", 75)
    wind = row.get("wind_speed", 10)

    if lmp >= 5000:
        return "uri_emergency"
    if lmp >= 500 and temp <= 20:
        return "winter_storm"
    if lmp >= 200 or (lmp >= 100 and temp >= 95):
        return "scarcity"
    if temp >= 90 and lmp >= 50:
        return "heat_dome"
    if lmp <= 5 and wind >= 20:
        return "wind_glut"
    if lmp <= 15 and spread <= 0:
        return "oversupply"
    return "normal"


def train():
    print("=" * 60)
    print(" Training Regime Classifier")
    print("=" * 60)

    df = pd.read_parquet(PARQUET)
    print(f"   Loaded {len(df)} rows")

    # Generate labels from rules
    df["regime"] = df.apply(classify_regime_rules, axis=1)
    print("\n   Label distribution:")
    print(df["regime"].value_counts().to_string())

    # Features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    feature_cols = ["lmp", "gas_price", "spread", "temp_f", "wind_speed",
                    "hour_sin", "hour_cos", "month", "weekday",
                    "lmp_6h_lag", "lmp_24h_lag", "lmp_trend_6h", "lmp_trend_24h"]

    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["regime"])

    # Time-based split (last 25% is test)
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n   Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "metric": "multi_logloss",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 5,
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
    y_pred = model.predict(X_test).argmax(axis=1)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n   Macro F1: {f1:.4f}")
    
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    present_classes = [le.classes_[i] for i in unique_labels]
    print(f"\n{classification_report(y_test, y_pred, labels=unique_labels, target_names=present_classes)}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model(os.path.join(MODELS_DIR, "regime_classifier.txt"))
    joblib.dump(le, os.path.join(MODELS_DIR, "regime_label_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "regime_features.pkl"))

    print(f"   ✅ Saved to {MODELS_DIR}/")
    print(f"   Classes: {le.classes_.tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    train()

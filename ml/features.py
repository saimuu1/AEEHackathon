"""Causal feature engineering.

Replaces the original pipeline's two hand-rolled lags with a proper feature set:
trailing rolling statistics, volatility, ramps, calendar encodings, and weather
degree-day transforms. The non-negotiable property is **causality** — a feature at
time *t* may use data at or before *t*, never after. Every rolling window is trailing
and every lag looks back, so nothing leaks the future into a training row.
``tests/test_features.py``-style checks assert this by truncation.

The engineered frame feeds ``ml.dataset.build_forecast_targets`` exactly like the raw
frame did; downstream code just sees more columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Windows (hours) for trailing rolling stats.
_ROLL_WINDOWS = (3, 6, 24)
_LMP_LAGS = (1, 3, 6, 24)
_SPREAD_LAGS = (1, 6, 24)


def _add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = df[ts_col].dt
    df["hour"] = ts.hour
    df["weekday"] = ts.weekday
    df["month"] = ts.month
    df["is_weekend"] = (ts.weekday >= 5).astype(int)
    # Cyclical encodings so the model sees 23:00 and 00:00 as adjacent.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df


def _add_weather_transforms(df: pd.DataFrame) -> pd.DataFrame:
    if "temp_f" in df:
        # Degree-hours capture the nonlinearity of cooling/heating demand.
        df["cdd"] = (df["temp_f"] - 65).clip(lower=0)   # cooling demand
        df["hdd"] = (65 - df["temp_f"]).clip(lower=0)   # heating demand
        df["temp_sq"] = df["temp_f"] ** 2
    return df


def add_causal_features(
    df: pd.DataFrame,
    site_col: str = "site_id",
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with strictly-causal features.

    Per-site (so windows never bridge two facilities), adds lags, trailing rolling
    mean/std/min/max, recent volatility, ramps, calendar, and weather transforms.
    NaNs from short warm-up windows are left in place — LightGBM handles them
    natively, and dropping rows would waste early history.
    """
    df = df.sort_values([site_col, ts_col]).reset_index(drop=True).copy()
    g = df.groupby(site_col, group_keys=False)

    # Lags (look strictly backward).
    for k in _LMP_LAGS:
        df[f"lmp_lag_{k}"] = g["lmp"].shift(k)
    for k in _SPREAD_LAGS:
        df[f"spread_lag_{k}"] = g["spread"].shift(k)

    # Trailing rolling stats (window ends at t inclusive → uses only data ≤ t).
    for w in _ROLL_WINDOWS:
        roll_lmp = g["lmp"].rolling(w, min_periods=1)
        df[f"lmp_roll_mean_{w}"] = roll_lmp.mean().reset_index(drop=True)
        df[f"lmp_roll_std_{w}"] = roll_lmp.std().reset_index(drop=True)
        df[f"lmp_roll_min_{w}"] = roll_lmp.min().reset_index(drop=True)
        df[f"lmp_roll_max_{w}"] = roll_lmp.max().reset_index(drop=True)
        df[f"spread_roll_mean_{w}"] = (
            g["spread"].rolling(w, min_periods=1).mean().reset_index(drop=True)
        )

    # Volatility + ramps.
    df["lmp_vol_24"] = g["lmp"].rolling(24, min_periods=2).std().reset_index(drop=True)
    df["lmp_ramp_1"] = df["lmp"] - g["lmp"].shift(1)
    df["lmp_ramp_6"] = df["lmp"] - g["lmp"].shift(6)
    df["spread_ramp_6"] = df["spread"] - g["spread"].shift(6)

    df = _add_calendar(df, ts_col)
    df = _add_weather_transforms(df)
    return df


# Engineered columns the model consumes (current-state + the derived features).
# Excludes identifiers, ts, gen_cost, and the target.
def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"site_id", "zone", "ts", "target_ts", "target_spread",
               "gen_cost", "org_id"}
    return [c for c in df.columns if c not in exclude and df[c].dtype != "O"]

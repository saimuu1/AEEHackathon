import numpy as np
import pandas as pd

from ml.features import add_causal_features, feature_columns


def _series(n=200, site="a"):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    lmp = 30 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame({
        "site_id": site, "ts": ts, "lmp": lmp, "gas_price": 2.5,
        "gen_cost": 22.0, "spread": lmp - 22.0,
        "temp_f": 70 + rng.normal(0, 5, n), "wind_speed": np.abs(rng.normal(8, 3, n)),
    })


def test_features_are_strictly_causal():
    """A feature at row t must not change when future rows are altered.

    Build features on the full series and on a truncated copy [:t+1]; the row-t
    feature values must be identical, proving no future data leaked in.
    """
    df = _series()
    full = add_causal_features(df)
    cols = feature_columns(full)
    t = 150
    truncated = add_causal_features(df.iloc[: t + 1].copy())
    a = full.loc[t, cols].astype(float).to_numpy()
    b = truncated.iloc[t][cols].astype(float).to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_windows_do_not_bridge_sites():
    df = pd.concat([_series(50, "a"), _series(50, "b")], ignore_index=True)
    out = add_causal_features(df)
    # First row of each site has no prior history -> lag_1 must be NaN, not borrowed.
    first_b = out[out["site_id"] == "b"].iloc[0]
    assert np.isnan(first_b["lmp_lag_1"])


def test_degree_day_transforms():
    df = _series()
    out = add_causal_features(df)
    # cdd and hdd are non-negative and never both positive in the same row.
    assert (out["cdd"] >= 0).all() and (out["hdd"] >= 0).all()
    assert not ((out["cdd"] > 0) & (out["hdd"] > 0)).any()


def test_feature_columns_excludes_target_and_ids():
    df = _series()
    out = add_causal_features(df)
    out["target_spread"] = 1.0
    out["target_ts"] = out["ts"]
    cols = feature_columns(out)
    for leaked in ("target_spread", "target_ts", "site_id", "ts", "gen_cost"):
        assert leaked not in cols

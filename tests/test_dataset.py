import numpy as np
import pandas as pd

from ml.dataset import build_forecast_targets, build_multi_horizon_targets


def _toy(site="a", n=10, start="2026-01-01"):
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "ts": ts,
        "site_id": site,
        "spread": np.arange(n, dtype=float),
        "lmp": np.arange(n, dtype=float) + 20,
    })


def test_exact_horizon_pairing():
    df = _toy(n=10)
    pairs = build_forecast_targets(df, horizon=3)
    # origin at t has target at t+3h -> target_spread = origin_spread + 3.
    assert np.allclose(pairs["target_spread"] - pairs["spread"], 3.0)
    assert (pairs["target_ts"] - pairs["ts"] == pd.Timedelta(hours=3)).all()
    # 10 rows, horizon 3 -> last 3 origins have no target.
    assert len(pairs) == 7


def test_gaps_do_not_shift_horizon():
    """If an hour is missing, the h-ahead target must still be h *hours* ahead,
    not h *rows* ahead. Drop hour index 5 and confirm no wrong pairing appears."""
    df = _toy(n=10)
    df = df.drop(index=5).reset_index(drop=True)  # remove the 06:00 row
    pairs = build_forecast_targets(df, horizon=3)
    # No pair may claim a 3h target that doesn't exist (the missing timestamp).
    for _, r in pairs.iterrows():
        assert r["target_ts"] == r["ts"] + pd.Timedelta(hours=3)
        assert r["target_ts"] in set(df["ts"])


def test_sites_are_isolated():
    df = pd.concat([_toy("a", 6), _toy("b", 6, start="2026-02-01")], ignore_index=True)
    pairs = build_forecast_targets(df, horizon=2)
    # A site's target must come from the same site (timelines don't bridge sites).
    for _, r in pairs.iterrows():
        same_site = df[df["site_id"] == r["site_id"]]
        assert r["target_ts"] in set(same_site["ts"])


def test_multi_horizon_stacks():
    df = _toy(n=12)
    stacked = build_multi_horizon_targets(df, horizons=[1, 6, 24])
    # horizon 24 exceeds the 12-row window -> contributes nothing, no crash.
    assert set(stacked["horizon"].unique()) == {1, 6}

"""Leakage-free construction of forecast (origin → target) pairs.

The model's job: standing at time *t* for a site, predict the spread *h* hours
later. We materialise that as one row per (site, origin time, horizon), pairing
the origin's features with the realized future spread — using an exact
timestamp join so gaps in the hourly series never silently shift the horizon.
"""
from __future__ import annotations

import pandas as pd

# Columns describing the state *as of the origin time*. Anything derived from
# the future must never appear here.
ORIGIN_FEATURES = [
    "lmp", "spread", "gas_price", "temp_f", "wind_speed",
    "hour", "month", "weekday",
    "lmp_6h_lag", "lmp_24h_lag", "lmp_trend_6h", "lmp_trend_24h",
]


def build_forecast_targets(
    df: pd.DataFrame,
    horizon: int,
    ts_col: str = "ts",
    site_col: str = "site_id",
    target_col: str = "spread",
) -> pd.DataFrame:
    """Return one row per origin that has a realized target exactly ``horizon``
    hours ahead, within the same site.

    The result carries the origin features (suffix-free), the origin timestamp,
    the target timestamp, and ``target_spread``. An exact ``ts + horizon`` join
    guarantees the label is truly h-ahead rather than "h rows ahead", which would
    be wrong wherever the series has missing hours.
    """
    out = []
    delta = pd.Timedelta(hours=horizon)
    for site, g in df.groupby(site_col):
        g = g.sort_values(ts_col)
        future = g[[ts_col, target_col]].rename(
            columns={ts_col: "_future_ts", target_col: "target_spread"}
        )
        future["_origin_ts"] = future["_future_ts"] - delta
        merged = g.merge(future, left_on=ts_col, right_on="_origin_ts", how="inner")
        merged["horizon"] = horizon
        merged = merged.rename(columns={"_future_ts": "target_ts"})
        out.append(merged)

    if not out:
        return pd.DataFrame()
    result = pd.concat(out, ignore_index=True)
    return result.drop(columns=["_origin_ts"], errors="ignore")


def build_multi_horizon_targets(
    df: pd.DataFrame, horizons: list[int], **kwargs
) -> pd.DataFrame:
    """Stack ``build_forecast_targets`` across several horizons."""
    frames = [build_forecast_targets(df, h, **kwargs) for h in horizons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

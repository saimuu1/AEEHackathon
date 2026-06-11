"""Idempotent, incremental ingestion into the warehouse.

The core primitive is ``upsert_dataframe``: insert rows, and on a primary-key
collision update in place. That single property makes every loader *idempotent*
(re-running never duplicates) and *incremental* (a backfill of an overlapping date
range is safe). The per-source ``load_*`` helpers just normalise a fetched frame
to the raw table's columns and upsert it.

Network fetching lives in ``scripts/`` (and needs the EIA key); these loaders take
an already-fetched DataFrame so they're pure and unit-testable without the network.
"""
from __future__ import annotations

import duckdb
import pandas as pd


def upsert_dataframe(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    key_cols: list[str],
) -> int:
    """Upsert ``df`` into ``table`` keyed by ``key_cols`` (INSERT ... ON CONFLICT).

    Returns the number of rows submitted. Idempotent: re-submitting the same rows
    leaves the table unchanged; submitting changed non-key values updates them.
    """
    if df.empty:
        return 0
    cols = list(df.columns)
    non_key = [c for c in cols if c not in key_cols]
    collist = ", ".join(cols)
    conflict = ", ".join(key_cols)
    action = (
        "DO UPDATE SET " + ", ".join(f"{c} = excluded.{c}" for c in non_key)
        if non_key else "DO NOTHING"
    )
    con.register("_staging", df)
    try:
        con.execute(
            f"INSERT INTO {table} ({collist}) SELECT {collist} FROM _staging "
            f"ON CONFLICT ({conflict}) {action}"
        )
    finally:
        con.unregister("_staging")
    return len(df)


def load_lmp(con, df: pd.DataFrame) -> int:
    """df columns: node, ts, lmp."""
    return upsert_dataframe(con, "raw_lmp", df[["node", "ts", "lmp"]], ["node", "ts"])


def load_gas(con, df: pd.DataFrame) -> int:
    """df columns: hub, date, price."""
    return upsert_dataframe(con, "raw_gas", df[["hub", "date", "price"]], ["hub", "date"])


def load_weather(con, df: pd.DataFrame) -> int:
    """df columns: site_id, ts, temp_f, wind_speed."""
    return upsert_dataframe(
        con, "raw_weather",
        df[["site_id", "ts", "temp_f", "wind_speed"]], ["site_id", "ts"],
    )

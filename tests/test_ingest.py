import os
import tempfile

import pandas as pd

from pipeline import warehouse
from pipeline.ingest import load_gas, upsert_dataframe


def _con():
    con = warehouse.connect(os.path.join(tempfile.mkdtemp(), "t.duckdb"))
    warehouse.init_schema(con)
    return con


def _gas(price=3.0):
    return pd.DataFrame({"hub": ["Waha"], "date": [pd.Timestamp("2026-01-01").date()], "price": [price]})


def test_upsert_is_idempotent():
    con = _con()
    load_gas(con, _gas())
    load_gas(con, _gas())  # same row again
    assert con.execute("SELECT count(*) FROM raw_gas").fetchone()[0] == 1


def test_upsert_updates_non_key_values():
    con = _con()
    load_gas(con, _gas(price=3.0))
    load_gas(con, _gas(price=4.5))  # same key, new price
    rows = con.execute("SELECT price FROM raw_gas").fetchall()
    assert rows == [(4.5,)]


def test_incremental_backfill_overlap_is_safe():
    con = _con()
    df = pd.DataFrame({
        "node": ["LZ_WEST"] * 3,
        "ts": pd.date_range("2026-01-01", periods=3, freq="h"),
        "lmp": [10.0, 20.0, 30.0],
    })
    upsert_dataframe(con, "raw_lmp", df, ["node", "ts"])
    # Re-ingest an overlapping window (2 old + 1 new) — no duplication.
    df2 = pd.DataFrame({
        "node": ["LZ_WEST"] * 2,
        "ts": pd.date_range("2026-01-01 02:00", periods=2, freq="h"),
        "lmp": [30.0, 40.0],
    })
    upsert_dataframe(con, "raw_lmp", df2, ["node", "ts"])
    assert con.execute("SELECT count(*) FROM raw_lmp").fetchone()[0] == 4


def test_empty_df_is_noop():
    con = _con()
    assert load_gas(con, _gas().iloc[0:0]) == 0

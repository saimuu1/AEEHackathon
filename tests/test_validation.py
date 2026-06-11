import os
import tempfile

import numpy as np
import pandas as pd

from pipeline import validation, warehouse


def _good_frame(n=5):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    lmp = np.linspace(20, 60, n)
    gen_cost = np.full(n, 21.5)
    return pd.DataFrame({
        "org_id": "demo", "site_id": "midland", "ts": ts,
        "lmp": lmp, "gas_price": 2.4, "gen_cost": gen_cost,
        "spread": lmp - gen_cost, "temp_f": 75.0, "wind_speed": 8.0,
    })


def test_clean_data_passes():
    res = validation.validate(_good_frame(), validation.feature_spread_checks())
    assert res.ok
    assert res.n_quarantined == 0


def test_catches_spread_inconsistency():
    df = _good_frame()
    df.loc[2, "spread"] = 999.0  # break spread == lmp - gen_cost
    res = validation.validate(df, validation.feature_spread_checks())
    assert not res.ok
    assert res.n_quarantined == 1
    assert "spread=lmp-gen_cost" in res.reasons


def test_catches_out_of_range_and_nulls():
    df = _good_frame()
    df.loc[0, "gas_price"] = -5.0      # impossible negative gas
    df.loc[1, "lmp"] = np.nan          # null required field
    res = validation.validate(df, validation.feature_spread_checks())
    assert res.n_quarantined == 2
    assert "range[gas_price]" in res.reasons
    assert "not_null[lmp]" in res.reasons


def test_catches_duplicate_primary_key():
    df = pd.concat([_good_frame(2), _good_frame(2)], ignore_index=True)  # dup PKs
    res = validation.validate(df, validation.feature_spread_checks())
    assert res.reasons.get("unique[org_id,site_id,ts]") == 2


def test_quarantine_persists_bad_rows():
    con = warehouse.connect(os.path.join(tempfile.mkdtemp(), "t.duckdb"))
    warehouse.init_schema(con)
    df = _good_frame()
    df.loc[0, "lmp"] = np.nan
    res = validation.validate(df, validation.feature_spread_checks())
    n = validation.quarantine_rows(con, "feature_spread", res.bad, "test")
    assert n == 1
    assert con.execute("SELECT count(*) FROM quarantine").fetchone()[0] == 1

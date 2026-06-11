import os
import tempfile

import pandas as pd

from pipeline import features, tenancy, warehouse
from pipeline.ingest import load_gas, load_lmp, load_weather
from pipeline.tenancy import SiteConfig


def _warehouse_with_one_site():
    con = warehouse.connect(os.path.join(tempfile.mkdtemp(), "t.duckdb"))
    warehouse.init_schema(con)
    tenancy.upsert_org(con, "demo", "Demo")
    tenancy.upsert_site(con, "demo", {
        "id": "midland", "zone": "ERCOT West",
        "settlement_point": "LZ_WEST", "gas_hub": "Waha",
        "lat": 32.0, "lng": -102.0, "name": "Midland", "state": "TX",
    })
    tenancy.upsert_site_config(con, "demo", "midland",
                               SiteConfig(heat_rate=7.5, o_and_m=3.5, facility_mw=100))
    ts = pd.date_range("2026-01-01", periods=3, freq="h")
    load_lmp(con, pd.DataFrame({"node": "LZ_WEST", "ts": ts, "lmp": [30.0, 40.0, 25.0]}))
    load_weather(con, pd.DataFrame({"site_id": "midland", "ts": ts,
                                    "temp_f": [70, 72, 68], "wind_speed": [5, 6, 7]}))
    load_gas(con, pd.DataFrame({"hub": ["Waha"], "date": [ts[0].date()], "price": [2.0]}))
    return con


def test_feature_build_computes_cost_and_spread():
    con = _warehouse_with_one_site()
    n = features.build_feature_spread(con, "demo")
    assert n == 3
    rows = con.execute(
        "SELECT lmp, gen_cost, spread FROM feature_spread ORDER BY ts"
    ).fetchall()
    # gen_cost = 2.0 * 7.5 + 3.5 = 18.5 ; spread = lmp - 18.5
    for lmp, gen_cost, spread in rows:
        assert abs(gen_cost - 18.5) < 1e-9
        assert abs(spread - (lmp - 18.5)) < 1e-9


def test_asof_gas_carries_forward():
    """Only day-1 gas exists; day-2 hours must reuse it (ASOF carry-forward)."""
    con = _warehouse_with_one_site()
    ts2 = pd.date_range("2026-01-02", periods=2, freq="h")
    load_lmp(con, pd.DataFrame({"node": "LZ_WEST", "ts": ts2, "lmp": [50.0, 55.0]}))
    load_weather(con, pd.DataFrame({"site_id": "midland", "ts": ts2,
                                    "temp_f": [80, 81], "wind_speed": [3, 4]}))
    features.build_feature_spread(con, "demo")
    # Day-2 rows still priced off the day-1 gas quote (2.0 -> gen_cost 18.5).
    day2 = con.execute(
        "SELECT gen_cost FROM feature_spread WHERE CAST(ts AS DATE) = '2026-01-02'"
    ).fetchall()
    assert all(abs(gc - 18.5) < 1e-9 for (gc,) in day2)


def test_uses_latest_config_version():
    con = _warehouse_with_one_site()
    # A more efficient turbine takes effect later -> lower gen_cost, higher spread.
    tenancy.upsert_site_config(con, "demo", "midland",
                               SiteConfig(heat_rate=6.0, o_and_m=3.5),
                               effective_from="2026-06-01")
    features.build_feature_spread(con, "demo")
    gen_cost = con.execute("SELECT DISTINCT gen_cost FROM feature_spread").fetchall()
    # 2.0 * 6.0 + 3.5 = 15.5 (latest config wins)
    assert any(abs(gc - 15.5) < 1e-9 for (gc,) in gen_cost)

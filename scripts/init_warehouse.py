"""Initialize the warehouse and run the full pipeline under the demo org.

Pipeline:
1. create schema,
2. seed the `demo` org + 13 sites + per-site config,
3. bootstrap raw tables (raw_lmp / raw_gas / raw_weather) from the legacy parquet,
4. build feature_spread deterministically from raw + config,
5. validate feature_spread; quarantine any bad rows.

Once live ingestion exists, step 3 is replaced by real fetches appending to the raw
tables — steps 4–5 stay identical. Idempotent: safe to re-run.

Usage: .venv/bin/python -m scripts.init_warehouse
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import features, tenancy, validation, warehouse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "backend", "data", "historical_spreads.parquet")


def bootstrap_raw_from_parquet(con, org_id: str) -> None:
    """Decompose the legacy merged parquet back into normalized raw tables.

    This is a one-time bridge: the parquet was a pre-merged feature file, so we
    reverse it into the raw system-of-record tables the pipeline now builds from.
    """
    safe = PARQUET.replace("'", "''")
    con.execute(f"CREATE OR REPLACE TEMP VIEW _legacy AS SELECT * FROM read_parquet('{safe}')")

    # raw_lmp: LMP belongs to a node (settlement point); collapse sites sharing a node.
    con.execute("""
        INSERT INTO raw_lmp (node, ts, lmp)
        SELECT zone AS node, ts, avg(lmp) AS lmp FROM _legacy GROUP BY zone, ts
        ON CONFLICT (node, ts) DO UPDATE SET lmp = excluded.lmp
    """)

    # raw_weather: per-site hourly observations.
    con.execute("""
        INSERT INTO raw_weather (site_id, ts, temp_f, wind_speed)
        SELECT site_id, ts, any_value(temp_f), any_value(wind_speed)
        FROM _legacy GROUP BY site_id, ts
        ON CONFLICT (site_id, ts) DO UPDATE SET
            temp_f = excluded.temp_f, wind_speed = excluded.wind_speed
    """)

    # raw_gas: daily price per hub (recovered via each site's configured gas hub).
    con.execute(f"""
        INSERT INTO raw_gas (hub, date, price)
        SELECT s.gas_hub AS hub, CAST(l.ts AS DATE) AS date, avg(l.gas_price) AS price
        FROM _legacy l
        JOIN sites s ON s.site_id = l.site_id AND s.org_id = '{org_id}'
        GROUP BY s.gas_hub, CAST(l.ts AS DATE)
        ON CONFLICT (hub, date) DO UPDATE SET price = excluded.price
    """)


def main() -> None:
    print("=" * 60)
    print("🏗️  Dispatch IQ warehouse — full pipeline")
    print("=" * 60)
    con = warehouse.connect()
    warehouse.init_schema(con)
    print("   ✅ schema ready")

    n_sites = tenancy.seed_demo_org(con)
    print(f"   ✅ seeded org '{tenancy.DEMO_ORG}' with {n_sites} sites + config")

    if os.path.exists(PARQUET):
        bootstrap_raw_from_parquet(con, tenancy.DEMO_ORG)
        print("   ✅ bootstrapped raw_lmp / raw_gas / raw_weather from parquet")

    n_feat = features.build_feature_spread(con, tenancy.DEMO_ORG)
    print(f"   ✅ built {n_feat:,} feature_spread rows from raw + config")

    # Validate the model-ready table; quarantine offenders.
    df = con.execute(
        "SELECT * FROM feature_spread WHERE org_id = ?", [tenancy.DEMO_ORG]
    ).fetchdf()
    result = validation.validate(df, validation.feature_spread_checks())
    if not result.ok:
        validation.quarantine_rows(con, "feature_spread", result.bad, "init validation")
    print(f"   {result.summary()}")

    print("\n   Table counts:")
    for t, c in warehouse.table_counts(con).items():
        print(f"     {t:<16} {c:>8,}")
    con.close()
    print(f"\n✅ Warehouse at {os.path.relpath(warehouse.DEFAULT_DB, ROOT)}")


if __name__ == "__main__":
    main()

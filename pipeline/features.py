"""Deterministic feature build: raw tables + tenant config -> feature_spread.

This is the transform that makes the model-ready table *regenerable*. Given the raw
market data (LMP, gas, weather) and a tenant's per-site economics, it recomputes
every hourly row's generation cost and spread. Re-running it is a pure function of
its inputs — the defining property of a reproducible pipeline.

Notable: gas is joined ``ASOF`` (most recent price on or before the hour's date),
which mirrors real settlement — you trade against the latest published gas price,
and weekends/holidays carry forward the prior quote.
"""
from __future__ import annotations

import duckdb

# Latest config version per (org, site): the row with the greatest effective_from.
_LATEST_CONFIG = """
SELECT org_id, site_id, heat_rate, o_and_m, facility_mw
FROM (
    SELECT *, row_number() OVER (
        PARTITION BY org_id, site_id ORDER BY effective_from DESC
    ) AS _rn
    FROM site_config
) WHERE _rn = 1
"""


def build_feature_spread(con: duckdb.DuckDBPyConnection, org_id: str) -> int:
    """Rebuild feature_spread for one org from raw tables + its site_config.

    Generation cost = gas_price * heat_rate + O&M (from the site's config row);
    spread = LMP - generation_cost. Idempotent via upsert on the PK.
    """
    safe_org = org_id.replace("'", "''")
    con.execute(f"""
        INSERT INTO feature_spread
            (org_id, site_id, ts, lmp, gas_price, gen_cost, spread, temp_f, wind_speed)
        SELECT
            s.org_id,
            s.site_id,
            l.ts,
            l.lmp,
            g.price                                    AS gas_price,
            g.price * c.heat_rate + c.o_and_m          AS gen_cost,
            l.lmp - (g.price * c.heat_rate + c.o_and_m) AS spread,
            w.temp_f,
            w.wind_speed
        FROM sites s
        JOIN ({_LATEST_CONFIG}) c
            ON c.org_id = s.org_id AND c.site_id = s.site_id
        JOIN raw_lmp l
            ON l.node = s.settlement_point
        JOIN raw_weather w
            ON w.site_id = s.site_id AND w.ts = l.ts
        ASOF LEFT JOIN raw_gas g
            ON g.hub = s.gas_hub AND CAST(l.ts AS DATE) >= g.date
        WHERE s.org_id = '{safe_org}'
        ON CONFLICT (org_id, site_id, ts) DO UPDATE SET
            lmp = excluded.lmp, gas_price = excluded.gas_price,
            gen_cost = excluded.gen_cost, spread = excluded.spread,
            temp_f = excluded.temp_f, wind_speed = excluded.wind_speed
    """)
    return con.execute(
        "SELECT count(*) FROM feature_spread WHERE org_id = ?", [org_id]
    ).fetchone()[0]

"""DuckDB warehouse: connection + schema.

DuckDB is chosen deliberately: zero-ops (a single file, no server), native parquet,
full SQL, and it scales fine for this volume. The schema is **tenant-first** — every
fact and dimension carries ``org_id`` so isolation is structural, not bolted on later.

Tables
------
orgs           — one row per customer (tenant).
sites          — a tenant's facilities (location + market wiring).
site_config    — versioned per-site economics (heat rate, O&M, MW). Versioned so a
                 customer's gas contract change is a new row, not a destructive update.
feature_spread — the model-ready hourly fact table (LMP, gas, weather, spread),
                 scoped by (org_id, site_id, ts).
"""
from __future__ import annotations

import os

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(ROOT, "backend", "data", "dispatch.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
    org_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    plan        TEXT NOT NULL DEFAULT 'free',
    created_at  TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sites (
    org_id            TEXT NOT NULL,
    site_id           TEXT NOT NULL,
    name              TEXT,
    state             TEXT,
    lat               DOUBLE,
    lng               DOUBLE,
    iso               TEXT,          -- ERCOT | CAISO | WECC
    settlement_point  TEXT,          -- market node for LMP
    gas_hub           TEXT,          -- pricing hub for fuel
    PRIMARY KEY (org_id, site_id)
);

CREATE TABLE IF NOT EXISTS site_config (
    org_id         TEXT NOT NULL,
    site_id        TEXT NOT NULL,
    heat_rate      DOUBLE NOT NULL,  -- MMBtu/MWh
    o_and_m        DOUBLE NOT NULL,  -- $/MWh variable O&M
    facility_mw    DOUBLE NOT NULL,  -- MW served by the dispatch decision
    effective_from TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, site_id, effective_from)
);

-- ── Raw source tables (shared public market data; NOT tenant-scoped) ──
-- These are the system of record. feature_spread is derived from them + a
-- tenant's site_config, so the model-ready table is always regenerable.
CREATE TABLE IF NOT EXISTS raw_lmp (
    node  TEXT NOT NULL,          -- settlement point / pricing node
    ts    TIMESTAMP NOT NULL,
    lmp   DOUBLE,
    PRIMARY KEY (node, ts)
);

CREATE TABLE IF NOT EXISTS raw_gas (
    hub   TEXT NOT NULL,          -- gas pricing hub (Waha, Henry Hub, ...)
    date  DATE NOT NULL,
    price DOUBLE,                 -- $/MMBtu
    PRIMARY KEY (hub, date)
);

CREATE TABLE IF NOT EXISTS raw_weather (
    site_id    TEXT NOT NULL,     -- keyed per site for now (coords-based key is a later refinement)
    ts         TIMESTAMP NOT NULL,
    temp_f     DOUBLE,
    wind_speed DOUBLE,
    PRIMARY KEY (site_id, ts)
);

-- ── Derived, tenant-scoped model-ready table ──
CREATE TABLE IF NOT EXISTS feature_spread (
    org_id      TEXT NOT NULL,
    site_id     TEXT NOT NULL,
    ts          TIMESTAMP NOT NULL,
    lmp         DOUBLE,
    gas_price   DOUBLE,
    gen_cost    DOUBLE,
    spread      DOUBLE,
    temp_f      DOUBLE,
    wind_speed  DOUBLE,
    PRIMARY KEY (org_id, site_id, ts)
);

-- ── Rejected rows from validation, kept for inspection instead of dropped ──
CREATE TABLE IF NOT EXISTS quarantine (
    table_name     TEXT,
    reason         TEXT,
    payload        TEXT,          -- JSON of the offending row
    quarantined_at TIMESTAMP DEFAULT now()
);
"""


def connect(db_path: str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (and create if needed) the warehouse file."""
    path = db_path or DEFAULT_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return duckdb.connect(path, read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotently create all tables."""
    con.execute(SCHEMA)


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Row counts per table, for quick sanity checks."""
    tables = [
        "orgs", "sites", "site_config",
        "raw_lmp", "raw_gas", "raw_weather", "feature_spread", "quarantine",
    ]
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}

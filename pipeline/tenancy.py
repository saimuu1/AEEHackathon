"""Tenant & site-config model — the demo→product shift.

Today the 13 sites are hardcoded and every facility shares the global
``HEAT_RATE`` / ``O_AND_M_COST`` constants in ``backend/config.py``. A real customer
needs *their* facility with *their* economics. This module turns that hardcoded list
into seed data for a single ``demo`` org, and gives the upsert helpers a real tenant
(or the onboarding flow) will use to register sites and config.

Generation cost is, from here on, a function of a site's config row — not a constant.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb

DEMO_ORG = "demo"

# Sensible defaults for the demo seed. A typical gas turbine ~7.5 MMBtu/MWh; these
# match the values the original pipeline baked in, now stored per-site so they can vary.
DEFAULT_HEAT_RATE = 7.5
DEFAULT_O_AND_M = 3.50
DEFAULT_FACILITY_MW = 100.0


@dataclass
class SiteConfig:
    heat_rate: float = DEFAULT_HEAT_RATE
    o_and_m: float = DEFAULT_O_AND_M
    facility_mw: float = DEFAULT_FACILITY_MW

    def generation_cost(self, gas_price: float) -> float:
        """$/MWh to self-generate: gas burned × heat rate + variable O&M."""
        return gas_price * self.heat_rate + self.o_and_m


def _iso_from_zone(zone: str) -> str:
    z = (zone or "").upper()
    if z.startswith("ERCOT"):
        return "ERCOT"
    if z.startswith("CAISO"):
        return "CAISO"
    if z.startswith("WECC"):
        return "WECC"
    return "UNKNOWN"


def upsert_org(con: duckdb.DuckDBPyConnection, org_id: str, name: str, plan: str = "free") -> None:
    con.execute(
        """
        INSERT INTO orgs (org_id, name, plan) VALUES (?, ?, ?)
        ON CONFLICT (org_id) DO UPDATE SET name = excluded.name, plan = excluded.plan
        """,
        [org_id, name, plan],
    )


def upsert_site(con: duckdb.DuckDBPyConnection, org_id: str, site: dict) -> None:
    con.execute(
        """
        INSERT INTO sites (org_id, site_id, name, state, lat, lng, iso, settlement_point, gas_hub)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (org_id, site_id) DO UPDATE SET
            name = excluded.name, state = excluded.state, lat = excluded.lat,
            lng = excluded.lng, iso = excluded.iso,
            settlement_point = excluded.settlement_point, gas_hub = excluded.gas_hub
        """,
        [
            org_id, site["id"], site.get("name"), site.get("state"),
            site.get("lat"), site.get("lng"), _iso_from_zone(site.get("zone", "")),
            site.get("settlement_point"), site.get("gas_hub"),
        ],
    )


# Baseline config version. Real customer changes append rows with a later
# effective_from; the seed always writes this fixed version so re-seeding is idempotent.
BASELINE_EFFECTIVE_FROM = "1970-01-01 00:00:00"


def upsert_site_config(
    con: duckdb.DuckDBPyConnection,
    org_id: str,
    site_id: str,
    cfg: SiteConfig,
    effective_from: str = BASELINE_EFFECTIVE_FROM,
) -> None:
    # The schema versions config via effective_from; a fixed value keeps the seed
    # idempotent, while a real update passes a new timestamp to append a version.
    con.execute(
        """
        INSERT INTO site_config (org_id, site_id, heat_rate, o_and_m, facility_mw, effective_from)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (org_id, site_id, effective_from) DO UPDATE SET
            heat_rate = excluded.heat_rate, o_and_m = excluded.o_and_m,
            facility_mw = excluded.facility_mw
        """,
        [org_id, site_id, cfg.heat_rate, cfg.o_and_m, cfg.facility_mw, effective_from],
    )


def seed_demo_org(con: duckdb.DuckDBPyConnection) -> int:
    """Register the demo tenant and its 13 sites from the legacy hardcoded list."""
    from backend.data.sites_data import SITES

    upsert_org(con, DEMO_ORG, "Dispatch IQ Demo", plan="pro")
    for site in SITES:
        upsert_site(con, DEMO_ORG, site)
        upsert_site_config(con, DEMO_ORG, site["id"], SiteConfig())
    return len(SITES)

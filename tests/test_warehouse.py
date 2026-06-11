import os
import tempfile

from pipeline import tenancy, warehouse
from pipeline.tenancy import SiteConfig


def _fresh_con():
    tmp = tempfile.mkdtemp()
    con = warehouse.connect(os.path.join(tmp, "test.duckdb"))
    warehouse.init_schema(con)
    return con


def test_generation_cost_uses_site_config():
    # $2.41/MMBtu * 7.5 + 3.50 = 21.575
    cfg = SiteConfig(heat_rate=7.5, o_and_m=3.50)
    assert abs(cfg.generation_cost(2.41) - 21.575) < 1e-9
    # A more efficient turbine (lower heat rate) generates cheaper.
    eff = SiteConfig(heat_rate=6.0, o_and_m=3.50)
    assert eff.generation_cost(2.41) < cfg.generation_cost(2.41)


def test_seed_is_idempotent():
    con = _fresh_con()
    tenancy.seed_demo_org(con)
    tenancy.seed_demo_org(con)  # second seed must not duplicate
    counts = warehouse.table_counts(con)
    assert counts["sites"] == 13
    assert counts["site_config"] == 13  # one baseline version per site


def test_config_versioning_appends():
    con = _fresh_con()
    tenancy.upsert_org(con, "demo", "Demo")
    tenancy.upsert_site(con, "demo", {"id": "s1", "zone": "ERCOT West"})
    tenancy.upsert_site_config(con, "demo", "s1", SiteConfig())
    # A real config change at a new effective_from is a *new* version, not an overwrite.
    tenancy.upsert_site_config(
        con, "demo", "s1", SiteConfig(heat_rate=6.0), effective_from="2026-01-01"
    )
    n = con.execute("SELECT count(*) FROM site_config WHERE site_id='s1'").fetchone()[0]
    assert n == 2


def test_tenant_isolation():
    con = _fresh_con()
    tenancy.upsert_org(con, "a", "Org A")
    tenancy.upsert_org(con, "b", "Org B")
    tenancy.upsert_site(con, "a", {"id": "shared_id", "zone": "ERCOT West"})
    tenancy.upsert_site(con, "b", {"id": "shared_id", "zone": "CAISO SP15"})
    # Same site_id under two orgs are distinct rows; a tenant-scoped read sees only its own.
    a_iso = con.execute("SELECT iso FROM sites WHERE org_id='a'").fetchone()[0]
    b_iso = con.execute("SELECT iso FROM sites WHERE org_id='b'").fetchone()[0]
    assert a_iso == "ERCOT" and b_iso == "CAISO"
    assert con.execute("SELECT count(*) FROM sites WHERE org_id='a'").fetchone()[0] == 1

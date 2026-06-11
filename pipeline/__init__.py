"""Dispatch IQ data platform (Phase 2).

Separates *ingestion* and *storage* from *serving*. Everything is tenant-scoped by
``org_id`` from the first table, so the single-tenant demo and a future multi-tenant
SaaS share one schema and one code path.
"""

__all__ = ["warehouse", "tenancy"]

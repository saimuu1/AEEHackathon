"""Data validation — a contract the feature table must satisfy before serving.

Bad data silently poisons a model. This module defines explicit checks (a schema
*contract*) over a DataFrame, separates the good rows from the bad, and writes the
offenders to the ``quarantine`` table with a reason instead of dropping them on the
floor. A clean run is the gate the pipeline must pass.

Checks are intentionally small and composable; each returns a boolean mask of
*valid* rows plus a label, so a row failing several checks is quarantined once with
all reasons recorded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    fn: callable            # df -> boolean mask of VALID rows
    description: str = ""


@dataclass
class ValidationResult:
    n_input: int
    n_valid: int
    n_quarantined: int
    reasons: dict[str, int] = field(default_factory=dict)
    valid: pd.DataFrame = None
    bad: pd.DataFrame = None

    @property
    def ok(self) -> bool:
        return self.n_quarantined == 0

    def summary(self) -> str:
        head = f"{self.n_valid}/{self.n_input} rows valid"
        if self.ok:
            return f"✅ {head}"
        detail = ", ".join(f"{k}={v}" for k, v in self.reasons.items())
        return f"⚠️ {head} | quarantined {self.n_quarantined} ({detail})"


# ── Reusable check builders ────────────────────────────────────────────────
def not_null(col: str) -> Check:
    return Check(f"not_null[{col}]", lambda df: df[col].notna(), f"{col} must not be null")


def in_range(col: str, lo: float, hi: float) -> Check:
    return Check(
        f"range[{col}]",
        lambda df: df[col].isna() | df[col].between(lo, hi),
        f"{col} within [{lo}, {hi}]",
    )


def unique_key(cols: list[str]) -> Check:
    def _fn(df: pd.DataFrame) -> pd.Series:
        return ~df.duplicated(subset=cols, keep="first")
    return Check(f"unique[{','.join(cols)}]", _fn, "primary key must be unique")


def consistent(name: str, fn, description: str = "") -> Check:
    """Row-level cross-field invariant, e.g. spread == lmp - gen_cost."""
    return Check(name, fn, description)


# ── The feature_spread contract ────────────────────────────────────────────
def feature_spread_checks() -> list[Check]:
    return [
        unique_key(["org_id", "site_id", "ts"]),
        not_null("lmp"),
        not_null("spread"),
        # LMP can be deeply negative (oversupply) or spike to the cap; bound generously.
        in_range("lmp", -250, 10_000),
        in_range("gas_price", 0, 100),
        in_range("temp_f", -60, 140),
        in_range("wind_speed", 0, 120),
        consistent(
            "spread=lmp-gen_cost",
            lambda df: df["spread"].isna() | df["gen_cost"].isna()
            | (np.abs(df["spread"] - (df["lmp"] - df["gen_cost"])) < 0.01),
            "spread must equal lmp - gen_cost",
        ),
    ]


def validate(df: pd.DataFrame, checks: list[Check]) -> ValidationResult:
    """Apply checks; partition into valid vs. quarantined rows with reasons."""
    n = len(df)
    valid_mask = pd.Series(True, index=df.index)
    reasons: dict[str, int] = {}
    for chk in checks:
        ok = chk.fn(df).astype(bool)
        failed = int((~ok).sum())
        if failed:
            reasons[chk.name] = failed
        valid_mask &= ok
    valid = df[valid_mask]
    bad = df[~valid_mask]
    return ValidationResult(
        n_input=n, n_valid=len(valid), n_quarantined=len(bad),
        reasons=reasons, valid=valid, bad=bad,
    )


def quarantine_rows(con, table_name: str, bad: pd.DataFrame, reason: str) -> int:
    """Persist rejected rows to the quarantine table (as JSON payloads)."""
    if bad is None or bad.empty:
        return 0
    rows = [
        (table_name, reason, json.dumps({k: _jsonable(v) for k, v in r.items()}))
        for r in bad.to_dict(orient="records")
    ]
    con.executemany(
        "INSERT INTO quarantine (table_name, reason, payload) VALUES (?, ?, ?)", rows
    )
    return len(rows)


def _jsonable(v):
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

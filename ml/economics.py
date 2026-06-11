"""Economic evaluation of a dispatch policy.

A forecast is only worth anything if *acting on it makes money*. The decision is
binary at each hour: GENERATE (run the on-site turbine) or IMPORT (buy from the
grid). The spread ``S = LMP - generation_cost`` is, by construction, the per-MWh
savings of generating instead of importing:

    S > 0  -> generating is cheaper; importing leaves S on the table
    S < 0  -> generating is more expensive; you should import

So, measured against an "always import" baseline (savings = 0):

    savings(policy) = Σ over hours the policy GENERATES of  S_h * MW

This gives four reference points:

    always_import     -> 0                         (the do-nothing baseline)
    always_generate   -> Σ S_h                      (can be negative)
    perfect_foresight -> Σ max(S_h, 0)              (theoretical ceiling)
    policy            -> Σ_{forecast says GENERATE} S_h

The headline number is **capture rate** = policy / perfect_foresight: the share of
all available arbitrage value the policy actually banked. That is the metric a
customer cares about, and it is denominated in dollars, not MAE.
"""
from __future__ import annotations

import numpy as np


def dispatch_decisions(point_forecast: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """GENERATE (True) when the forecasted spread clears the threshold."""
    return np.asarray(point_forecast, dtype=float) > threshold


def economic_value(
    realized_spread: np.ndarray,
    decisions: np.ndarray,
    mw: float = 100.0,
) -> dict[str, float]:
    """Dollars captured by a GENERATE/IMPORT policy vs. the reference strategies.

    Parameters
    ----------
    realized_spread : actual S = LMP - gen_cost at each decision's target hour ($/MWh).
    decisions       : bool array, True = the policy chose to GENERATE that hour.
    mw              : facility size; each hourly decision moves ``mw`` MWh.
    """
    s = np.asarray(realized_spread, dtype=float)
    d = np.asarray(decisions, dtype=bool)
    mask = ~np.isnan(s)
    s, d = s[mask], d[mask]
    if s.size == 0:
        return {}

    policy = float(np.sum(s[d]) * mw)
    perfect = float(np.sum(np.maximum(s, 0.0)) * mw)
    always_gen = float(np.sum(s) * mw)

    # How good is the *decision*, ignoring magnitude? (Did we pick the right side?)
    optimal_choice = s > 0
    decision_accuracy = float(np.mean(d == optimal_choice))

    return {
        "policy_savings": policy,
        "perfect_foresight": perfect,
        "always_generate": always_gen,
        "always_import": 0.0,
        "capture_rate": float(policy / perfect) if perfect > 0 else float("nan"),
        "decision_accuracy": decision_accuracy,
        "n_hours": int(s.size),
        "mw": mw,
    }

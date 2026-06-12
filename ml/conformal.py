"""Conformalized Quantile Regression (CQR).

Phase 1 found the LightGBM intervals were overconfident: the p10–p90 band covered
only ~66% of outcomes versus the 80% it advertised. CQR (Romano, Patterson &
Candès, 2019) fixes this with a finite-sample *coverage guarantee* and no
distributional assumptions.

Mechanism: hold out a calibration set the model never trained on. Compute, per
calibration point, the conformity score

    E_i = max( q_lo(x_i) - y_i ,  y_i - q_hi(x_i) )

— how far outside its predicted band the truth fell (negative if inside). Take the
``(1-α)`` empirical quantile of those scores (with the standard finite-sample
correction) and widen every future band by it:

    [ q_lo(x) - Q ,  q_hi(x) + Q ]

The widened interval is guaranteed to cover at the nominal rate in expectation.
"""
from __future__ import annotations

import numpy as np


class ConformalInterval:
    """Calibrates one lower/upper quantile pair to hit nominal coverage."""

    def __init__(self, lo_q: float = 0.1, hi_q: float = 0.9) -> None:
        self.lo_q = lo_q
        self.hi_q = hi_q
        self.nominal = hi_q - lo_q
        self.adjustment_: float | None = None

    def calibrate(self, y_true, q_lo_pred, q_hi_pred) -> ConformalInterval:
        """Learn the band widening from a held-out calibration set."""
        y = np.asarray(y_true, dtype=float)
        lo = np.asarray(q_lo_pred, dtype=float)
        hi = np.asarray(q_hi_pred, dtype=float)
        scores = np.maximum(lo - y, y - hi)
        n = scores.size
        # Finite-sample correction: the ceil((n+1)(1-α))/n empirical quantile.
        level = min(1.0, np.ceil((n + 1) * self.nominal) / n)
        self.adjustment_ = float(np.quantile(scores, level, method="higher"))
        return self

    def apply(self, q_lo_pred, q_hi_pred):
        """Widen predicted bands by the calibrated adjustment."""
        if self.adjustment_ is None:
            raise RuntimeError("calibrate() must be called before apply()")
        lo = np.asarray(q_lo_pred, dtype=float) - self.adjustment_
        hi = np.asarray(q_hi_pred, dtype=float) + self.adjustment_
        return lo, hi

import numpy as np

from ml.conformal import ConformalInterval
from ml.metrics import interval_coverage


def test_conformal_fixes_undercoverage():
    """Start with deliberately too-narrow bands; after CQR, coverage hits ~80%."""
    rng = np.random.default_rng(0)
    y_cal = rng.normal(0, 1, 4000)
    y_test = rng.normal(0, 1, 4000)

    # Predicted bands far too tight (±0.2 instead of the true ~±1.28 for 80%).
    lo_cal, hi_cal = np.full_like(y_cal, -0.2), np.full_like(y_cal, 0.2)
    lo_test, hi_test = np.full_like(y_test, -0.2), np.full_like(y_test, 0.2)

    before = interval_coverage(y_test, lo_test, hi_test)
    assert before < 0.4  # badly undercovering

    conf = ConformalInterval(0.1, 0.9).calibrate(y_cal, lo_cal, hi_cal)
    lo_adj, hi_adj = conf.apply(lo_test, hi_test)
    after = interval_coverage(y_test, lo_adj, hi_adj)
    assert abs(after - 0.8) < 0.03  # calibrated to nominal


def test_apply_before_calibrate_raises():
    import pytest
    conf = ConformalInterval(0.1, 0.9)
    with pytest.raises(RuntimeError):
        conf.apply([0.0], [1.0])


def test_adjustment_widens_symmetrically():
    y = np.linspace(-3, 3, 1000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)
    conf = ConformalInterval(0.1, 0.9).calibrate(y, lo, hi)
    lo_adj, hi_adj = conf.apply(lo, hi)
    assert conf.adjustment_ > 0
    assert np.allclose(hi_adj - hi, lo - lo_adj)  # same widening each side

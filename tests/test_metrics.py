import numpy as np

from ml import metrics


def test_pinball_known_values():
    # Over-prediction (ŷ > y) at high quantile is cheap; at low quantile costly.
    assert metrics.pinball_loss([10.0], [8.0], 0.9) == 0.9 * 2
    assert metrics.pinball_loss([10.0], [8.0], 0.1) == 0.1 * 2
    # Under-prediction (ŷ < y) flips the asymmetry.
    assert abs(metrics.pinball_loss([6.0], [8.0], 0.9) - 0.2) < 1e-12


def test_pinball_minimised_at_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(size=10_000)
    true_p90 = np.quantile(y, 0.9)
    loss_at_truth = metrics.pinball_loss(y, np.full_like(y, true_p90), 0.9)
    loss_off = metrics.pinball_loss(y, np.full_like(y, true_p90 + 0.5), 0.9)
    assert loss_at_truth < loss_off


def test_calibration_and_interval_coverage():
    y = np.arange(100, dtype=float)
    # A pred sitting at the 30th value should cover ~30% from below.
    assert abs(metrics.calibration(y, np.full_like(y, 29.0), 0.3) - 0.30) < 0.02
    low = np.full_like(y, 10.0)
    high = np.full_like(y, 89.0)
    # values 10..89 inclusive -> 80 of 100 inside.
    assert abs(metrics.interval_coverage(y, low, high) - 0.80) < 1e-9


def test_nan_handling():
    y = [1.0, np.nan, 3.0]
    p = [1.0, 5.0, 2.0]
    # The NaN row is dropped pairwise; MAE over the two valid rows = (0 + 1)/2.
    assert metrics.mae(y, p) == 0.5


def test_scorecard_keys():
    y = np.linspace(0, 10, 50)
    preds = {0.1: y - 3, 0.5: y, 0.9: y + 3}
    card = metrics.quantile_scorecard(y, preds)
    assert "mean_pinball" in card
    assert "interval_coverage" in card
    assert abs(card["nominal_coverage"] - 0.8) < 1e-9
    # p50 sits exactly on truth -> zero point error.
    assert card["mae_p50"] == 0.0

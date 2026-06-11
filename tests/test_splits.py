import numpy as np
import pandas as pd
import pytest

from ml.splits import rolling_origin_folds


def _timeline(n_hours=24 * 60, sites=("a", "b")):
    """A two-site hourly timeline sharing the same global clock."""
    base = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    ts = np.concatenate([base.values for _ in sites])
    return pd.Series(pd.to_datetime(ts))


def test_no_time_leakage_with_embargo():
    ts = _timeline()
    horizon = 72
    folds = rolling_origin_folds(ts, n_folds=4, horizon=horizon)
    embargo = pd.Timedelta(hours=horizon)
    for f in folds:
        train_ts = ts[f.train]
        test_ts = ts[f.test]
        # Every training row is at least `horizon` hours before the test window,
        # so a training origin's target cannot fall inside the test period.
        assert train_ts.max() <= f.test_start - embargo
        # No test row precedes the test window start.
        assert test_ts.min() >= f.test_start


def test_train_and_test_never_overlap():
    ts = _timeline()
    folds = rolling_origin_folds(ts, n_folds=5, horizon=24)
    for f in folds:
        assert not np.any(f.train & f.test)


def test_folds_move_forward_in_time():
    ts = _timeline()
    folds = rolling_origin_folds(ts, n_folds=5, horizon=24)
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)


def test_expanding_window_grows():
    ts = _timeline()
    folds = rolling_origin_folds(ts, n_folds=4, horizon=24, expanding=True)
    sizes = [f.train.sum() for f in folds]
    assert sizes == sorted(sizes)  # each fold trains on at least as much history


def test_rejects_impossible_config():
    ts = _timeline(n_hours=48)
    with pytest.raises(ValueError):
        rolling_origin_folds(ts, n_folds=5, horizon=72, min_train_frac=0.9)


def test_demonstrates_the_old_split_was_leaky():
    """The original pipeline grouped rows by site, then sliced X[:0.75].

    With two sites concatenated, that slice puts site 'a' in train and site 'b'
    in test while their *timestamps fully overlap* — i.e. the model is tested on
    the same time period it trained on. This asserts the bug exists in the old
    approach, justifying the new splitter.
    """
    ts = _timeline()
    n = len(ts)
    old_train_ts = ts[: int(n * 0.75)]
    old_test_ts = ts[int(n * 0.75):]
    # The "test" period is not in the future at all — it overlaps training time.
    assert old_test_ts.min() < old_train_ts.max()

import numpy as np
import pandas as pd

from ml import regime


def _series(spreads, site="a"):
    ts = pd.date_range("2026-01-01", periods=len(spreads), freq="h")
    return pd.DataFrame({"site_id": site, "ts": ts, "spread": spreads,
                         "lmp": np.array(spreads) + 20.0})


def test_label_uses_future_window_exclusive_of_now():
    # Spike (>=50) only at index 3. With horizon=2:
    spreads = [0, 0, 0, 99, 0, 0]
    df = regime.build_spike_labels(_series(spreads), horizon=2, threshold=50)
    lab = dict(zip(df["ts"].dt.hour, df["spike_next"], strict=False))
    # origin t=1 sees window (t2,t3] -> includes the spike -> 1
    assert lab[1] == 1.0
    # origin t=2 sees (t3,t4] -> spike at t3 -> 1
    assert lab[2] == 1.0
    # origin t=3 sees (t4,t5] -> no spike, and its OWN spike doesn't count -> 0
    assert lab[3] == 0.0
    # origin t=0 sees (t1,t2] -> no spike -> 0
    assert lab[0] == 0.0


def test_labels_do_not_bridge_sites():
    a = _series([0, 0, 99], "a")
    b = _series([0, 0, 0], "b")
    df = regime.build_spike_labels(pd.concat([a, b], ignore_index=True),
                                   horizon=5, threshold=50)
    # site b never spikes; none of its labels should be 1 from site a's data.
    assert (df[df["site_id"] == "b"]["spike_next"] == 0).all()


def test_average_precision_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert regime.average_precision(y, perfect) == 1.0
    # Random-ish scorer ~ prevalence (0.5 here), well below 1.
    assert regime.average_precision(y, np.array([0.5, 0.6, 0.4, 0.5])) < 1.0


def test_roc_auc_separation():
    y = np.array([0, 0, 1, 1])
    assert regime.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(regime.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) - 0.0) < 1e-9


def test_brier_and_pr_at_threshold():
    y = np.array([0, 1, 1])
    assert regime.brier_score(y, np.array([0.0, 1.0, 1.0])) == 0.0
    m = regime.precision_recall_at(y, np.array([0.1, 0.9, 0.9]), 0.5)
    assert m["precision"] == 1.0 and m["recall"] == 1.0

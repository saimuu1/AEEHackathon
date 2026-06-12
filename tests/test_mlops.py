import os
import tempfile

import numpy as np
import pandas as pd

from mlops import versioning
from mlops.registry import ModelRegistry


# ── versioning ──────────────────────────────────────────────────────────────
def test_dataframe_hash_is_deterministic_and_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    h1 = versioning.dataframe_hash(df)
    h2 = versioning.dataframe_hash(df.copy())
    assert h1 == h2  # same content -> same hash
    changed = df.copy()
    changed.loc[0, "a"] = 99
    assert versioning.dataframe_hash(changed) != h1  # changed content -> new hash


def test_dataset_hash_roundtrip(tmp_path=None):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.parquet")
    pd.DataFrame({"a": np.arange(5)}).to_parquet(p)
    assert len(versioning.dataset_hash(p)) == 64  # sha256 hex


# ── registry + promotion gate ───────────────────────────────────────────────
def _reg():
    return ModelRegistry(registry_dir=tempfile.mkdtemp())


def _register(reg, capture, pinball=2.0):
    return reg.register(
        data_hash="abc123", metrics={"capture_rate": capture, "mean_pinball": pinball},
        params={}, created_at="2026-01-01T00:00:00",
    )


def test_first_model_auto_becomes_production():
    reg = _reg()
    _register(reg, capture=0.50)
    assert reg.production()["version"] == 1


def test_better_candidate_is_promoted():
    reg = _reg()
    _register(reg, capture=0.50)          # v1 -> prod
    _register(reg, capture=0.60)          # v2 candidate, better
    promoted, _ = reg.promote(2, metric="capture_rate", higher_is_better=True)
    assert promoted and reg.production()["version"] == 2


def test_worse_candidate_is_rejected():
    reg = _reg()
    _register(reg, capture=0.60)          # v1 -> prod
    _register(reg, capture=0.40)          # v2 candidate, worse
    promoted, reason = reg.promote(2, metric="capture_rate", higher_is_better=True)
    assert not promoted
    assert reg.production()["version"] == 1  # production unchanged
    assert "does not beat" in reason


def test_promotion_respects_lower_is_better():
    reg = _reg()
    _register(reg, capture=0.5, pinball=3.0)  # v1
    _register(reg, capture=0.5, pinball=2.0)  # v2 lower pinball = better
    promoted, _ = reg.promote(2, metric="mean_pinball", higher_is_better=False)
    assert promoted and reg.production()["version"] == 2


def test_registry_persists_to_disk():
    d = tempfile.mkdtemp()
    reg = ModelRegistry(registry_dir=d)
    _register(reg, capture=0.5)
    # A fresh registry over the same dir sees the registered version.
    assert ModelRegistry(registry_dir=d).production()["version"] == 1

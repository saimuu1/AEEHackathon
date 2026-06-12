"""Dataset versioning by content hash.

A model is only reproducible if you know *exactly* which data it was trained on.
Full DVC/lakeFS is the production tool; here we use a lightweight, dependency-free
equivalent: a stable SHA-256 over the dataset's content. The hash is logged with the
model (and in MLflow), so any model can be traced back to the precise data snapshot
— and a silent data change is detectable because the hash moves.
"""
from __future__ import annotations

import hashlib
import os

import pandas as pd


def dataset_hash(path: str) -> str:
    """Stable content hash of a parquet/CSV dataset (order-independent of file mtime)."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return dataframe_hash(df)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame's content + schema."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(",".join(map(str, df.dtypes)).encode())
    # pandas hashes rows deterministically; sum-free concatenation keeps it stable.
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h.update(row_hashes.tobytes())
    return h.hexdigest()


def short(hash_str: str, n: int = 12) -> str:
    return hash_str[:n]


def file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / 1e6, 2) if os.path.exists(path) else 0.0

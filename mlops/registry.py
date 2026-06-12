"""A minimal, file-backed model registry with a promotion gate.

Models are versioned (v1, v2, …), each stored with its metrics, the dataset hash it
trained on, and its params. Exactly one version is marked ``production``. Promotion is
*gated*: a candidate only becomes production if it beats the incumbent on the agreed
metric — the registry refuses a regression. This is the governance MLflow's registry
provides in the large; implemented here in the small so it's transparent and testable.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGISTRY_DIR = os.path.join(ROOT, "models", "registry")


@dataclass
class ModelVersion:
    version: int
    created_at: str
    data_hash: str
    metrics: dict
    params: dict = field(default_factory=dict)
    artifact_dir: str = ""
    notes: str = ""


class ModelRegistry:
    """JSON index + per-version artifact directories under ``registry_dir``."""

    def __init__(self, registry_dir: str = DEFAULT_REGISTRY_DIR) -> None:
        self.dir = registry_dir
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, "registry.json")
        self._index = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                return json.load(f)
        return {"production": None, "versions": []}

    def _save(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def next_version(self) -> int:
        return max([v["version"] for v in self._index["versions"]], default=0) + 1

    def register(
        self, data_hash: str, metrics: dict, params: dict,
        created_at: str, source_artifacts: list[str] | None = None, notes: str = "",
    ) -> ModelVersion:
        """Add a new version; copy its artifacts into the registry dir."""
        version = self.next_version()
        rel_dir = f"v{version}"  # stored relative to the registry dir (portable)
        artifact_dir = os.path.join(self.dir, rel_dir)
        os.makedirs(artifact_dir, exist_ok=True)
        for src in source_artifacts or []:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(artifact_dir, os.path.basename(src)))
        mv = ModelVersion(version, created_at, data_hash, metrics, params,
                          rel_dir, notes)
        self._index["versions"].append(asdict(mv))
        if self._index["production"] is None:  # first model becomes production
            self._index["production"] = version
        self._save()
        return mv

    def production(self) -> dict | None:
        v = self._index["production"]
        return next((x for x in self._index["versions"] if x["version"] == v), None)

    def get(self, version: int) -> dict | None:
        return next((x for x in self._index["versions"] if x["version"] == version), None)

    def versions(self) -> list[dict]:
        return list(self._index["versions"])

    def promote(
        self, candidate_version: int,
        metric: str = "capture_rate", higher_is_better: bool = True,
    ) -> tuple[bool, str]:
        """Promote candidate to production iff it beats the incumbent on ``metric``.

        Returns (promoted, reason). The first model is auto-production at register time;
        thereafter promotion is gated.
        """
        cand = self.get(candidate_version)
        if cand is None:
            return False, f"version {candidate_version} not found"
        incumbent = self.production()
        if incumbent is None or incumbent["version"] == candidate_version:
            self._index["production"] = candidate_version
            self._save()
            return True, "no incumbent; candidate set as production"

        c, i = cand["metrics"].get(metric), incumbent["metrics"].get(metric)
        if c is None or i is None:
            return False, f"metric '{metric}' missing on a model"
        better = c > i if higher_is_better else c < i
        if better:
            self._index["production"] = candidate_version
            self._save()
            return True, (f"promoted v{candidate_version}: {metric} {c:.4f} beats "
                          f"v{incumbent['version']} {i:.4f}")
        return False, (f"rejected v{candidate_version}: {metric} {c:.4f} does not beat "
                       f"v{incumbent['version']} {i:.4f}")

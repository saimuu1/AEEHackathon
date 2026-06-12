# Phase 4 — MLOps: reproducible, tracked, versioned, governed

Turns model training from a notebook one-off into a governed lifecycle. Everything
below runs from one command: `make train`.

## The pieces

| Concern | Implementation | Why it matters |
|---|---|---|
| **Config-driven training** | `scripts/train.py` + `configs/train_spread.yaml` | One seeded entrypoint; no hidden notebook state. Reproducible by construction. |
| **Experiment tracking** | MLflow (SQLite backend) — params, metrics, artifacts per run | A searchable logbook of every run; compare candidates over time. |
| **Dataset versioning** | `mlops/versioning.py` — SHA-256 content hash, logged with each model | Ties a model to the *exact* data it saw; a silent data change moves the hash. |
| **Model registry** | `mlops/registry.py` — JSON index + per-version artifacts | Versioned models (v1, v2, …); exactly one is `production`. |
| **Model card** | `mlops/model_card.py` — auto-generated per train | Documents data window, metrics vs baseline, intended use, limits. Regenerated so it never drifts. |
| **Promotion gate** | `registry.promote()` + `scripts/promote_model.py` | A candidate reaches production **only** if it beats the incumbent on the chosen metric. |

## Promotion gate — demonstrated

```
$ make train                 # v1: rich features (tuned)  -> pinball 2.555, auto-production
$ python -m scripts.train --config configs/train_spread_legacy.yaml --register
                             # v2: legacy features          -> pinball 2.567
$ python -m scripts.promote_model --candidate 2 --metric mean_pinball
🚫 rejected v2: mean_pinball 2.5674 does not beat v1 2.5553
```

The gate refused to ship the weaker model. Note v2 had *higher* capture rate (60.2%)
but worse pinball — so the **choice of gate metric is itself a decision**, made
explicit rather than buried.

## What this demonstrates
The full MLOps lifecycle — tracking, data+model versioning, reproducible config-driven
training, a model card, and an automated promotion gate that enforces "no regression
ships." This is the layer that separates "I can train a model" from "I can operate ML
in production," and it's rare in junior portfolios.

## Reproduce
```
make train      # train + track + register the production model
python -m scripts.promote_model --list   # inspect the registry
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs (optional)
```

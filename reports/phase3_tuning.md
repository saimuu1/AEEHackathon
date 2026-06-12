# Phase 3 — Hyperparameter Tuning (Optuna)

_TPE sampler · p50 pinball objective · leakage-free time split (56,984 train / 28,017 val) · embargo ≥ 72h._

| | val pinball (p50) ↓ |
|---|---:|
| Default params | 3.5450 |
| Optuna-tuned | 3.3310 |

## Verdict
Tuning improved val pinball by **6.0%** — the engineered features pay off once the model is regularized/subsampled for them.

## Best params
```yaml
learning_rate: 0.019784722270991845
num_leaves: 177
max_depth: 11
min_data_in_leaf: 43
feature_fraction: 0.6974894281382357
bagging_fraction: 0.8052449397258459
lambda_l1: 0.004533156484733439
lambda_l2: 2.2176453859645493
```

Saved to `configs/spread_lgbm.yaml` for config-driven, reproducible training in Phase 4 (MLOps).

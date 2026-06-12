# Model Card — Spread Forecaster v2

- **Created:** 2026-06-11T21:35:02
- **Dataset hash:** `40df73c1b4e8fae8…` (22,534 rows, 2026-01-19 → 2026-04-19)
- **Model:** LightGBM quantile boosters (p10/p50/p90) over causal features.
- **Task:** forecast the gas-vs-grid spread $/MWh at horizons 1–72h; output a probabilistic band used to make the GENERATE/IMPORT dispatch call.

## Evaluation (leakage-free rolling-origin backtest)

| Metric | Value |
|---|---:|
| mean pinball ↓ | 2.567 |
| interval coverage (→80%) | 67% |
| capture rate (dollars) ↑ | 60.2% |

### vs. persistence baseline
| Metric | Model | Persistence |
|---|---:|---:|
| mean pinball | 2.567 | 3.635 |
| capture rate | 60.2% | 36.1% |

## Intended use
Decision-support for a behind-the-meter operator choosing whether to self-generate or import power. **Not** an automated control signal — a human reviews the call.

## Limitations
- Trained on ~90 days of ERCOT/CAISO history; not yet validated across a full seasonal cycle (summer scarcity, winter storms).
- Long-horizon (48–72h) forecasts are materially less sharp than short-horizon.
- Spike/scarcity events are rare; tail accuracy is the weakest area (see `reports/phase3_regime.md`).

## Hyperparameters
```yaml
learning_rate: 0.019784722270991845
num_leaves: 177
max_depth: 11
min_data_in_leaf: 43
feature_fraction: 0.6974894281382357
bagging_fraction: 0.8052449397258459
lambda_l1: 0.004533156484733439
lambda_l2: 2.2176453859645493
num_boost_round: 300
seed: 42
```

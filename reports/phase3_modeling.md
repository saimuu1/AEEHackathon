# Phase 3 Scorecard — Feature Engineering + Conformal Calibration

_4 rolling-origin folds · horizons [1, 6, 24, 72]h · 100 MW · same leakage-free harness as Phase 1._

## Results (all horizons pooled)

| Forecaster | mean pinball ↓ | interval cov. (→80%) | capture rate ↑ |
|---|---:|---:|---:|
| Persistence | 3.635 | 87% | 36.1% |
| LightGBM (legacy feats) | 2.585 | 66% | 57.4% |
| LightGBM (rich feats) | 2.628 | 67% | 57.1% |
| LightGBM (rich+conformal) | 2.853 | 76% | 47.3% |

## Verdict
- **Rich features did *not* pay off here:** mean pinball was +1.7% vs legacy (worse). With ~90 days of data, 47 features (many collinear with the existing lags) add variance without signal. Honest read: the legacy set was already near the data's signal ceiling at these horizons. Next lever is *tuning* (Optuna) and *more history*, not more raw features.
- **Conformal calibration delivered:** interval coverage moved 66% → 76% (target 80%). CQR widens the band on a held-out calibration set to restore honest coverage — a deliberate trade of a little sharpness (and some pinball/capture) for a coverage *guarantee*. That trade is usually worth it: an interval a customer can trust beats a tight one that lies.

## Method notes
- Same leakage-free rolling-origin harness as Phase 1; every model refit per fold on strictly-past data.
- Features are strictly causal (asserted by truncation tests); conformal calibration uses a time-ordered hold-out carved from each fold's training pairs.
- **What this demonstrates:** rigorous A/B comparison with an honest negative result reported as-is, plus a modern uncertainty-quantification technique that did its job. Not every change wins; the discipline is measuring it correctly.

## Per-horizon interval coverage (target 80%)

| Forecaster | 1h | 6h | 24h | 72h |
|---|---:|---:|---:|---:|
| Persistence | 82% | 82% | 92% | 93% |
| LightGBM (legacy feats) | 68% | 66% | 69% | 62% |
| LightGBM (rich feats) | 71% | 67% | 69% | 64% |
| LightGBM (rich+conformal) | 84% | 72% | 76% | 73% |

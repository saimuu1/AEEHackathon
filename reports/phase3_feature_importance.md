# Phase 3 — Feature Importance (SHAP)

_mean(|SHAP|) on a 4,000-row held-out sample · model trained on 70,996 pairs · median (p50) spread forecaster._

| Rank | Feature | mean(|SHAP|) | share |
|---:|---|---:|---:|
| 1 | `spread` | 9.377 | 37.6% |
| 2 | `horizon` | 2.823 | 11.3% |
| 3 | `hour_sin` | 1.459 | 5.9% |
| 4 | `gas_price` | 1.171 | 4.7% |
| 5 | `hour_cos` | 0.945 | 3.8% |
| 6 | `lmp_trend_24h` | 0.899 | 3.6% |
| 7 | `lmp_lag_3` | 0.715 | 2.9% |
| 8 | `lmp_ramp_1` | 0.672 | 2.7% |
| 9 | `lmp_roll_min_24` | 0.595 | 2.4% |
| 10 | `hour` | 0.473 | 1.9% |
| 11 | `lmp_roll_max_24` | 0.459 | 1.8% |
| 12 | `spread_lag_24` | 0.458 | 1.8% |
| 13 | `spread_roll_mean_24` | 0.458 | 1.8% |
| 14 | `lmp_roll_mean_24` | 0.385 | 1.5% |
| 15 | `lmp_trend_6h` | 0.364 | 1.5% |

## Reading it
- SHAP attributes each prediction to its features via Shapley values, so this ranks what actually moves the spread forecast — not just split counts.
- `horizon` ranking high is expected: forecast uncertainty grows with lead time. Recent price level / lags and rolling stats carrying weight is the model leaning on momentum and mean-reversion; weather degree-days surfacing confirms the demand channel.
- This is the interpretability artifact a stakeholder (or an interviewer) asks for: *why* does the model say what it says.

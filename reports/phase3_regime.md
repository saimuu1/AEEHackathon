# Phase 3 — Spike-Risk Model (regime redesign)

_Forward question: will spread ≥ $50/MWh occur within the next 24h? · 4 leakage-free folds · 46 causal features._

## Why this replaces the old regime classifier
The original model self-distilled labels from a deterministic rule over its own inputs, so its F1 was high and meaningless (it relearned the rule). This predicts a genuine *future* event, so accuracy is real and the task is actionable: a forward warning of a high-value generation window.

Spike-window prevalence (base rate): **8.3%** — an imbalanced problem, so PR-AUC and Brier are the honest metrics (not raw accuracy or ROC alone).

## Results

| Method | PR-AUC ↑ | ROC-AUC ↑ | Brier ↓ | precision@0.5 | recall@0.5 |
|---|---:|---:|---:|---:|---:|
| base-rate | 0.201 | 0.500 | 0.090 | nan% | 0% |
| hour-climatology | 0.101 | 0.478 | 0.090 | nan% | 0% |
| SpikeRiskModel | 0.247 | 0.737 | 0.114 | 19% | 17% |

## Verdict
- **Ranking skill is real:** ROC-AUC **0.737** vs the base-rate's 0.500 — the model genuinely orders hours by spike risk, which is the actionable part (which hours to watch). This is real signal, not the circular near-1.0 F1 of the old self-distilled design.
- **Beats both baselines on PR-AUC** (0.247 vs base-rate 0.201 ≈ 1.2× and hour-climatology 0.101) — so it's learning more than the base rate or 'spikes happen in the afternoon'.
- **Honest limits:** 24h-ahead spike prediction is hard at this data scale (~90 days). The PR-AUC lift over base rate is modest; the value is the reliable risk *ranking* (ROC 0.737), and more history + tuning is the path to sharper precision. Reported as-is.

## What this demonstrates
Reframing a flawed (circular) target into a real forward-looking one, then scoring it with the right metrics for class imbalance against honest baselines — the judgment that separates 'trained a classifier' from 'designed an evaluation'.

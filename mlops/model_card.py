"""Auto-generated model card.

Every trained model ships with a card documenting what it is, the data it saw, how
it scored against baselines, and its known limits — regenerated on each train so it
never drifts from reality. This is both good ML governance and a strong portfolio
artifact (it shows you think about a model as a product, not a one-off).
"""
from __future__ import annotations


def render(
    version: int,
    created_at: str,
    data_hash: str,
    data_window: str,
    n_rows: int,
    params: dict,
    metrics: dict,
    baseline_metrics: dict | None = None,
) -> str:
    lines = [
        f"# Model Card — Spread Forecaster v{version}",
        "",
        f"- **Created:** {created_at}",
        f"- **Dataset hash:** `{data_hash[:16]}…` ({n_rows:,} rows, {data_window})",
        "- **Model:** LightGBM quantile boosters (p10/p50/p90) over causal features.",
        "- **Task:** forecast the gas-vs-grid spread $/MWh at horizons 1–72h; output a "
        "probabilistic band used to make the GENERATE/IMPORT dispatch call.",
        "",
        "## Evaluation (leakage-free rolling-origin backtest)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean pinball ↓ | {metrics.get('mean_pinball', float('nan')):.3f} |",
        f"| interval coverage (→80%) | {metrics.get('interval_coverage', float('nan')):.0%} |",
        f"| capture rate (dollars) ↑ | {metrics.get('capture_rate', float('nan')):.1%} |",
    ]
    if baseline_metrics:
        lines += [
            "",
            "### vs. persistence baseline",
            "| Metric | Model | Persistence |",
            "|---|---:|---:|",
            f"| mean pinball | {metrics.get('mean_pinball', float('nan')):.3f} | "
            f"{baseline_metrics.get('mean_pinball', float('nan')):.3f} |",
            f"| capture rate | {metrics.get('capture_rate', float('nan')):.1%} | "
            f"{baseline_metrics.get('capture_rate', float('nan')):.1%} |",
        ]
    lines += [
        "",
        "## Intended use",
        "Decision-support for a behind-the-meter operator choosing whether to self-generate "
        "or import power. **Not** an automated control signal — a human reviews the call.",
        "",
        "## Limitations",
        "- Trained on ~90 days of ERCOT/CAISO history; not yet validated across a full "
        "seasonal cycle (summer scarcity, winter storms).",
        "- Long-horizon (48–72h) forecasts are materially less sharp than short-horizon.",
        "- Spike/scarcity events are rare; tail accuracy is the weakest area (see "
        "`reports/phase3_regime.md`).",
        "",
        "## Hyperparameters",
        "```yaml",
        *[f"{k}: {v}" for k, v in params.items()],
        "```",
        "",
    ]
    return "\n".join(lines)

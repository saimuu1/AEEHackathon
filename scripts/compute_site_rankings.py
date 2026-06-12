"""
Compute site rankings from historical spread data.
Aggregates parquet → site_rankings.json with real metrics.

Usage: /opt/anaconda3/bin/python -m scripts.compute_site_rankings
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARQUET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "backend", "data", "historical_spreads.parquet")
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "backend", "data", "site_rankings.json")


def classify_regime_rules(row):
    """Same regime rules used in training."""
    lmp = row["lmp"]
    spread = row["spread"]
    temp = row.get("temp_f", 75)
    wind = row.get("wind_speed", 10)

    if lmp >= 5000: return "uri_emergency"
    if lmp >= 500 and temp <= 20: return "winter_storm"
    if lmp >= 200 or (lmp >= 100 and temp >= 95): return "scarcity"
    if temp >= 90 and lmp >= 50: return "heat_dome"
    if lmp <= 5 and wind >= 20: return "wind_glut"
    if lmp <= 15 and spread <= 0: return "oversupply"
    return "normal"


def compute():
    print("=" * 60)
    print("📊 Computing Site Rankings")
    print("=" * 60)

    df = pd.read_parquet(PARQUET)
    df["regime"] = df.apply(classify_regime_rules, axis=1)

    # ── Phase 1: Collect raw metrics for every site ──
    raw_metrics = {}
    for site_id in sorted(df["site_id"].unique()):
        s = df[df["site_id"] == site_id]

        avg_spread = round(s["spread"].mean(), 1)
        positive_pct = round((s["spread"] > 0).mean() * 100)
        p90_spread = round(s["spread"].quantile(0.90), 1)  # 90th percentile upside
        std = s["spread"].std()

        # Volatility bucket
        if std < 20: vol = "Low"
        elif std < 50: vol = "Medium"
        else: vol = "High"

        # Monthly spreads
        monthly = s.groupby("month")["spread"].mean()
        monthly_spreads = [round(monthly.get(m, 0), 1) for m in range(1, 13)]

        # Regime breakdown
        regime_counts = s["regime"].value_counts(normalize=True) * 100
        regime_breakdown = {r: round(regime_counts.get(r, 0)) for r in
                          ["normal", "heat_dome", "wind_glut", "scarcity", "oversupply", "winter_storm"]}

        # Best/worst hours
        best_hour = s.loc[s["spread"].idxmax()]
        worst_hour = s.loc[s["spread"].idxmin()]

        raw_metrics[site_id] = {
            "avg_spread": avg_spread,
            "positive_hours_pct": positive_pct,
            "p90_spread": p90_spread,
            "volatility": vol,
            "monthly_spreads": monthly_spreads,
            "regime_breakdown": regime_breakdown,
            "best_hour": f"LMP=${best_hour['lmp']:.0f}, Spread=+${best_hour['spread']:.0f}/MWh at {best_hour['ts']}",
            "worst_hour": f"LMP=${worst_hour['lmp']:.0f}, Spread=${worst_hour['spread']:.0f}/MWh at {worst_hour['ts']}",
            "data_rows": len(s),
        }

    # ── Phase 2: Normalize across all sites and compute composite score ──
    all_avg = [m["avg_spread"] for m in raw_metrics.values()]
    all_pct = [m["positive_hours_pct"] for m in raw_metrics.values()]
    all_p90 = [m["p90_spread"] for m in raw_metrics.values()]

    def norm(val, vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return 50.0
        return (val - mn) / (mx - mn) * 100.0

    rankings = {}
    for site_id, m in raw_metrics.items():
        n_pct = norm(m["positive_hours_pct"], all_pct)    # 0-100 normalized
        n_p90 = norm(m["p90_spread"], all_p90)             # 0-100 normalized
        n_avg = norm(m["avg_spread"], all_avg)             # 0-100 normalized

        # Weighted composite: 35% frequency + 35% peak upside + 30% avg economics
        raw_score = n_pct * 0.35 + n_p90 * 0.35 + n_avg * 0.30
        score = int(min(95, max(10, round(raw_score))))

        rankings[site_id] = {
            "avg_spread": m["avg_spread"],
            "positive_hours_pct": m["positive_hours_pct"],
            "volatility": m["volatility"],
            "composite_score": score,
            "monthly_spreads": m["monthly_spreads"],
            "regime_breakdown": m["regime_breakdown"],
            "best_hour": m["best_hour"],
            "worst_hour": m["worst_hour"],
            "data_rows": m["data_rows"],
        }

        print(f"   {site_id:15s}  avg=${m['avg_spread']:+.1f}  p90=${m['p90_spread']:+.1f}  +hours={m['positive_hours_pct']}%  score={score}")

    # Rank by composite score
    sorted_sites = sorted(rankings.items(), key=lambda x: x[1]["composite_score"], reverse=True)
    for rank, (site_id, data) in enumerate(sorted_sites, 1):
        rankings[site_id]["rank"] = rank

    with open(OUTPUT, "w") as f:
        json.dump(rankings, f, indent=2, default=str)

    print(f"\n   ✅ Saved to {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    compute()

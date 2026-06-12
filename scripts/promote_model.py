"""Promotion gate CLI: only let a model reach production if it earns it.

Lists registered versions and, given a candidate, promotes it to production *only*
if it beats the incumbent on the chosen metric (default: capture rate, the dollars
metric). This is the governance step that stops a worse model from shipping.

Usage:
    .venv/bin/python -m scripts.promote_model --list
    .venv/bin/python -m scripts.promote_model --candidate 2 [--metric capture_rate]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops.registry import ModelRegistry  # noqa: E402

_LOWER_IS_BETTER = {"mean_pinball", "baseline_pinball"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--candidate", type=int)
    ap.add_argument("--metric", default="capture_rate")
    args = ap.parse_args()

    reg = ModelRegistry()
    prod = reg.production()
    prod_v = prod["version"] if prod else None

    if args.list or args.candidate is None:
        print(f"Registered versions (production = v{prod_v}):")
        for v in reg.versions():
            m = v["metrics"]
            star = " ⭐ PROD" if v["version"] == prod_v else ""
            print(f"  v{v['version']}  pinball={m.get('mean_pinball', float('nan')):.3f}  "
                  f"capture={m.get('capture_rate', float('nan')):.1%}  "
                  f"data={v['data_hash'][:12]}{star}")
        if args.candidate is None:
            return

    higher_better = args.metric not in _LOWER_IS_BETTER
    promoted, reason = reg.promote(args.candidate, metric=args.metric,
                                   higher_is_better=higher_better)
    print(("✅ " if promoted else "🚫 ") + reason)


if __name__ == "__main__":
    main()

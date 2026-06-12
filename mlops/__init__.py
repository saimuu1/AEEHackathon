"""MLOps layer (Phase 4): reproducible, tracked, versioned model lifecycle.

Turns "I trained a model once in a notebook" into a governed process: every model
is trained from a config, tied to the exact dataset hash it saw, scored on the
leakage-free backtest, tracked in MLflow, registered with a version, and only
promoted to production if it beats the incumbent.
"""

__all__ = ["versioning", "registry", "model_card"]

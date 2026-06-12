# Dispatch IQ — developer & pipeline commands.
# Uses the project venv at .venv. Run `make help` to list targets.

PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install warehouse data evaluate test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies into the venv
	$(PIP) install -q -r backend/requirements.txt -r requirements-dev.txt

warehouse:  ## Build the DuckDB warehouse: schema, seed demo org, raw -> features, validate
	$(PY) -m scripts.init_warehouse

data: warehouse  ## Alias: end-to-end data pipeline (currently == warehouse)

evaluate:  ## Run the leakage-free backtest and write reports/phase1_scorecard.md
	$(PY) -m scripts.evaluate

train:  ## Config-driven training: track to MLflow + register the production model
	$(PY) -m scripts.train --config configs/train_spread.yaml --register

registry:  ## List registered model versions and which is production
	$(PY) -m scripts.promote_model --list

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint with ruff (if installed)
	$(PY) -m ruff check ml pipeline scripts || true

clean:  ## Remove the warehouse (regenerable via `make warehouse`)
	rm -f backend/data/dispatch.duckdb backend/data/dispatch.duckdb.wal

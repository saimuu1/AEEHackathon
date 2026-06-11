# Dispatch IQ — Engineering Roadmap

> Turning a hackathon demo into a portfolio-grade, **SaaS-shaped** product.
> Target competency signal: **time-series ML done rigorously + production data
> engineering + a real multi-tenant product shell**. The forecasting is the moat;
> the SaaS layer (auth, multi-tenancy, billing, per-customer config) is the
> product wrapper that proves you can ship, not just model.
>
> Ambition: **portfolio piece built like a real SaaS** — revenue isn't the goal,
> but it's built realistically enough to convert if a design partner ever bites.
> The SaaS work is folded into the existing phases, not bolted on as a separate
> track; Phases 1–5 remain the ML/DE core.

---

## 0. Where we are starting from (honest baseline)

**What already works and is genuinely good**
- Real domain with a real economic equation (`spread = LMP − (gas × heat_rate + O&M)`).
- Live data collection from ERCOT, CAISO (OASIS), EIA, Open-Meteo (`scripts/build_historical_dataset.py`).
- Trained LightGBM quantile forecasters (p10/p50/p90) + a regime classifier.
- A working FastAPI + React product on top of the models.
- Current dataset: **22,534 rows, 13 sites, ~90 days hourly** (`backend/data/historical_spreads.parquet`).

**What makes it read as "hackathon", and what each gap maps to**

| Gap | Evidence | Fixed in |
|---|---|---|
| 🔴 Data leakage in "time-based" split | `train_spread_forecaster.py:80` slices `X[:0.75]` over site-grouped, overlapping pairs | Phase 1 |
| 🔴 Wrong headline metric | Reports MAE for a *quantile* model; ad-hoc coverage | Phase 1 |
| 🔴 No baseline to beat | No persistence/seasonal comparison | Phase 1 |
| 🔴 No economic evaluation | Never checks if the dispatch signal makes money | Phase 1 |
| 🔴 Leaked secret | Hardcoded EIA key `build_historical_dataset.py:28` | Phase 2 |
| 🟠 No reproducible/versioned data | Refetches live; single parquet overwritten | Phase 2 |
| 🟠 No data validation | No schema/quality gates | Phase 2 |
| 🟠 Regime model self-distills from rules | `train_regime_classifier.py:57` learns a deterministic function of its own inputs → meaningless F1 | Phase 3 |
| 🟠 Thin features | Only lag6/lag24; no rolling stats, calendar, interactions | Phase 3 |
| 🟠 No experiment tracking / registry | Models saved as `.txt`, overwritten, unversioned | Phase 4 |
| 🟠 No tests / CI / Docker / deploy | Two scratch scripts at repo root | Phase 5 |
| 🟣 Hardcoded to 13 demo sites | `backend/data/sites_data.py` — not customer-configurable | Phase 2 (config) |
| 🟣 Single-tenant, no auth/billing | No orgs, users, isolation, or payments | Phase 5 (SaaS shell) |

**Guiding principles for the whole project**
1. *No model ships unless it beats the baselines on a leakage-free backtest.*
2. *The headline metric is economic value (dollars), not MAE.*
3. *Everything reproducible: seeded, config-driven, versioned data.*
4. *Each phase ends with something demoable and a written result.*
5. *Tenant-first: every data row, feature, and model lookup is scoped by an
   `org_id`/`site_id` from the start, even while there's only one tenant — so the
   single-tenant demo and the multi-tenant SaaS share one code path.*

## The SaaS layer (where each piece lands)

The product wrapper is **folded into the ML/DE phases**, not a separate track:

| SaaS concern | Lands in | Why there |
|---|---|---|
| Per-tenant **data model** (`org_id`, `site_id` everywhere) | Phase 2 | It's a schema decision in the warehouse |
| **Per-site / per-customer config** (sites, heat rate, gas hub, tariff) | Phase 2 | Turns the hardcoded demo into a configurable product |
| Tenant-scoped **ingestion & features** | Phase 2–3 | Pipelines read a tenant's site list |
| Tenant-scoped **model serving** | Phase 5 | API resolves the caller's org before predicting |
| **Auth** (login, orgs, RBAC) | Phase 5 | Part of the serving/product shell |
| **Billing** (Stripe, plans, metering) | Phase 5 | Last — only meaningful once there's a product |
| **Onboarding** flow (add a site, see a forecast) | Phase 5 | The "aha" path for a new tenant |

---

## Phase 1 — Trustworthy evaluation (the foundation)

**Goal:** be able to produce one honest scorecard for any forecaster. Nothing else
in the roadmap means anything until measurement is trustworthy.

**Deliverables**
- `ml/metrics.py` — pinball loss, calibration, interval coverage/width, MAE/RMSE, `quantile_scorecard`. ✅ *(done)*
- `ml/splits.py` — `rolling_origin_folds` with an embargo ≥ horizon (leakage-free). ✅ *(done)*
- `ml/dataset.py` — `build_forecast_targets` via exact `ts + horizon` join. ✅ *(done)*
- `ml/baselines.py` — `PersistenceForecaster`, `ClimatologyForecaster`. ✅ *(done)*
- `tests/` — unit tests for metrics, splits, dataset (assert no time leakage). *(in progress)*
- `pyproject.toml` — pytest config + dev deps.
- `ml/economics.py` — dispatch P&L: realized $ from following a forecast's signal vs.
  `always-import`, `always-generate`, and `perfect-foresight` oracles.
- `ml/backtest.py` — runs any forecaster through the folds, emits metric scorecard +
  economic scorecard as a table and JSON.
- `scripts/evaluate.py` — CLI: backtest the baselines **and** the existing trained
  LightGBM models the honest way; write `reports/phase1_scorecard.md`.

**Acceptance criteria**
- `pytest` green; a leakage test fails on the *old* split and passes on the new one.
- A scorecard table comparing Persistence / Climatology / current-LightGBM on
  pinball, calibration error, interval coverage, and economic value, per horizon bucket.
- A written paragraph: "honest numbers vs. the previously reported numbers — here's
  how much leakage inflated them."

**SWE/ML signal:** you understand evaluation, leakage, proper scoring rules, and that
the business metric ≠ the loss function. This is the #1 thing that separates real ML
engineers from notebook tutorial followers.

---

## Phase 2 — Reproducible data pipeline

**Goal:** separate *ingestion* from *serving*; make the dataset reproducible, validated,
and continuously growing instead of refetched live in the request path.

**Deliverables**
- **Secret hygiene:** remove the hardcoded EIA key; `.env.example`; document rotation.
- **Storage:** move from a single overwritten parquet to a local warehouse
  (**DuckDB** recommended — zero-ops, SQL, parquet-native). Tables: `lmp`, `gas`,
  `weather`, `spread_features` — **every fact table carries `org_id` + `site_id`**.
- **Tenant & config model (the demo→product shift):** an `orgs` / `sites` /
  `site_config` schema where a customer defines their facility — coordinates,
  settlement point, gas hub, **heat rate, O&M, utility tariff**. The hardcoded
  `backend/data/sites_data.py` becomes seed data for one demo org. Generation cost
  is computed from the tenant's config, not constants.
- **Ingestion jobs:** idempotent, incremental loaders per source (`ingest/ercot.py`,
  `ingest/caiso.py`, `ingest/eia.py`, `ingest/weather.py`) that upsert by
  `(source, zone, timestamp)` and can backfill a date range.
- **Data validation:** `pandera` (or Great Expectations) schemas — types, ranges
  (LMP can be negative but not −$1e6), null budgets, monotonic timestamps, no dup keys.
  A failing batch is quarantined, not merged.
- **Feature build as a pure transform:** `features/build.py` turns raw tables into the
  model-ready frame deterministically (so the parquet is regenerable from raw).
- **Orchestration:** a simple `Makefile`/`prefect` flow: `ingest → validate → features`,
  schedulable (cron / GitHub Action) so history actually accumulates.

**Acceptance criteria**
- `make data` rebuilds the feature table from raw with one command, deterministically.
- Validation suite runs and blocks bad data; a deliberately corrupted row is caught.
- No secrets in source; `git log -p` review confirms none added (old one rotated).
- Adding a *new* site for a tenant via config (no code change) makes it flow through
  ingestion → features end-to-end; generation cost reflects that site's heat rate/tariff.

**SWE/DE signal:** ETL design, idempotency, data contracts, warehouse modeling,
orchestration — the core of a data-engineering interview.

---

## Phase 3 — Modeling depth

**Goal:** earn the accuracy. Richer features, honest model comparison, calibrated
uncertainty, and a regime model that isn't circular.

**Deliverables**
- **Feature engineering** (`features/`): rolling mean/std/min/max over 3/6/24h,
  hour/day/holiday calendar, weather×load interactions, recent volatility, gas basis,
  ramp features — all strictly causal (no future leakage), unit-tested for it.
- **Model bake-off** through the Phase-1 backtest: LightGBM quantile (tuned) vs. a
  global gradient-boosted model with exogenous vars vs. a linear quantile baseline;
  pick per-horizon winners. Hyperparameter search with time-series CV (Optuna).
- **Conformalized quantile regression** — wrap the model so interval coverage is
  *guaranteed* (split-conformal), then report the sharpened, calibrated intervals.
  (This is an advanced flex that reads as "knows modern UQ".)
- **Regime model redesign:** drop self-distillation. Define regimes from *future-relevant*
  outcomes (e.g. realized price spikes / negative-price episodes) and predict those, or
  reframe as anomaly detection. Evaluate with proper class-imbalance metrics.
- **Per-horizon error analysis** + feature importance / SHAP writeup.

**Acceptance criteria**
- Chosen model beats both baselines on economic value at the horizons that matter,
  shown in the scorecard; intervals hit nominal coverage within tolerance.
- `reports/phase3_modeling.md` with the bake-off table, calibration plots, SHAP.

**SWE/ML signal:** feature engineering discipline, model selection, modern uncertainty
quantification, and the judgment to redesign a flawed target.

---

## Phase 4 — MLOps (reproducibility & lifecycle)

**Goal:** make training a tracked, versioned, one-command, reproducible process.

**Deliverables**
- **Experiment tracking:** MLflow — log params, metrics (the full scorecard), and
  artifacts for every run.
- **Config-driven training:** Hydra/YAML configs; one seeded entrypoint
  `scripts/train.py --config configs/spread_lgbm.yaml`.
- **Data + model versioning:** DVC (or lakeFS) pins the exact dataset snapshot to each
  model; models registered with a version + the dataset hash they were trained on.
- **Automated eval report + model card:** every train regenerates a model card
  (data window, metrics vs. baselines, intended use, known limits).
- **Promotion gate:** a model is only "promoted" to the API if it beats the current
  production model on the backtest (a script enforces it).

**Acceptance criteria**
- `dvc repro` reproduces a model bit-for-comparable from a clean checkout.
- MLflow UI shows the run history; the registry has a "production" model tied to a data hash.
- Promotion script refuses a worse model.

**SWE/ML signal:** the MLOps lifecycle — tracking, versioning, reproducibility,
governance. Rare in junior portfolios; strongly differentiating.

---

## Phase 5 — Productionization, serving & the SaaS shell

**Goal:** anyone can run it, it's deployed live, it's monitored, and it's a real
multi-tenant product with login and billing.

**Deliverables**
- **Serving refactor:** API loads the *registered* production model; predictions cached;
  raw data served from the warehouse, not live fetches. Tighten CORS.
- **Multi-tenant serving:** every request resolves the caller's `org_id`; queries and
  model lookups are tenant-scoped; row-level isolation enforced (and tested — a
  cross-tenant read must fail).
- **Auth:** login + organizations + roles via a managed provider (**Clerk** or
  **Supabase Auth** recommended for speed); protected API + frontend routes.
- **Onboarding flow:** sign up → create org → add a site (coords, heat rate, gas hub,
  tariff) → see a live forecast. This is the product's "aha" path.
- **Billing:** Stripe (test mode) — a free tier + a paid plan, subscription per site or
  usage-based; a billing portal; feature gating by plan. (Test mode is enough for a
  portfolio; it demonstrates the full integration without real charges.)
- **Per-tenant API keys + metering** if exposing forecasts via API (rate-limited).
- **Tests + CI:** `pytest` (unit + a small integration test hitting the API), GitHub
  Actions running tests + lint (ruff) + the data-validation suite on every PR.
- **Docker:** multi-stage backend image + frontend build; `docker compose up` runs the
  whole stack incl. DuckDB volume.
- **Deploy:** backend to Fly.io/Render, frontend to Vercel/Netlify — a real public URL.
- **Monitoring & drift:** scheduled job scores yesterday's forecasts as truth arrives,
  logs realized pinball/economic value, and runs a drift check (PSI/KS) on incoming
  features; alert/report when the live model degrades.

**Acceptance criteria**
- Green CI badge; `docker compose up` works from a clean clone; live demo URL.
- A monitoring page/report showing rolling live forecast accuracy and a drift signal.
- A second org can sign up, add a site, and get an isolated forecast; a cross-tenant
  data access attempt is rejected (tested). Stripe test-mode checkout completes and
  unlocks a gated feature.

**SWE signal:** testing, CI/CD, containerization, deployment, production ML
monitoring, **plus** multi-tenancy, auth, and billing — the "can ship and operate a
SaaS" checklist on top of the ML/DE core.

---

## Suggested order & sequencing notes

1. **Phase 1 first, always** — it's the measuring stick every later phase reports against.
2. Phases 2 and 3 can interleave: you need *some* clean data (Phase 2) before serious
   features (Phase 3), but the bake-off can start on the existing parquet.
3. Phase 4 wraps Phase 3's modeling once you have a model worth versioning.
4. Phase 5 is last but partially parallelizable (CI/tests/Docker can land early).

## What "done" looks like (the portfolio narrative)

> "I took an energy-arbitrage forecaster, found and fixed a data-leakage bug that was
> overstating accuracy, rebuilt the evaluation around proper scoring rules and a dollars-
> based economic backtest, engineered causal time-series features, added conformal
> calibration so the uncertainty bands are trustworthy, made the whole pipeline
> reproducible with DVC + MLflow, and shipped it as a multi-tenant SaaS with auth,
> per-customer configuration, Stripe billing, CI, Docker, a live deploy, and drift
> monitoring." — that sentence is a hireable ML/DE engineer *who can also ship product*.

---

## Progress log

- **Phase 1 — ✅ COMPLETE.** Leakage-free evaluation foundation shipped:
  - `ml/metrics.py` (pinball, calibration, interval coverage), `ml/splits.py`
    (rolling-origin folds + embargo), `ml/dataset.py` (exact-horizon pairs),
    `ml/baselines.py` (persistence, climatology), `ml/economics.py` (dispatch P&L /
    capture rate), `ml/forecasters.py` (refit-per-fold LightGBM), `ml/backtest.py`.
  - `scripts/evaluate.py` CLI → `reports/phase1_scorecard.md`.
  - 15 unit tests green (incl. a test proving the *old* split was leaky).
  - **Headline result:** on the honest backtest, LightGBM beats both baselines —
    mean pinball **2.585** vs 3.64 (persistence) / 3.69 (climatology), and captures
    **57%** of available arbitrage value vs 36% / 8%. *The model genuinely earns its
    place.*
  - **Finding that feeds Phase 3:** LightGBM's p10–p90 interval coverage is only
    **66%** vs the 80% nominal — it's overconfident. → motivates conformal
    calibration in Phase 3.
- **Phase 2 — ✅ COMPLETE.**
  - ✅ **Secret hygiene:** removed the hardcoded EIA key from all 3 source files; added
    `.env` + `.env.example`; key gone from working tree. *User to rotate at eia.gov and
    paste into `.env`.* (Old key remains in git history but is dead once rotated.)
  - ✅ **DuckDB warehouse** (`pipeline/warehouse.py`): tenant-first schema. Shared raw
    tables (`raw_lmp`, `raw_gas`, `raw_weather`) as system-of-record; tenant-scoped
    `orgs` / `sites` / `site_config` (versioned) / `feature_spread`; a `quarantine` table.
  - ✅ **Tenant & config model** (`pipeline/tenancy.py`): the demo→product shift —
    generation cost derives from per-site heat rate / O&M, not global constants.
  - ✅ **Idempotent ingestion** (`pipeline/ingest.py`): generic `upsert_dataframe`
    (INSERT…ON CONFLICT) → every loader is idempotent + backfill-safe.
  - ✅ **Deterministic feature build** (`pipeline/features.py`): raw + config →
    `feature_spread` via SQL, with an **ASOF join** on gas (carry-forward pricing).
    Rebuild matches the legacy parquet to **$0.002 mean abs error** (100% within $1).
  - ✅ **Data validation** (`pipeline/validation.py`): schema contract (ranges, nulls,
    unique PK, `spread == lmp - gen_cost`); bad rows go to `quarantine`, not dropped.
  - ✅ **Orchestration:** `Makefile` (`make warehouse|data|evaluate|test|lint|clean`)
    and `scripts/init_warehouse.py` runs the full pipeline (seed → bootstrap raw →
    build features → validate). All offline-reproducible.
  - **31 tests green** (+12: ingest idempotency/backfill, validation/quarantine,
    feature-build correctness incl. ASOF + config-versioning).
  - **Deferred (small):** live network ingestion still runs via the existing
    `scripts/build_historical_dataset.py` fetchers; wiring those to append into the raw
    tables on a schedule is a thin follow-up. `raw_weather` is keyed by `site_id`
    (coords-based shared key is a future refinement).
- **Phase 3 — 🚧 IN PROGRESS (core done).**
  - ✅ **Causal feature engineering** (`ml/features.py`): trailing rolling stats (3/6/24h),
    volatility, ramps, calendar + cyclical encodings, weather degree-days (CDD/HDD).
    Strictly causal — asserted by a truncation test (feature at t unchanged when future
    rows change). 47 engineered features.
  - ✅ **Conformalized Quantile Regression** (`ml/conformal.py` + `ConformalizedForecaster`):
    split-conformal interval calibration with finite-sample correction. **Directly fixes
    the Phase 1 finding** — interval coverage 66% → 76% toward the 80% nominal.
  - ✅ **Model bake-off** (`scripts/evaluate_phase3.py` → `reports/phase3_modeling.md`):
    Persistence vs legacy-feat vs rich-feat vs rich+conformal, through the Phase-1 harness.
  - 🔬 **Honest finding:** rich features did **not** beat the legacy 2-lag set
    (pinball 2.628 vs 2.585) — ~90 days of data + collinear features add variance, not
    signal. Reported as-is in the scorecard; next lever is tuning + more data, not more
    features. (This negative result is itself a strong portfolio signal.)
  - **7 new tests** (feature causality, no site-bridging, degree-days; conformal coverage
    restoration). **38 tests total, green.**
  - **Remaining in Phase 3:** regime-model redesign (drop self-distillation from rules →
    predict a future-relevant outcome like a 24h spike); hyperparameter tuning (Optuna)
    to give the rich features a fair shot; SHAP / feature-importance writeup.

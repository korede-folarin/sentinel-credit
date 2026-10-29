# Testing Sentinel Credit

Added in v2 step V2-01 ([v2 Engineering Decision Log](docs/v2_engineering_decision_log.md)).
v1 itself is frozen at Git tag `v1.0-evaluated`; these tests and checks protect it and the
code that serves it.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

With coverage:

```bash
pytest --cov=serving --cov=pipeline --cov=data_generator --cov=models --cov=v2 --cov-report=term
```

Lint (new code only, so v1 files stay untouched):

```bash
ruff check v2 tests
```

Prove v1 is unchanged:

```bash
python v2/tools/v1_manifest.py verify --repo   # committed files (what CI runs)
python v2/tools/v1_manifest.py verify --full   # also database, MLflow, model files (local)
```

## What is tested

| Area | File | What it proves |
|---|---|---|
| Synthetic data generator | `tests/test_generator.py` | The repayment state machine: default happens at exactly 90 days past due (60 is not enough) and is absorbing; payments clear the oldest instalment first (FIFO), so DPD moves to the age of the next arrear; a customer who never misses never defaults; default can land after the schedule ends, during collections; the affordability-ratio formula |
| Feature engineering | `tests/test_features.py` | The model uses exactly the 13 frozen features (v1 Decision 27), the same as the shipped champion; `pct_late_payments` is excluded; no outcome-tied, exposure-time, generator-internal or protected field can reach the model; new-applicant features are built correctly; bad input is rejected; the feature-store SQL defines every model feature |
| Champion scoring path | `tests/test_scoring.py` | Reference applicants score exactly as the API and demo publish them (0.009456 and 0.031834); scoring is deterministic; the decision follows the 0.0657 cutoff; the committed model copy is byte-identical to the governed package (local) |
| Drift calculation | `tests/test_drift.py` | PSI is about 0 for identical distributions, grows with the size of the shift, is never negative, handles categorical features and missing values, and bands at the configured thresholds |
| Approval Gate | `tests/test_approval_gate.py` | Passes only with recent evidence and no drift alerts; missing, stale or alerting evidence blocks; the bad-rate shift is reported but never blocks; older timestamps without a UTC offset are read correctly |
| FastAPI endpoints | `tests/test_api.py` | `/health`, `/model/info`, `/score` for a new applicant, the scoring page; missing fields and wrong types are rejected (422); an existing customer is scored from the feature store (local); **the optional `SENTINEL_MODEL_DIR` gives identical answers to the default** (local) |
| v1 integrity | `tests/test_v1_integrity.py` | Nothing in the v1 manifest has changed: committed artefacts and config everywhere; database row counts and schemas, MLflow runs, model files and key metrics locally |

## Where the tests run

| Where | Tests | Skipped, and why |
|---|---|---|
| Your machine (database running, `SENTINEL_DB_PASSWORD` set) | 50 pass | none |
| GitHub Actions CI | 46 pass | 4: they need the local PostgreSQL database or the git-ignored model package `models/artifacts_v1_1` |

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request:
lint, tests with coverage, the v1 integrity check, and a container build plus smoke test of the
scoring API (the container must score the reference applicant at 0.0318).

## Coverage, honestly

Line coverage on GitHub CI, measured over `serving`, `pipeline`, `data_generator`, `models` and `v2`:

| Module | Coverage |
|---|---:|
| `serving/features.py` | 100% |
| `models/model_features.py` | 100% |
| `serving/app.py` | 87% |
| `serving/reasons.py` | 84% |
| `pipeline/approval_gate.py` | 44% |
| `v2/tools/v1_manifest.py` | 41% |
| `data_generator/bnpl_data_generator.py` | 30% |
| `pipeline/monitor_drift.py` | 30% |
| **All measured code** | **16%** |

The total is low because most of the v1 code is one-off scripts with no tests at all.

## What is not covered

- **Training, evaluation and cutoff selection scripts** (`models/train_baseline_models.py`, `evaluate_models.py`, `select_business_cutoff.py`, `explain_models.py`, `govern_model_features.py` and others): 0%. Their outputs are frozen in the v1 manifest, but the code itself is untested.
- **Champion promotion** (`pipeline/promote_champion.py`): 0%. Only the Approval Gate's decision logic is tested, not the promotion flow or the MLflow registry calls.
- **The feature-store SQL** (`sql/02_feature_store.sql`): only checked to define every model feature. Its aggregation logic and point-in-time correctness are not tested against a database in CI.
- **The generator's population sampling** (customers, transactions, calibration): only the repayment state machine and the affordability formula are tested.
- **Drift monitoring's database and MLflow steps** (loading populations, writing logs): only the PSI maths and banding are tested.
- **The Streamlit dashboard, Power BI report and Airflow DAGs**: not unit-tested. The dashboard has a separate local AppTest check; the demo site has its own deploy gate (`website/tests/browser_check.py`).
- **Existing-customer scoring** runs only locally, because it needs the database.

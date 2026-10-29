# Sentinel Credit v2 — Engineering Decision Log

This log records every consequential decision in **Sentinel Credit v2**, the extension of the
finished platform. It is separate from the v1 [Engineering Decision Log](engineering_decision_log.md)
so the two never mix:
- v2 entries are numbered **V2-01, V2-02, …**;
- a reference to the original log is written **"v1 Decision N"**.

**Ground rules for all v2 work** (owner, 29th October 2026):
- **v1 is frozen.** It is everything built up to Git tag `v1.0-evaluated` (v1 Decisions 1–35). v2 never modifies a v1 table, MLflow run, model, feature definition, evaluation result or decision log entry.
- **v1 Decisions 18 and 19 apply to v1 data only.** On any new dataset, the feature set and the business cutoff are re-derived under new V2 decisions.
- **v2 uses versioned names:** its own schema or prefix, one MLflow experiment per phase, and a `dataset_version` tag on every run.
- **The v1.1 XGBoost champion stays champion** unless a challenger passes the Approval Gate. The outcome is logged either way, including "challenger did not win".
- **Power BI, the API and the Streamlit dashboard keep reading v1** until v2 is validated.
- **v2 is framed honestly:** production-grade architecture validated on synthetic data at realistic scale or on a simulated stream, never real customers or live traffic.
- **Every component is labelled** *running*, *tested locally* or *designed only*, and only real numbers from real runs are reported.
- **No credentials in code.**

**Plan:**
1. V2-01: protect v1, tests, CI.
2. Phase 1: PyTorch challenger.
3. Phase 4: PyFlink streaming.
4. Phase 2: Spark at a realistic laptop scale.
5. Phase 3: AWS on ECS Fargate, priced before deployment.

---

## V2-01 — Protect v1, Add Tests and Continuous Integration

### Challenge
Before extending the platform, it had to be possible to *prove* that v2 changes nothing in v1.
Until then, three things made that impossible:
- the project had no test suite (`tests/` was empty);
- nothing ran automatically on a push except the demo site's deploy gate;
- "v1 is unchanged" could only be asserted, not checked.

### Decision
1. **v1 is frozen at a Git tag.** `v1.0-evaluated` was created on commit `e098ca0`, the last
   commit of v1 (*running*: created locally, to be pushed by the owner).
2. **v1 is fingerprinted.** `v2/tools/v1_manifest.py` records everything that defines v1 in
   `v2/manifests/v1_manifest.json`:
   - SHA-256 hashes of 25 committed artefacts and definitions: the champion copy, the ONNX model,
     the model contract, explanation rules and wording, feature definitions, feature-store SQL
     and the generator;
   - a hash of the v1 sections of `config.yaml`, so new v2 sections can be added;
   - hashes of the 16 git-ignored model-package files;
   - exact row counts for the raw and feature-store tables;
   - column definitions for 12 tables and views;
   - all 103 MLflow runs (params and metrics hashed);
   - the champion's key metrics: Gini 0.5618, AUC 0.7809, Brier 0.0254, ECE 0.0015,
     cutoff 0.0657.

   `verify --repo` checks the committed part; `verify --full` checks everything locally. Both
   pass (*running*).
   - **Not frozen:** the reporting tables' row counts, because monitoring appends to them. Their
     column definitions are frozen.
   - **MLflow runs:** new runs may appear, but none recorded may change or disappear.
3. **Tests.** 50 pytest tests in `tests/` (*running*), covering:
   - the generator's repayment state machine: the exact 90-days-past-due trigger, the
     oldest-instalment-first (FIFO) rule, the absorbing default state and post-schedule
     collections;
   - the 13 frozen features and exclusion of leaking or protected fields;
   - the champion scoring path, with reference applicants scoring exactly as published;
   - the PSI drift calculation;
   - the Approval Gate's decision logic;
   - the FastAPI endpoints;
   - the v1 manifest.
4. **One approved change to v1 code.** `serving/app.py` accepts an optional
   `SENTINEL_MODEL_DIR`, so CI and containers can load the committed copy of the champion.
   - Unset, the API behaves exactly as before.
   - A test proves the default and the override give identical answers (local), and that the
     committed copy is byte-identical to the governed package.
5. **Continuous integration** (`.github/workflows/ci.yml`) on every push and pull request:
   - lint of new code only (`v2/`, `tests/`; v1 files are never linted or reformatted);
   - tests with coverage;
   - the v1 integrity check, which fails if any committed v1 artefact or v1 config section
     changes;
   - a container build of the scoring API (`v2/docker/api.Dockerfile`), with a smoke test that
     requires the reference applicant to score 0.0318.

   *Running* once pushed; all four were run locally first.
6. **Honest coverage** is published in [TESTING.md](../TESTING.md), with the gaps listed.

### Evidence (real runs, 29th October 2026)
| Check | Result |
|---|---|
| Full test suite, local (database and model package present) | 50 passed |
| Clean copy of the pushed files, no database, no model package (as on CI) | 46 passed, 4 skipped, each with the reason it needs local resources |
| `v1_manifest.py verify --full` / `--repo` | v1 unchanged / v1 unchanged |
| Lint (`ruff check v2 tests`) | all checks passed |
| API container built from `v2/docker/api.Dockerfile`, run, and scored | `/health` ok; reference applicant 0.0318, decision pass; runs as a non-administrator user |
| Line coverage, CI scope (serving, pipeline, data_generator, models, v2) | **16% overall**. `serving/features.py` 100%, `models/model_features.py` 100%, `serving/app.py` 87%, `serving/reasons.py` 84%, `pipeline/approval_gate.py` 44%, generator 30%, drift 30%. Training, evaluation, cutoff and promotion scripts 0% |

### Finding (v1, recorded and not fixed)
**The drift-monitoring MLflow runs cannot be read by the project's own MLflow.** The 22 runs in
the `sentinel_credit_monitoring` experiment were written by the newer MLflow inside the Airflow
container. Their run records lack the `run_uuid` field that the project's MLflow 2.13 expects, so
`MlflowClient().search_runs` fails on them. The manifest therefore reads the MLflow files directly,
read-only. Aligning the two MLflow versions would change v1, so it is left for a future decision.

### Limits
- **CI cannot see the local database, MLflow store or git-ignored model package,** so the full
  check runs only locally. In CI, the committed champion copy stands in, and its byte-equality
  with the governed package is tested locally.
- **The manifest is itself a committed file.** CI stops accidental change; it cannot stop a
  deliberate rewrite. The tag `v1.0-evaluated` keeps the original manifest's commit history, and
  any rebuild of the manifest must be its own logged V2 decision.
- **Coverage is low by design of v1,** whose training and evaluation code is one-off scripts.
  Tests were added where behaviour is reused (scoring, features, gate, drift, generator rules),
  not to inflate the number.

### Component status
| Component | Status |
|---|---|
| Tag `v1.0-evaluated` | Running (local; owner pushes) |
| v1 manifest and verifier | Running |
| Test suite | Running (local) |
| CI workflow | Tested locally; running once pushed |
| API container image | Tested locally |

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 0.1 | 29th October 2026 | Log created, separate from the v1 log. Ground rules and plan recorded. Added V2-01: v1 frozen at tag `v1.0-evaluated` and fingerprinted (25 committed artefacts, 16 model files, row counts, 12 schemas, 103 MLflow runs, key metrics). 50 pytest tests. CI (lint new code, tests with coverage, v1 integrity, API container smoke test). One approved, additive change to `serving/app.py` (`SENTINEL_MODEL_DIR`), proven identical to the default. Honest coverage of 16% with its gaps listed. Finding: monitoring runs written by the Airflow container's MLflow are unreadable by the project's MLflow 2.13. |

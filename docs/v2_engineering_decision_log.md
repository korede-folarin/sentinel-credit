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

## V2-02 — Phase 1: PyTorch Challenger, Autoencoder, Approval Gate (No Promotion)

### Challenge
Build a neural-network challenger to the v1.1 XGBoost champion and judge it fairly. Also test
whether an anomaly detector adds anything. None of this may promote a model or change v1.

Two traps in v1's own tooling had to be avoided:
- `pipeline/promote_champion.py`, run as a script, would register the challenger under the **v1**
  registered model and overwrite v1's promotion record;
- v1's promotion threshold (`min_gini_delta`) is unset on purpose, so that script can never
  recommend a promotion at all.

### Decision (owner-approved, 29th October 2026)
1. **Same footing as the champion.** The challenger uses:
   - the champion's 13 features (v1 Decision 27);
   - the same 41,604 customers;
   - the champion's stratified split, rebuilt from the frozen ID files;
   - the out-of-time split, rebuilt by the same date rule;
   - v1's median fill;
   - the same Platt calibration (5-fold sigmoid);
   - the 0.0657 cutoff;
   - v1's own metric code (`models/evaluate_models.py`, imported read-only).

   Four checks confirmed the rebuilt splits match v1 exactly. Recomputing the champion through
   v2's code reproduced its published Gini, 0.5618 and 0.5611.
2. **Model** (`v2/phase1/torch_models.py`). A small MLP (multilayer perceptron) wrapped as a
   scikit-learn estimator:
   - inputs standardised;
   - the rare default class weighted, as XGBoost's `scale_pos_weight` does;
   - early stopping on a held-back slice;
   - fixed seeds.

   Size was chosen by 5-fold CV AUC on the training partition only, from three fixed options (the
   owner asked for light tuning, because of overfitting risk).
3. **Autoencoder** trained on training-partition non-defaulters only, one per split, so the
   out-of-time test never leaks into training.
4. **Explainability.** Captum Integrated Gradients on the uncalibrated network, set beside the
   champion's v1 SHAP ranking.
5. **Logging.** MLflow experiment `sentinel_v2_phase1_pytorch`, with the champion's tag keys and
   `dataset_version = full_horizon_2025_v1`. Models are registered under a separate name,
   `sentinel_credit_v2_challengers` (versions 1 and 2). v1's `sentinel_credit_champion` entry is
   untouched.
6. **Results in PostgreSQL**, in a new `v2_results` schema (`v2/sql/phase1_results.sql`):
   - `model_comparison`, `segment_metrics`, `anomaly_deciles`, `feature_attributions`,
     `gate_outcomes`;
   - append-only, keyed by run label;
   - the writer refuses any other schema;
   - `gate_outcomes.promoted` has `CHECK (promoted = false)`, so Phase 1 cannot record a
     promotion.
7. **Promotion evaluation** (`v2/phase1/evaluate_challenger.py`). It calls v1's
   `find_champion_run`, `get_run_metrics`, `evaluate_promotion`, `load_monitoring_evidence` and
   `evaluate_gate` read-only, and never their writers. It logs the outcome whatever it is.

### Evidence (run `phase1_20261029T225916`, real runs)
| Held-out | Champion (XGBoost) | Challenger (MLP) |
|---|---:|---:|
| Stratified Gini / AUC | 0.5618 / 0.7809 | 0.5329 / 0.7664 |
| Stratified ECE / Brier | 0.0015 / 0.0254 | 0.0008 / 0.0255 |
| Stratified recall at cutoff / sent to review | 57.96% / 15.90% | 47.35% / 13.92% |
| Out-of-time Gini / AUC | 0.5611 / 0.7806 | 0.5700 / 0.7850 |
| Out-of-time recall at cutoff / sent to review | 40.32% / 11.90% | 40.32% / 10.93% |

**Paired bootstrap** (2,000 resamples of the held-out customers), Gini difference, challenger
minus champion:

| Split | Difference | 95% CI | Reading |
|---|---:|---|---|
| Stratified | -0.0289 | [-0.0521, -0.0042] | Challenger genuinely worse |
| Out-of-time | +0.0089 | [-0.0205, +0.0374] | Not distinguishable from zero (challenger ahead in 72.8% of resamples) |

**Architecture search** (CV AUC on training data):
- the three options were within about 0.004 of each other on both splits;
- chosen: one hidden layer of 32 (stratified) and 64 to 32 (out-of-time);
- training time: 207 s and 217 s on CPU; 459 s in total.

**Same blind spot, different model.** At the cutoff, the challenger also misses **100%** of
established-file defaulters on both splits. So v1's blind spot (v1 Decision 31) comes from the data
and the shared probability scale, not from XGBoost. On thin files, the challenger misses more
defaulters than the champion on the stratified split (false-negative rate 35.2% against 20.6%) and
the same share out of time (44.4%).

**Explainability.** Captum's ranking against the champion's SHAP ranking: Spearman 0.758.
- **Same top feature:** thin file, ranked 1 by both.
- **Same last feature:** existing obligations, ranked 13 by both.
- **Main disagreements:** the MLP leans more on order value (rank 2 against 4) and on number of
  credit accounts (6 against 12), and less on the affordability ratio (11 against 7).

**Autoencoder: unusual customers do not default more.**
- **On its own as a default signal:** AUC 0.461 (stratified) and 0.420 (out-of-time), *below*
  0.5. Higher reconstruction error goes with slightly *lower* default.
- **Against the champion's score:** Spearman -0.24 on both splits.
- **Among customers the champion does not send to review:** the most unusual decile defaulted at
  1.65% against 1.32% for the rest (stratified), but 1.49% against 1.76% (out-of-time). So it
  does not consistently catch the champion's misses.
- **Conclusion:** in this synthetic population, "unusual" means extreme but often safe profiles,
  not risk. The detector is reported, not used.
- **Not fully converged:** it stopped at 97 and 100 epochs against a limit of 100.

**Approval Gate and outcome.**
- **Gate: PASS.** Monitoring evidence was 23.1 hours old, with no drift alerts. The Airflow drift
  run had executed that day.
- **Recorded outcome:** *"challenger did not win: no promotion threshold is set (v1
  champion_promotion.min_gini_delta is unset on purpose), so no Gini difference can count as a
  material win. Gini difference: stratified -0.0289, temporal +0.0089. Approval Gate: PASS."*
- **Independently of the threshold,** the bootstrap shows the challenger is worse on the stratified
  split, so it would not merit promotion even with a threshold.
- **Nothing promoted.**

**v1 unchanged:** `v1_manifest.py verify --full` passed after all Phase 1 writes. 57 tests pass
locally (7 new Phase 1 tests); lint is clean.

### Findings
1. **On Windows, PyTorch must be imported before MLflow, XGBoost and pandas.** Otherwise its DLLs
   fail to initialise (WinError 1114). The first training run failed at import and wrote nothing;
   the scripts now import PyTorch first.
2. **MLflow could not infer pip requirements** for the logged models and fell back to
   `scikit-learn==1.5.0, cloudpickle==3.1.2`. The PyTorch dependency is not recorded in the logged
   model's environment. Anyone loading it must install torch 2.13.0 themselves.

### Limits
- **One run, one seed.** Seed variance was not measured; the bootstrap covers sampling variance of
  the test set only.
- **The out-of-time model trains on 36,747 customers,** some of whom are in the stratified test
  set. This is v1's design (audit A2), kept for comparability.
- **Power BI is not yet pointed at `v2_results`.** The tables are ready. The owner hasn't yet
  described pages 2 and 4, so their shape follows the documented Fairness and Segments page and a
  model-comparison view.

### Component status
| Component | Status |
|---|---|
| MLP challenger, calibration, evaluation | Tested locally (real run) |
| Autoencoder | Tested locally (real run) |
| Captum explanations | Tested locally (real run) |
| `v2_results` schema and rows | Running (local PostgreSQL) |
| MLflow experiment and v2 registry entry | Running (local MLflow) |
| Promotion evaluation and Approval Gate | Tested locally; outcome logged, nothing promoted |
| Power BI on `v2_results` | Designed only |

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 0.1 | 29th October 2026 | Log created, separate from the v1 log. Ground rules and plan recorded. Added V2-01: v1 frozen at tag `v1.0-evaluated` and fingerprinted (25 committed artefacts, 16 model files, row counts, 12 schemas, 103 MLflow runs, key metrics). 50 pytest tests. CI (lint new code, tests with coverage, v1 integrity, API container smoke test). One approved, additive change to `serving/app.py` (`SENTINEL_MODEL_DIR`), proven identical to the default. Honest coverage of 16% with its gaps listed. Finding: monitoring runs written by the Airflow container's MLflow are unreadable by the project's MLflow 2.13. |
| 0.2 | 29th October 2026 | Added V2-02 (Phase 1). PyTorch MLP challenger on the champion's 13 features, splits, calibration and cutoff; the splits were proven identical to v1. Stratified Gini 0.5329 against 0.5618, worse, with a 95% CI excluding zero. Out-of-time 0.5700 against 0.5611, a CI including zero. The challenger shares the champion's 100% established-file miss rate. The autoencoder is not a default signal (AUC 0.46/0.42). Captum against SHAP: Spearman 0.758. Approval Gate PASS. Outcome: challenger did not win, nothing promoted. Results are in the new v2_results schema; models in a separate registry entry. v1 is verified unchanged. Findings: PyTorch DLL import order on Windows; MLflow did not record torch in the logged model's requirements. |

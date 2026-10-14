# Sentinel Credit — Pre-Continuation Audit Findings

**Date:** 12th October 2026
**Scope:** Full audit requested in the Claude Code handoff, run before any
"Remaining work" item. Findings were surfaced, not fixed. The only file this
audit wrote is this one.
**Tags:** **(a)** genuine bug/gap to fix · **(b)** decision log needs updating
to match reality · **(c)** verified fine, no action

> **Database-dependent checks are still pending.** `SENTINEL_DB_PASSWORD` is
> not set in this session's environment at any scope. Everything below was
> verified from code, saved artifacts, and file timestamps. The checks listed
> in the final section need a live connection.

---

## Progress log

### 13th October 2026: Steps 1 and 2 (Decision 24). Model untouched.

**Fixed:**
- **A1.** `log_drift_history()` now inserts once and doesn't crash.
  - Each run gets a `run_id`, written to the rows, the metadata JSON and
    the MLflow tags.
  - The DDL moved to `sql/reporting/create_feature_drift_log.sql`, and
    the `run_id` column was added to the live table.
  - The 14 rows from 13 Aug keep `run_id = NULL`. No id was invented
    for them.
- **A5 (badge part).** `governance_status_latest` now derives
  PASS → STALE at read time, using the row's own
  `max_monitoring_age_hours`. It read STALE (1,488.9h) before the fresh
  run.
- **A9.** `segment_key` was folded into `create_reporting_views.sql`.
  Re-applying the file against the live views succeeded.

**DAG leftover code (handoff item):** none was present.
`infra/dags/monitor_drift_dag.py` matched the committed version, apart
from line endings, with no stray `log_*_history` functions or
pandas/sqlalchemy imports. The leftover code was only in
`pipeline/monitor_drift.py`. The DAG `description` was changed so it
no longer claims daily monitoring.

**Why nothing ran from 14 Aug to 12 Oct:**
- MLflow's monitoring experiment holds exactly three runs: 11 Aug 12:07
  (manual), and 12 Aug 00:00 and 13 Aug 00:00 (scheduled, UTC).
- Docker Desktop has `AutoStart: False`.
- On launch today, Docker Desktop crashed with "starting services:
  initializing Inference manager… remove …\Docker\run\dockerInference:
  The file cannot be accessed by the system". It's a stale socket file
  (reparse point) last written 11 Aug 11:46. The `dockerEthernetVfkit`
  and `userAnalyticsOtlpHttp.sock` files beside it date from the same
  time.
- Conclusion: the scheduler only runs while the stack is up, and the
  stack has been down. **Unverified:** whether the DAG was paused, and
  Airflow's own run history. Both need Docker, which currently can't
  start. Docker was not repaired, because that's outside Steps 1–2.

**Fresh evidence (13 Oct 00:54–00:55):**
- The monitor ran natively with `--skip-mlflow`. It exited 0 with
  0 of 14 features alerting, and logged run
  `drift_20261013T005435_21a21fce` (14 rows).
- The gate then wrote PASS, with evidence 0.01h old.
- `SELECT COUNT(*) FROM reporting.feature_drift_log` returns 28: two
  runs of 14.
- `governance_status_log` has 3 rows. The latest is PASS, 13 Oct 00:55.
- **This run has no MLflow record**, deliberately (Decision 20's
  no-Windows-MLflow rule).
- Without a working scheduler, the badge will turn STALE again around
  15 Oct 00:55. That's intended behaviour.

**Airflow health (13 Oct, with Docker running). Supersedes the
"unverified" notes above.**
- `sentinel_credit_drift_monitoring` is **unpaused**. Variables
  `sentinel_db_host` (= `host.docker.internal`) and
  `sentinel_db_password` exist; the password value was not read.
- **Run history:**
  - 11 Aug: three failed setup runs, then a manual success.
  - 12 and 13 Aug: scheduled successes.
  - **14 Aug to 11 Oct: no runs and no `feature_drift_log` /
    `governance_status_log` rows on any of those 59 days.**
  - Then catch-up runs `scheduled__2026-08-14` and
    `scheduled__2026-10-12`, both started 12 Oct 23:53 UTC **at the
    same time**.
  - Then `scheduled__2026-10-13` at 00:00 UTC.
  - All October runs: `compute_drift` success, `evaluate_alerts`
    success (no drift).
- **Cause:** the Astro containers restart automatically with Docker,
  but Docker Desktop's AutoStart is off. When Docker is down, nothing
  runs. The launch error dialog appeared, but the engine did come up.
- **`feature_drift_log`: 70 rows = 5 runs × 14.** These are the
  13 Aug legacy run, three October container runs, and one host run.
  Every feature has exactly **one** distinct PSI and comparison mean
  across all 5 runs, which confirms A3 with data.
- **`governance_status_log`:** rows only on 13 Aug (×2) and 13 Oct
  (×1, host run). The DAG doesn't run the gate, so container runs
  never refresh the badge.

**New findings (not fixed):**
- **A17. `logged_at` mixes UTC (container) and London (host) times.**
  The 13 Oct 01:00 BST container run is stored as `00:00:12` and sorts
  before the 00:54 BST host run. Same root cause as A15.
- **A18. Concurrent catch-up runs.** Two DAG runs wrote the same CSV
  and metadata JSON at the same time. The DAG doesn't set
  `max_active_runs=1`.
- **A19. The gate is not scheduled** (A5 restated with evidence).
- **A15. Timestamp skew confirmed with data.**
  - The 13 Aug gate row (01:00:23 London) evaluated the 12 Aug evidence,
    because the container's 00:00:35 run was UTC, i.e. 01:00:35 London.
  - It recorded the evidence age as 25.0h against a true age of about
    24h.
  - Gate runs from the container would make the view read up to 1h
    young during BST.
- **A16. Docker Desktop shows a launch error** about stale socket
  files. It was running again on 13 Oct, but it isn't set to start
  automatically, and that's the real reason for the gap.

### 13th October 2026 (evening): gate scheduled, timestamps fixed (Decision 24 Addendum)

**Fixed:**
- **A15 / A17.** `evaluated_at`, `created_at` and `logged_at` are now
  `timestamptz`.
  - Existing rows were converted from their evidenced zone by
    `sql/reporting/migrations/2026_10_13_timestamptz.sql`.
  - Writers send offset-aware values.
  - The badge view compares against `now()`.
  - Host and container rows now sort in true order.
- **A18.** `max_active_runs=1` is set on the DAG.
- **A19 / A5.** `run_approval_gate` is a DAG task.
  - `approval_gate.py`'s exit codes are now 0 PASS, 1 BLOCK, and 2
    could not complete or record.
  - With `--require-db-write`, a failed DB write fails the task.

**New findings (not fixed):**
- **A20. Airflow's API server hung.** The worker got `ReadTimeout`
  from `api-server:8080/execution/`, and a triggered run sat in
  `queued` for over 3h. Restarting the `api-server` container fixed
  it. Root cause not established. Nothing alerts on a stuck-queued
  run.
- **A21. `monitor_drift.py` stalled for ~9.5 min in disk I/O**
  (state `D`, wait channel `p9_client_rpc`) on the WSL2 9P bind mount.
  - OneDrive cloud-only files are **ruled out**: 0 of 3,261 `mlruns`
    files.
  - Fresh reads on the same mount were instant during the stall, so
    it was one stuck 9P request, not a dead mount.
  - It cleared on its own. Root cause not established.
  - **Mitigated:** the DAG now sets `execution_timeout=20min`, so a
    hung task can't block later runs under `max_active_runs=1`.

**Verified end-to-end (21:44–21:58 UTC):**
- all three DAG tasks succeeded
- `feature_drift_log` = 84 rows (6 runs)
- `governance_status_log` row 4 = PASS at 21:58:20 UTC, written from
  the container, with a correct evidence age of 0.03h
- the badge view reads PASS
- **A22. Deprecation.** The DAG uses `airflow.models.Variable`, which
  Airflow 3 warns is deprecated in favour of `airflow.sdk.Variable`.

### 14th October 2026: decision-log clean-up, drift overlap, pct_late_payments evidence

**Fixed:**
- **B1.** Decision 23 was written retrospectively. The Summary Table now
  runs to Decision 26, and the log is at v3.26.
- **B2–B5.** Decisions 3, 5, 6, 7, 8 and 9 now have dated
  Implementation Status notes. Decision 2 has a filename note.
- **B9.** The stale temporal-split comment in `config.yaml` was
  corrected, and a stale `recalibrate_threshold.py` reference in
  `train_baseline_models.py` was fixed.
  - **Not changed, deliberately:** `model_governance.pipeline_stage`
    and `decision_log_version`. `explain_models.py` and
    `govern_model_features.py` check them against
    `champion_metadata.json` and refuse to run on a mismatch, so they
    are documented as identity values that change only with a retrain.
- **A2.** The drift comparison now excludes reference customers:
  1,003 unseen customers instead of 4,857, of whom 3,854 overlapped.
  Several PSI values rose once self-comparison was removed. All are
  still stable.
- **A3 (partly).** A new `reporting.feature_drift_by_cohort` view gives
  per-quarter PSI with a noise floor, keyed by origination quarter.
  `feature_drift_log` itself is unchanged, and its rows are still not
  a trend. Decision 25.
- **A7.** Closed as checked, no change warranted (Decision 26):
  - the champion's train/test gap is 0.067, 95% CI [0.001, 0.137]
  - the champion's fixed hyperparameters beat depth 2/3 × learning
    rate 0.005/0.01 with early stopping on validation PR-AUC

**Evidence complete, decision open:**
- **A6.** The 13-feature challenger is registered as v2. Removal costs
  −0.0054 repeated-CV Gini; the held-out difference is −0.0038, and its
  CI includes 0. Decision 21's Gini-win rule cannot promote a
  validity fix. Decision 26 lists what needs deciding.

**New findings (not fixed):**
- **A23 (c): intended behaviour, confirmed by the owner 14 Oct.**
  `promote_champion.py` found 3 matching champion runs per split and
  used the latest. This is by design: re-running the same champion
  should reproduce the same result, so the latest run stands for all
  of them.
  - **Verified 14 Oct:** all 3 runs per split have identical Gini
    (stratified 0.5656, temporal 0.5675) and the same `git_commit`
    (`0aeceb3`).
  - One run per split has no `eval_*` metrics, because evaluation was
    never logged on it. If it were the latest,
    `get_run_metrics()` would raise an error rather than use it, so
    the failure mode is loud, not silent.
  - **Rollback rule added 14 Oct (owner's rule, Decision 21
    addendum).** Same Gini → latest; latest worse → roll back to the
    previous run; latest better → latest, flagged. Unscored runs are
    skipped. Tested on fake runs (7/7 pass) and on the real store,
    where the selection is unchanged.
- **A25. A worse rerun is already the served model.** The rollback
  above only affects the promotion comparison. `serving/app.py`
  serves `models/artifacts/champion_calibrated.joblib`, and
  `train_baseline_models.py` overwrites that file on every rerun.
  Not fixed. Options: serve the registry `champion` alias, or have
  training refuse to overwrite the package when the result is worse.
- **A24.** The 2024Q4 cohort shows moderate PSI (0.106) on
  merchant-category mix. It is plausibly a short-history artefact of
  the partial final quarter, but this is not verified.

### 14th October 2026 (later): owner decision, remove pct_late_payments (Decision 27)

The owner ruled that leakage is a valid reason for removal. Choices:
- **promotion:** an evidence override
- **model:** the champion's hyperparameters, held fixed
- **files:** a new package folder selected by config

**Built (the active champion is unchanged until the switch):**
- **Package location is a setting.** `config.yaml`
  `model_governance.artifacts_dir`, read by all 10 scripts and the
  API, which previously hardcoded `models/artifacts`. All modules
  import cleanly and still resolve to `models/artifacts`.
- **Partly closes A25.** `train_baseline_models.py` refuses to write
  into a package folder that belongs to a different champion identity.
  The August package can no longer be overwritten by a retrain under
  a new identity.
- **Pinned settings.** `train_baseline_models.py --pin-xgboost-params
  <champion_metadata.json>` holds XGBoost to a given champion's
  settings, recorded as the `xgboost_params_source` tag and metadata.
- **A14 (prompt).** `--no-prompt` removes the blocking `input()` for
  non-interactive runs.
- **Override path.** `promote_champion.py --validity-override
  "<Decision N>" --validity-evidence <file>`:
  - permitted only if the decision-log heading exists and the evidence
    file describes these exact challenger runs
  - and if, on BOTH splits, the 95% paired-bootstrap CI of the Gini
    difference reaches zero or above
  - the approval gate and typed `PROMOTE` still apply
  - on apply, the registry version is tagged with the path, decision
    and removed feature
- **Temporal CI.** `evaluate_feature_removal.py` now also computes the
  temporal paired-bootstrap CI and writes `promotion_evidence.json`.

**Switched (v1.1 live, Decision 27 outcome):**
- **Promotion.** The owner typed `PROMOTE`; registry v3 is champion.
- **Package and cutoff.** The v1.1 package is in
  `models/artifacts_v1_1/`; v1 is untouched (checksums). The cutoff is
  0.0657.
- **A6 closed.** `pct_late_payments` is out of the model and out of
  the API.
- **A25 partly closed.** Training can no longer overwrite another
  champion's package. The API still loads files, not the registry
  alias.

**New findings (not fixed):**
- **A26.** `reporting.customer_risk_snapshot` still exposes
  `pct_late_payments`, contrary to the view's own "no excluded fields"
  rule. Kept until the Power BI report is confirmed not to bind to it,
  then drop it.
- **A27. Narrative docs still describe v1** (14 features, 0.0671):
  - `docs/shap_explainability_report.md`
  - `docs/model_evaluation_report.md`
  - `docs/training_summary.md`
  - `docs/feature_governance_report.md`
  - possibly `README.md`

  The generated evidence in `reports/` is v1.1, and v1's is archived
  in `reports/_archive_v1_14f_2026-08/`.
- **Still open:** A13 (the owner's sign-off of the 13-feature
  governance review) and the thin-file FNR governance item
  (Decision 19).

---

## Findings that could change the "Remaining work" priorities

These come first because building Power BI history tables or visuals on top
of them would mean displaying numbers that are about to change, or trends
that can't exist.

### A6 (a) `pct_late_payments` is not independent of the default path
`data_generator/bnpl_data_generator.py` assigns `payment_status = "Late"` in
three places:
- line 696: the independent 5% "paid a few days late" draw from `CURRENT`
- line 673: a **full cure** from `DELINQUENT` (clearing arrears)
- line 682: a **partial cure** from `DELINQUENT`

So most of the `Late` rows behind a delinquent customer's score are arrears
resolution events. That's the mechanism default emerges from. Separately, every
instalment after default is written as `Defaulted` and sits in the
`AVG(...)` denominator in `sql/02_feature_store.sql:135`. That pulls the
feature down for defaulters by an amount that depends on how long the loan ran
after default, which is information that only exists after the outcome.

This is the same "two generator events under one label" problem used to
exclude `avg_days_late` / `max_days_late` / `total_days_late` as
`RESOLUTION_ADJACENT_EXCLUDED`. The Decision 12 addendum's claim that the
late-payment features are "drawn from a process independent of the default
path" was true of the pre-Decision-10 generator but is not true of the current
one. The model gives it 5.40% of mean |SHAP| (rank 7). The SHAP report's
false-negative example shows a defaulter with `pct_late_payments` contributing
−0.48, which fits the denominator-dilution effect.

**Why this changes priorities:** if `pct_late_payments` has to come out, v1
needs retraining, which means a new cutoff (Decision 19), new SHAP, and a new
governance review. Every Performance/Fairness number on Power BI would change.

### A7 (a) Every selected XGBoost hyperparameter is on the edge of the grid
`models/artifacts/champion_metadata.json` against `params.yaml`:

| Param | Grid | Selected | Position |
|---|---|---|---|
| `learning_rate` | 0.01, 0.05, 0.1 | 0.01 | minimum |
| `max_depth` | 3, 5, 7 | 3 | minimum |
| `n_estimators` | 100, 200, 300 | 300 | maximum |

All three point the same way: more regularisation (shallower trees, smaller
steps, more rounds) was still improving CV PR-AUC when the grid ran out. The
true optimum is probably outside the grid. GridSearchCV was run without
`return_train_score=True`, so no train/validation gap was recorded for the
search. The train-vs-test check needs the DB (pending, below). This would also
mean retraining, so it should be investigated together with A6, in a single
retrain.

### A2 (a) 79.3% of the drift "comparison" population is also in the reference
`monitor_drift.py` uses the champion's stratified training partition as the
reference and "everyone with `first_transaction_date >= 2024-09-01`" as the
comparison. Checked against the saved ID files: **3,854 of the 4,857
comparison customers (79.3%) are in the reference partition.** PSI is
mechanically pulled toward zero. "0 of 14 features alerting" is therefore much
weaker evidence than Decision 20 presents it as.

### A3 (a) Daily monitoring on a static snapshot produces identical numbers every run
Neither population changes between runs (the data is a fixed generated
snapshot, and no scoring stream exists). Every scheduled run computes the same
PSI values. Once `feature_drift_log` works, it will collect the same 14 rows
each day. **A PSI "trend" visual would be a flat line by construction. That's
not evidence of stability, it just means the inputs never change.** The same
applies to a `model_comparison_log`: `evaluate_models.py` output only changes
when the model is retrained. Neither history table can show a real trend
without a source of new data, e.g. scoring batches by origination month. Under
the "don't fabricate a trend" rule, this needs a decision before Remaining
work items 1, 2 and 6 are built.

---

## (a) Genuine bugs and gaps

**A1. `log_drift_history()` crashes after inserting.** `pipeline/monitor_drift.py:296-350`
(uncommitted). The psycopg2 insert runs and commits, and then the function
continues into leftover code from the earlier failed attempt:
`create_engine(...)`, which is never imported, so it raises `NameError`.
File timestamps match this: `feature_drift.csv` (written just before the call)
is from **13 Aug 23:41**. `monitoring_run_metadata.json` (written just after)
is from **13 Aug 01:00**. That last run therefore crashed. Its rows may be in
the table, but there's no metadata JSON or MLflow run, and under Airflow
`compute_drift` would go red. Simply adding the import would make it insert
every row twice. Smaller issues in the same function:
- `CREATE TABLE` sits in runtime code rather than `sql/reporting/`.
- There's no run identifier to group the rows from one run.
- A tz-aware `utcnow()` goes into a `TIMESTAMP` column, which Postgres converts
  to the session timezone.

**A4. Airflow is not running, so nothing is scheduled.** Docker isn't running.
The last successful monitoring run was 13 Aug 2026, about 60 days ago. The
approval gate's 48h freshness check would BLOCK today. "Runs daily" describes
the design, not the current state.

**A5. The governance status feed is real but not live.**
`approval_gate.py`'s new DB write (uncommitted) only runs when
`approval_gate.py` is run by hand or from `promote_champion.py`. Nothing
schedules it. `governance_status_latest` shows the last manual evaluation, not
current status.

**A8. ECE is computed on bins that the scores barely use.**
`evaluate_models.py:397` uses ten uniform bins on [0, 1]. The calibrated
champion never predicts above about 0.15, so only 1–2 bins are ever populated.
Reported ECE is 0.0008. With equal-frequency deciles on the same stratified
test predictions it's **0.0040**, five times higher. Calibration is still
genuinely good (see C1). But Decision 13's "0.3552 → 0.0026, ~150×" is partly
an artifact: uncalibrated scores spread across all ten bins, while calibrated
ones collapse into one. Brier skill vs a constant base-rate prediction is only
about 4% (0.0254 vs 0.0264), so the model is well-calibrated but has low
resolution. That's worth stating, not hiding.

**A9. Live reporting views have drifted from the canonical SQL.** The
root-level `fix_customer_risk_snapshot.sql` / `fix_segment_key_final.sql` add
`segment_key` to both views. `sql/reporting/create_reporting_views.sql`
doesn't have it. Re-running the canonical file would fail, because
`CREATE OR REPLACE VIEW` can't drop a column, and on a fresh database it would
build views without the column Power BI expects. `sql/reporting/` is also
untracked.

**A10. `customer_risk_snapshot` has no model score column.** The drillthrough
view exposes features and the outcome, but no calibrated PD. Where Power BI
gets its risk score from is still unknown. `inventory_tables.sql` mentions a
`champion_predictions` table. It may be a CSV load, which the handoff says is
not production data (pending DB check).

**A11. The delivery layer is not version-controlled.** `serving/`,
`dashboard/`, `sql/reporting/` and `sentinel dashboard.pbix` are all untracked.
The handoff's `api/app.py` doesn't exist: the API is `serving/app.py`.

**A12. Swagger styling is applied, but with a different navy.** Remaining work
item 5 is already in the file (`docs_url=None, redoc_url=None` are set, and
`serving/docs_ui.py` is wired in). It uses `#243B5A` / `#14233B`, not the
specified `#0F2747`.

**A13. The 14-feature governance review was never signed off.**
`reports/feature_governance/feature_governance_review.md` (generated 10 Aug)
lists all 14 features as `PENDING_MANUAL_REVIEW`. Decision 18 required this
pass "before v1 is called final", and `config.yaml` still says
`pipeline_stage: post_scope_decision_pending_retrain`.

**A14. Failure points.**
- `train_baseline_models.py:700` ends with `input()`, which hangs or raises EOF
  in any non-interactive run (Airflow, CI).
- `train_baseline_models.py:load_data` ignores `SENTINEL_DB_HOST`, unlike
  `monitor_drift.py`, the API and the gate, so it would fail in the container.
- Every script defaults the password to `""` when the env var is missing. The
  result is a generic auth error rather than a clear "SENTINEL_DB_PASSWORD not
  set" message. `user="postgres"` is hardcoded everywhere.
- `split_data` median-fills using the full frame, test rows included. It's a
  no-op today (0 nulls), but if nulls ever appear it becomes silent leakage.
  The API doesn't impute at all, so training and serving would then diverge
  without any error.
- `serving/app.py` has no handling for DB errors (unhandled 500). For existing
  customers it silently ignores `order_value_gbp` / `n_instalments` /
  `merchant_category`.
- `approval_gate.py` compares a naive local `datetime.now()` against a
  timestamp that may have been written in the container's UTC, giving a 1h
  skew during BST.
- `tests/` is empty. Decisions 20/22 say logic was "unit-verified", but no
  tests are in the repo. The Astro example test requires `retries=2` and the
  DAG sets 1, so `astro dev pytest` would fail.

---

## (b) Decision log entries to update

**B1. Decision 23 doesn't exist.** Commit `e59a042` says "Decision 23:
automated approval gate", but the log ends at Decision 22. The header says
v3.24 and the handoff says v3.25. The version-history rows are out of order
(3.24 is listed before 3.22), and the Summary Table stops at Decision 12.

**B2. Decision 3 (and the Summary Table) say "PostgreSQL Feature Store on AWS
RDS".** It's local Postgres. Nothing is deployed.

**B3. Decisions 2/10 and Appendices B/D refer to `validate_synthetic.py`.** The
file is `validation/validation.py`.

**B4. Decision 5 describes drift monitoring that was never built.** It
describes concept drift, prediction drift (PSI on scores), weekly cohorts,
monthly revalidation including fairness, automatic escalation, and
`psi_threshold` / `drift_alert_threshold` config keys. What was built is
daily feature PSI, data-quality checks and a bad-rate delta. There's no score
PSI, and those config keys don't exist. Decision 20 doesn't say which parts of
Decision 5 it closes and which remain open.

**B5. Decisions 6/7/8/9 describe components that don't exist:**
- error classifier
- adaptive LLM engine with three audience outputs
- audit-trail PDF export
- BentoML/Docker/AWS deployment
- approver ID recorded in MLflow (`promote_champion.py` records no approver
  identity)

Decision 8's promotion criteria (Gini, KS, PR-AUC, ECE and fairness) were
silently narrowed by Decision 21 to Gini on both splits, with no note on
Decision 8. These entries need "planned, not built" markers in the style
already used for CTGAN.

**B6. Decision 12 addendum's "late-payment features are independent of default"
claim** is contradicted by the current generator (A6).

**B7. Decision 13's ECE improvement figure** needs the binning caveat (A8).

**B8. Decision 20.** It needs the 79% overlap (A2), the static-snapshot
limitation (A3), and the "not currently running" status (A4). The
`monitor_drift.py` and DAG docstrings and the metadata `note` still say the
approval gate is "not yet built".

**B9. Stale `config.yaml` comments and values:**
- `split_strategy` says the temporal split "produces a degenerate single-class
  test set… skipped automatically". It doesn't: the temporal test set has
  4,857 customers and is evaluated.
- `decision_log_version: "3.20"` (champion metadata says 3.19, log is 3.24).
- `pipeline_stage` is stale.
- References to the deleted `recalibrate_threshold.py` remain, also in
  `train_baseline_models.py:66`.

**B10. Handoff corrections (not the log, but they change the work plan):**
- Remaining work item 2: `model_comparison_stratified.csv` is written by
  `models/evaluate_models.py:794`, not by the promotion workflow.
- Remaining work item 3: KS, PR-AUC, ECE, Brier and decile lift **already
  exist** as CSVs (`reports/model_comparison_*.csv`,
  `reports/lift_tables_*.csv`). What's actually missing is any Postgres copy,
  a calibration-curve table, persisted precision/recall at the cutoff, and a
  corrected ECE (A8).

---

## (c) Verified fine

**C1. Calibration fitting discipline.** `CalibratedClassifierCV(method="sigmoid", cv=5)`
is fit on `X_train` only (`train_baseline_models.py:299-300`). The test
partition is touched once, for prediction. Held-out reliability by decile is
good:

| Split | Mean pred | Observed | Top decile pred / obs | ECE (deciles) |
|---|---|---|---|---|
| Stratified (champion) | 2.78% | 2.72% | 9.75% / 10.82% | 0.0040 |
| Temporal (temporal-trained model) | 2.32% | 2.55% | 9.18% / 9.47% | 0.0042 |

Some deciles in the middle and low range are noisy: 3–6 defaults per bin out
of about 830.

**C2. Cutoff derivation matches deployment.** `select_business_cutoff.py` uses
training-only OOF with nested CV, the same sigmoid calibration and frozen
hyperparameters, and no held-out reads. Every consumer uses `>= cutoff`
(evaluate, explain, diagnostics, API) and reads the same config value. The
held-out stratified flag rate is 15.51% against the 15.00% target. The API
scores with the same `champion_calibrated.joblib`. Its new-customer path is
openly labelled as out-of-scope (Decision 18). The 15% capacity figure is
still a placeholder (Decision 19 already states this).

**C3. The API's new-customer merchant risk tiers match the SQL definitions
exactly** (`sql/02_feature_store.sql:110-117`).

**C4. Git hooks (Decision 11) are installed** and `docs/audit_trail.log` exists.

**C5. Approval gate and promotion logic are sound:**
- 48h staleness check
- never recommends promotion with `min_gini_delta` unset
- `PROMOTE` confirmation required
- config is never auto-written

---

## Database checks (run 13th October 2026, read-only)

1. **`reporting.feature_drift_log` exists with 14 rows from one run**
   (2026-08-13 23:41). That confirms A1: the insert committed, then the script
   crashed. No other run has ever logged.
2. **`reporting.governance_status_log` has 2 rows. The latest is PASS at
   2026-08-13 01:00.** Power BI's gate badge is still showing a two-month-old
   PASS, but re-running the gate today would BLOCK because the evidence is
   stale. The badge is misleading as it stands (extends A4/A5).
3. **No model predictions are stored anywhere in Postgres.** The database
   contains only `raw.*` (3 tables), `feature_store.customer_features`, the two
   reporting log tables, and 5 reporting views. There's no
   `champion_predictions`, `model_comparison_*`, or score column. Power BI
   currently has no calibrated PD to show (confirms A10).
4. **The live views have `segment_key` but the canonical SQL doesn't.** This
   confirms A9.
5. **Train vs test gap:**

   | Model | Train Gini | Test Gini | Gap |
   |---|---|---|---|
   | Calibrated champion (5-fold ensemble) | 0.6321 | 0.5656 | +0.067 |
   | Explanation base (full-train fit) | 0.6293 | 0.5663 | +0.063 |

   With 226 test defaults, the standard error on test Gini is roughly
   ±0.03–0.04. A gap of about 0.065 is therefore modest, not a sign of
   material overfitting. **A7 is downgraded:** the grid-edge observation
   stands, but the expected gain from widening the grid is small. It's worth
   doing only as a cheap confirmation during the A6 retrain, not as a
   priority in its own right.
6. **A6 decomposition (rows in the modelable population):**

   | | Independent `Late` | Cure-event `Late` | Cure share of `Late` | `Defaulted` rows in denominator |
   |---|---|---|---|---|
   | Non-defaulters | 22,040 | 6,538 | 23% | 0 |
   | Defaulters | 645 | 421 | **39%** | 723 (4.2% of their rows) |

   The mean `pct_late_payments` is almost the same for defaulters and
   non-defaulters (0.0601 vs 0.0611). The share with any `Late` row is higher
   for defaulters (59.8% vs 47.5%). The arrears-derived component and the
   post-default dilution push the mean in opposite directions, which is why
   univariate IV looked weak while the feature still carries trajectory
   information. **The structural contamination is confirmed. The effect on
   model performance still needs the repeated-CV ablation.**
7. **Zero nulls** across all 14 model features on 41,604 modelable customers
   (Decision 17's claim confirmed).

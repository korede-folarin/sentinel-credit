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

## V2-03 — Phase 4: Streaming Features and Scoring with PyFlink (Shadow Mode)

### Challenge
v1 scores from a batch feature store: `sql/02_feature_store.sql` rebuilds every customer's
features from their whole history at once. A BNPL lender decides while the customer waits at
checkout, so features must be current the moment an application arrives. Real event streams
also misbehave:
- events arrive twice;
- events arrive late, or out of order;
- machines fail mid-stream.

The phase had to show this working without touching v1, and without implying real traffic.

### Decision (owner-approved, 29th October 2026; design revised on measured evidence, 30th October)
1. **A simulated stream of v1's own synthetic history.**
   - `v2/streaming/simulator.py` reads chosen customers' profiles (`raw.customers`) and
     applications (`raw.transactions`), read-only.
   - It sends them to Kafka in the order they happened, at a fixed rate of 10, 100 or 1,000
     events per second. Each event's time is the replay clock, so years of history play in
     true order, compressed.
   - **To test the misbehaviour:** 1% of events are sent twice, and 2% are delivered late, by 1
     second to 10 minutes.
   - **Watermark heartbeats:** it sends a heartbeat to every partition every 250 ms, so event
     time keeps moving when no events are due.
   - **Whole histories only:** each run replays every event of the chosen customers, so the
     final state can be compared with the v1 feature store.
2. **The PyFlink job** (`v2/streaming/flink_job.py`; Flink 1.20, Kafka 3.8, in Docker) has two
   steps.
   - **Step 1: features, per customer (keyed state).**
     - Each customer's running totals live in Flink state (`stream_features.py`): profile,
       maximum affordability ratio, approved count, and sums for the averages and category
       shares.
     - The features follow the SQL exactly, including PostgreSQL's rounding (exact decimals,
       half away from zero).
     - Every event is applied as soon as it arrives. Kafka keeps one customer's events in order
       (one key, one partition).
   - **Event time and watermarks decide lateness.** An event is flagged `late` when it is:
     - behind the watermark (2 seconds behind the newest event time); or
     - older than an event already applied for that customer.

     Late events are still applied, because the final state is sums, counts and a maximum and
     so does not depend on order. Only their own point-in-time score is marked.
   - **Duplicates** are recognised by event id, dropped, and recorded.
   - **An application that arrives before its customer's profile** is held, then scored when
     the profile arrives (`delayed_profile`).
   - **Step 2: scoring in micro-batches.**
     - The unchanged champion (`dashboard/demo_data/model`, byte-identical to the governed
       package) scores up to 500 applications or 100 ms at a time, over 8 bucket keys.
     - It uses the 0.0657 cutoff and the API's decision labels.
     - XGBoost runs on one thread per call, and the input is a plain array in the champion's
       checked feature order. Both changes are faster, with identical probabilities (measured
       below).
   - **Revised design.** The first design buffered each on-time event until the watermark
     passed it (one event-time timer per event). It was replaced after measurement:
     - it topped out at about 340 events per second here;
     - it added a fixed 2.4-second wait.
3. **Point-in-time features.** When an application arrives, the model sees the profile,
   everything before this application, and this order as the candidate.
   - This is how the v1 API builds a first-time applicant's features (`serving/features.py`),
     extended to customers with history. A unit test proves the two agree for a first
     application.
   - The historical approve/decline flag of the current application is never used to score it.
4. **Failure handling.**
   - **Checkpoints every 10 seconds** save all state and the Kafka read positions together.
     Flink restarts from the last checkpoint after a failure (fixed delay, up to 10 attempts).
   - **Output is at-least-once, and the sink's writes are idempotent:**
     - `ON CONFLICT DO NOTHING` on (run, event);
     - "update only if newer" on customer state;
     - Kafka positions committed only after the database commit.
5. **Storage.**
   - **PostgreSQL:** a new `v2_stream` schema (`v2/sql/phase4_stream.sql`). Every row is keyed by
     replay `run_id`, and per-event timestamps record each stage.
   - **A local audit folder laid out like an S3 bucket**
     (`bucket=sentinel-v2-audit/stream/run_id=…/dt=…/part-….jsonl`), git-ignored. It stands in
     for S3 until Phase 3.
   - **`replay_runs.mode` has `CHECK (mode = 'shadow')`,** so this phase cannot record a live
     decision.
6. **Checks** (`v2/streaming/stream_check.py`), per run:
   - **completeness:** every event processed once, every duplicate dropped;
   - **parity:** every customer's final state against `feature_store.customer_features`, feature
     by feature, within 1e-9;
   - **speed:** latency by stage, throughput and consumer lag;
   - **scoring parity:** Flink's probability against the champion on the host;
   - **drift:** PSI using v1's own function (`pipeline/monitor_drift.py`, read-only);
   - **gate:** the v1 Approval Gate status, read-only.
7. **Orchestration.** A new Airflow DAG, `sentinel_v2_stream_replay`, does four things:
   - checks the Flink job is running;
   - writes a replay request;
   - waits for the replay agent to finish it;
   - runs the checks.

   Airflow never talks to Kafka, so its image needs no new packages. The v1 drift DAG is
   unchanged.
8. **Shadow mode whatever the gate says.** The gate governs promotion, which this phase never
   attempts. The stream does not replace the v1 API, and Power BI, the API and Streamlit still
   read v1.

### Evidence (real runs, 30th October 2026; final design, 8 parallel workers)
Every run used the configured 1% duplicates and 2% late deliveries (up to 10 minutes late).

| Run | Events (customers) | Processed once | Duplicates dropped | Customers matching v1 | Scoring parity (max \|PD difference\|) |
|---|---:|---:|---:|---:|---:|
| `stream_10eps_final` | 1,803 (400) | 1,803 | 20 / 20 | 400 / 400 | 1.9e-10, 0 decisions differ |
| `stream_100eps_final` | 17,872 (4,000) | 17,872 | 183 / 183 | 4,000 / 4,000 | 3.0e-9, 0 decisions differ |
| `stream_1000eps_final` | 224,421 (50,000) | 224,421 | 2,191 / 2,191 | **50,000 / 50,000** | 1.0e-8, 0 decisions differ |
| `stream_100eps_failure_r2` | 17,872 (4,000) | 17,872 | 183 / 183 | 4,000 / 4,000 | 3.0e-9, 0 decisions differ |

**Speed.** These figures cover events delivered on schedule, excluding applications held for a
late profile. Send-to-processed times are in seconds.

| Rate | Kept up? | p50 | p95 | p99 | Last event processed |
|---|---|---:|---:|---:|---|
| 10/s | Yes | 3.1 | 5.7 | 6.2 | 7 s after the last send |
| 100/s | Yes | 4.3 | 6.9 | 7.8 | 6 s after the last send |
| 1,000/s | **No** | 220 | 367 | 379 | 382 s after the last send |

- **At 1,000/s,** 216,819 events were sent in 224 seconds and fully processed by second 606:
  about 360 events per second sustained, with a best 60-second rate of 549. The backlog peaked
  at 158,654 events sent but not yet stored. Nothing was lost, and parity held for all 50,000
  customers.
- **Where the time goes at 10 and 100/s** (medians): 1.9 to 2.2 s reaching the feature step,
  1.1 to 2.1 s from features to scored, and 0.3 s to storage. This is mostly PyFlink's default
  bundle time (up to 1 s per Python step). Shortening it lowered throughput (experiment B
  below), so throughput was chosen over latency.
- **Late events:** 2% were delivered late, and 72–77% of those were flagged late (22/30,
  276/359, 3,351/4,528). The rest arrived late but before any later event for the same customer
  and ahead of the watermark, so their point-in-time score was unaffected.
- **Applications held for a late profile:** 23, 226 and 3,110.

**Failure test** (`stream_100eps_failure_r2`, 100/s, 4,000 customers):
- **Flink worker:** the TaskManager container was killed 60 s into the stream. Flink restored
  the job from checkpoint 334, with all 24 tasks running again 115 s after the kill. That
  includes 15 s waiting for Docker before the script restarted the container.
- **Sink:** the sink container was killed 3 minutes later and restarted.
- **Result:** every event processed once, every duplicate dropped, and all 4,000 customers
  matching v1. The cost was latency during recovery (p50 42.5 s).
- **Restarts:** Docker's `unless-stopped` policy does not restart a container stopped with
  `docker kill` (it counts as manual), so the script restarted both, as an orchestrator would.

**Drift** (v1 PSI function and bands):
- **Final state against the training population** (modelable customers only): PSI at most
  0.0001 on the 50,000-customer run, which is expected given exact parity.
- **Point-in-time features against the training population:** the category shares show a
  *moderate shift* (`pct_high_risk_category` 0.126, `pct_medium_risk_category` 0.117 at 10/s;
  0.10 and 0.08 at 100/s; 0.096 and 0.078 at 1,000/s). The champion was trained on
  end-of-history aggregates, but at decision time a customer's share comes from fewer orders and
  is more extreme. This is training/serving skew that the v1 batch design could not show (v1
  already noted "true application-time underwriting is deferred").

**Approval Gate:** PASS in every run. The mode stayed shadow.

**Airflow:** the DAG ran end to end (run `manual__2026-10-30T03:09:07`, 16.8 minutes): it found
the Flink job running, requested a 50-customer replay at 10/s, waited for the agent, then ran
the checks inside Airflow. Every event was processed once, every duplicate dropped, and 50 of 50
customers matched v1. Scoring parity was recorded as skipped there, because Airflow's image has
scikit-learn 1.9.0 and XGBoost 3.4.0, not the champion's versions.

**How the design was reached** (all runs real; the failures are kept in `v2_stream.replay_runs`
with notes):

| Step | What was run | Result | What it showed |
|---|---|---|---|
| Smoke test | 60 customers, 50/s | 140 of 241 events processed | The watermark stops when the stream stops, so the last 2 s of buffered events never left. Fixed with heartbeats. |
| Scoring cost | Champion in the Flink image | 55 ms per single-row call | XGBoost thread start-up; 9 ms on one thread, 15 ms for 500 rows. Hence micro-batches. |
| `stream_1000eps_r1` | First design, 2 workers | about 108 events/s; 181,124 of 224,421 processed when stopped | Fell behind. 47k events were judged late because the source's idleness timeout let the watermark run ahead of unread events under backpressure. Idleness removed. |
| `stream_1000eps_r2`, `_r3` | 4 workers, larger state cache | 40–60 events/s; stopped | The scoring step re-read its whole waiting batch on every application (quadratic). Fixed with a running count; still slow. |
| Experiment A | Burst of 22,255 events (5,000 customers at 1,000/s), first design, Flink defaults | 339 events/s, complete, parity 5,000/5,000 | — |
| Experiment B | As A, with short bundles and buffers (20 ms, 10 ms) | 217 events/s | Low-latency settings cost throughput. Defaults kept. |
| Experiment C | Process on arrival, no per-event timers | 436 events/s, complete, parity 5,000/5,000 | The per-event timers were the larger cost. Design adopted. |
| Experiment H | C, 8 workers, 8 scoring keys, array input | 467 events/s | Final configuration. |

- **Noise:** the same configuration run twice gave 297 and 442 events per second (experiments F
  and G), so only large differences between experiments mean anything.
- **The ceiling:** about 400–470 events per second, whatever the setting tried.
- **Why:** every record crosses between Java and Python several times, and there is one Python
  process per slot.

**v1 unchanged:** `v1_manifest.py verify --full` passed after all Phase 4 runs. 66 tests pass
locally (9 new); lint is clean.

### Findings
1. **PyFlink 1.20 needs numpy below 1.25** (its Apache Beam dependency), while the project uses
   1.26.4. The Flink image uses 1.24.4. Scoring parity shows the effect on the champion:
   - probabilities differ from the host by at most 1.0e-8;
   - no decision differs.

   (Earlier smoke runs, scoring one row at a time from a DataFrame, matched exactly. The batched
   array path accounts for the last digits; it was not investigated further.)
2. **The Flink configuration file replaces the image's default,** which carries the Java 17
   module flags. Without them, job submission fails. They are copied into
   `flink-conf/config.yaml`.
3. **Flink's REST metrics refresh every 10 seconds.** I first read their 10-second jumps as a
   PyFlink fault. The per-event timestamps stored in `v2_stream` are the reliable measure.
4. **The first replay after a job starts is slower** (Python worker warm-up). Each reported set
   began with a small warm-up run.
5. **Windows tooling:**
   - Windows PowerShell 5.1 treats Docker's progress output as an error when
     `$ErrorActionPreference = "Stop"` (`stream.ps1` checks exit codes instead).
   - Git Bash rewrites container paths unless `MSYS_NO_PATHCONV=1` is set.
6. **The Kafka container keeps no volume.** `stream.ps1 stop` keeps topics; recreating the
   container (`teardown`, or a changed setting) deletes them, and `start` creates them again.
7. **The first version of the DAG failed on its first run.** A task argument named `run_id`
   collides with a name Airflow reserves. The request and the replay itself worked; only the
   waiting task failed. It was renamed `replay_id` and the DAG run again from the start.
8. **The first failure test did not test anything.** It sent `kill -9` to PID 1 *inside* each
   container, which Linux ignores, so nothing failed. It is recorded as a clean run, and the
   test was redone from the host.

### Limits
- **1,000 events per second is not sustained** on this laptop: about 360 sustained, against
  1,000 offered. The job falls behind and catches up without loss. Reaching it would need more
  workers on more machines, or moving the per-customer step out of Python (Java/SQL operators).
  That is designed only, not tested.
- **Latency of 3 to 4 seconds at low rates** is a PyFlink default trade-off, not a tuned
  real-time figure.
- **One machine.** Docker has 8 GB shared with the v1 Airflow stack; the fraud-detection stack
  was stopped during the runs.
- **The stream is v1's own synthetic history** on a compressed replay clock: no real customers,
  no live traffic. Rates are events per second of that replay.
- **One run per rate.** Run-to-run variation was not measured for the final runs; the
  experiments showed up to about 50% between identical bursts.
- **Lag figures are coarse.** Flink commits Kafka positions only at checkpoints, so the
  "committed" figure moves every 10 seconds; the "sent but not stored" figure is the better
  guide.
- **Power BI is not pointed at `v2_stream`.**

### Component status
| Component | Status |
|---|---|
| Kafka, Flink cluster, sink, replay agent (Docker) | Running (local) |
| PyFlink job: keyed state, event time, de-duplication, micro-batched scoring | Tested locally (real runs) |
| Checkpoint recovery after a lost TaskManager; idempotent sink after a crash | Tested locally (failure test) |
| Parity with the v1 feature store | Tested locally (50,000 / 50,000 customers) |
| `v2_stream` schema and audit folder | Running (local PostgreSQL and disk) |
| Airflow DAG `sentinel_v2_stream_replay` | Tested locally (one successful run; manual trigger only, paused) |
| 1,000 events per second sustained | Not achieved locally; scaling path designed only |
| Power BI on `v2_stream` | Designed only |

---

## V2-04 — Phase 2: The Feature Store at Scale (PostgreSQL, pandas, Spark)

### Challenge
v1 builds its feature store with one SQL script over 50,000 customers. Three questions follow
for a lender with more customers:
- Does the same logic hold up at millions of rows?
- Can Spark, the usual answer to scale, give exactly the same features?
- At what size does Spark start to pay off on one machine?

### Decision (owner-approved, 30th October 2026)
1. **Features at scale only.** No model is trained on the scaled data. A new dataset would mean
   re-deriving the feature set and cutoff under new V2 decisions (ground rule: v1 Decisions 18
   and 19 apply to v1 data only), which is a separate, larger piece of work.
2. **Scaled datasets made by v1's own generator, unchanged.**
   - `v2/phase2/generate_scaled.py` imports `data_generator/bnpl_data_generator.py` read-only. It
     changes two settings in memory only: the number of customers (50,000, 250,000, 1 million,
     2 million) and the random seed (2026, the owner's choice, so the data is clearly new and
     not a superset of v1's customers).
   - v1's `config.yaml`, code and save step are never touched.
   - **In chunks.** The generator holds each repayment row as a Python dictionary (about 40 GB
     for 2 million customers in one go, against 15.4 GB of RAM). So applications and repayments
     are generated in chunks of 100,000 customers, in 4 parallel processes. Chunk k uses seed
     2026 + 1000k, and chunk 0 uses exactly the single-run seed. The 50,000-customer dataset was
     checked identical to a single run (175,103 applications, 485,952 repayments).
   - Each dataset has a `dataset_version` (`scaled_synthetic_v2_seed2026_n<N>`) and a manifest.
   - **The data lives outside the repository and outside OneDrive** (`SENTINEL_V2_DATA_DIR`): about
     1 GB of Parquet.
3. **Three engines, one definition.**
   - **PostgreSQL (the reference):** v1's own `sql/02_feature_store.sql`, read-only, pointed at the
     dataset's own schema (`v2_scale_raw_n<N>`, with v1's column types via `CREATE TABLE (LIKE
     raw.*)`). That script begins with `DROP TABLE feature_store.customer_features`, so the
     rewrite refuses to run if any v1 schema name survives it (unit-tested).
   - **pandas:** one Python process (`pandas_engine.py`).
   - **Spark 3.5:** local mode in Docker, 8 cores, a 6 GB container (`spark_engine.py`,
     `feature_store_spark.sql`). The Spark SQL is a transcription of v1's SQL with five marked
     differences, all about getting PostgreSQL's exact answers (exact decimal inputs, averages
     over high-precision decimals, `DATEDIFF` for date subtraction).
4. **Exactness is the bar, not "close".** Every engine's table is compared with PostgreSQL's for
   every customer and all 63 columns (`compare.py`):
   - counts, dates, flags and text must be equal;
   - decimals within 1e-9;
   - NULL must match NULL.

   pandas computes every rounded average in whole units with integer arithmetic, rounding half
   away from zero as PostgreSQL does. Ordinary float rounding would round ties such as 75.125
   the other way.
5. **The proof first.** Before any timing counts, Spark and pandas must rebuild v1's own feature
   store exactly from v1's raw data (copied to Parquet with SELECTs only).
6. **Fair timing.**
   - **What is timed:** for pandas and Spark, read Parquet, compute, write Parquet. For
     PostgreSQL, the `CREATE TABLE AS`.
   - **Fixed costs are recorded separately:** loading into PostgreSQL, staging files into
     Spark's Docker volume, starting a Spark session.
   - **Local disk for every engine:** Spark reads a Docker volume, not a Windows folder share.
   - **Repeats:** 3 runs per engine and size; the median is reported.
   - **Quiet machine:** the fraud-detection stack was stopped during the runs.
7. **Results:** `v2_scale.benchmark_runs`, `.setup_costs` and `.parity`; MLflow experiment
   `sentinel_v2_phase2_spark` (one run per engine and size, tagged `dataset_version`);
   `v2/reports/phase2/`.

### Evidence (real runs `phase2_20261030T201352` and `phase2_20261030T213930`)
**Proof on v1's own data:** Spark and pandas each reproduced `feature_store.customer_features`
with 0 of 3,150,000 cells different.

**Parity on the scaled data:** every engine matched PostgreSQL at every size, with no cell
different.

| Customers | Applications | Repayment rows | Cells compared per engine | Cells different |
|---:|---:|---:|---:|---:|
| 50,000 | 175,103 | 485,952 | 3,150,000 | 0 |
| 250,000 | 874,990 | 2,424,729 | 15,750,000 | 0 |
| 1,000,000 | 3,501,153 | 9,710,535 | 63,000,000 | 0 |
| 2,000,000 | 6,998,911 | 19,431,048 | 126,000,000 | 0 |

**Time to build the feature store** (median of 3, seconds; the three runs in brackets):

| Customers | PostgreSQL | pandas (1 core) | Spark (8 cores) | pandas peak memory |
|---:|---:|---:|---:|---:|
| 50,000 | **2.7** (2.73–2.92) | **2.2** (1.75–2.52) | 24.6 (16.3–39.2) | 1.0 GB |
| 250,000 | 18.9 (18.3–20.4) | **12.9** (12.8–13.4) | 49.2 (45.8–84.7) | 3.5 GB |
| 1,000,000 | 82.5 (77.8–129.7) | **69.9** (52.6–73.3) | 190.7 (178.5–238.3) | 7.7 GB |
| 2,000,000 | **211** (197–321) | 302 (299–372) | 354 (288–385) | 9.1 GB |

**Fixed costs at 2 million customers:** loading into PostgreSQL took 354 s; staging for Spark
16 s; starting a Spark session 3.4 s.

**Where Spark pays off — the honest answer on this machine: not yet, by 2 million customers.**
- **Spark was slowest at every size.** At 50,000 its fixed overhead (scheduling, shuffles, a JVM)
  dominates: 24.6 s against 2.2 s for pandas.
- **Spark scales best.** From 1 to 2 million customers its time rose 1.9×, PostgreSQL's 2.6× and
  pandas' 4.3×. At 2 million it is within 17% of pandas, and pandas needed 9.1 GB of the
  machine's 15.4 GB.
- **On these trends,** pandas runs out of memory and Spark overtakes it somewhere beyond 2
  million customers. That point was not measured (owner's size limit) and is not claimed.
- **PostgreSQL, the existing v1 tool, was fastest at 2 million.** On one machine the database
  already scales further than either alternative here.
- **Spark's real advantages lie outside this test:** it would add machines, and it ran inside a
  fixed 6 GB container.

### Findings
1. **Spark could not read the first scaled datasets.** pandas had written `registration_date`
   as nanosecond timestamps, which Spark 3.5 rejects ("Illegal Parquet type: INT64
   (TIMESTAMP(NANOS,false))"). pandas and PostgreSQL had read it without complaint, so only
   Spark's first runs at 50,000, 250,000 and 1 million failed. They are recorded as failed in
   `v2_scale.benchmark_runs`.
   - **Fix:** the generator now writes dates as dates, matching v1's raw `date` type. The four
     existing customer files were rewritten with the same function (only the column type
     changes; values do not).
   - **Rerun:** Spark was rerun for those three sizes. At 2 million it had already picked up the
     corrected file.
2. **A common pandas pitfall cost most of its time.** Parquet dates arrive as Python `date`
   objects, so `min`/`max` on them run as a pure-Python loop: 24 of 27 seconds under the
   profiler on v1's data. Converted to datetime64, the build went from 21 s to 6 s.
3. **The generated data first landed inside OneDrive.** It was moved out before the benchmark,
   and the location is now set by `SENTINEL_V2_DATA_DIR`.
4. **The first run of each engine is slower** (cold caches): PostgreSQL 130 s against 78–83 s at
   1 million. Hence medians of 3.

### Limits
- **One machine, and the engines get different resources.** pandas uses 1 core, PostgreSQL its
  own parallel workers, and Spark 8 cores in a 6 GB Docker container on an 8 GB VM shared with
  the v1 Airflow stack. That is a realistic laptop comparison, not a tuned cluster benchmark.
- **Spark was not tuned** beyond 24 shuffle partitions and 4 GB of driver memory. Its exactness
  work (decimal averages) has a cost that was not measured separately.
- **pandas' peak memory is sampled every 50 ms,** so short peaks can be missed.
- **Three runs per size;** identical runs varied by up to 60% (PostgreSQL at 1 million).
- **Synthetic data from v1's generator,** with the same behaviour at every size. Real data would
  have skew (heavy customers, hot keys) that changes Spark's picture.
- **No model was trained on the scaled data.**

### Component status
| Component | Status |
|---|---|
| Scaled datasets (50k to 2M customers, seed 2026) | Running (local Parquet, outside the repo) |
| PostgreSQL build from v1's SQL (`v2_scale_raw_n*`, `v2_scale.customer_features_n*`) | Tested locally (real runs) |
| pandas engine, exact to PostgreSQL | Tested locally (0 cells different at every size) |
| Spark engine, exact to PostgreSQL | Tested locally (0 cells different at every size) |
| Benchmark results in `v2_scale` and MLflow | Running (local) |
| Spark on more than one machine | Designed only (Phase 3 territory) |
| Model training at scale | Not in scope (owner's decision) |

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 0.1 | 29th October 2026 | Log created, separate from the v1 log. Ground rules and plan recorded. Added V2-01: v1 frozen at tag `v1.0-evaluated` and fingerprinted (25 committed artefacts, 16 model files, row counts, 12 schemas, 103 MLflow runs, key metrics). 50 pytest tests. CI (lint new code, tests with coverage, v1 integrity, API container smoke test). One approved, additive change to `serving/app.py` (`SENTINEL_MODEL_DIR`), proven identical to the default. Honest coverage of 16% with its gaps listed. Finding: monitoring runs written by the Airflow container's MLflow are unreadable by the project's MLflow 2.13. |
| 0.2 | 29th October 2026 | Added V2-02 (Phase 1). PyTorch MLP challenger on the champion's 13 features, splits, calibration and cutoff; the splits were proven identical to v1. Stratified Gini 0.5329 against 0.5618, worse, with a 95% CI excluding zero. Out-of-time 0.5700 against 0.5611, a CI including zero. The challenger shares the champion's 100% established-file miss rate. The autoencoder is not a default signal (AUC 0.46/0.42). Captum against SHAP: Spearman 0.758. Approval Gate PASS. Outcome: challenger did not win, nothing promoted. Results are in the new v2_results schema; models in a separate registry entry. v1 is verified unchanged. Findings: PyTorch DLL import order on Windows; MLflow did not record torch in the logged model's requirements. |
| 0.3 | 30th October 2026 | Added V2-03 (Phase 4, PyFlink streaming, shadow mode). Kafka 3.8 and Flink 1.20 in Docker replay v1's synthetic history at 10, 100 and 1,000 events/s, with 1% duplicates and 2% late events (up to 10 minutes). Per-customer keyed state, event-time lateness, de-duplication, micro-batched scoring with the unchanged champion. Every run processed every event once, dropped every duplicate, and matched the v1 feature store for every customer (50,000 / 50,000 at 1,000/s); scores within 1.0e-8 of the host, no decision differed. It kept up at 10 and 100/s (p50 3.1 and 4.3 s), not at 1,000/s (about 360/s sustained, caught up without loss). It recovered from a killed TaskManager (checkpoint restore) and a killed sink with no loss or double counting. Point-in-time category features show a moderate shift against training (PSI 0.10–0.13). The design was revised on measured evidence (per-event timers replaced); the failed runs are recorded. New schema v2_stream; new DAG sentinel_v2_stream_replay. v1 verified unchanged. |
| 0.4 | 30th October 2026 | Added V2-04 (Phase 2, features at scale). v1's generator (unchanged, in-memory overrides, seed 2026) made 50k, 250k, 1M and 2M-customer datasets (up to 19.4M repayment rows). The feature store was built by PostgreSQL (v1's own SQL, guarded against touching v1), pandas and Spark 3.5. Spark and pandas rebuilt v1's feature store exactly, and every engine matched PostgreSQL in every cell at every size (126M cells at 2M). Median build at 2M: PostgreSQL 211 s, pandas 302 s (9.1 GB), Spark 354 s (8 cores). Spark was slowest at every size but scaled best (1.9× from 1M to 2M against 4.3× for pandas); it did not pay off by 2M on one machine. Findings: nanosecond Parquet timestamps that Spark cannot read (fixed, Spark rerun), a pandas date pitfall, data moved out of OneDrive. v1 verified unchanged. |

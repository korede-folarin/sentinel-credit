# Sentinel Credit
## Engineering Decision Log

**Project:** Sentinel Credit — Research-Informed CreditOps
Architecture for BNPL Lending
**Author:** Sentinel Credit Project
**Version:** 3.31
**Last Updated:** 15th October 2026 (v3.31)
**Status:** Living document — updated as decisions are made

---

## Purpose

This document records every significant engineering
decision made during the design and build of
Sentinel Credit.

For each decision it records the challenge that
motivated it, the research that informed it, the
engineering implication that followed, the design
decision that was made, and the regulatory context
that validates it.

Regulation appears last in each entry deliberately.
Engineering problems were identified first. Research
was consulted. Decisions were made. Regulatory
alignment was then confirmed. The FCA did not
prescribe this architecture. Reading FCA publications
helped clarify what a well-governed credit system
should look like in practice.

This document should be read alongside the
[Project Rationale](project_rationale.md). The Project
Rationale explains why Sentinel Credit exists, the
engineering questions it explores, and the research
that informed its overall design. This Engineering
Decision Log records how those questions were
translated into specific technical and architectural
decisions.

---

## Relationship to the Project Rationale

The Project Rationale provides the high-level
motivation for choosing BNPL, using synthetic data,
and building a connected CreditOps architecture. It
also holds the research and regulatory reference
material: calibration assumptions, the regulatory
source mapping and the regulatory reference summary.

This document is the record of engineering decisions.
Each entry focuses on the implementation challenge,
the evidence consulted, the engineering implication,
the resulting design decision, and its regulatory
context. The short research and regulatory notes
inside each entry stay here because they explain that
specific decision. Material that supports many
decisions lives with the document it belongs to (see
[Reference Material](#reference-material)).

Related documents:
- [Project Rationale](project_rationale.md): why the
  project exists, research and regulatory foundations
- [Validation Results](validation_results.md):
  synthetic-data validation results and methodology
- [Model Evaluation Report](model_evaluation_report.md):
  evaluation results and metric definitions
- [Audit Findings](audit_findings_2026-10-12.md): the
  October 2026 audit of the log against the code

---
## Decision 1 — Synthetic Data Generation

### Challenge
No publicly available dataset represents UK BNPL
customers. Kaggle credit datasets are either
PCA-transformed, destroying feature correlations
or from different markets with different
regulatory contexts. Neither can simulate a
realistic UK BNPL lending environment.

### Research Observation
Khalili et al. 2026 (Engineering Reports) confirmed
using real BNPL data that BNPL customers are
predominantly thin-file and that digital behaviour
variables are statistically significant default
predictors — informing which features our synthetic
population should contain.

Kiran and Saravana Kumar 2024 (IEEE Access) tested
11 synthetic data generators and identified CTGAN
and PATECTGAN as strongest for tabular
classification — providing a benchmark for our
validation comparison.

### Engineering Implication
A realistic synthetic dataset enables reproducible
experimentation across the complete ML lifecycle
without exposing personal information. Features
must reflect real BNPL customer characteristics —
thin-file indicators, affordability ratios,
vulnerability signals, and alternative data signals.

### Design Decision
Develop a synthetic BNPL dataset using a parametric
domain-expert generator calibrated against published
regulatory and industry statistics. This is a
deliberate methodological choice — using a learned
generator trained on a PCA-transformed proxy dataset
would import correlations from a different market,
not UK BNPL customer behaviour.

The generator is calibrated against a set of
documented assumptions derived from published
statistics. These assumptions support transparent,
domain-faithful experimentation rather than claim
precise representation of the UK BNPL market.
Detailed calibration assumptions and derivations
are provided in [Calibration Assumptions](project_rationale.md#synthetic-data-calibration-assumptions).

The dataset supports feature engineering, model
development, A/B testing, drift monitoring,
explainability, and governance experimentation
across the full platform.

### Regulatory Context
The synthetic data generation approach is informed
by the joint guidance published by the FCA, the
ICO, and the Alan Turing Institute on responsible
use of synthetic data (June 2023, updated June
2026), which defines three validation dimensions —
utility, fidelity, and privacy — implemented in
validate_synthetic.py.

*Having generated the dataset, the next challenge
was proving it was statistically sound and fit for
purpose.*

---

## Decision 2 — Synthetic Data Validation Framework

### Challenge
Generating synthetic data is not sufficient.
Without validation there is no basis for trusting
that the dataset preserves the statistical
properties needed for meaningful model development,
or that it does not introduce re-identification
risks. No real UK BNPL dataset exists for direct
comparison.

### Research Observation
The FCA, ICO, and Alan Turing Institute (June 2023,
updated June 2026) defined three validation
dimensions: utility, fidelity, and privacy. The
paper identified lack of industry consensus on
validation as a critical barrier to synthetic data
adoption in financial services.

Kiran and Saravana Kumar 2024 provided empirical
validation methodology using Hellinger distance,
KL divergence, and Euclidean distance to compare
synthetic and real distributions.

### Engineering Implication
Validation must be structured, documented, and
repeatable across all three dimensions. The absence
of a real UK BNPL reference dataset must be
acknowledged and handled transparently.

### Design Decision
Implement a validation module in
validate_synthetic.py covering:

Fidelity: KS tests, Hellinger distance, and
Jensen-Shannon divergence against published
regulatory statistics used as calibration ground
truth.

Utility: Train-on-Synthetic Test-on-Real (TSTR)
Gini comparison using Home Credit Default Risk as
an imperfect proxy. Limitation explicitly
documented: PCA-transformed, different market.
Tests directional signal preservation not exact
fidelity. Where no real reference file is present,
an internal held-out check runs instead, and the
result explicitly records that real TSTR was not
performed — see Decision 10 for how this branch
was completed.

Privacy — singling-out and nearest-neighbour
distance analysis for spontaneous re-identification
risk. No differential privacy applied — generator
uses aggregate published statistics not
individual-level data.

CTGAN benchmark — **planned, not yet implemented.**
The intent is to compare the parametric generator's
output against CTGAN trained on Home Credit, as
supporting evidence of realistic statistical
structure. Descoped for now to avoid spending time
on a supporting check before the core pipeline is
complete; revisit once the platform is otherwise
finished. `validate_synthetic.py` currently
implements Fidelity, Utility, and Privacy only.

Internal consistency — five generation runs with
different seeds. KS tests confirm stable output.

Detailed methodology provided in [Validation Methodology](validation_results.md#validation-methodology).

### Regulatory Context
Directly implements the three dimensions defined
in the FCA/ICO/Alan Turing Institute guidance.
Privacy assessment follows the FCA risk-based
model — acknowledging that zero risk equals zero
utility and documenting the trade-off explicitly.

### Addendum — Validation Config Restructuring (30th July 2026)
Privacy thresholds were hardcoded directly in
`singling_out_check()` (0.20) and
`nearest_neighbour_check()` (0.01) in
`validate_synthetic.py`, while a single
`privacy_threshold: 0.05` sat unused in
`config.yaml` — an ambiguous key that could not
correctly govern two structurally different checks
with two different natural ceilings. Replaced with
two explicit config keys,
`singling_out_threshold` and
`near_duplicate_threshold`, read directly by the
code. This changes only where the acceptance
criteria are written down, not the underlying data
or the computed rates — verified by an actual
validation run producing unchanged results (7.90%
singling-out, 0.00% near-duplicate, both passing).

`gini_tolerance` was previously declared in config
but never read by any function. It is now used: the
absolute gap between the synthetic and real-data
Gini scores is computed and reported against this
tolerance in the TSTR utility check, as supporting
context rather than the sole pass/fail gate, since a
genuinely harder or easier real-market dataset can
widen the gap without indicating a broken generator.
`n_runs` and `ks_threshold` remain declared but
unimplemented, tied to the internal consistency
check noted as planned, not done, above.

### File Name Note (audit, 12th October 2026)
`validate_synthetic.py`, named throughout this
decision and in Appendices B/D, is
`validation/validation.py` in the repository.

*With validated data available, the next decision
was how to engineer features consistently across
training and production.*

---

## Decision 3 — PostgreSQL Feature Store on AWS RDS

### Challenge
Feature engineering implemented differently in
training and scoring environments produces
inconsistent predictions — a failure mode known
as training-serving skew. Notebook-based feature
engineering cannot solve this at production scale.

### Research Observation
Immaneni 2022 documents that automating deployment
pipelines and implementing systematic processes
in financial services ML ensures consistent model
performance while maintaining compliance standards.
Training-serving skew is a widely documented
failure mode in production ML systems.

### Engineering Implication
Centralised feature management improves
consistency, traceability, and reproducibility
across the model lifecycle. Features built at the
database level are available to all components
without re-implementation.

### Design Decision
Implement a PostgreSQL feature store on AWS RDS
containing 30+ engineered features built in SQL.
Features include affordability ratios, payment
velocity, Weight of Evidence binnings,
vulnerability composite scores, and merchant
risk signals. SQL-based features run identically
in training and production — eliminating
training-serving skew by design.

### Regulatory Context
The feature store supports the documentation and
reproducibility expectations described in PRA
SS1/23 Principle 3. It does not itself satisfy
regulatory requirements — it provides the
infrastructure that makes documentation and
consistency achievable in practice.

### Implementation Status (audit, 12th October 2026)
**Not on AWS RDS.** The feature store is a local
PostgreSQL 18 instance (`localhost:5432`). Nothing in
this project is deployed to AWS. The design above
(SQL-built features shared by training and scoring)
is implemented; the hosting is not. Any move to RDS
needs its own decision entry.

*With consistent features available, the next
decision was which models to build and how to
make the performance-interpretability tradeoff
explicit.*

---

## Decision 4 — Three-Model Comparison

### Challenge
Credit risk modelling involves balancing predictive
performance with interpretability, operational
requirements, and fairness. Choosing a single model
without comparison leaves the tradeoffs
undocumented and the model selection indefensible
to regulators or stakeholders.

### Research Observation
Demajo et al. 2020 (University of Malta)
demonstrated XGBoost achieves state-of-the-art
performance on credit scoring datasets and that
different people in different situations require
different types of explanation — global, local
feature-based, and local instance-based. This
informed both model selection and explainability
design.

Logistic regression has historically been preferred
in regulated credit scoring because coefficients
have direct interpretable meaning. This is a widely
documented industry practice.

### Engineering Implication
Comparing models makes selection evidence-based
and the comparison itself becomes audit evidence.
Fairness must be evaluated alongside performance
— a model accurate on average but systematically
less accurate for thin-file or vulnerable customers
would cause disproportionate harm.

### Design Decision
Benchmark three models:

Logistic Regression — regulatory baseline.
Coefficients are directly interpretable and
auditable. Makes adverse action explanations
easier to generate because coefficients have
directly interpretable meaning and can be
communicated without relying on post-hoc
explanation techniques.

Decision Tree — interpretable rules. Reveals
structurally how feature overreliance errors
occur — the error type identified in Belton
et al. 2025.

XGBoost — performance champion. Higher predictive
accuracy per Demajo et al. 2020. Requires SHAP
values for explanation.

Each model evaluated on predictive performance,
calibration, business threshold performance,
stability, and fairness. Metric definitions
provided in [Metric Definitions](model_evaluation_report.md#metric-definitions).

### Regulatory Context
Comparative evaluation with documented metrics
supports transparent model validation. PRA SS1/23
Principle 3 requires models to be fit for purpose
and documented. Benchmarking three models with
documented metrics provides evidence that model
selection was deliberate.

### Addendum — Modelling Scope: PD Only (30th July 2026)
This platform models Probability of Default (PD).
Loss Given Default (LGD) and Exposure at Default
(EAD) — the other two components of Expected Credit
Loss under IFRS 9 (Expected Loss = PD × LGD × EAD) —
are out of scope. The generator does not currently
model post-default recovery behaviour (partial
recovery over time, collections success rate,
write-off timing), which an honest LGD build would
require; `outstanding_balance_gbp` exists in
`raw.repayments` but nothing simulates what happens
to it after default. Building LGD/EAD without that
groundwork would risk the same "stated as
implemented but isn't real" problem corrected
elsewhere in this log (Decision 2/3.3). Stated here
explicitly as a scope boundary, not an omission.

*Selecting a champion is not the end of the
modelling lifecycle. Credit portfolios and customer
behaviour change over time. The next challenge was
ensuring the champion continued to perform reliably
after deployment.*

---

## Decision 5 — Drift Monitoring with PSI Thresholds

### Challenge
Customer behaviour and economic conditions change
after deployment, causing model performance to
deteriorate over time. A model that performed well
at deployment can silently degrade without
continuous monitoring.

### Research Observation
Kurian and Allali 2024 (Journal of Data Information
and Management) formally derived PSI as a variant
of KL divergence and validated thresholds in a
banking credit score scenario:
- Below 0.10: stable
- 0.10 to 0.25: monitor
- Above 0.25: immediate action required

The paper demonstrated that ETL processes alone
cannot detect content drift.

### Engineering Implication
Monitoring must be integrated into normal platform
operations. Threshold breaches should trigger
automated governance workflows — not just alerts.

### Design Decision
Three types of drift monitored continuously:

Data drift — input feature distribution changes.
PSI per feature using Kurian and Allali 2024
thresholds.

Concept drift — relationship between features and
default outcomes changes. Tracked by comparing
actual default rate versus predicted probability
of default across cohorts.

Prediction drift — score distribution changes
over time. PSI on model output scores.

Scheduled cadence: daily PSI checks, weekly
cohort comparisons, monthly full revalidation
including fairness metrics. Breach triggers
automatic escalation — not just notification.

Config.yaml:
  psi_threshold: 0.25
  drift_alert_threshold: 0.10

### Regulatory Context
Continuous monitoring supports supervisory
expectations for ongoing model oversight. FCA
Mills Review July 2026 confirms this is a core
expectation for AI in retail financial services.
PRA SS1/23 Principle 4 requires ongoing
independent validation — automated monthly
revalidation implements this continuously.

### Implementation Status (audit, 12th October 2026)
Decision 20 implemented part of this design:
- **Built:** data drift (feature PSI), data-quality
  checks, and a portfolio bad-rate delta, run by the
  Airflow DAG while the local stack is up.

**Planned, not built:**
- concept drift
- prediction drift (PSI on model scores)
- weekly cohort comparisons
- monthly revalidation including fairness
- automatic escalation

The `psi_threshold` / `drift_alert_threshold` keys
shown above do not exist in `config.yaml`. The PSI
bands live under `feature_governance.thresholds`.

*Detecting drift is only useful if it triggers
the right response. That required connecting
monitoring to governance automatically.*

---

## Decision 6 — Closed-Loop CreditOps Architecture

### Challenge
Production ML activities — monitoring, validation,
governance, deployment — are often implemented as
separate processes owned by separate teams. The FCA
CRA multi-firm review May 2026 highlights examples
of governance weaknesses including surveillance
gaps and disconnected processes, reinforcing the
importance of continuous oversight. Manual handoffs
between teams introduce delay, inconsistency, and
gaps in the audit record.

### Research Observation
Immaneni 2022 demonstrates that automating
deployment pipelines and continuous monitoring
enables financial institutions to ensure consistent
model performance while maintaining compliance.

Kurshan et al. 2020 (JP Morgan AI Research)
identified multiple structural challenges in AI
model governance and proposed a self-regulating
framework with continuous monitoring and
human-in-the-loop alerting. Governance reviews
under manual processes take over a year.

Brahmandam 2025 demonstrated that MLOps automation
in financial services improved model performance
outcomes, suggesting that automated retraining can
be valuable in appropriately governed financial
ML systems.

### Engineering Implication
Event-driven workflows reduce operational friction
by connecting monitoring, governance, and lifecycle
management while preserving human oversight for
high-impact decisions.

### Design Decision
Implement a closed-loop CreditOps architecture
where monitoring events trigger orchestration
pipelines, governance records are automatically
updated, and model lifecycle events are
version-controlled through MLflow before
deployment approval.

Eight steps:

1. Drift detected — PSI breach triggers Airflow
2. Airflow DAG runs — daily, weekly, monthly
3. Error classifier — identifies error type
4. Adaptive LLM engine — three audience outputs
5. Human approval gate — deployment programmatically
   prevented until an authorised reviewer approves
   promotion
6. MLflow governance log — approver, timestamp,
   version recorded
7. Audit trail export — PDF for regulatory review
8. Model deployed — BentoML, Docker, AWS

Steps 1 to 4 automated. Step 5 mandatory
human-in-the-loop. Steps 6 to 8 complete
automatically after approval.

### Regulatory Context
This architecture supports governance,
accountability, and monitoring principles in FCA
and PRA publications without claiming to replicate
any specific firm's implementation. The FCA did
not prescribe this architecture — reading FCA
publications helped clarify what well-governed
credit systems require.

### Implementation Status (audit, 12th October 2026)
| Step | Status |
|---|---|
| 1. Drift detected | **Built** (Decision 20) |
| 2. Airflow DAG | **Built** (daily setting only; runs while the local stack is up, Decision 24) |
| 3. Error classifier | Planned, not built |
| 4. Adaptive LLM engine | Planned, not built |
| 5. Human approval gate | **Partly built**: typed `PROMOTE` confirmation plus an automated evidence gate (Decision 23); no approver identity recorded |
| 6. MLflow governance log | **Partly built**: Model Registry aliases (Decision 21); no approver or rationale recorded |
| 7. Audit-trail PDF export | Planned, not built |
| 8. Deploy (BentoML, Docker, AWS) | Planned, not built. The scoring API runs locally only. |

*With the governance loop in place, the next
decision was how to make individual decisions
explainable to three different audiences.*

---

## Decision 7 — Adaptive Explainability Engine

### Challenge
Different stakeholders require different
explanations for ML credit decisions. A single
explanation format is technically insufficient
and — as published research demonstrates —
potentially harmful to consumer outcomes when
the wrong format is applied.

### Research Observation
Demajo et al. 2020 established that different
people in different situations require different
explanation types — global, local feature-based,
and local instance-based.

Belton et al. FCA Research Note February 2025.
8,860 UK adults. Randomised controlled trial.
Providing more detailed explanations decreased
participants' ability to identify model errors
by up to 20 percentage points. Three error
scenarios:

Scenario 3 — data input error: features-based
explanation most effective.

Scenario 4 — feature overreliance: showing
decision rule caused 18pp accuracy decrease.

Scenario 5 — missing feature: combination
genres caused 20pp accuracy decrease.

Conclusion: genre should be selected based on
error type and audience — not applied uniformly.

### Engineering Implication
Error type must be identified before genre is
selected. The same explanation that helps
consumers spot a data error actively prevents
them from spotting a model logic error.

### Design Decision
Detect error type first. Select appropriate
genre. Generate three audience outputs.

Data input error — features-based explanation.
Feature overreliance — data-centric only, no
decision rule shown.
Missing feature — data-centric, human review
triggered.
Clean decision — combination explanation
appropriate.

Three outputs per decision:

Customer — plain English, genre matched to
error type. Supports consumer ability to
challenge incorrect decisions.

Risk officer — full SHAP values, error
classification, escalation flag, model version.

Regulatory — structured documentation that
could support regulatory review and internal
audit activities. Methodology version, approver
ID, divergence flag, full decision lineage.

### Regulatory Context
FCA Consumer Duty PS22/9 requires understandable
information supporting informed decisions. The
adaptive approach ensures explanation
appropriateness rather than just volume. GDPR
right to explanation and ECOA principles cited
in Demajo et al. 2020 further support
audience-appropriate explanations in credit
decisions.

### Implementation Status (audit, 12th October 2026)
**Planned, not built.** The SHAP foundation exists
(Decision 14). No error-type detection, genre
selection, or three-audience output exists in the
code; Decision 14 already records the audience layer
as "not yet built". The design above is the intended
next layer, not a description of current behaviour.

**Update, 15th October 2026 (Decision 30): partly built.** The customer
output now exists for new applicants. It gives plain-English reasons in
the features-based genre, with no decision rule shown, from the API, the
dashboard and the public site. Error-type detection, genre selection,
the regulatory output and existing-customer explanations are still not
built.

**Update, 15th October 2026 (Decision 31): built.**
- Three error types are detected (data input, missing information,
  over-reliance), with rules computed from the training data and
  thresholds declared in config.
- The genre is chosen to match. A detected issue means "Risk officer
  review required", stated rather than routed.
- Three outputs are returned for new applicants and existing customers
  alike: applicant, risk officer, decision record.
- Still not built: an approver identity in the record (Decision 23),
  and user testing of the genres.

*With explainability producing three audience
outputs, the next challenge was ensuring every
model version and governance decision was
fully traceable.*

---

## Decision 8 — Model Registry and Governance

### Challenge
Without centralised version control, it is
impossible to determine which model made which
decision. Champion/challenger testing without a
registry produces no audit trail of why the
champion was chosen or who approved promotion.

### Research Observation
Kurshan et al. 2020 identified version
traceability and governance audit trails as
structural challenges in AI model governance
and proposed systematic lifecycle tracking as
part of the self-regulating AI framework.

Immaneni 2022 demonstrates that systematic
tracking in deployment pipelines reduces
downtime and improves regulatory alignment.

### Engineering Implication
Every model decision must be traceable to a
specific version, training run, and approval
record. The registry is the governance
evidence base.

### Design Decision
Use MLflow to manage champion/challenger models,
versions, approvals, and lifecycle metadata.

A new model must outperform the champion on
Gini, KS, Precision-Recall AUC, and Expected
Calibration Error, and pass fairness checks,
before the human approval gate opens.

Every experiment logs parameters, metrics, data
version, feature importance, fairness scores,
and training timestamp. Model registry records
version, approver, timestamp, performance delta,
methodology divergence, and rejection reason
where applicable.

### Regulatory Context
PRA SS1/23 Principle 1 requires model
identification and inventory — the MLflow
registry is that inventory. Principle 6 requires
internal audit evidence — MLflow logs and audit
export provide that evidence.

### Implementation Status (audit, 12th October 2026)
- **Built:** MLflow tracking of every training and
  evaluation run, and the alias-based Model Registry
  (Decision 21).
- **Superseded without note until now:** Decision 21
  narrowed the promotion rule to a material Gini win on
  both splits. KS and PR-AUC are reported as
  corroborating evidence. ECE and fairness are not
  promotion criteria.
- **Planned, not built:** registry records of approver,
  methodology divergence and rejection reason, and the
  audit export.

*With governance records in place, the next
decision was how to ensure no model reaches
production without explicit human review.*

---

## Decision 9 — Human Approval Gate

### Challenge
Automated deployment alone is insufficient for
high-impact credit decisions. A model that has
drifted, failed fairness checks, or diverged
from documented methodology must not reach
production without human review.

### Research Observation
Adari and Sarabu 2021 established that
human-in-the-loop compliance is mandatory for
high-risk decisions including loan rejections,
and that three lines of defence require
independent challenge at each governance level.

Kurshan et al. 2020 proposed human-in-the-loop
alerting as a core component — automated
monitoring should surface issues to humans for
decision, not resolve them autonomously.

PRA SS1/23 Principle 4 requires ongoing,
independent, and effective challenge to model
development and use. The approving body must
ensure validation recommendations are actioned.

### Engineering Implication
Governance must include an explicit approval
checkpoint. Approval must be recorded with
identity and timestamp to constitute audit
evidence. The gate must programmatically prevent
deployment — not just notify.

### Design Decision
Human approval gate programmatically prevents
BentoML deployment until an authorised reviewer
approves promotion. Cannot be bypassed.

Approval logged in MLflow: approver ID,
timestamp, model version, performance delta,
fairness results, decision rationale.

Rejection returns model to challenger status
and logs rejection reason.

Three lines of defence in code:
First line — data science team builds and
documents.
Second line — risk function independently
reviews and approves.
Third line — audit trail provides evidence
for review.

Important clarification: the human approval
gate provides accountability, challenge, and
auditability. It does not automatically ensure
compliance, fairness, or good consumer outcomes.
Those are properties of the model and governance
process as a whole. The gate provides a mandatory
checkpoint where a qualified person reviews the
evidence. That review, properly conducted, creates
the conditions for compliance and fairness.

### Regulatory Context
SR 11-7 Federal Reserve requires effective
challenge by individuals independent of model
development. PRA SS1/23 Principle 2 requires
independent validation and senior management
ownership. Principle 4 requires the approving
body to ensure validation recommendations are
actioned before deployment.

### Implementation Status (audit, 12th October 2026)
**Partly built:**
- `promote_champion.py --apply` refuses unless the
  evidence supports promotion, and then requires typing
  `PROMOTE`.
- An automated evidence gate (Decision 23) blocks on
  stale or drifting monitoring.

**Not built:**
- BentoML deployment (there is nothing to prevent
  deploying, because nothing is deployed)
- approver ID, rationale and rejection reason logged
  to MLflow
- separate first and second line roles

The current gate is a single-person confirmation, not
independent challenge.

*With every deployment gated on human review,
the final open question was whether the
underlying data itself represented default
correctly — a gap that surfaced once the
database was rebuilt against real repayment
behaviour.*

---

## Decision 10 — Hazard-Based Default Modelling

### Challenge
The original generator inverted causality. It drew a
`default_probability` per transaction first, then wrote
"missed payment" and "defaulted" statuses into the
repayments table to match that pre-decided outcome. In
real BNPL lending, causality runs the other way: a customer
takes credit, hits difficulty, misses a payment, misses
more, and only then defaults. Missed payments are the early
warning signal that precedes and predicts default. With
default assigned first, every repayment status was
downstream of the answer rather than a genuine predictor of
it — a structural leakage problem, not a data quality one.
This also left `run_utility`, the TSTR validation function,
incomplete: it handled the case where a real reference file
is present but had no `else` branch for its absence, so it
silently returned `None` in every run rather than recording
that real TSTR had been skipped.

### Engineering Observation
A value assigned in advance cannot be distinguished, by any
downstream model, from a signal that genuinely preceded the
outcome in time. For repayment behaviour to be usable as a
feature at all, default had to become the emergent result of
that behaviour, not an input fixed before it.

### Engineering Implication
Default needed rebuilding as a process, not a label, and the
schema, generator, calibration, and validation logic all had
to change together: the repayments table needed to carry the
process itself — arrears opening, deepening, and resolving
over time — not just its endpoint.

### Design Decision

**The state machine.** `default_probability` was removed
entirely, from both the generator and the schema. Every
transaction now moves through an explicit state machine:
`CURRENT` (no unpaid instalments), `DELINQUENT` (one or more
unpaid, oldest under 90 days past due), `DEFAULT` (oldest
unpaid has reached 90 DPD — absorbing, no return). At each
due date the customer pays, pays late, or misses; a miss
moves `CURRENT` to `DELINQUENT`. Each subsequent cycle a
delinquent customer either fully cures (clears all arrears,
returns to `CURRENT`), partially cures (clears the oldest
arrear only, stays delinquent), or cures nothing (arrears
deepen). Payments always clear the oldest unpaid instalment
first, so days-past-due tracks the age of the next-oldest
outstanding item rather than resetting to zero on any
payment — a chronically-behind customer can stay
persistently delinquent without necessarily defaulting,
which matches real repayment behaviour.

**Two separate hazards, not one.** `miss_hazard()` — the
per-instalment probability of missing — is driven by credit
risk factors: thin-file (×2.0), vulnerability (×1.5),
affordability ratio above 0.35 (×1.3), credit score below
550 (×1.4), and the merchant risk multiplier, capped at
0.60. `financial_resilience()` is a separate latent
variable measuring capacity to recover once behind, built
from savings buffer, disposable income buffer, and
employment stability. The two are deliberately not inverses
of each other: a customer can be likely to miss yet able to
recover, or unlikely to miss yet unable to recover if they
do. `resolve_arrears()` combines resilience with an
arrears-depth cure table (full/partial/no-payment
probabilities that shrink as depth increases) and a DPD
decay factor, so cure chances halve every
`dpd_decay_halflife` days the oldest arrear ages.

**The default and SICR thresholds, with their limits stated
honestly.** `default_dpd_threshold: 90` follows the CRR/EBA
90-days-past-due framework in principle; the config
explicitly notes that full regulatory implementation also
considers materiality and unlikeliness-to-pay, which are out
of scope here. `sicr_dpd_threshold: 30` is retained as an
IFRS 9-style delinquency indicator, not as default — IFRS 9
treats 30 DPD as a rebuttable presumption rather than an
automatic trigger, and this generator uses it as a fixed
threshold, a simplification the config documents rather than
hides.

**The calibration investigation.** Once default became
emergent, the portfolio default rate had to be tuned rather
than set. It initially hit a hard floor near 1.2% that no
amount of hazard or cure-generosity adjustment could break.
`diagnose_default_timing()` traced the cause: 42% of all
defaults were customers who missed their *final* instalment,
leaving zero remaining due dates at which to cure before the
schedule simply ended. Real BNPL collections continue past
the contractual schedule; this generator hadn't modelled
that. The fix was post-schedule collection cycles — arrears
continue accumulating cure opportunities every 30 days past
the final due date, until default or the observation window
closes. The floor disappeared once this was in place.

Calibration then proceeded as a two-stage sweep using
`calibrate_grid()`: a wide sweep across `base_miss_hazard`,
`cure_scale`, and `dpd_decay_halflife` to locate a plausible
region, followed by a narrower sweep within it
(`calibrate.py`). Each cell was judged on two competing
constraints, not one — the delinquency and default rate
against published references, and the number of positive
(defaulted) cases retained, since a calibration that hits
the target rate but leaves too few defaults to train on has
failed at the dataset's actual purpose; under roughly 300
positives was treated as making evaluation unstable. The
landed values: `base_miss_hazard: 0.005`,
`cure_scale: 1.30`, `dpd_decay_halflife: 45`, producing a
0.93% observed default rate, ~5% ever-missed (matching
Klarna's published ~5% late-fee figure), and 796 defaulted
customers.

**The default rate was never a target.** `base_default_rate`
(the raw 0.4% Klarna figure) is not read anywhere in the
current generator — every function that used to consume it
was replaced when the hazard engine was built. It is kept in
config only as a documented calibration reference, not a
value the code depends on. This distinction matters: every
other parameter in the population and repayment process was
chosen independently from published UK statistics, so the
emergent default rate is what falls out of those
independently-reasoned parts, not a number the system was
fitted to hit. It sits defensibly between Klarna's 0.4% (a
different market segment, different metric — a value-weighted
loss rate, not a transaction-count default rate) and the
Woolard Review's finding of roughly one in ten BNPL customers
in arrears.

**The delinquency spectrum is coarse by product design.**
Roughly 80% of delinquent transactions sit in the 30-44 DPD
band, with a mean episode length of 1.23 cycles — most
customers who miss either cure immediately or head to
default, with little in between. This is structural, not a
flaw: payments are monthly, so DPD only advances in 30-day
steps at each due date, and the dominant instalment plan
(Pay in 3) gives a customer who misses only two remaining due
dates to cure before hitting 90 DPD. A longer-tenor product
would produce a smoother ladder; this one produces a coarse
one because that is what a short, monthly product genuinely
does. `max_dpd` and `reached_sicr` remain real signal, but
carry less variation than they would on a 12-month loan, and
that limit is worth stating alongside the feature, not
discovering later.

**The A/B mechanism silently broke, and was fixed at the
causal layer.** `group_B_default_uplift` had been designed to
add flat to the old `default_probability` draw; once that
draw was removed, the key had nothing to attach to and Group
B showed no measurable difference from Group A. It was
replaced with `group_B_hazard_multiplier: 1.60`, applied
multiplicatively inside `miss_hazard()` rather than to any
default value — the correct place for it, since Group B's
looser affordability threshold means its customers are more
financially stretched, and that effect should enter through
repayment behaviour rather than being asserted directly on
the outcome. Result: A_strict 0.65% default, B_relaxed 1.15%,
a +0.50pp difference in the expected direction, comfortably
above the powered minimum detectable effect.

**Reject inference — considered, deliberately not
implemented.** Customers declined on every application never
generate repayments, so labelling them "did not default"
would be false; their outcome is genuinely unknown, not
negative. Reject inference (statistically inferring outcomes
for declined applicants from similar approved ones) requires
overlap between the approved and rejected populations. This
generator's approval rule is a deterministic affordability
cutoff, so no such overlap exists — attempting reject
inference here would only reconstruct the approval rule
itself, not add information. The `has_outcome` /
`outcome_observed` flag was used instead: the feature store
retains every customer, but the modelling layer trains only
where an outcome is genuinely observable.

**Right-censoring.** Transactions originated late in the
window may not have completed the 90-day performance period
needed to observe an outcome. The generator splits the
observation period into an origination window (when
applications can be made) and a performance window (how long
repayment behaviour is then watched); a transaction is mature
only if its final instalment plus the 90-day default window
falls within the performance window. Roughly a third of
transactions were censored at the calibration point described
above. `outcome_observed` governs modelling eligibility;
`default_flag` remains `NOT NULL DEFAULT 0` as a storage
convention only, so a 0/0 pair must always be read as "not
yet known," never as "did not default."

**Config structure was cleaned up alongside the logic.**
`arrears_resolution` and `dpd_decay_halflife` had been
misplaced under the `ab_test` block while the code read them
from `repayment_process`; both were moved to the correct
block. A duplicated `start_date` key was removed. A stale
`end_date` key was removed once the code moved to
`origination_end_date` / `data_end_date`, along with
`base_cure_probability`, `min_cure_probability`,
`max_cure_probability`, and `persistence_penalties` — all
dead once the arrears-depth cure table replaced the earlier
continuous-resilience cure formula.

**`run_utility` was completed to match.** The missing `else`
branch now runs an internal held-out check when no real
reference file is present, and returns a result dict with an
explicit `mode` (`"TSTR"` or `"INTERNAL"`) and
`tstr_performed` flag, so the absence of real TSTR is
recorded rather than silently producing `None`. Verified
result on the internal held-out path: Gini 0.2142, AUC
0.6071, n 8,607.

**The database was rebuilt from scratch, not migrated.**
Because the ID format changed to natural composite keys and
columns were both added and dropped, a clean drop-and-rebuild
of the raw schema was judged simpler and lower-risk than
altering the existing tables in place — the source CSVs meant
nothing was lost in doing so. Rebuilding also surfaced a
load-order lesson: `\copy` maps CSV columns to table columns
by position, not by header name, and the repayments CSV order
had diverged from the schema's logical order because the
generator appends outcome columns in a second pass. The fix
is to always name columns explicitly in `\copy` statements.
Post-fix, all three tables verified clean: 50,000 customers /
174,421 transactions / 483,579 repayments, with `default_flag`
confirmed as holding 0/1 integers (480,108 non-default /
3,471 default instalment rows) rather than text.

### Regulatory Context
The 90-DPD default definition aligns the schema with the
CRR/EBA regulatory definition of default, with the config
itself documenting where this implementation is a
simplification (materiality and unlikeliness-to-pay are out
of scope). The `reached_sicr` field aligns the delinquency
ladder with IFRS 9's significant-increase-in-credit-risk
concept, again documenting that 30 DPD is used here as a
fixed threshold rather than IFRS 9's rebuttable presumption.
Together these mean the schema's central label — default —
is now defensible against external regulatory standards, with
its simplifications stated rather than implied.

### Addendum — Out-of-Time Split: Tenor-Maturity Bias, Deliberately Left Unfixed (30th July 2026)
An out-of-time (OOT) validation split was added alongside the
stratified split in `models/train_baseline_models.py`
(`split_strategy: "temporal"`), following the EBA guidance on
PD model validation naming out-of-time analysis as a required
element. `temporal_split_date` was set using
`percentile_disc(0.8)` on `first_transaction_date` across the
modelable population — a statistical property of the
generated data, not a calculation using the maturity
constraint below.

**Finding.** The resulting temporal test set showed a bad rate
of 0.33% against a 2.12% training bad rate — a 6x gap. Traced
to right-censoring interacting with variable product tenor: a
12-instalment plan's final due date falls around day 330, and
default cannot be confirmed until 90 further days past due,
so full maturity requires roughly 420 days from origination.
Near `data_end_date` (2024-12-31), only short-tenor (3-month)
loans have had time to reach `has_outcome = 1` at all — longer
tenors originating in the same window are absent from the
modelable population entirely, not merely censored in the test
set specifically. The temporal split's test population was
therefore skewed toward short-tenor loans, not a representative
slice of "customers who arrived later."

**Considered fix.** A maturity-curated split date, calculated
directly from the generator's own rules rather than the data's
distribution: `data_end_date` minus the full maturity
requirement for the longest tenor (330 + 90 = 420 days),
giving a cutoff around 2023-11-06 instead of the current
2024-09-01. This would guarantee every tenor in the product mix
has had its full window to mature ahead of the split point.

**Decision: not implemented.** Two reasons. First, moving the
cutoff back by roughly ten months shrinks the training
population and grows the test population as a direct mechanical
consequence of relocating the train/test boundary — a real,
non-trivial cost, not a side effect. Second, and more
fundamentally: `has_outcome = 1` filtering happens globally,
before any split. A longer-tenor loan originating near the end
of the data window is absent from the modelable population
entirely, not merely present-but-censored in a specific test
set. Any split date chosen close to `data_end_date` will
therefore reproduce a version of this bias — a maturity-curated
date only relocates where the loss of long-tenor visibility
begins, it does not eliminate the underlying constraint. This
is a structural limit of working from a single fixed dataset
snapshot: a live system with continuously arriving performance
data does not face the same bind, since new data keeps
extending the observable window forward.

**Retained as-is:** the current percentile-based
`temporal_split_date`, with this limitation documented rather
than hidden. Temporal Gini/AUC results should be read as
directional, not as a like-for-like comparison against the
stratified split, given the known tenor-mix skew in the test
population.

**Future work:**
1. Recompute `temporal_split_date` using the maturity formula
   above if a fairer (though smaller) OOT test set is wanted.
2. The more complete fix: redefine the default label around a
   fixed observation horizon (the Basel/IRB convention is a
   fixed 12-month PD, applied uniformly regardless of product
   tenor) rather than one tied to each loan's own resolution
   timeline. This would remove the tenor-coupling that causes
   the bias in the first place, rather than working around it
   with split-date selection. Not yet built — planned only.

### Addendum — Identifier Design: Pseudonymised Customer IDs, Natural Keys Below Them (recorded 15th October 2026)
**Why customer IDs are hashed (owner's account, corroborated by the
code).** A raw sequential customer number should not be left exposed
in every table, export and report. The generator therefore derives
each customer ID from an MD5 digest of an internal key
(`generate_customer_id()`, `data_generator/bnpl_data_generator.py`),
whose docstring records the purpose as GDPR-style pseudonymisation.
The second aim was that IDs be unambiguously unique.

**The lesson: a truncated digest is only as unique as its width.**
Truncating a hash trades length for collision risk. The same
docstring records that the digest was **widened to 16 hex characters
"so the truncated digest cannot collide across the population"**, so
an earlier, shorter truncation preceded it. The README's Data
Foundation section records that a collision **was** observed, and
where: "Loading the data into PostgreSQL exposed an identifier
collision that Python validation had not detected, leading to a
redesign of the identifier strategy." The primary-key constraint
caught what the Python validation missed. The earlier digest width
was not recorded and is not reconstructed here. At 16 hex characters (64 bits), the chance of
any collision among 50,000 customers is about n²/2⁶⁵ ≈ 7×10⁻¹¹.
Uniqueness also does not rest on the hash alone: the digest input
includes the customer's index, so distinct customers hash distinct
inputs.

**Below the customer, natural composite keys instead of more hashes.**
Transaction and repayment IDs are built from their parent, not hashed:
- `T-{customer_id}-{n}` for transactions
- `R-{transaction_id}-{instalment}` for repayments

They are unique by construction, and readable: an ID shows which
customer and transaction it belongs to. This is the "ID format changed
to natural composite keys" behind the from-scratch rebuild described
above.

**Verified (15th October 2026, live database).** Every ID is distinct,
and each is a declared primary key:

| Table | Rows | Distinct IDs |
|---|---:|---:|
| `raw.customers` | 50,000 | 50,000 |
| `raw.transactions` | 174,421 | 174,421 |
| `raw.repayments` | 483,579 | 483,579 |

---

## Decision 11 — Automated Audit-Trail Enforcement

### Challenge
The decision log's audit value depended entirely on human
memory: nothing prevented a config, generator, or schema
change from being committed without a corresponding log
entry. Decision 8 and 9 cite PRA SS1/23 Principle 6 (internal
audit evidence) as something the architecture provides, but
the actual mechanism for keeping the log itself accurate was
"remember to update it" — the same manual, memory-dependent
process the FCA CRA review (Decision 6's evidence) criticises
in real firms. This was confirmed as a live problem, not a
hypothetical one: the CTGAN benchmark had been stated as
implemented in Decision 2 and the validation methodology (then Appendix B) for some time before
a direct check against `validate_synthetic.py` showed it
didn't exist.

### Engineering Observation
Two different things were being asked of one document: the
reasoning behind a decision (irreplaceable, must come from a
person) and the mechanical fact that a change happened
(replaceable, a tooling problem with a tooling solution).
Conflating them meant the log could silently drift from the
codebase with no warning.

### Engineering Implication
Forgetting needed to become structurally difficult rather
than merely discouraged. A local git hook was the lowest-cost
way to enforce this for a solo-maintained repository, without
requiring CI infrastructure that matters more for multi-
contributor teams.

### Design Decision
Two git hooks, installed to `.git/hooks/`:

`pre-commit` — blocks any commit that touches an audit-
relevant file (`config.yaml`, `bnpl_data_generator.py`,
`validate_synthetic.py`, any `sql/*.sql`) unless
`docs/engineering_decision_log.md` is staged in the same
commit. Bypassable with `git commit --no-verify` for
genuinely trivial changes (typo, comment-only), by design —
the hook enforces discipline, it doesn't remove judgement.

`post-commit` — appends a timestamped, append-only record to
`docs/audit_trail.log` for every commit that touched a
watched file: commit hash, commit message, whether the
decision log was updated in the same commit, and which files
changed. This is the mechanical half — proof that a change
happened and when — separate from and complementary to the
log's narrative half.

Verified working: a test commit changing only `config.yaml`
was correctly blocked with the expected message before a
corresponding log entry existed.

CI enforcement of the same check (as a required PR check) was
considered and deliberately deferred — it matters most when
multiple contributors could each skip installing the local
hook, which does not apply to a solo-maintained repository at
this stage.

### Regulatory Context
This directly strengthens the audit-evidence claim already
made in Decision 8 and 9. PRA SS1/23 Principle 6 requires
internal audit evidence; before this decision, that evidence
depended on the log being remembered to be updated, with no
mechanism enforcing it. The hooks make the log's own
completeness auditable, not just its content.

---

## Decision 12 — Feature Store Stage 3: Repayment and Delinquency Features

### Challenge
Decision 3 committed to 30+ engineered features across
affordability, payment behaviour, credit history, merchant
risk exposure, and vulnerability. Under the original
generator, that target was unreachable honestly. Repayment
status values of `Missed` and `Defaulted` were assigned only
after the default outcome was already decided — a customer's
instalments were backfilled to match a coin-flip result drawn
in advance. Counting missed or defaulted instalments as
features would therefore have reconstructed the target
directly: proxy leakage, not genuine signal. Only the
independent `Late` payment path (a flat, default-unrelated
5% per-instalment draw) was safe to use, capping Stage 3 at
roughly ten features and leaving the 30+ target out of reach
without either lowering the bar or accepting leakage.

### Engineering Observation
The constraint blocking Stage 3 was never really about SQL —
it was downstream of the same causal flaw Decision 10 fixed.
Repayment history could not be trusted as a feature source
until it was genuinely produced *before* default, not
backfilled to match it.

### Engineering Implication
Once Decision 10 rebuilt default as an emergent outcome of a
hazard-based state machine, the delinquency ladder columns
(`max_dpd`, `reached_sicr`, `cycles_delinquent`,
`arrears_depth`, `terminal_state`) genuinely precede default
in time by construction — they are the mechanism default
emerges from, not a label written backwards. Stage 3 could be
rebuilt honestly, using the same leakage discipline already
established rather than relaxing it.

### Design Decision
`sql/02_feature_store.sql` gained a `repayment_behaviour` CTE,
aggregating `raw.repayments` to customer level and joined into
`feature_store.customer_features` alongside the existing
spine, application-behaviour, and exposure features. Two
families:

Late-payment path (unchanged from the original honest Stage
3, still leakage-free on its own terms): `n_late_payments`,
`pct_late_payments`, `avg_days_late`, `max_days_late`,
`total_days_late`, `n_transactions_with_late_payment`, plus
structural counts.

Delinquency ladder (newly legitimate under the hazard model):
`max_dpd_ever`, `avg_max_dpd`, `n_reached_sicr`,
`pct_reached_sicr`, `max_cycles_delinquent`,
`avg_cycles_delinquent`, `max_arrears_depth`,
`avg_arrears_depth_when_behind`, and the terminal-state
resolution mix (`n_terminal_current`, `n_terminal_delinquent`,
`n_terminal_default`).

This brings the total past the 30+ target committed to in
Decision 3, without reopening the leakage question Decision
10 closed — every new column genuinely precedes default in
the data-generating process, not just in name.

### Regulatory Context
Supports the same PRA SS1/23 Principle 3 documentation
expectation Decision 3 was built against — the feature store
remains the formal leakage boundary, and this stage extends
it rather than weakening it. The delinquency ladder features
are the direct SQL-level expression of the CRR/EBA and IFRS 9
alignment established in Decision 10.

### Addendum — Temporal Leakage in Resolution-Tied Features (30th July 2026)
A distinct problem from the one this decision originally
solved was found before any model was trained on this table.
`max_dpd_ever` and the terminal-state counts
(`n_terminal_current`, `n_terminal_delinquent`,
`n_terminal_default`) are aggregated across a customer's full
repayment history through resolution — including the point of
default itself. Default is defined as reaching 90 DPD
(Decision 10), so for any defaulted customer `max_dpd_ever`
is ≥90 by construction, and `n_terminal_default` is a direct
count of default events. Neither predicts the target; both
restate it. `cycles_delinquent` is tied to the same resolved
trajectory and carries the same problem.

This is a different failure from the one Decision 10 fixed.
Decision 10 corrected causal *generation* — repayment history
now genuinely precedes default in how the data is produced.
This addendum concerns feature *aggregation*: even causally
valid history becomes leakage if it is aggregated across a
customer's entire lifetime, including the resolution event,
rather than cut off at a point before the outcome is known. A
production scoring model sees only what is observable at the
moment of scoring, never the loan's eventual full trajectory.

Two fixes were considered. A fixed observation-window
snapshot (aggregate repayment history only up to a set point
post-origination, e.g. day 30 or 60, scoring against the
eventual outcome) is the production-realistic pattern and is
logged as planned future work, not yet built — same
planned-not-done discipline as the CTGAN benchmark (Decision
2). For the current model, the resolution-tied columns are
excluded at feature selection rather than deleted from the
table, following the exact precedent already set for
`default_probability` and `merchant_risk_multiplier`
(Decision 3, "Label and leakage handling"): the feature store
remains the complete record for auditing, explainability
(Decision 7), drift monitoring (Decision 5), and portfolio
reporting, while `MODEL_FEATURES` in the training code
defines what the model is actually allowed to see. The
distinction is worth stating precisely: `default_probability`
and `merchant_risk_multiplier` are generator-internal fields
never meant to represent real-world observable data at all;
`max_dpd_ever` and the terminal-state counts are real,
meaningful outcome data excluded specifically because they
are aggregated across the resolution event, not because they
are synthetic artifacts. Same boundary mechanism, different
underlying reason — the two should not be cited
interchangeably if this is asked about directly.

Independent late-payment features (`n_late_payments`,
`avg_days_late`, etc.) do not have this problem, since they
are drawn from a process independent of the default path
(Decision 2/CREDIT 2 reasoning) — they remain in
`MODEL_FEATURES`. They do carry a milder, separate limitation
worth naming: they too are aggregated over the customer's
full lifetime rather than a fixed early window, so in a real
production system they would not all be available at the
point a lending decision is made. Unlike the resolution-tied
columns, this is a deployability-realism gap, not a validity
gap — late-payment counts are independent noise relative to
the target, not derived from it, so they do not artificially
inflate model performance the way `max_dpd_ever` would.

### Addendum — Vulnerability Fields Excluded from Model Training (30th July 2026)
`vulnerability_score` and `is_vulnerable` are excluded from
`MODEL_FEATURES` entirely, on a different basis from the
temporal-leakage exclusions above — this is a fairness and
regulatory boundary, not a data-quality one. Two separate
grounds apply. First, the composite score folds in the health
driver, and `mental_health_flag` is already documented in the
generator itself as never permitted as a direct model input
under the Equality Act 2010; including the composite would
risk reintroducing that protected characteristic indirectly.
Second, and more broadly, FCA Consumer Duty's vulnerable-
customer provisions require vulnerability status to inform
*treatment* — enhanced monitoring, forbearance, tailored
explanation genre (Decision 7) — not *scoring*. Using it as a
model input would risk the model penalising the customers
the framework exists to protect, the opposite of its intent.

These fields remain in `feature_store.customer_features` for
their intended purpose: Consumer Duty monitoring and fairness
analysis run *against* the model's outputs (are outcomes
equitable across vulnerable and non-vulnerable segments), not
*inside* the model as a predictor. `age_bracket`, `income_band`,
`employment_status`, and `ab_test_group` are excluded from
`MODEL_FEATURES` for the same reason — segmentation for
analysis, never a scoring input — consistent with how they
are already labelled directly in the Stage 2 SQL comments.

### Addendum — Segment-Aware Thresholding: Considered, Deliberately Not Implemented (30th July 2026)
Baseline model evaluation at the F1-optimal cutoff (0.05)
surfaced a severe error-rate disparity by `thin_file_flag`: a
false-negative rate of 100% for `thin_file_flag=0` customers
across all three models, against a false-positive rate for
`thin_file_flag=1` customers roughly 1,000x higher than for
`thin_file_flag=0`. An experiment removing `thin_file_flag`
from `MODEL_FEATURES` and re-running training and evaluation
produced an identical gap — `credit_score` (r=-0.75 with
`thin_file_flag`) reconstructs the same segmentation on its
own. `thin_file_flag` was restored to `MODEL_FEATURES`; the
disparity is a property of a single global cutoff applied to
segments with genuinely different risk distributions, not an
artefact of any one input feature.

The mitigation this naturally suggests — computing the
F1-optimal threshold separately per segment rather than one
global cutoff — was considered and deliberately not built.
The reasoning: the engineering finding has already delivered
its value. A single global threshold produces materially
different operating characteristics across thin-file and
non-thin-file customers; that has been discovered, measured,
and documented. Implementing segment-aware thresholds would
not be a further modelling optimisation — it would mean two
different lending policies applied to the same model output
by customer segment, which immediately raises questions a
model-tuning decision cannot answer on its own: why are two
customers with the same predicted PD treated differently, who
approved that policy, how is it monitored, and under what
conditions would the thresholds change. Those are governance
and Consumer Duty policy questions, not modelling questions,
and answering them requires explicit policy approval and
ongoing monitoring design, not a code change to
`evaluate_models.py`.

**Retained:** one global threshold; fairness analysis by
segment; full error-rate comparison (FPR, FNR, precision,
recall) per segment; documentation of the disparity found.

**Not implemented:** different thresholds by `thin_file_flag`
or any other segment.

**Future work:** investigate whether segment-aware
thresholding or alternative decision policies can reduce this
operational disparity while remaining consistent with
explainability requirements (Decision 7) and governance
expectations (Decision 9). Any implementation would require
separate policy approval, a formal fairness assessment, and
ongoing monitoring — treated as a governance decision, not a
modelling optimisation, should it ever be pursued.

### Addendum — has_outcome Correction: Maturity Was Never Actually Checked (30th July 2026)
`feature_store.customer_features`'s `has_outcome` field was
set to 1 for any customer with repayment records at all —
meaning "was approved for at least one transaction." It never
referenced `outcome_observed`, the row-level flag Decision 10
built specifically to distinguish "confirmed no default" from
"not yet known." A customer with a still-maturing (censored)
loan therefore had `default_flag = 0` (the storage convention
for "not yet observed," per Decision 10) folded into
`MAX(default_flag) = 0`, and was silently labelled a confirmed
good outcome rather than an unresolved one.

Verified against the live database: only 44.93% of the
population previously marked `has_outcome = 1` actually had a
fully matured outcome (18,694 of 41,604). Corrected `has_outcome`
logic now requires either a confirmed default
(`default_flag = 1`, regardless of any other censored
transaction that customer may also hold) or that every one of
the customer's repayment rows has `outcome_observed = 1` — the
same `MAX(defaulted) OR MIN(all_observed)` rule already used
correctly in the generator's own `calibrate_grid()` function;
this SQL simply never carried that rule over to the feature
store. `sql/02_feature_store.sql` rebuilt accordingly.

**Effect**: modelable population 41,604 → 18,694; observed bad
rate 1.91% → 4.26%. Every result produced earlier in the
session (EDA, both trained models, calibration, threshold
selection, fairness analysis) was computed against the
uncorrected population and needs re-running against the
corrected one. `EPV` (events per variable) check on the
corrected population: 796 defaults ÷ ~24 candidate features ≈
33 events per variable, comfortably above the conventional
10-per-variable minimum (Peduzzi et al.) — the smaller,
corrected population remains statistically adequate for
modelling.

### Addendum — Unequal Observation Opportunity: A Shared Measurement Problem, Not a List of Bad Features (30th July 2026)
A separate `has_outcome` bug was found and fixed the same day
(see "has_outcome Correction" addendum above): the field
previously qualified any customer with repayment records at
all, never checking `outcome_observed`. Corrected, the
modelable population dropped from 41,604 to 18,694 and the
true bad rate rose from 1.91% to 4.26%.

Re-running EDA against the corrected population surfaced a
second, related problem. Twelve features crossed the 0.5
leakage-investigation threshold, up from five. Investigation
found this was not a set of independently weak or leaky
features, but a single upstream cause expressing itself across
an entire feature family.

**Root cause.** The corrected `has_outcome` logic has two
qualifying paths, and they are asymmetric: a defaulted
customer qualifies regardless of timing (`default_flag = 1`
alone is sufficient), but a non-defaulted customer only
qualifies if every one of their repayments has fully matured —
which mechanically requires having originated early enough in
the window. Verified directly: `target_default = 1` customers
average 313.5 days of `transaction_span_days` against 204.1
for `target_default = 0` — non-defaulters are structurally
forced to be "older" in the observation window than
defaulters. Every cumulative exposure feature (transaction
counts, approval counts, total spending, transaction span,
maxima, distinct-category counts) inflates simply from having
more calendar time available, independent of actual risk.
Observed total spending is behaviour multiplied by time
available, not behaviour alone.

**Confirmed and excluded now**, via per-day-of-exposure
normalisation: `n_approved` (raw gap 2.59 vs 4.04; normalised
by days, 0.012654 vs 0.012874 — near-identical, ~100%
artifact) and `days_since_last_transaction` (IV 1.11, the
highest in the dataset — mechanically implausible as genuine
signal given the same cause). Both moved to
`EXPOSURE_TIME_EXCLUDED` in `models/model_features.py`.

**Suspected, not yet confirmed or excluded**:
`total_order_value_gbp` (raw gap 77%, but only ~15 percentage
points survive per-day normalisation — meaning most, not all,
of its strength is artifact; a genuine behavioural signal may
exist underneath), `transaction_span_days` (not merely another
suspected feature — it is the exposure-time variable driving
the contamination in the others), `max_order_value_gbp` and
`n_merchant_categories` (mechanically different — a maximum
and a distinct-count inflate from having more transactions,
not directly from calendar time — not conclusively shown
contaminated when normalised per-transaction rather than
per-day, but not cleared either). All four moved to
`EXPOSURE_TIME_SUSPECTED`, retained in `MODEL_FEATURES` for
the current training round.

**Decision: defer full resolution, do not finalise the feature
set as final.** A sensitivity comparison is required before
this is closed — Feature Set A (conservative: all four
suspected features excluded) versus Feature Set B (exploratory:
retained) compared on Gini, KS, PR-AUC, calibration, feature
importance, and train/test stability. If Set B performs
dramatically better than Set A, that is evidence the
observation-window asymmetry is contributing to apparent
model performance, not evidence the features are genuinely
predictive. This comparison is deferred until after SHAP
(Decision 7), the MLflow promotion workflow (Decision 8/9),
and drift monitoring (Decision 5) are built — the feature set
should be locked once, deliberately, immediately before those
layers are built on top of it, not revised mid-build.

**The production-realistic long-term fix**: rebuild the
cumulative exposure family on a common, comparable observation
basis rather than raw lifetime totals — candidates include
`order_value_per_observed_day`, `transactions_per_observed_month`,
`approvals_per_application`, `merchant_categories_per_transaction`,
or counts within a fixed 30/60/90-day window from origination.
A fixed window is preferable to normalising by lifetime days,
which behaves awkwardly for customers with very short
histories. This is the same underlying fix as the temporal
split addendum above — a fixed observation horizon, decoupled
from each customer's own variable exposure time — surfacing
independently through a different investigation on the same
day, which is itself evidence the two findings share a common
root cause rather than being coincidentally similar.

**Not yet implemented**: the rate-normalised feature rebuild,
the Feature Set A/B comparison, and any resulting change to
`MODEL_FEATURES`. Current training results should be read with
this caveat attached — apparent model performance may be
partly inflated by the observation-time artifact until this is
resolved.

---

## Decision 13 — Baseline Model Training, Evaluation, and the Business Cutoff as Policy

### Challenge
Three models needed training and comparing (Decision 4), with
a defensible evaluation suite and a decision threshold. The
naive approach — fit each model once with hand-picked
hyperparameters, report Gini, pick a threshold that looks
reasonable — produces numbers that cannot be defended under
review.

### Decision

**Hyperparameter selection.** `GridSearchCV` with 5-fold
`StratifiedKFold` over search grids declared in `params.yaml`,
not single fixed values. Stratified rather than plain K-Fold
because the class imbalance (4.26% positive) means a random
fold split could leave very few positives in one fold, making
that fold's score unreliable. Cross-validation runs on the
training set only; the test set is touched once, at the end.

**Scoring metric for selection: `average_precision` (PR-AUC),
not `roc_auc`.** Under imbalance, ROC-AUC's false positive
rate is diluted by a large true-negative denominator, so a
model can accumulate real false positives with little visible
effect on the curve. This reasoning was already recorded in
the metric definitions (then Appendix C) but had not been wired into the code until this
build. Gini/AUC-ROC remain the REPORTED headline metrics,
since that is the industry and regulatory convention;
selection metric and reporting metric are deliberately
different tools for different jobs.

**Class balancing is mandatory, not searched.** `class_weight`
and `scale_pos_weight` are fixed rather than offered as grid
options. At this prevalence, training without balancing risks
the model favouring the majority class; this is a data-
integrity requirement, not an empirical question to be settled
by whichever setting scores marginally higher.

**Probability calibration.** Balanced class weighting fixes
the model's ability to LEARN from the rare class but distorts
the probability SCALE — predicted PDs come out systematically
inflated. Measured: ECE 0.3552 before calibration, with the
top decile predicting 80% against an actual 8.8%. Fixed with
`CalibratedClassifierCV` (Platt scaling), fitted on the
training set only. Result: ECE 0.0026 for XGBoost, roughly a
150x improvement. Ranking (Gini/AUC/KS) is unaffected by
calibration; only the probability scale changes.

**Threshold selection is found, not guessed.** Both F1-optimal
(scanning the precision-recall curve) and Youden's J-optimal
(scanning the ROC curve) are computed and reported per model
in `evaluate_models.py`, with an automatic warning when the
configured cutoff drifts from F1-optimal by more than 0.02.

### The Business Cutoff Is a Policy Parameter, Not a Model Property
This is the substantive decision. `business_cutoff` does not
belong to XGBoost, Logistic Regression, or any other
algorithm. It belongs to the decision engine, downstream of
scoring:

```
raw data -> feature store -> champion model -> risk score
         -> decision engine -> human review
```

The model predicts risk. The business decides where to draw
the line. Those are separate engineering responsibilities, and
the architecture already assumes the threshold is configurable
rather than learned (Decision 9's human approval gate,
Decision 7's explainability requirements, Consumer Duty
outcome monitoring).

Accordingly: **the business cutoff is treated as a configurable
policy parameter rather than a property of any individual
algorithm. For this benchmark it is calibrated using the
champion XGBoost model, but future champion-challenger
promotion would require threshold recalibration rather than
assuming transferability across models.**

Set to `0.145`, from XGBoost's F1-optimal (0.1449) on the
corrected population. Explicitly NOT set by averaging the
three models' optima (0.0993 / 0.1159 / 0.1449) — a midpoint
has no engineering justification, being neither F1-optimal,
expected-loss optimal, nor policy-derived. Since each model
produces a different score distribution, a cutoff calibrated
for one is not valid for another; recalibration is therefore
part of promotion, not an afterthought to it.

### Results (stratified split, corrected population)

| Model | Gini | AUC | KS | PR-AUC | ECE |
|---|---|---|---|---|---|
| XGBoost | 0.6493 | 0.8247 | 0.5061 | 0.2052 | 0.0026 |
| Logistic Regression | 0.6241 | 0.8120 | 0.4849 | 0.1929 | 0.0067 |
| Decision Tree | 0.6083 | 0.8041 | 0.4651 | 0.1756 | 0.0031 |

Notably, every metric improved after the `has_outcome`
correction reduced the modelable population from 41,604 to
18,694 and raised the true bad rate from 1.91% to 4.26%.
Correcting the labels produced a stronger model, not a weaker
one — the common expectation that more realistic data degrades
measured performance did not hold here, because the previous
numbers were measuring against a partially mislabelled target.

### Regulatory Context
Separating the scoring model from the decision threshold
mirrors how model risk management frameworks (PRA SS1/23, SR
11-7) treat model outputs versus their operational use: the
model is validated on its own terms, while the policy applied
to its output is governed, approved, and monitored separately.
Treating the cutoff as a config-level policy parameter, with
recalibration required at promotion, makes that separation
explicit in the codebase rather than implicit.

---

## Decision 14 — SHAP Explainability, and Resolving the Exposure-Time Features on Feature Validity

### Challenge
Two open items converged. Decision 7 required a SHAP layer
before XGBoost could be defended for any customer-facing use.
Decision 12's addendum left four `EXPOSURE_TIME_SUSPECTED`
features retained-but-unresolved, pending a sensitivity
comparison. Resolving the second required the first: SHAP
could measure how much the model *depends* on those features,
but only a controlled ablation could test whether that
dependence was *useful*.

### The governed model package
`train_baseline_models.py` now persists two intentional model
artifacts from the same canonical training run:

- `champion_calibrated.joblib` — the calibrated model whose
  probabilities the decision engine consumes.
- `champion_explanation_base.joblib` — an uncalibrated
  XGBoost fitted on the same training partition with the same
  selected hyperparameters, saved specifically for SHAP.

Plus `champion_metadata.json`, `feature_list.json`, and the
saved split IDs. `explain_models.py` and
`compare_feature_sets.py` load from this package rather than
retraining their own approximations with hardcoded
hyperparameters, which would silently go stale whenever the
grid search selects different winners.

The explanation base exists because
`CalibratedClassifierCV` cannot be SHAP-explained directly: it
holds one fold estimator per CV fold, each fitted on a fold
rather than the full training set, so explaining it yields
several explanations per prediction with no principled way to
reconcile them. The consequence is stated explicitly in every
output: **SHAP explains the uncalibrated risk-ranking logic,
not the calibrated probability that drives the decision.**
Global rankings stay directionally consistent since
calibration is monotonic; local magnitudes must not be
described as changes to the calibrated PD. The metadata
records `valid_uses` and `not_valid_for` lists so this cannot
be assumed away by a later reader.

### SHAP validation
Run before any output was used. Additivity max absolute error
2.384e-06 against a 1e-4 tolerance; feature list matched both
the saved model and the current `model_features.py`; zero
nulls; no excluded fields present.

### Finding: dependence is not the same as usefulness
SHAP showed the model leaning heavily on the suspected
features — **32.1% of total mean absolute SHAP importance**,
with `total_order_value_gbp` alone at 20.8%, the single
largest contributor of any feature.

The ablation then tested whether that dependence earned its
place. Feature Set A (18 features, suspected removed) versus
Feature Set B (22, full), identical hyperparameters and seed
so the feature set was the only variable, across
RepeatedStratifiedKFold (5 folds x 3 repeats):

| Metric | Set A | Set B | Diff (A−B) | Folds A better |
|---|---|---|---|---|
| Gini | 0.6479 | 0.6511 | −0.0032 | 6/15 |
| AUC | 0.8240 | 0.8256 | −0.0016 | 6/15 |
| KS | 0.5127 | 0.5132 | −0.0006 | 9/15 |
| PR-AUC | 0.2111 | 0.2374 | −0.0263 | 2/15 |
| train/test Gini gap | 0.1529 | 0.1643 | −0.0114 | 14/15 |

An earlier single-split comparison had suggested Set A was
slightly better on Gini and notably better on KS. **The
repeated CV did not support that** — the KS advantage
(0.5218 vs 0.4998 on one split) collapsed to 0.5127 vs 0.5132
across 15 folds. This is recorded because the single-split
result was nearly written up as a conclusion, and repeated
cross-validation is what prevented that.

### Decision: exclude, on feature validity
Set B extracts modestly more predictive signal — the PR-AUC
advantage is real and consistent (13 of 15 folds), and PR-AUC
is the metric deliberately chosen for this problem's
imbalance, so it is not minimised here. Set A generalises more
consistently (narrower train/test gap in 14 of 15 folds).
Gini, AUC and KS are effectively ties.

**The exclusion decision therefore rests primarily on feature
validity rather than predictive performance.** The
observation-window analysis identified these variables as
structurally confounded with unequal exposure time — they do
not measure the phenomenon they were intended to measure. The
repeated cross-validation shows that removing them carries a
modest cost in PR-AUC, but does not materially change Gini,
AUC or KS.

A supporting consideration: removing them changes **9.04% of
individual decisions** at the business cutoff. The features
are not merely present-but-harmless; they materially alter who
gets flagged, while providing no demonstrated out-of-sample
benefit outside PR-AUC and being structurally difficult to
defend.

`total_order_value_gbp`, `transaction_span_days`,
`max_order_value_gbp` and `n_merchant_categories` move from
`EXPOSURE_TIME_SUSPECTED` to `EXPOSURE_TIME_EXCLUDED`.
`MODEL_FEATURES` goes from 22 to 18.

### Methodological note
The sequence matters more than the outcome: SHAP identified
the dependency, a controlled ablation tested whether that
dependency was useful, and repeated cross-validation
established how stable the answer was. Excluding features
because their IV or SHAP importance merely *looked*
suspicious would not have been defensible. Neither would
retaining them because a single split favoured doing so.

### Outcome after end-to-end retraining
The ablation above deliberately froze hyperparameters at the
champion's selected values, so the feature set was the only
variable. The production pipeline then re-ran everything:
feature selection, grid search, calibration, threshold
optimisation, SHAP. That is a different experiment, and it
produced a different result.

XGBoost on the canonical stratified split:

| Metric | 22 features | 18 features |
|---|---|---|
| Gini | 0.6493 | 0.6550 |
| AUC | 0.8247 | 0.8275 |
| KS | 0.5061 | 0.5178 |
| Recall at cutoff | 45.91% | 47.80% |
| Top-decile lift | 5.03 | 5.16 |
| PR-AUC | 0.2052 | 0.1956 |
| Highest VIF | 32.23 | 18.24 |
| Exposure-time share of SHAP importance | 32.1% | 0% |

Part of the difference is that the grid search re-selected
hyperparameters for the smaller set — `learning_rate` moved
from 0.1 to 0.05. The optimal configuration for 18 features is
not the optimal configuration for 22, and the ablation could
not see that by construction.

**Revised conclusion.** The exposure-time investigation
identified structurally confounded features whose apparent
predictive value arose partly from unequal observation
opportunity. Removing them produced a cleaner feature set,
eliminated their influence from the explanation layer
(32.1% to 0% of SHAP importance), and substantially reduced
multicollinearity (highest VIF 32.23 to 18.24). On the
canonical split, the retrained end-to-end model also improved
on Gini, KS, recall and lift, with a modest reduction in
PR-AUC.

The decision still rests primarily on feature validity. The
performance evidence is supportive rather than decisive: the
end-to-end comparison is a single stratified split, and the
same evidential weakness at Stage 1 of this investigation is
precisely what the repeated CV was introduced to correct. A
0.0057 Gini difference on one split is not stated with more
confidence than the 15-fold result was.

### Engineering lesson — ablations are not the final criterion
**A feature ablation should not be the final decision
criterion for a production pipeline.**

An isolated ablation answers one question: how much predictive
information does this feature family contain, holding
everything else fixed? A full retraining answers a different
one: how does the entire pipeline behave without it, including
hyperparameter re-selection, calibration, and threshold
recalibration?

Both are needed, and this investigation demonstrates why.
Stopping after the ablation would have understated the benefit
of removing structurally confounded features — the recorded
conclusion at that point was that exclusion carried a
measurable cost, which the end-to-end retraining did not
support. Equally, stopping after the single-split end-to-end
result would have overstated it, which is what the repeated CV
guarded against at the earlier stage. The sequence — SHAP for
dependency, ablation for isolated usefulness, repeated CV for
stability, full retraining for pipeline behaviour — is what
makes the conclusion defensible, not any one of those steps
alone.

### Not yet built
The audience-specific explanation layer (customer, risk
officer, regulatory narratives) sits on top of this SHAP
foundation and is deliberately separate — it involves
disclosure policy, causal-language avoidance, and
adverse-action reason derivation from correlated features,
none of which should be entangled with the numerical layer.
The rate-normalised replacements for the excluded exposure
features (`order_value_per_observed_day`,
`transactions_per_observed_month`, or fixed-window variants)
remain the production-realistic long-term fix and are not
implemented.

---

## Decision 15 — A/B Test Analysis, and the Maturity Collapse

> **Superseded in part by Decision 16 (3rd August 2026).** The
> maturity collapse described below was resolved by extending
> the observation horizon. The causal reasoning about
> post-treatment selection remains valid and is retained: it is
> why the problem was identified rather than absorbed. The
> mature-cohort construction it produced is no longer needed,
> because right-censoring no longer exists.

### Challenge
Decision 1 defined an affordability-threshold experiment
(A_strict 0.45 vs B_relaxed 0.60, `group_B_hazard_multiplier`
1.60, powered for 80% at 5% significance) that had never been
formally analysed. Bad rates by arm had appeared in EDA output
throughout the project, but with no significance test, no
confidence interval, and no assessment of whether the design
target had actually been met.

### The causal problem with the obvious analysis population
The natural instinct is to analyse the modelable population
(`has_outcome = 1`). That is wrong, and the reason is
important: **`has_outcome` is a post-treatment variable.**

Group B's hazard multiplier changes repayment behaviour, which
changes default timing, which changes whether an outcome has
become observable by `data_end_date`. A fast-defaulting B
customer becomes observable; a slow-maturing B non-defaulter
does not. The maturity *rule* is identical across arms, but its
*effect* is not, because treatment alters the outcome process
itself. Conditioning on it compares two selected
post-treatment populations rather than preserving the
randomisation.

Measured directly: observability is **34.08% in A_strict
against 40.77% in B_relaxed**, a 6.69pp difference at p < 0.001.
Not a marginal concern — the filter is substantially
differential.

**Primary population used instead**: a fully mature cohort,
defined as customers whose *last* transaction falls on or
before `data_end_date` minus 420 days (330-day longest tenor
plus the 90-day default confirmation window). Anchored on the
last transaction, not the first, because `target_default` is
`MAX(default_flag)` across all repayments — a customer who
originated early but borrowed again late has an immature
outcome that a first-date anchor would silently admit.

**Residual limitation, stated rather than hidden**:
`last_transaction_date` is itself partly post-treatment, since
treatment affects approvals, which affects loan count and
timing. Measured: B_relaxed averages 3.225 approved loans
against A_strict's 3.029, and a 226-day span against 209 —
so B customers are marginally more likely to be excluded
(11.19% vs 12.32% survival). This trades a large selection
problem for a small one; it does not eliminate it. The clean
fix is a fixed observation horizon in the label definition,
already logged under Decision 10.

Randomisation itself is confirmed sound: origination timing is
indistinguishable between arms (670.9 vs 661.6 average days
from window start, earliest dates one day apart, identical
latest date), and all pre-treatment covariate SMDs are below
0.02. The chi-square on arm sizes flags 50.6/49.4 as
"imbalanced" (p = 0.0075), but at n = 50,000 that test detects
trivial deviations; the SMDs are the appropriate metric and
they are clean.

### Primary result — and why it cannot be interpreted
On the mature cohort (4,871 customers):

| | A_strict | B_relaxed |
|---|---|---|
| Defaults / n | 24 / 2,343 | 39 / 2,528 |
| Default rate | 1.02% | 1.54% |

Absolute difference +0.52%, 95% CI [−0.11%, +1.15%], relative
risk 1.506x, p = 0.110.

**This is not a null result. It is an underpowered one.**
Achieved power: **35.85%** against the 80% target. The minimum
detectable effect at this sample size is 0.91%, larger than the
effect appears to be. The confidence interval is compatible
with both no effect and a ~1pp increase — which is exactly what
low power looks like, and why the p-value alone would be
misleading here.

### Root cause: the simulation horizon, not the design
The mature cohort contains only 4,871 of 50,000 customers
(9.7%). Diagnostic investigation
(`sql/diagnostics/check_maturity_collapse.sql`) established
why:

**85.6% of last transactions fall in 2024.** The transaction
distribution ramps steeply through the simulation window and
the maturity cutoff (2023-11-07) slices through Q4 2023,
leaving nothing eligible after it. The cohort is not small
because the maturity rule is strict — it is small because the
simulation stopped collecting 420 days before most of the
activity it generated could resolve.

Extending `data_end_date` restores it:

| Extension | Cutoff | Eligible | % |
|---|---|---|---|
| 0 months | 2023-11-07 | 4,871 | 9.7% |
| 6 months | 2024-05-06 | 12,328 | 24.7% |
| 9 months | 2024-08-06 | 20,969 | 41.9% |
| **12 months** | **2024-11-06** | **41,604** | **83.2%** |
| 15 months | 2025-02-04 | 41,604 | 83.2% |
| 18 months | 2025-05-06 | 41,604 | 83.2% |

An 8.5x increase in eligible cohort, with **no change to the
causal design**. The plateau at 15 and 18 months is a computed
sufficiency point rather than an estimate: 41,604 is the total
number of customers who ever transacted, so 12 months is
precisely enough for every loan in the dataset to mature.

### Decision
**Extend the simulation horizon and re-run.** `data_end_date`
moves from 2024-12-31 to 2025-12-31; `origination_end_date`
stays at 2024-10-31, so the same customers and the same
transactions are generated — the change purely gives existing
loans time to resolve, isolating outcome observability rather
than also altering the population.

Deliberately **not** implemented in the same session as the
diagnosis. Extending the horizon will raise the observed
default rate, since more loans reach 90 DPD with 12 further
months to resolve. That may push it outside the
BNPL-plausible band the generator was calibrated against
(Decision 10, anchored to a published ~5% late-fee reference),
requiring a re-sweep of `base_miss_hazard` and `cure_scale`.
A calibration sweep is judgement-heavy work and was not
started at the end of a long session in which the feature
store, feature set, models and threshold had already been
rebuilt.

**Not yet done**: the regeneration itself, the resulting
recalibration check, and the re-run of every downstream stage
(feature store, EDA, training, SHAP, thresholds,
documentation).

### Engineering note
The value here is that the unexpected result was investigated
rather than accepted. A 35.85%-power experiment could have
been written up as a limitation — defensible, and quietly
wrong, since the cause was fixable and the fix was one config
line. Diagnosing *why* the cohort collapsed distinguished a
design flaw from a horizon flaw, and only the second was
actually true.

That the fix is a single `data_end_date` change, propagating
through the pipeline without touching analysis code, is also
evidence the architecture is sound. The same correction in a
notebook-based project would mean manually editing every
downstream stage.

---

## Decision 16 — Extending the Observation Horizon: An Architectural Simplification

### What changed
One config value:

```yaml
data_generator:
  origination_end_date: "2024-10-31"   # UNCHANGED
  data_end_date:        "2025-12-31"   # was 2024-12-31
```

`origination_end_date` was deliberately left alone. The same
customers and the same transactions are generated — the change
gives existing loans time to resolve rather than adding
population, so the effect is isolated to outcome observability.

### Why
Decision 15 established that only 9.7% of customers (4,871 of
50,000) qualified for a causally valid A/B analysis population,
leaving the experiment at 35.85% power against an 80% target.
The diagnostic
(`sql/diagnostics/check_maturity_collapse.sql`) showed the
cause was the simulation horizon, not the experimental design:
85.6% of last transactions fell in 2024, while the maturity
cutoff sat at 2023-11-07. A 12-month extension was computed as
the sufficiency point — eligible cohort rising to 41,604 and
plateauing thereafter, since 41,604 is every customer who ever
transacted.

Leaving it unfixed was considered. It was defensible — the
limitation could have been documented — but the cause was known
and the fix was one line. Knowingly shipping an underpowered
primary experiment when the correction was understood is not a
position that survives review.

### Effect on the data

| | Before | After |
|---|---|---|
| Right-censored transactions | ~33% | **0.00%** |
| Modelable customers | 18,694 | **41,604** |
| Customer default rate | 4.26% | 2.72% |
| Transaction default rate | — | 0.90% |
| A/B effect (absolute) | +0.52pp | +0.41pp |
| A/B analysis population | 4,871 (mature cohort) | ~50,000 (full) |

**Right-censoring is now zero.** Every approved loan reaches an
observed outcome.

The customer default rate fell from 4.26% to 2.72%. This is
consistent with the removal of the previous maturity selection
— the old figure was computed on a population where only 37%
of customers qualified — but that explanation has not been
demonstrated. Verifying it would require comparing the
composition (repayment counts, approval mix, customer
characteristics) of the old and new modelling populations.
Recorded as a plausible account, not an established one.

### Effect on statistical power
Recomputed from the regenerated data: **99.85%** achieved
power, with a minimum detectable effect of 0.233pp against an
observed effect of 0.410pp. The experiment can now detect an
effect less than half the size of the one present.

The previous 35.85% figure is **not** a baseline this improved
on. Regeneration changed the data-generating process — the
horizon, the observed outcomes, and the event rate all moved —
so the two numbers describe different experiments. Power was
recomputed from first principles rather than compared.

### Validation after regeneration
Fidelity 8/8, utility PASS, privacy 2/2. Fidelity scores are
identical to the previous run (Hellinger 0.0019 / 0.0043 /
0.0029; thin-file 31.22%; resilience 24.08%; health 16.95%),
which is the expected result: the horizon change affects when
outcomes resolve, not who the customers are.

One movement worth recording: TSTR retention fell from 0.7865
to 0.7398 and the Gini gap widened from 0.0844 to 0.1080. Both
remain within the documented pass criteria (the gap is
reported as supporting context, not a gate — Decision 2
addendum). Plausibly a consequence of the changed target rate.
Not investigated further.

### The architectural consequence
This is the substantive outcome, larger than the power fix.

**Before:**
```
generate -> load -> determine maturity -> build modelling
cohort -> feature store -> models
```

**After:**
```
generate -> load -> feature store -> models
```

An entire stage is gone. With it goes a class of post-treatment
selection risk that had to be reasoned about at every step:
`has_outcome` was a post-treatment variable (observability
differed 34.08% vs 40.77% between A/B arms), the mature-cohort
filter existed to work around that, and the temporal split's
tenor-maturity bias came from the same source. None of those
apply now, because the underlying cause was removed rather than
managed.

### What this means for the code
The distinction applied is **diagnostics stay, cohort
construction goes**:

| Component | Fate | Reasoning |
|---|---|---|
| Observability check | Retained | A validation safeguard. Proves the property holds rather than assuming it, and would catch a regression cheaply. |
| Power calculation | Retained | Relevant to any experiment |
| Assignment and covariate balance, SMDs | Retained | Standard for any randomised comparison |
| `build_mature_cohort()` | Archived | Exists only to work around a problem eliminated at source |
| Single-class test-set guard | Reviewed | Was added because the temporal split produced 16 customers, all defaulters; that condition no longer arises |

Deliberately NOT made conditional. A branch of the form
`if maturity_rate < 0.99: use_mature_cohort()` would carry
complexity forward to guard a state this platform has designed
out. Sentinel Credit is not intended to support incomplete
observation windows; the code should reflect that rather than
hedge against it. The archived logic and the reasoning behind
it remain in this log and in `sql/diagnostics/`.

### Open question, not resolved
The generator reports a transaction-level default rate of
0.90% against a reference of approximately 0.40%. That
reference may have been chosen under the previous, shorter
follow-up. If so, 0.90% under full observation may be correct
rather than too high, and recalibrating toward 0.40% would be
actively wrong — forcing the simulator to match a target set
under different observation conditions. **Whether the reference
still applies must be established before any recalibration
decision.** No recalibration has been performed.

### Addendum — Signal Recovery: What the Corrected Data Revealed
EDA on the regenerated dataset produced the clearest evidence
yet that the exposure-time contamination was real, and it came
from a direction that could not have been engineered.

Information Value moved in two opposite directions:

| Feature | Censored | Full observation | Change |
|---|---|---|---|
| `n_applications` | 0.5829 | 0.2344 | **−60%** |
| `bnpl_exposure_to_monthly_income_ratio` | 0.6224 | 0.2839 | **−54%** |
| `avg_order_value_gbp` | 0.3691 | 0.1414 | **−62%** |
| `pct_late_payments` | 0.2045 | 0.0567 | **−72%** |
| `n_late_payments` | 0.2493 | 0.0743 | **−70%** |
| | | | |
| `savings_balance_gbp` | 0.6538 | 0.7692 | +18% |
| `thin_file_flag` | 0.6884 | 0.7535 | +9% |
| `credit_score` | 0.5718 | 0.6294 | +10% |
| `months_credit_history` | 0.5811 | 0.6075 | +5% |

**Why this happened.** Under right-censoring, a customer only
entered the modelable population if their outcome had resolved.
Resolution took time, and more activity meant more opportunity
to resolve — so activity-related features were correlated with
*being observable at all*, not only with defaulting. That
correlation inflated their apparent predictive power. Removing
censoring removes the selection, and those features fall back
to what they are actually worth.

The customer attributes moved the other way for the
complementary reason. `savings_balance_gbp`, `credit_score`,
`thin_file_flag` and `months_credit_history` are generated
before any repayment simulation and feed directly into
`miss_hazard()` and `financial_resilience()`. They were always
genuine. What changed is that the artifact inflating everything
around them is gone, so real signal is more visible relative to
the noise that was crowding it.

**Why this matters beyond the numbers.** Decision 14 excluded
four exposure-time features on feature-validity grounds, using
SHAP for dependence, an ablation for usefulness, and repeated
CV for stability. That decision was made while the
contamination was still present and could only be reasoned
about indirectly. This EDA run is independent confirmation
obtained by removing the cause rather than measuring around it:
the same collapse appears in the surviving relatives of the
excluded features, which is what the diagnosis predicted would
happen and could not have been arranged after the fact.

Two consequences follow:

**Five features now exceed the 0.5 investigation threshold,
down from seven, and all five are day-zero customer
attributes.** No behavioural or exposure feature crosses the
line. The IV distribution is now interpretable in one sentence:
the model's strongest signals are things known about a customer
before they borrow.

**Events per variable improved to roughly 63** (1,132 defaults,
18 features), up from 44, despite the bad rate falling from
4.26% to 2.72% — the 2.2x larger population more than offsets
the lower rate.

**One question opened, not closed.**
`existing_monthly_obligations_gbp` now has an IV of 0.0018 and
`age` 0.0085, both below the 0.02 drop threshold. This is a
different question from the exposure-time investigation — no
contamination, just weak signal — and both are conventional
credit attributes a reviewer would expect to see considered
rather than silently dropped. Flagged for a deliberate
decision; not removed.

### Addendum — Artifact Persistence
Both `EDA/eda_baseline.py` and `validation/validation.py`
previously printed their results to console and saved almost
nothing. Every re-examination therefore required re-executing
the script and re-connecting to the database.

Both now persist full artifact sets:

- `reports/eda/` — target summary, missingness, IV screening,
  VIF, full correlation matrix, correlated pairs as a flat
  list, outliers, split stability, and `eda_metadata.json`
  recording the feature list, thresholds and population that
  produced them.
- `reports/validation/` — fidelity, utility, privacy, and
  `validation_metadata.json`.

The metadata files matter more than the CSVs: a saved result
is only useful if it can be traced to the feature set and
population that generated it. Without that, a stale artifact is
worse than none.

### Engineering note
Production engineers are judged not only by what they add, but
by what they can safely remove. Having eliminated the root
cause, the correct response was to delete the machinery built
to manage its symptoms — not to keep it behind a flag. The
historical reasoning lives in the log; the pipeline does not
have to carry it.

---

## Decision 17 — Feature Store Inventory: Every Column Accounted For

### Challenge
`MODEL_FEATURES` reached 18 through a sequence of individual
exclusions made across Decisions 12, 14 and 16. Each was
individually documented, but the list had never been *derived*
from a complete inventory — nobody had established what the
full candidate set actually was, or confirmed that every
column in the feature store had a stated role.

That gap matters for two reasons. A reviewer asking "why isn't
X in the model?" needs an answer for every X, not only for the
ones that happened to come up. And features that were never
screened at all are invisible: they neither pass nor fail,
they simply don't appear.

### Method
`sql/diagnostics/check_feature_store_inventory.sql` profiles
every column in `feature_store.customer_features` — type, null
rate, and distinct-value count on the modelable population.
Each column was then classified by role.

**The feature store and `MODEL_FEATURES` are different layers,
and conflating them is the source of the confusion:**

```
Feature store (64 columns)
  the governed analytical record — model development,
  feature screening, fairness analysis, A/B evaluation,
  monitoring, diagnostics, explainability, audit traceability

        |  eligibility and governance review
        v

MODEL_FEATURES (18 columns)
  the scoring contract — only predictors available at
  decision time, not targets or identifiers, not policy
  assignments, not regulatory monitoring fields, not
  resolution-tied, and passing the current governance review
```

A column can legitimately belong in the feature store and not
in the model. That is the design, not an oversight.

### Inventory result — 64 columns

| Category | Count | Basis |
|---|---|---|
| **CANDIDATE (in `MODEL_FEATURES`)** | **18** | Passed all gates |
| Excluded — resolution-tied | 12 | Only observable after outcome resolves |
| Excluded — redundant or exact identity | 11 | Duplicates information already present |
| Excluded — exposure-time confounded | 6 | Feature validity (Decision 14) |
| Excluded — regulatory or segmentation | 4 | Not lawful/appropriate as scoring inputs |
| Never screened — decision required | 4 | See below |
| Monitoring only | 2 | `is_vulnerable`, `ab_test_group` |
| Structural (ID, target, filter, dates) | 5 | Not predictors by definition |
| Excluded — constant | 1 | `n_terminal_delinquent` |
| Excluded — direct leakage | 1 | `n_terminal_default` |

The 18 CANDIDATE columns are exactly the current
`MODEL_FEATURES`. The allowlist was correct; it simply had
never been shown to be complete.

### Why each exclusion category exists

**Resolution-tied (12)** — `max_dpd_ever`, `avg_max_dpd`,
`n_reached_sicr`, `pct_reached_sicr`, `max_cycles_delinquent`,
`avg_cycles_delinquent`, `max_arrears_depth`,
`avg_arrears_depth_when_behind`, `n_terminal_current`,
`avg_days_late`, `max_days_late`, `total_days_late`. These
describe how a loan resolved, so they are only knowable after
the outcome they would be predicting. Retained in the feature
store for post-hoc explanation, portfolio monitoring and
delinquency reporting — the boundary pattern established in
Decision 12: keep in the table, exclude at the model layer.

**Redundant or exact identity (11)** — `annual_income_gbp`
(= monthly x 12), `n_transactions` (= `n_approved`, r=1.0000),
`n_declined` (= applications - approved), `approval_rate`,
`pct_affordability_flagged`, `n_affordability_flags`
(= `n_declined`), `pct_low_risk_category` (three proportions
sum to 1), `total_instalment_exposure_gbp` (r=0.974),
`n_transactions_with_late_payment` (r=0.949),
`avg_affordability_ratio` (r=0.93), `debt_to_income_ratio`
(dominant term inside affordability ratio). Feeding both sides
of an identity to Logistic Regression makes its coefficients
unstable and meaningless, defeating the reason Decision 4
chose it.

**Exposure-time confounded (6)** — `n_approved`,
`days_since_last_transaction`, `total_order_value_gbp`,
`transaction_span_days`, `max_order_value_gbp`,
`n_merchant_categories`. Excluded on feature validity in
Decision 14 and independently confirmed by the horizon
extension (Decision 16 addendum), where the same collapse
appeared in their surviving relatives once censoring was
removed.

**Regulatory or segmentation (4)** — `vulnerability_score`
(composite includes a health driver, Equality Act 2010),
`age_bracket`, `income_band`, `employment_status`. The
banded/categorical forms are for analysis and fairness
segmentation, not scoring. Note `age` and `monthly_income_gbp`
are retained in continuous form — the exclusion is of the
banded representations used for segment reporting, not of the
underlying attributes.

**Monitoring only (2)** — `is_vulnerable` and `ab_test_group`
are comparison-group labels. Using either as a predictor would
mean scoring customers on their vulnerability status or their
experiment assignment.

**Constant (1)** — `n_terminal_delinquent` has a single
distinct value across the entire modelable population. Under
the extended horizon every transaction resolves to CURRENT or
DEFAULT; none remains still-behind. Zero signal is
mathematically possible. Worth recording because it is also a
data property confirming the horizon extension worked.

**Direct leakage (1)** — `n_terminal_default` is a restatement
of the target.

### Four columns never screened — decision required
These exist in the feature store and were never added to the
allowlist or explicitly excluded. They are neither passing nor
failing; they are simply absent from every prior review:

| Column | Prior suspicion |
|---|---|
| `n_instalments_total` | Cumulative count — likely exposure-time confounded |
| `n_transactions_with_repayments` | Likely identical to `n_approved` |
| `total_scheduled_repayment_gbp` | Cumulative — likely near-duplicate of `total_order_value_gbp` |
| `avg_instalment_amount_gbp` | Per-transaction average, so NOT obviously exposure-confounded and not obviously duplicating anything included |

Three are expected to fail on structural grounds.
`avg_instalment_amount_gbp` is the genuine open question: as
an average rather than a cumulative sum it does not inflate
with exposure time, and it has no obvious duplicate in the
current set. **Not excluded on suspicion** — to be tested with
correlation and IV in the governance review, held to the same
standard as everything else.

### Two retained features flagged, not removed
`age` (IV 0.0085) and `existing_monthly_obligations_gbp`
(IV 0.0018) both sit below the 0.02 drop threshold on the
regenerated data. **Both are retained for this training run.**

IV is a screening tool, not a decision rule. It measures
univariate separation only: a feature can be weak alone and
still contribute through interactions or non-linear splits,
particularly in a gradient-boosted model. This project has
already had IV mislead in both directions — `thin_file_flag`
read 0.0000 under a binning bug, and the exposure-time family
read 0.58-0.62 under a selection artifact. Acting on IV alone
would apply a weaker evidential standard than Decision 14 used
for the exposure features, in the same project, on the same
day.

Both are also conventional credit attributes. A reviewer
asking whether existing financial commitments were considered
should receive "yes, tested, contributed negligibly, excluded
with evidence" — not "IV was below 0.02."

They will be decided in the governance review using IV, VIF,
SHAP and a repeated-CV ablation together, with the outcome
recorded whichever way it goes.

### Missingness treatment
Two columns carry substantial nulls by design:
`avg_days_late` (52.21%) and
`avg_arrears_depth_when_behind` (79.93%) — both undefined for
customers who never paid late or never fell behind. Both are
already excluded as resolution-tied, so no imputation decision
is required for the model. Recorded because a nullable field
in a feature store needs a stated reason, and "genuinely
undefined" is a different situation from "pipeline failure."

All 18 model features have **zero nulls** in the modelable
population.

### Standing principle
**Keep broadly in the feature store. Exclude deliberately at
the model boundary.** Deleting a column the moment it fails
model eligibility would destroy the evidence for its own
exclusion — the identity between `n_transactions` and
`n_approved`, and the IV of 1.11 on
`days_since_last_transaction`, were only discoverable because
those columns existed to be measured.

The right test for a feature store column is not "does the
model use it" but "does it have a documented role." A
64-column store where every field has a stated purpose is
better designed than a 30-column one where nobody knows why
half of them exist.

---

## Decision 18 — Model Scope, and the v1 Feature Freeze

### Challenge
Decision 16's extended horizon and Decision 17's inventory left
six features (`n_applications`, `avg_n_instalments`,
`pct_high_risk_category`, `pct_medium_risk_category`,
`bnpl_exposure_to_monthly_income_ratio`, `avg_order_value_gbp`)
flagged `conditional_as_of_logic` — unconfirmed whether they
are computed point-in-time or as full-lifetime aggregates —
alongside `pct_late_payments`, `max_affordability_ratio` and
`n_late_payments` carrying the same open question. This could
not be resolved as a feature-level question in isolation: it
depends on what Sentinel Credit is claiming to model, and that
had never been stated explicitly.

Two distinct model types were being conflated. A **customer-level
portfolio-risk model** classifies risk using behaviour across a
defined observation period; lifetime-aggregate features are
legitimate here provided the scope is stated. An
**application-time underwriting model** scores one specific new
application using only information available before that
application, which requires point-in-time reconstructed
features and a training population rebuilt as one row per
application rather than one row per customer. The two require
different features, different populations, and support
different claims. Continuing to freeze `MODEL_FEATURES` without
choosing between them would freeze a feature set that is
answerable to neither standard cleanly.

### Decision — scope
**Sentinel Credit v1 is a customer-level portfolio-risk model,
not an application-time underwriting model.** Application-time
underwriting is recorded as a future redesign (see below), not
as an expansion of the current scope.

### Method — testing the six for exposure-time contamination
`sql/diagnostics/check_point_in_time.sql` correlates each
flagged feature against exposure-window length, on the
reasoning that a feature drawing on information beyond a
consistent observation point should tend to grow with how long
a customer has been active.

**Confirmed exposure-time contaminated** (correlation with
exposure length): `n_applications` (r=0.47),
`bnpl_exposure_to_monthly_income_ratio` (r=0.34),
`n_late_payments` (r=0.22) — all rise monotonically with tenure.

**No material association found**: `pct_late_payments`,
`pct_medium_risk_category`, `pct_high_risk_category`,
`avg_n_instalments`, `max_affordability_ratio`. `avg_order_value_gbp`
showed a mild upward drift (r=0.08), below the threshold treated
as material here.

This diagnostic tests only for growth with exposure length. A
near-zero correlation shows the feature is not accumulating
over time; it does not establish that the feature's value was
fixed as of any particular decision point — `max_affordability_ratio`,
for example, is a lifetime maximum, and a low correlation with
tenure says nothing about *when within that lifetime* the
maximum occurred. That distinction only matters under an
application-time scope. Under the portfolio-risk scope adopted
above, it does not need to be resolved for v1: the diagnostic
is retained as supporting evidence that these six show no
material exposure-time confound, not as proof of point-in-time
validity, and the wording throughout this decision is
deliberately kept to that weaker, defensible claim.

### Decision — four features removed
1. **`age`** — removed on regulatory defensibility grounds
   (Equality Act 2010 proxy risk), not on IV. This corrects the
   Decision 17 framing, which had proposed removal on "weak IV"
   before noticing `existing_monthly_obligations_gbp` scores
   lower on the same metric and was being retained — an
   inconsistency once the two are placed side by side.
2. **`n_applications`** — removed. Confirmed exposure-time
   contaminated (r=0.47); counts applications across the
   customer's full observed history, including activity after
   any consistent observation point.
3. **`bnpl_exposure_to_monthly_income_ratio`** — removed.
   Confirmed exposure-time contaminated (r=0.34); the lifetime
   version accumulates exposure beyond a fixed point.
4. **`n_late_payments`** — removed. Confirmed exposure-time
   contaminated (r=0.22); can include late payments occurring
   after the point risk should have been assessed. Its exact
   0.000000 SHAP contribution (Decision 17 governance review) is
   recorded as corroborating, supporting evidence — the removal
   rests on the confirmed confound, not on the SHAP result alone.

### Decision — six features retained
`pct_late_payments`, `pct_medium_risk_category`,
`pct_high_risk_category`, `avg_n_instalments`,
`max_affordability_ratio`, `avg_order_value_gbp` are **retained**
for v1. Their construction is consistent with the customer-level
portfolio-risk scope, and the exposure-window diagnostic found
no material association requiring removal. This diagnostic is
supporting evidence, not proof of point-in-time validity, and is
not being cited as such.

### Resulting feature set
`MODEL_FEATURES` moves from 18 to **14** for v1. This is
recorded as a scope-driven freeze, not a performance-driven one:
no retraining evidence motivated any of the four removals.

### Governance evidence retained, not discarded
The 285-fit repeated-CV leave-one-out review (Decision 17) is
**not superseded as evidence** — it remains the governance record
for the 18-feature baseline and documents why the final set
changed. It is explicitly marked as pre-freeze evidence, not the
review of the 14-feature v1 model. A fresh governance pass on
the 14-feature set is queued as a follow-on step before v1 is
called final.

### Future work — application-time underwriting
Recorded, not built: a v2 redesign scoring a specific application
using only prior history, requiring point-in-time reconstructions
of the excluded and flagged features (`n_prior_applications`,
`prior_avg_order_value_gbp`, `max_prior_affordability_ratio`, and
equivalents for the remaining flagged features) and a training
population rebuilt as one row per application. None of the
current governance evidence would carry over unchanged to that
redesign.

---

## Decision 19 — Business Cutoff Selection: Training-Only OOF, Review-Capacity Constrained

### Challenge
`business_cutoff: 0.137` had been provisional and increasingly
dead since Decision 16's horizon extension changed the
population's bad rate. Evaluation on the frozen 14-feature v1
champion (`models/artifacts/champion_metadata.json`, stratified
and temporal splits) confirmed this concretely rather than
provisionally: at 0.137, XGBoost flagged 0 of 8,321 stratified
test customers and 3 of 4,857 temporal test customers. The
configured cutoff was not merely miscalibrated, it was
non-functional — no evaluation of precision, recall, or
fairness at an actual operating point was possible until it was
replaced.

Two constraints on how the replacement could be selected,
carried over from the project's standing discipline:
1. **Training-only.** Selecting a threshold on either held-out
   evaluation set would contaminate that set's use as an
   independent performance estimate — the same reason GridSearchCV
   never touches the test partition (Decision 4).
2. **No invented cost assumptions.** `evaluate_models.py`'s
   docstring records that a cost-matrix approach using assumed
   unit costs was considered and declined for the main evaluation
   script, since this platform is PD-only (Decision 4 addendum)
   and has no real LGD/EAD basis for a false-negative:false-positive
   cost ratio. Manufacturing one to select a threshold would
   produce a cutoff that looks governed without being defensible.

### Method
`models/select_business_cutoff.py` (retiring an earlier draft,
`recalibrate_threshold.py`, which duplicated this purpose and
was deleted rather than merged) reconstructs the exact canonical
stratified training partition from `split_ids_train.csv`,
freezes the champion's already-selected XGBoost hyperparameters
(no re-search), and generates calibrated out-of-fold
probabilities via a double cross-validation structure: 5 outer
folds partition the training data for OOF prediction, each
fitted on a `CalibratedClassifierCV(method="sigmoid")` with its
own 5-fold inner split for calibration — the same calibration
method the deployed champion uses, so the threshold is selected
on the probability scale the decision engine actually outputs,
not on raw uncalibrated XGBoost scores. Neither held-out test
split was read at any point in this script.

Three candidate thresholds were generated from the same
underlying OOF precision-recall table, each answering a
different question:

| Method | Threshold | Flagged | Precision | Recall | Bad rate if approved |
|---|---:|---:|---:|---:|---:|
| F1-optimal | 0.0868 | 7.82% | 11.02% | 31.68% | 2.02% |
| **Review-capacity (15%)** | **0.0671** | **15.00%** | **9.35%** | **51.55%** | **1.55%** |
| Youden's J | 0.0376 | 24.02% | 7.74% | 68.32% | 1.13% |

A fourth candidate, expected-cost-minimizing, was configured but
deliberately left unset (`threshold_recalibration.cost_assumptions.
false_negative_cost` / `false_positive_cost`, both `null` in
`config.yaml`) and consequently skipped — no defensible cost
ratio exists yet, and inventing one to manufacture a fourth
number would have violated the same PD-only boundary
`evaluate_models.py` already declined to cross.

### Decision
**`business_cutoff` is set to `0.0671`**, the review-capacity-
constrained threshold, selected over F1 and Youden because it is
the only one of the three that answers the operational question
directly: given a stated review capacity of 15% of applications,
how much risk can the model capture. F1 leaves the majority of
available review capacity unused (7.82% flagged) and misses
roughly two-thirds of defaults (31.68% recall). Youden exceeds
the stated capacity by more than half again (24.02% flagged) —
a statistically balanced operating point, but not one the
review process was described as able to absorb. At 0.0671,
15.00% of the training population is flagged, capturing 51.55%
of observed defaults at 9.35% precision, with a 1.55% residual
bad rate among approved customers.

F1 and Youden are retained as statistical comparators in
`reports/threshold_selection/threshold_candidates.csv`, not
discarded — they document that the capacity constraint was an
active choice among alternatives, not the only threshold
computed.

**This selection is conditional on 15% being a genuine
operational review-capacity figure.** `config.yaml`'s
`review_capacity.max_flag_rate: 0.15` is currently documented as
an illustrative placeholder demonstrating the mechanism ("the
review team can handle at most 15%" was given as an example, not
confirmed). If the real figure differs, `select_business_cutoff.py`
should be rerun with the corrected value before this cutoff is
treated as final — the OOF evidence and calibration do not need
to be regenerated, only the capacity constraint and the
resulting candidate.

### Held-out confirmation (both splits)
`evaluate_models.py` was rerun on both stratified and temporal at
`business_cutoff: 0.0671`. Stratified landed close to the OOF
selection estimate (recall 54.87% vs 51.55% OOF, flagged rate
15.51% vs 15.00% target) — expected, since stratified is drawn
from the same population distribution the cutoff was selected
against. Temporal came in lower (recall 38.71%, flagged rate
10.99%), reflecting a genuinely out-of-time population with a
different score distribution, not a selection failure. Cutoff-
independent metrics (Gini, AUC, KS, PR-AUC, ECE, Brier) were
unchanged from Section 6 of the training summary on both splits,
as expected — they do not depend on `business_cutoff`.

### Operating-policy review: thin_file_flag segment finding
The `thin_file_flag` fairness table at 0.0671 surfaced a material
pattern requiring its own investigation before this cutoff could
be signed off, not merely noted: false-negative rate for
`thin_file_flag=0` was 100% on both splits — confirmed via both
the targeted segment diagnostics and, subsequently, full
`evaluate_models.py` runs on stratified (FNR 100.00%, FPR 0.00%)
and temporal (FNR 100.00%, FPR 0.03%) — every defaulted customer
in that segment was missed on both. `thin_file_flag=1` carried
both the recall and a substantial false-positive burden (FPR
46.53% stratified, 32.93% temporal). Logistic Regression and
Decision Tree showed the same pattern, which corroborates but
does not independently confirm the finding — the three models
share the same population and heavily overlapping predictors, so
agreement between them is expected even under a shared ranking or
scale issue, not three separate lines of evidence.

`thin_file_flag` is a credit-profile segment, not itself a
protected characteristic; the finding is an operating-policy
concern, not a formally established disparate-impact claim.

Three checks (`models/segment_threshold_diagnostics.py`) were run
on both splits before drawing any conclusion:

1. **Score distribution among defaulted customers, by segment.**
   `thin_file_flag=0` defaulters' calibrated scores topped out at
   0.0344 (stratified, n=61) and 0.0367 (temporal, n=34) — roughly
   half the cutoff, with 0% above it on either split. This is a
   hard ceiling, not a few borderline cases: the entire segment's
   defaulted-customer score distribution sits below any threshold
   that was on the table.
2. **Threshold sweep by segment, single global threshold.** No
   threshold exists that recovers `thin_file_flag=0` recall
   without collapsing `thin_file_flag=1`'s false-positive rate to
   75-86%+ in the process. This rules out the threshold choice
   itself as the cause — F1 (0.0868) and Youden (~0.02-0.06
   depending on split) sit at or above the same collapse point, so
   no candidate on the table would have avoided this.
3. **Within-segment AUC, with bootstrap 95% CI** (given the small
   defaulted-customer counts):

| Split | Segment | n | Defaults | AUC | 95% CI | Gini | KS |
|---|---|---:|---:|---:|---|---:|---:|
| Stratified | thin=0 | 5,648 | 61 | 0.6409 | 0.5661-0.7220 | 0.2819 | 0.2584 |
| Stratified | thin=1 | 2,673 | 165 | 0.6904 | 0.6532-0.7286 | 0.3808 | 0.3293 |
| Temporal | thin=0 | 3,294 | 34 | 0.6885 | 0.6148-0.7613 | 0.3770 | 0.3621 |
| Temporal | thin=1 | 1,563 | 90 | 0.6667 | 0.6159-0.7159 | 0.3333 | 0.2772 |

`thin_file_flag=0`'s CI lower bound clears 0.5 on both splits
(0.5661 stratified, 0.6148 temporal), and the point estimate does
not collapse on temporal — it rises (0.64 to 0.69). This is
**interpretation (1): a policy-scale problem, not a model-signal
failure.** The model ranks risk within `thin_file_flag=0`
comparably to how it ranks within `thin_file_flag=1` — the two
segments' CIs overlap substantially on both splits. What collapses
is the *absolute* probability scale: a rare event (1.03% bad
rate) within an already-low-base-rate segment gets compressed
toward zero under one shared calibration, even for the
segment's genuinely higher-risk members. Checks 1-2 already
established no single global threshold can fix this; check 3
establishes why — the signal exists, it does not reach the scale
a shared cutoff operates on.

This is not new: it echoes the earlier finding (Decision 12
addendum) that removing `thin_file_flag` as a model input did not
close this gap, because `credit_score` reconstructs the same
segmentation on its own (r=-0.75). The mechanism recurring under
a different diagnostic is corroborating, not incidental.

**What this finding does not license.** Segment-specific
thresholds are not implemented here, and this evidence is not
grounds to implement them unilaterally — that remains the
explicit governance decision Decision 12 already deferred.
Ranking surviving within `thin_file_flag=0` also does not reduce
this to "just calibration": a single globally calibrated model
producing this pattern is consistent with the segment carrying
real, if compressed, risk information — what policy to build on
that (percentiles within segment, segment-adjusted cutoffs, or
leaving the global cutoff as-is with the limitation documented)
is a separate decision, not resolved by this diagnostic.

### Decision (resolved)
`business_cutoff: 0.0671` is **confirmed** for the global-decision
question — held-out evaluation on both splits is consistent with
the OOF selection evidence, and no alternative global threshold
on the table would have avoided the segment pattern above. A
**separate governance item is opened, not resolved here**: whether
segment-aware review capacity is warranted, given ranking survives
within `thin_file_flag=0` but the shared probability scale
suppresses it to zero recall in practice. This item requires
explicit policy sign-off before any segment-differentiated
treatment is built, per Decision 12.

---

## Decision 20 — Drift Monitoring: PSI + Data Quality, Scheduled via Airflow (Astro)

### Challenge
Drift monitoring had been queued since the project's earliest
handoff notes (Decision 5) but never implemented — every prior
evaluation of population stability (Decision 17's temporal PSI,
Decision 19's segment diagnostics) was a one-off check against a
fixed snapshot, not a repeatable process that could catch
degradation between deliberate governance passes. `evaluate_models.py`'s
own docstring already anticipated the intended mechanism
("PSI / drift monitoring — belongs to Decision 5's scheduled
Airflow pipeline") without anything existing to run it.

### Method
`pipeline/monitor_drift.py` compares the canonical stratified
training partition (reference) against the temporal test split
(comparison — the most temporally recent population available in
the absence of a live scoring stream, stated explicitly as a
proxy in the script's own docstring, not presented as a live
production monitor). Three check types: PSI per feature, reusing
`govern_model_features.py`'s exact calculation and thresholds
(`feature_governance.thresholds.psi_moderate` / `psi_high`) so a
"material_shift" flag means the same thing in both places;
null-rate delta, out-of-range rate, and mean-shift-in-training-sigma-units
as data-quality checks PSI's binning can occasionally miss;
and portfolio bad-rate drift, the single number most directly
tied to business impact. All logic was unit-verified against
synthetic data with known shift magnitudes before being run
against real data (identical distributions -> PSI ~0; a 2-sigma
shift -> correctly flagged "material_shift").

The script does not gate deployment, retrain, or promote anything
by itself — it produces evidence and a structured alert flag for
the not-yet-built Approval Gate and Champion/Challenger workflow.

**Scheduling**: implemented via Astro CLI (local Airflow-in-Docker),
not GitHub Actions — reconsidered explicitly after initially leaning
toward the lighter option, because Airflow is the correct tool for
this specific job in a way that isn't just stylistic: a real
drift-monitoring pipeline needs DAG-structured task dependencies
(check drift -> branch on the result -> possibly trigger downstream
evaluation), which GitHub Actions handles awkwardly once a job
needs more than "run this one script on a schedule." The project's
own `requirements.txt` already pinned `apache-airflow==2.9.1`
before this decision, confirming Airflow was the intended tool
from the start, not a late change.

`infra/dags/monitor_drift_dag.py` runs the script as a subprocess
rather than reimplementing its logic — the DAG task exercises the
exact same code path a manual run would, so there is no separate
"DAG version" of the monitoring logic that could drift out of sync
with what is documented and tested. Two tasks are deliberately
separated: `compute_drift` fails only on an infrastructure problem
(the script itself erroring); `evaluate_alerts` reads the metadata
`compute_drift` already wrote and fails only if real drift was
found. This keeps "the pipeline broke" and "the pipeline found
something" as visibly distinct failure states in Airflow's UI
rather than one conflated red X — the same distinction the
architecture's Monitoring block implies with "Alerting" as a
separate concern from the check itself.

### Environment issues resolved along the way
Standing up Astro-in-Docker against this project surfaced a
cluster of environment problems worth recording, since each was a
real fix, not noise:
- Exact package pins in `infra/requirements.txt` (`pyyaml==6.0.1`,
  `mlflow==2.13.0`, `psycopg2-binary==2.9.9`) conflicted with the
  Astro Runtime image's own Airflow 3.3 core dependencies and its
  Python 3.14 environment (no prebuilt wheel for the old
  `psycopg2-binary` pin). Resolved by loosening to `>=` bounds —
  every affected package is used through long-stable APIs in
  `monitor_drift.py`, so allowing a newer compatible version was
  safe, not a compromise.
- `docker-compose.override.yml` initially referenced a `webserver`
  service that does not exist in this Astro Runtime version's
  compose definition, invalidating the whole file silently until
  removed — only `scheduler` needed the project volume mount.
- Container networking required an explicit `SENTINEL_DB_HOST`
  environment-variable override in `monitor_drift.py` (added
  symmetrically with the pre-existing `SENTINEL_DB_PASSWORD`
  pattern), since `localhost` inside the container refers to the
  container itself, not the Windows host running PostgreSQL —
  `host.docker.internal` is the correct target, set as an Airflow
  Variable rather than hardcoded.
- The newer MLflow version resolved inside the container required
  `MLFLOW_ALLOW_FILE_STORE=true` to keep using the same file-based
  `mlruns/` tracking store the rest of the project already relies
  on, rather than forcing a database-backend migration mid-task.
- The `sentinel_credit_monitoring` MLflow experiment's first
  creation happened during a manual Windows-native test run before
  the container setup was complete, which baked an absolute
  Windows path into that experiment's `meta.yaml`. The container's
  Linux process later inherited that path via the shared `mlruns/`
  mount and tried to create a literal `/C:` directory at its
  filesystem root, failing with a misleading `PermissionError`
  that was not actually a permissions problem. Resolved by deleting
  that one experiment's folder so it was recreated fresh,
  entirely from inside the container. Established going forward:
  `monitor_drift.py` should be triggered through Airflow, not run
  manually from Windows PowerShell, to avoid reintroducing a
  mixed-path experiment.

### Decision
Drift monitoring is implemented and verified working end-to-end:
`pipeline/monitor_drift.py`, scheduled daily via
`infra/dags/monitor_drift_dag.py` under local Astro/Airflow,
confirmed with a real triggered run (`compute_drift` and
`evaluate_alerts` both succeeded; `reports/monitoring/monitoring_run_metadata.json`
shows 0 of 14 features alerting, generated fresh from the
container-triggered run, not stale data). This closes the Decision
5 item that had been queued since the project's earliest planning.
Feeding this evidence into an automated Approval Gate or the
Champion/Challenger promotion workflow remains future work, not
resolved here.

### Addendum — "Daily" Corrected (13th October 2026)
"Scheduled daily" above describes the DAG's `@daily` setting, not
what happened. The schedule only fires while the local Astro/Docker
stack is running. Scheduled runs occurred on 12th and 13th August
2026 and then stopped when the stack went down. Docker Desktop does
not auto-start on this machine, so there were no runs from 14th
August to 11th October. Monitoring is therefore described as **scheduled while the
local stack is running**. Two further limitations are recorded in
Decision 24 and `docs/audit_findings_2026-10-12.md` (A2/A3):
- the comparison population overlaps the reference by 79.3%
- both populations are fixed snapshots, so repeated runs produce
  near-identical results

---

## Decision 21 — Champion/Challenger Promotion Workflow (Evidence-Only, MLflow Registry)

### Challenge
The architecture's Modelling Engine names "MLflow Model Registry:
Champion, Challenger, Versioning" as a standing capability, but
nothing in the project had ever called `mlflow.register_model` —
every prior script logged runs to the tracking experiment without
registering any model version. No mechanism existed to compare a
future retrained or alternative model against the current champion
on governed terms, or to record that comparison anywhere.

### Method
`pipeline/promote_champion.py` auto-discovers the current champion's
stratified and temporal runs via the `is_champion` tag
`train_baseline_models.py` already sets, combined with the standard
governance-identity filter every other script in this project
checks against. The challenger is deliberately NOT auto-discovered
— its two run IDs (stratified, temporal) are required CLI arguments,
since unlike "the current champion" there is no safe default for
"which run is the challenger."

Promotion requires a material Gini improvement on BOTH splits, not
one — reusing the temporal-stability concern central to this
project since Decision 16/18. PR-AUC and KS are reported as
corroborating evidence, not gating criteria, matching how the
current champion was itself selected on combined evidence rather
than a single dominant metric.

`champion_promotion.min_gini_delta` is left unset in config.yaml,
deliberately — no real promotion has happened yet to calibrate a
defensible threshold against, the same position taken for the
cost-assumption and review-capacity thresholds elsewhere. Until
set, the script computes and reports every delta but
`material_win` is unconditionally `False` — verified directly: a
configured threshold of `None` was tested against a materially
better synthetic challenger and correctly still returned
`material_win: False`, confirming the script cannot accidentally
recommend a promotion by falling back to an implicit default.

Uses MLflow's alias-based Model Registry (`champion`/`challenger`
aliases on one registered model name), not the deprecated
stage-based API, consistent with `mlflow>=2.13.0`. Registration is
idempotent on run_id — reruns reuse the existing version rather
than creating duplicates.

`--apply` refuses outright if promotion criteria are not met,
regardless of the flag being set, and even when met requires
typing `PROMOTE` verbatim — no `-y`/`--yes` shortcut exists.
Even on a successful apply, `config.yaml` and `champion_metadata.json`
are explicitly NOT updated by this script — the same "evidence
generated, human applies" boundary used for `business_cutoff`
(Decision 19), stated in the script's own printed output.

### Verification
No real challenger exists yet in this project — self-tested by
comparing the current champion against itself (its own stratified
and temporal run IDs passed as the "challenger"). Result: delta_gini
0.0000 on both splits, `material_win: False` on both,
"promotion NOT supported," and the challenger correctly registered
under `sentinel_credit_champion` version 1 with the `challenger`
alias. This is the first registered model version this project has
ever created.

### Decision
The workflow is implemented and verified working, but has not been
exercised against a genuine challenger, since none currently
exists. `min_gini_delta` remains unset until a real promotion
decision requires it — recorded as an open input, not a defect.

### Addendum — Champion Run Selection and Rollback (14th October 2026)
Re-running training for the same champion creates more runs carrying
the same champion tags; three per split existed by October. Selecting
the latest is intended: a rerun of the same champion should reproduce
it. The owner set the rule explicitly: **if the reruns have the same
Gini, use the latest; if the latest has worse Gini, roll back to the
previous run and do not adopt the latest.**

`find_champion_run()` now implements it:
- Only runs with `eval_gini` logged (scored by `evaluate_models.py`)
  are considered. Unscored runs are skipped and listed.
- **Same Gini** (within 0.0001, the 4 decimal places `eval_gini` is
  logged at, i.e. rounding, not a threshold): use the latest.
- **Latest lower:** use the previous run, with a warning. The rollback
  is recorded in `promotion_recommendation.json` under
  `champion_run_selection.rule = ROLLED_BACK_latest_worse`.
- **Latest higher:** use the latest, flagged, because reruns of the
  same champion are expected to match.

Tested on fake runs for every case, and on the real store, where it
selects the same runs as before (`3d169840` stratified, `e97c1eee`
temporal; all reruns 0.5656 / 0.5675 from git `0aeceb3`).

**Scope limit.** This rollback governs which run the promotion
comparison treats as the champion. It does **not** roll back the
served model. `serving/app.py` loads
`models/artifacts/champion_calibrated.joblib`, which
`train_baseline_models.py` overwrites on every rerun, so a worse
rerun is already the live model before this rule runs. Rolling back
what is served would need the API to load a registry version (the
`champion` alias), or training to refuse to overwrite the package
when the result is worse. Recorded as open (audit A25).

Caveat: choosing between two runs by their test-set Gini is a
selection on the test set. It is acceptable here only because the
runs are meant to be identical, so a difference signals a fault to
investigate, not a better model to prefer.

---

## Decision 22 — Portfolio Forecasting (Prophet), LSTM Explicitly Deferred

### Challenge
The architecture's Intelligence Core names "Forecasting: Prophet +
LSTM, macro scenarios, stress testing, IFRS 9 impact" as a
capability. Before building anything, a direct check of the
database (`raw.transactions`, `raw.repayments`, `raw.customers`,
`feature_store.customer_features` — four tables, confirmed via
`information_schema.tables`) established that no macro/economic
data exists anywhere in this project — nothing for unemployment,
interest rates, CPI, or any other external series a real stress-
testing or IFRS 9 macro overlay would require. This split
Forecasting into two genuinely different pieces of work: portfolio
time-series forecasting (buildable immediately from existing raw
data) and macro-scenario stress testing (blocked on a real data-
sourcing decision, deliberately not fabricated to fill the gap).

### Method
`forecasting/build_portfolio_timeseries.py` constructs two monthly
series from `raw.transactions`/`raw.repayments` directly — a
portfolio-level view, independent of the customer-level modelling
pipeline. Originations: transactions grouped by month, restricted
to `application_approved = 1`. Default rate: NOT a naive count of
defaults recorded per month — every repayment row is joined back to
its transaction to recover the ORIGINATION month, since a default
surfacing in month N from a cohort originated in month N-4 belongs
to that earlier cohort's risk, not the recording month. Only
`outcome_observed = 1` transactions contribute, the same right-
censoring discipline as Decision 10/12 applied at cohort level.
Verified against synthetic data with a known unresolved case before
running against real data: January cohort correctly showed 1/3
defaulted (a late-surfacing default correctly attributed to its
origination month, not the month it was recorded), and a
deliberately unresolved February transaction was correctly excluded
rather than counted as a non-default.

Against real data: 34 monthly cohorts, January 2022 through October
2024, zero unresolved transactions (consistent with Decision 16's
confirmed 0% right-censoring). Originations grow near-exponentially
(60 to 17,111, ~300x). Default rate is noisier but shows a genuine
mild upward drift (~0.5-0.8% early, ~0.8-1.1% later).

`forecasting/forecast_portfolio.py` fits Prophet on each series,
backtesting against the last 6 held-out months before producing
any forward forecast, and reports MAE and MAPE (guarded against
near-zero denominators) rather than a single potentially-misleading
metric. Forecast output beyond the data's own end date is visually
and programmatically distinguished (`is_beyond_data` flag, separate
plot color/style) from the backtested portion, since nothing
validates it against anything.

**LSTM was deliberately not built**, not omitted silently: ~34
monthly points is far too little history to train a neural sequence
model without it memorizing noise rather than learning a real
pattern — a confidently-wrong LSTM would be a worse outcome than no
LSTM, the same reasoning already applied to declining invented cost
ratios elsewhere in this project. Recorded as future work once
meaningfully more monthly history exists.

**The originations forecast carries a caveat load-bearing enough to
be verified empirically, not just asserted**: `config.yaml`'s
`data_generator.origination_end_date` (2024-10-31) and
`n_customers` (50000) are fixed generation-schedule parameters, not
market dynamics — the near-exponential growth reflects the
synthetic population's onboarding schedule reaching its endpoint.
The backtest confirms this directly: the final held-out months
(2024-05 through 2024-10) curve sharply above Prophet's fitted
linear trend, and the actual October value lands far outside what
the model predicted — visible in `forecast_origination_volume.png`,
not just stated in a docstring. MAPE (24.45%) is high for exactly
this reason. Default rate does not carry the same problem to the
same degree — it reflects portfolio risk evolving over the observed
period, a legitimate forecasting target, though still subject to
this project's general synthetic-data and PD-only scope (Decision
4 addendum).

### Verification
Run twice against identical real data — once via container-relayed
values, once natively from PowerShell — producing matching results
(default rate MAE 0.001633 vs 0.001634, MAPE 18.43% vs 18.44%;
origination MAE and MAPE identical to four decimal places).
Confirmed reproducible, not a one-off artifact.

### Decision
Portfolio time-series forecasting (Prophet, both series, backtested)
is implemented and verified. LSTM and macro-scenario stress testing
/ IFRS 9 macro overlay are recorded as explicit future work, gated
respectively on more historical data and a real macro-data-sourcing
decision — neither fabricated to appear complete.

---

## Decision 23 — Automated Approval Gate (Evidence Gate on Promotion)

> **Written retrospectively, 14th October 2026.** The gate was built
> and committed on 11th August 2026 (`e59a042`, whose message names
> "Decision 23"), but its log entry was never written. The 12th
> October audit found the gap (B1). This entry records what the code
> committed on 11th August actually does, read from
> `pipeline/approval_gate.py` and `pipeline/promote_champion.py`. It
> does not reconstruct reasoning that was not written down at the
> time. Later changes to the gate are recorded in Decision 24.

### Challenge
Decision 21's promotion workflow compared a challenger's Gini against
the champion's, but nothing connected that comparison to monitoring.
A challenger with a material Gini win could have been promoted while
the population it would score was drifting, or while the monitoring
evidence was weeks old. Decision 20's own entry left "feeding this
evidence into an automated Approval Gate" as future work.

### Decision
`pipeline/approval_gate.py` reads the evidence `monitor_drift.py`
already writes (`reports/monitoring/monitoring_run_metadata.json`)
and returns PASS or BLOCK. It recomputes nothing. Gating checks:
1. **Monitoring evidence exists.**
2. **The evidence is recent:** no older than
   `approval_gate.max_monitoring_age_hours` (48h). This is the one
   governance threshold in the project that is defaulted rather than
   left unset, because it is derived from an operational fact already
   decided (the DAG's `@daily` schedule, tolerating one missed run),
   not from a business judgement nobody has made.
3. **Zero active drift or data-quality alerts.**

Portfolio bad-rate shift is **reported, not gated**. No defensible
threshold for acceptable bad-rate drift exists, and inventing one is
the same mistake already declined for `cost_assumptions` and
`min_gini_delta`.

**Wiring:** `promote_champion.py` runs the gate as a subprocess
(`check_approval_gate()`) and reads its result file. A PASS is
**necessary, not sufficient**: Decision 21's Gini criteria still apply
independently. A BLOCK overrides everything, so an otherwise-material
Gini improvement is not applied during active drift.

### Scope, stated plainly
This is an **automated evidence gate**, not the human approval
Decision 9 describes. Human approval in the current code is the
typed `PROMOTE` confirmation in `promote_champion.py --apply`. That
confirmation records no approver identity, so Decision 9's "approver
ID logged in MLflow" is not implemented (audit B5).

### Regulatory Context
PRA SS1/23 Principle 4 expects model use to be challenged by current
evidence. The gate makes stale or drifting monitoring evidence block
a promotion mechanically, rather than relying on a reviewer to
notice it.

---

## Decision 24 — Drift History Logging Fixed, and Gate Status Derived at Read Time

> **Numbering note.** Commit `e59a042` describes "Decision 23:
> automated approval gate", but no Decision 23 entry had been written
> when this entry was created. Decision 23 was back-filled
> retrospectively on 14th October 2026 (above).

### Challenge
The 12th October 2026 audit (`docs/audit_findings_2026-10-12.md`)
found two problems in the monitoring-to-governance path.

First, `log_drift_history()` in `pipeline/monitor_drift.py` inserted
a run's rows into `reporting.feature_drift_log` and then crashed:
leftover code from an earlier `pandas.to_sql` attempt called an
unimported `create_engine`. The table held exactly one run (14
rows, 13th August 2026 23:41), and that run never wrote its
metadata JSON or MLflow record.

Second, `reporting.governance_status_latest` returned whatever
`approval_gate.py` last wrote. Its latest row was a PASS from 13th
August 2026. On 12th October that PASS was still displayed on the
Power BI badge, although re-running the gate would have blocked on
evidence roughly 62 days old. The status was real data, but it was
frozen at write time.

### Why the scheduled runs stopped
Airflow's run history (`airflow dags list-runs`, read 13th October
2026) and the MLflow monitoring experiment agree:
- **11th August:** failed setup runs, then a manual success at 12:07.
- **12th and 13th August 00:00 UTC:** scheduled successes.
- **14th August to 11th October:** no runs.
- **12th October 23:53 UTC:** when the stack next came up, Airflow ran
  two catch-up runs (`scheduled__2026-08-14` and
  `scheduled__2026-10-12`) **concurrently**.
- **13th October 00:00 UTC:** the scheduled run.

All of the October runs succeeded. The DAG is unpaused (`is_paused =
False`). Both Airflow Variables exist, and `sentinel_db_host` is
`host.docker.internal`.

The cause is the host, not the DAG.
- The Astro containers restart automatically whenever Docker is
  running, but Docker Desktop's `AutoStart` setting is off.
- On 12th October, Docker Desktop also showed a launch error about
  stale socket files in `%LOCALAPPDATA%\Docker\run\` (last written
  11th August), though the engine did come up afterwards.

No Docker means no scheduler, and no runs.

**Consequence for how this project describes scheduling:** the DAG's
`@daily` schedule only fires while the local Astro/Docker stack is
up. It is described as **scheduled while the local stack is
running**, not as daily monitoring.

### Decision
1. **Drift logging.** `log_drift_history()` now inserts once via
   raw psycopg2 (the approach the earlier attempts landed on) and
   no longer creates the table at runtime. Each run gets a
   `run_id`, written to every row, `monitoring_run_metadata.json`,
   and the MLflow run's tags. The DDL lives in
   `sql/reporting/create_feature_drift_log.sql`, including the
   `ADD COLUMN IF NOT EXISTS run_id` migration.

   The 14 pre-existing rows keep `run_id = NULL`. They are a real run,
   but no id was recorded for it, and one is not invented after the
   fact.
2. **Gate status at read time.** `governance_status_latest` now
   computes the evidence's age *now*:
   `monitoring_age_hours` plus the time since the gate ran. A PASS
   whose evidence exceeds the row's own `max_monitoring_age_hours`
   reads `STALE`.
   - No second threshold is introduced. The view uses the gate's
     stored value, so the two cannot disagree.
   - A BLOCK stays BLOCK, because age cannot make it less blocking.
   - `gate_status` keeps its column name so existing report bindings
     still work.
   - New columns: `recorded_gate_status`, `evidence_age_hours_now`,
     `is_stale`.

   Because Power BI queries the view via DirectQuery, the badge
   changes without any pipeline run. A stopped pipeline now shows up
   as STALE instead of an indefinite PASS.
3. **Canonical SQL matches the live database.** `segment_key`, applied
   earlier by the one-off `fix_*.sql` scripts, is folded into
   `sql/reporting/create_reporting_views.sql`. The scripts themselves
   are kept as a record of the fix, moved from the project root to
   `sql/reporting/maintenance/`; `inventory_tables.sql` moved to
   `sql/diagnostics/`. Re-applying the file
   against the live views succeeded, which confirms the columns
   match.

### Limitations, stated rather than hidden
- **Time basis (resolved — see Addendum below).** `evaluated_at` is a naive local timestamp. Gate runs
  on the Windows host match the server's `Europe/London`
  `localtimestamp`. Runs inside the UTC container read up to one hour
  young during BST. The 13th August rows show the skew is real:
  - container `generated_at` 00:00:35 UTC = 01:00:35 London
  - a gate run at 01:00:23 London evaluated the previous day's
    evidence and recorded its age as 25.0h, against a true age of
    about 24h

  Fixed by the timestamptz migration in the Addendum below.
- **Accumulated drift rows are not a trend.** Reference and comparison
  are fixed snapshots, and 79.3% of the comparison customers are in
  the reference partition. Repeated runs log near-identical values.
  This is unchanged by this decision (audit A2/A3).
- **The same skew affects `feature_drift_log.logged_at` (resolved — see Addendum below).** Container
  rows are stored in UTC and host rows in London time. The 13th
  October 01:00 BST container run is stored as `00:00:12`, which
  sorts before a host run made at 00:54 BST. Order by `logged_at`
  across sources is unreliable until timestamps carry a time zone.
- **Five logged runs, identical values.** Every feature has exactly
  one distinct PSI and one distinct comparison mean across all five
  runs (13th August to 13th October). This confirms that accumulated
  rows on fixed snapshots are not a trend.
- **The gate is not scheduled (resolved — see Addendum below).** Container runs write drift rows but
  no governance rows. The badge stays current only when
  `approval_gate.py` is run, and otherwise goes STALE after 48h, by
  design.

### Regulatory Context
PRA SS1/23 Principle 4 expects ongoing monitoring to be effective,
not merely present. A governance indicator that keeps reporting
PASS after its evidence has lapsed overstates assurance. Deriving
staleness at read time means the indicator fails safe when the
monitoring it depends on stops.

### Addendum — Gate Scheduled, Timestamps Made Absolute (13th October 2026)
Two items from the Limitations above are resolved.

**The gate now runs in the DAG.** `infra/dags/monitor_drift_dag.py`
gained a third task, `run_approval_gate`, downstream of
`compute_drift` and alongside `evaluate_alerts`. Found drift
therefore still produces a recorded BLOCK row, not a missing row.
If `compute_drift` fails, the gate is skipped, and the badge turns
STALE on its own after 48h, as designed.

`approval_gate.py`'s exit codes were made unambiguous: 0 PASS,
1 BLOCK, 2 could not complete or could not record. Previously an
unhandled error also exited 1, which a caller could not tell apart
from a recorded BLOCK. The new `--require-db-write` flag (used by
the DAG) makes a failed `governance_status_log` insert exit 2 and
fail the task, rather than surfacing two days later as STALE.
`promote_champion.py` reads the result file, not the exit code, so
it is unaffected.

`max_active_runs=1` is set, so catch-up runs after downtime no
longer run concurrently against the same CSV and metadata JSON.

**Timestamps are absolute instants.**
`governance_status_log.evaluated_at` / `created_at` and
`feature_drift_log.logged_at` are now `timestamptz`, via
`sql/reporting/migrations/2026_10_13_timestamptz.sql`. Each existing
row was converted from the zone it was actually written in,
established from evidence (Airflow run records for the three
container runs; host runs for everything else), not assumed.

Writers now send offset-aware values:
- `approval_gate.py`: `evaluated_at`, and the evidence-age
  calculation itself
- `monitor_drift.py`: `generated_at` and `logged_at`
- `run_id` is now stamped in UTC (`...Z`)

The badge view compares against `now()`. After migration, the host
run of 12th October 23:54:35 UTC correctly sorts before the
container runs at 23:59 and 00:00 UTC; previously it sorted after
them.

**Airflow incident during verification.** The first triggered
verification run (`manual__2026-10-13T18:13:46`) never started.
- Airflow's API server had hung. The worker received `ReadTimeout`
  from `http://api-server:8080/execution/`, and the task sat in
  `queued` for over three hours.
- This was unrelated to the DAG change: the task failed before any
  project code ran.
- Restarting the `api-server` container restored the execution API
  (HTTP 200), and the run was cleared via the REST API.

On re-run, `compute_drift` logged run `drift_20261013T214704Z_181478be`
at 21:47:05 UTC (correct, from the UTC container). The subprocess
then stalled for about 9.5 minutes in disk I/O (process state `D`,
wait channel `p9_client_rpc`): a single request to the WSL2 9P file
server backing the Windows bind mount went unanswered. Fresh reads
on the same mount, including a full walk of `mlruns/`, completed
instantly during the stall, and no files were OneDrive cloud-only.
The stall cleared on its own, and the cause is not established.

**Verified end-to-end (13th October 2026, 21:44–21:58 UTC).**
- All three tasks succeeded: `compute_drift`, `evaluate_alerts`,
  `run_approval_gate`.
- `feature_drift_log` holds 84 rows (6 runs).
- The gate, running inside the UTC container, wrote
  `governance_status_log` row 4: PASS at 21:58:20 UTC, evidence age
  0.03h. That age is correct; before this change, a container gate
  run would have used naive local times and mis-measured it.
- `governance_status_latest` reads PASS with evidence 0.0h old.

**Hardening added after the stall.** With `max_active_runs=1`, a
task that never returns would block every later scheduled run. The
DAG's `default_args` now set `execution_timeout = 20 minutes`
(healthy runs take about a minute), so a hung task fails and frees
the schedule. The badge's STALE logic remains the backstop.

---

## Decision 25 — Drift Comparison Without Self-Overlap, and a Cohort Time Axis

### Challenge
The 12th October audit found two reasons the drift monitor said
less than it appeared to (A2, A3):
1. **79.3% of the comparison population was also in the reference.**
   3,854 of the 4,857 "temporal test" customers were in the
   champion's stratified training partition. PSI between a
   population and a largely identical copy of itself is pulled
   toward zero.
2. **Both populations were fixed snapshots.** Five logged runs two
   months apart had exactly one distinct PSI value per feature.
   Accumulated rows in `feature_drift_log` were therefore not a trend.

### Decision
1. **Overlap excluded.** `load_comparison_population()` now removes
   any customer that is also in the training reference, and reports
   the count (`n_comparison_overlap_excluded`). The default comparison
   is now the 1,003 later-originating customers the model never saw.

   Effect on the 13th October data:
   - every feature still "stable", 0 alerts
   - several PSI values rose once self-comparison was removed:
     `avg_order_value_gbp` 0.042 → 0.063, `savings_balance_gbp`
     0.003 → 0.016, `num_credit_accounts` 0.002 → 0.016
   - the previous numbers understated drift
2. **Origination-quarter cohort view.** Each run also computes PSI
   per feature for every origination quarter of held-out customers
   (the stratified test partition) against the training reference.
   - Written to `reporting.feature_drift_by_cohort`, keyed by
     `cohort_period`, so the time axis is when customers arrived.
     That is real variation in the data, computed from actual cohorts.
   - It is not a run history and is not backfilled. Each run replaces
     the table with its own computation, identified by `run_id`.
   - Quarters under 200 held-out customers are listed, not scored.
   - Each row carries `psi_noise_floor`, the PSI expected from
     sampling alone at that size (~(bins−1)·(1/n + 1/N)), so a small
     cohort's PSI can be read against its own noise level.
   - Informational only: it does not change the alert count or the
     approval gate.

### First reading (13th October 2026, observation not conclusion)
- 11 quarters scored; 2022Q1 is too small.
- Most quarters sit at 2–5× their noise floor, with max PSI 0.02–0.08.
- **2024Q4 reaches the moderate band:**
  - `pct_medium_risk_category` 0.106, `pct_high_risk_category` 0.098
  - about 6× its noise floor
  - 2024Q4 is a partial quarter (origination ends 31st October 2024)
- A plausible but unverified explanation: customers who have made
  only one or two purchases have category percentages at 0 or 1
  rather than averages. That would make this a short-history effect,
  the same family as the exposure-time problem (Decisions 12/14), not
  market drift.

### Regulatory Context
PRA SS1/23 Principle 4 expects monitoring to be capable of detecting
change. A comparison that overlaps its reference by four-fifths
cannot, and a log of identical rows can be mistaken for evidence of
stability. Both are corrected here, and the remaining limits are
stated rather than implied.

---

## Decision 26 — pct_late_payments: Removal Evidence (decided in Decision 27)

> **Status: decided 14th October 2026 — remove (see Decision 27).**
> The owner ruled that leaking outcome information into the model is
> a valid reason for removal, independent of the small Gini cost.
> The three choices below were resolved in Decision 27. The evidence
> in this entry stands as recorded.

### Challenge
Audit finding A6: the generator writes `payment_status = 'Late'` for
three different events:
- the independent 5% late draw
- **full cure payments** from arrears
- **partial cure payments** from arrears

Instalments after default are written as `Defaulted` and sit in the
`AVG()` denominator. Decomposed on the modelable population:
- **39%** of defaulters' `Late` rows are cure events (23% for
  non-defaulters)
- defaulters carry 723 post-default rows in the denominator

So `pct_late_payments` partly measures the arrears process default
emerges from, and partly information that only exists after the
outcome. Decision 12's claim that late-payment features are
"independent of the default path" no longer holds under the
Decision 10 generator. This is the same mixed-event basis on which
`avg/max/total_days_late` were excluded as resolution-adjacent.

### How it was judged
On **feature validity**: would this information exist at decision
time, independent of the outcome? A leaky feature usually looks good
in cross-validation, so a Gini drop on removal is **not** a reason to
keep it. Performance was measured to quantify the cost of removal,
not to decide it. Script: `models/evaluate_feature_removal.py`.
Evidence: `reports/feature_removal/pct_late_payments/`.

### Evidence (champion hyperparameters held fixed)
| | 14 features (champion) | 13 features (no `pct_late_payments`) |
|---|---|---|
| Repeated CV Gini (5×3) | 0.5533 | 0.5479 (lower in 14/15 folds, mean −0.0054) |
| Repeated CV PR-AUC | 0.0911 | 0.0928 (higher in 10/15 folds) |
| Held-out stratified Gini [95% bootstrap CI] | 0.5656 [0.500, 0.626] | 0.5618 [0.494, 0.622] |
| Gini difference, paired bootstrap | — | −0.0038 [−0.0148, +0.0079], includes 0 |
| Held-out temporal Gini | 0.5675 | 0.5611 (−0.0064) |
| Temporal PR-AUC | 0.0876 | 0.0782 |
| Train/test Gini gap [95% CI] | 0.0665 [0.001, 0.137] | 0.0625 [−0.003, 0.135] |
| ECE, decile bins / Brier | 0.0040 / 0.0254 | 0.0037 / 0.0254 |
| Decisions flipping at 0.0671 | — | 135 of 8,321 (1.62%) |

The cost of removal is small, consistent in repeated CV, and not
distinguishable from zero on the held-out split. Calibration is
unaffected.

**Train/test gap (audit A7).** The champion's 0.067 gap has a 95%
CI of [0.001, 0.137]. It only just excludes zero, and its width is
dominated by the 226 test defaults. This is not evidence of material
overfitting.

**Hyperparameter check (audit A7).** 5-fold CV on the training
partition; early stopping on an inner 15% slice of each fold.
- The champion's fixed configuration had the best validation PR-AUC
  (the selection metric) for both feature sets: 0.096–0.099, against
  0.081–0.088 for depth 2/3 × learning rate 0.005/0.01 with early
  stopping.
- Depth 2 roughly halves the train/validation Gini gap (≈0.04–0.05
  vs ≈0.096) at similar validation Gini, but with lower PR-AUC.
- Early stopping chose 118–565 trees, not more than 300.
- Caveat: the early-stopped configurations trained on 85% of each
  fold, a small handicap.
- Conclusion: the grid-edge selection is not leaving an obvious gain
  on the table. A7 is closed as "checked, no change warranted",
  with depth 2 noted as the lower-variance alternative if
  generalisation becomes the priority.

### Workflow outcome
`promote_champion.py` (Decision 21) registered the challenger and
returned **"promotion NOT supported"**. The rule requires a material
Gini **win** on both splits, and `min_gini_delta` is unset.

### Open for decision
1. **Rule conflict.** Decision 21's promotion rule can only promote a
   model that is *better* on Gini. A validity-driven removal will
   almost always cost a little Gini, so under the current rule a
   validity fix can never be promoted. Adopting the 13-feature model
   needs either an amendment to Decision 21 (for example, a
   non-inferiority criterion, a pre-declared maximum acceptable Gini
   loss when removing an invalid feature, approved by a human) or an
   explicit, logged override. This is a governance choice and is not
   made here.
2. **If adopted:**
   - recalibrate the business cutoff for the new score distribution
     (Decision 13/19)
   - regenerate SHAP and the 14→13 governance review, then sign it off
     (still pending for the 14-feature set, audit A13)
   - remove `pct_late_payments` from `serving/app.py`, where the API
     hardcodes `0.0` for new customers, from the
     `customer_risk_snapshot` view comment ("the 14 frozen model
     features"), and from `govern_model_features.py`'s per-feature
     metadata
   - write a Decision 18 supersession entry
3. **Longer term:** a point-in-time late-payment feature computed from
   instalments before the first miss would not have this problem. That
   belongs to the v2 application-time redesign (Decision 18), not to
   v1.

---

## Decision 27 — Remove pct_late_payments (Leakage); v1.1 Champion via a Validity Promotion Path

> **Supersedes Decision 18's frozen feature set** (14 → 13 features).
> Amends Decision 21 by adding a second, evidence-bound promotion path.

### Decision
**`pct_late_payments` is removed from the model** because it leaks
outcome information (Decision 26):
- part of what it counts is the arrears-cure process default emerges
  from
- part is post-default rows that only exist after the outcome

The owner ruled on 14th October 2026 that leakage is a valid reason
for removal on its own. A small Gini cost on removal is the leakage's
contribution disappearing, not lost signal. The feature stays in the
feature store, moved to `RESOLUTION_ADJACENT_EXCLUDED` alongside the
`days_late` features excluded on the same mixed-event grounds.

### How it is applied (owner's choices)
1. **Promotion path: evidence override.** Decision 21's rule promotes
   only on a material Gini win, which a validity removal can never
   meet. `promote_champion.py` gains `--validity-override "<Decision
   N>" --validity-evidence <file>`, allowed only if all of these hold:
   - the cited decision exists in this log
   - the evidence file describes these exact challenger runs
   - on **both** splits, the 95% paired-bootstrap CI of the Gini
     difference (challenger minus champion) reaches zero or above,
     i.e. the loss is not distinguishable from zero

   The approval gate and typed `PROMOTE` still apply. No new numeric
   threshold is introduced; the bar is the evidence's own confidence
   interval. A removal with a real, measurable Gini loss is refused.
2. **Model: champion settings held fixed** (learning rate 0.01,
   depth 3, 300 trees), so the change is attributable to the feature
   alone. The Decision 26 hyperparameter check found these settings
   still best on the selection metric.
   `train_baseline_models.py --pin-xgboost-params` reproduces this in
   the governed training pipeline. Logistic regression and decision
   tree are re-searched as usual, as comparators.
3. **Files: new package folder.** The v1.1 champion is written to
   `models/artifacts_v1_1/`, selected by `config.yaml`
   `model_governance.artifacts_dir`, which every script and the API now
   read. The August v1 package in `models/artifacts/` is never
   overwritten: training refuses to write into a folder holding a
   different champion identity. Rolling back means pointing
   `artifacts_dir` and the identity values back at v1.

### Evidence for the override (`promotion_evidence.json`)
Challenger runs `936b28cf` (stratified) and `2fa5d7af` (temporal):

| Split | Gini difference | 95% CI | Loss distinguishable from zero? |
|---|---|---|---|
| Stratified | −0.0038 | [−0.0148, +0.0079] | No |
| Temporal | −0.0015 | [−0.0164, +0.0150] | No |

The temporal figure here pairs both arms at identical settings. The
−0.0064 shown by the plain promotion comparison is against the
champion's temporal run, whose hyperparameters were searched
separately on the temporal training set, so it mixes the feature
change with a settings difference.

### Sequence and status
1. Evidence: done.
2. **Owner runs** `promote_champion.py --validity-override "Decision 27" ... --apply`
   and types `PROMOTE`. This is the human approval step; it is not
   performed on the owner's behalf.
3. Then, in one change set:
   - `MODEL_FEATURES` 14 → 13
   - config identity (v1.1, `13_features_v1_1`, `artifacts_dir`)
   - pinned retrain into the new folder
   - evaluation on both splits
   - business cutoff recalibrated (training-only OOF, Decision 19
     method)
   - SHAP and governance review regenerated
   - API's hardcoded `pct_late_payments = 0.0` removed
   - monitoring and the gate re-run

### Regulatory Context
PRA SS1/23 Principle 3 requires model inputs to be appropriate for
their intended use. An input that partly encodes the outcome is not,
however much it helps a metric. The override keeps that judgement
bounded by evidence and by a recorded human decision.

### Outcome — v1.1 Live (14th October 2026)
**Promotion.** The owner ran the validity-override promotion and
typed `PROMOTE` at 22:58. Registry `sentinel_credit_champion`
version 3 (run `936b28cf`) holds the `champion` alias, tagged:
- `promotion_path=validity_override`
- `decision_ref=Decision 27`
- `removed_feature=pct_late_payments`
- `replaced_champion_run_stratified=3d169840`

**Switch (one change set):**
- **Features.** `MODEL_FEATURES` 14 → 13. `pct_late_payments` moved to
  `RESOLUTION_ADJACENT_EXCLUDED`; the column stays in the feature store.
- **Config identity.** `v1.1`, `13_features_v1_1`,
  `v1_1_validity_promotion`, `artifacts_dir: models/artifacts_v1_1`.
- **Retrain.** `train_baseline_models.py --pin-xgboost-params` from the
  v1 metadata. XGBoost held at the v1 settings; logistic regression
  and decision tree re-searched. The new package's XGBoost reproduces
  the promoted challenger exactly (Gini 0.5618 stratified, 0.5611
  temporal), so the served model is the approved model.
- **v1 preserved.** `models/artifacts/` unchanged (checksums verified
  before and after). v1's report evidence copied to
  `reports/_archive_v1_14f_2026-08/` before regeneration.
- **Cutoff.** Recalibrated by the Decision 19 method (training-only
  OOF, 15% review capacity, still a placeholder): **0.0657** (v1:
  0.0671). OOF recall 50.11% at 9.09% precision. F1 (0.0764) and
  Youden (0.0400) kept as comparators.

**Held-out results at the operating cutoff:**

| | v1 (14 features, 0.0671) | v1.1 (13 features, 0.0657) |
|---|---|---|
| Stratified Gini / PR-AUC | 0.5656 / 0.0897 | 0.5618 / 0.0909 |
| Stratified recall / precision | 54.87% / 9.60% | 57.96% / 9.90% |
| Temporal Gini | 0.5675 | 0.5611 |
| Temporal recall / precision | 38.71% / — | 40.32% / 8.65% |
| Calibration (ECE uniform bins, stratified) | 0.0008 | 0.0015 |

**Unchanged.** The Decision 19 operating-policy finding persists:
`thin_file_flag=0` defaulters are still 100% missed on both splits
(FNR 1.0000). That governance item stays open.

**SHAP (v1.1).** Additivity passes (max error 1.8e-06). Ranking:
`thin_file_flag` 29.0%, `savings_balance_gbp` 18.4%,
`avg_n_instalments` 12.1%, `avg_order_value_gbp` 11.3%,
`pct_high_risk_category` 10.4%, then the rest.
`existing_monthly_obligations_gbp` is lowest at 0.33%, and the
governance review flags it as a `simplification_candidate`.

**Governance review.** Regenerated for the 13-feature set
(`reports/feature_governance/feature_governance_review.md`). Every
feature is `PENDING_MANUAL_REVIEW` **by design**: final retain/remove
decisions are the owner's to sign off and are not taken here (audit
A13, still open). Segment diagnostics were also regenerated.

**Downstream:**
- **API.** Serves v1.1 (smoke-tested: `/model/info` reports v1.1, 13
  features, cutoff 0.0657; new- and existing-customer scoring return
  200).
- **API code.** The hardcoded new-customer `pct_late_payments = 0.0`
  is removed. If a future package expects a feature the API can't
  build, it now fails with a clear error.
- **Monitoring.** Drift monitoring runs on 13 features (0 alerts); the
  gate is PASS.
- **Power BI view.** `customer_risk_snapshot` still carries
  `pct_late_payments` as a known, documented exception until the
  report is confirmed not to use it (audit A26).

**Rollback** to v1 is documented in `config.yaml` next to
`artifacts_dir`. It means restoring the v1 identity values and cutoff
0.0671, restoring the feature in `MODEL_FEATURES` (git `e42c108`), and
moving the registry alias back.

---

## Decision 28 — One Scoring Path for API and Dashboard; Dashboard Rebuilt on Evidence Only

### Challenge
Adding an interactive Credit Risk Assessment page to the Streamlit
dashboard meant a second place that turns an applicant into model
features. Copying the FastAPI service's feature construction would
create two implementations that could drift apart: the same applicant
scored differently depending on the screen used. That is the
training-serving skew Decision 3 exists to prevent, moved to serving.

The audit also found the dashboard partly broken and overstated:
- the Model Performance page crashed on load under the pinned
  Streamlit 1.35 (`st.dataframe(width='stretch')`, a newer API)
- the home page advertised explainability, fairness and drift pages
  that did not exist
- the governance tile read the gate's JSON file, so it could never
  show STALE (Decision 24)

### Decision
1. **One feature-construction module.** `serving/features.py` holds
   the first-application feature logic and the merchant risk tiers
   (mirroring `sql/02_feature_store.sql`). `serving/app.py` and the
   dashboard's `dashboard/scoring.py` both call it. The champion is
   loaded from `model_governance.artifacts_dir`, and the decision uses
   `model.business_cutoff` with `>=`, as everywhere else.
2. **The dashboard shows evidence, never placeholders.** Every page
   reads a live query, `config.yaml`, or evidence the pipeline wrote.
   When evidence is missing, the page says which script produces it.
   Seven sections replace the previous three, each mapped to something
   that exists:
   - Executive Overview
   - Credit Risk Assessment (new)
   - Portfolio Intelligence (absorbs Forecasting)
   - Model Performance
   - Explainability
   - Model Monitoring
   - Governance & Audit
3. **Honest outputs on the assessment page.** It shows the calibrated
   PD and a review recommendation at the cutoff ("refer for manual
   review" / "below review threshold"). It does not show a credit
   score, a limit or an approval.

   The context around the score comes only from real sources:
   - percentile among the champion's held-out test predictions
   - SHAP drivers from the explanation model, labelled as raw margin
   - warnings when an input falls outside the training range

   The Decision 18 scope warning is always visible. Nothing typed is
   stored.
4. **One design system.** `.streamlit/config.toml` plus
   `dashboard/theme.py` hold the palette, typography, Plotly template
   and components. Status is never shown by colour alone. The
   assessment page uses a separate, quieter ivory register.

### Verification
- **All seven pages:** run headless under `streamlit.testing`
  `AppTest` with no exceptions.
- **Score parity:** a submitted new-applicant assessment matches the
  FastAPI service exactly (PD 0.0318, same decision).
- **Session state:** the result persists across reruns.
- **Lookups:** unknown customer IDs are reported as not found, and
  existing-customer scoring matches the API (0.1117).
- **API unchanged:** API scores before and after the refactor are
  identical (0.033 / 0.0091 / 0.1117).
- **Rendering:** checked at 1,440 px and 375 px widths, with no
  horizontal overflow on any page.

Found and fixed while verifying:
- a form/session-state key collision that broke every new-applicant
  assessment
- a segment table overflowing its column
- clipped chart labels and titles at phone width
- the sidebar covering the page on phones

### Limits
- The dashboard runs locally; nothing is hosted (Decision 6).
- The SHAP explanations are of the uncalibrated explanation model
  (Decision 14).
- The 15% review capacity behind the cutoff is still a placeholder
  (Decision 19).

---

## Decision 29 — Public Demo Scores In the Browser (ONNX), Gated on Two Parity Checks

### Challenge
The project needs a public, recruiter-facing demonstration where a
visitor can enter an application and get a genuine score from the
real champion. Two routes existed (handoff):
1. Host the FastAPI service and have the site call it. That means real
   infrastructure with cost and upkeep, and an endpoint exposed to
   strangers.
2. Convert the model to ONNX and score in the visitor's browser with
   onnxruntime-web. No server and no database; a permanent static link.

Route 2 carries one specific risk: a converted model that is subtly
different from the governed one. A demo that quietly scores
differently from the champion would undermine the project's central
claim.

### Decision
**Route 2, in-browser ONNX, chosen by the owner on 15th October 2026.**
Nothing is built on the converted model until it passes two parity
checks against the Python champion. The pass rule was fixed before
running: max |Δ probability| below 1e-5, and **zero** decision flips at
the configured cutoff.

`website/tools/export_model.py` converts the full champion — five
XGBoost models, each with its own Platt calibrator, probabilities
averaged — into one ONNX graph (skl2onnx 1.17, onnxmltools 1.12,
opset 17 / ai.onnx.ml 3).

**Conversion lesson.** onnxmltools requires XGBoost split features named
`f0..fN`, and the champion's boosters carry real feature names.
Names are stripped on a **copy** only. Column order is unchanged, and
parity is measured against the original, unmodified model, so a
mapping error could not pass silently.

### Evidence
| Check | Engine | n | max \|Δ\| | Decision flips |
|---|---|---:|---:|---:|
| Modelable population | onnxruntime 1.19.2 (Python) | 41,604 | 5.96e-08 | 0 |
| Edge cases (input extremes + 500 random points) | onnxruntime 1.19.2 (Python) | 529 | 8.94e-09 | 0 |
| Browser replay, incl. the 200 customers closest to the cutoff | onnxruntime-web 1.19.2 (WASM, Chromium) | 1,988 | 2.53e-08 | 0 |

The browser scored all 1,988 vectors in about 115 ms. Evidence:
`website/public/model/python_parity_report.json`, and
`website/public/parity.html`, which re-runs the browser check for any
visitor.

### Contract
`website/public/model/model_contract.json` publishes what the site
must respect:
- feature order and the cutoff, with its `>=` rule
- training ranges, for out-of-range warnings
- the Decision 18 scope statement and the synthetic-data label

The site may show a probability and a review recommendation only. No
credit score, limit or approval is invented.

### Not yet decided
- **Hosting.** GitHub Pages is proposed but not yet chosen.
- **Browsers.** The site pages are not built yet, and the browser
  check has run in Chromium only. Firefox and Safari should be checked
  before launch.

### Addendum — Site Built, Hosted on GitHub Pages, Deploys Gated on Browser Checks (15th October 2026)
**Hosting: GitHub Pages** (owner's choice). The repository is public,
so Pages is free. The site is fully static, which suits Pages: the
model runs in the visitor's browser, and no request reaches a server,
database or API.

**No third-party requests.** onnxruntime-web 1.19.2 (MIT) is bundled
from the npm package, with its integrity checked against the registry
checksum. The fonts are self-hosted (SIL OFL). A visit loads only the
site's own files.

**The repository's no-data rule is respected.** Nothing published is a
customer record:
- the parity vectors are **generated** applicants
- the simulator ships **aggregated counts** per score band × segment ×
  outcome (8 KB), whose rebuilt confusion matrix and segment error
  rates at the cutoff **exactly** equal `evaluate_models.py`'s
- the full-population parity evidence stays offline

**Pages:**
- **Live assessment:** the score, a comparison with a typical
  applicant, and the smallest single change that crosses the cutoff.
  All are real model evaluations in the browser.
- **Cutoff and fairness simulator:** workload, recall and segment error
  rates at any cutoff, using Decision 19's evidence.
- **Verification:** re-runs both parity checks for any visitor.

The JavaScript feature construction is a port of `serving/features.py`,
checked on 240 generated applications: **zero feature mismatches** and
no decision changed.

**Bugs found and fixed while verifying:**
1. **The browser model never loaded.** onnxruntime-web resolves a
   relative `wasmPaths` against its own script, not the page, so the
   page waited forever without an error. Fixed with an absolute URL,
   plus a 45-second load timeout that surfaces an error instead of
   hanging.
2. **Comparator buttons jumped to 0%.** Band edges are stored to 8
   decimal places while thresholds are not, so an exact-match lookup
   returned −1. Fixed with nearest-edge matching. The configured cutoff
   (0.0657) and the review-capacity threshold it was rounded from
   (0.065738) are now one button rather than two near-duplicates.
3. **The first boundary-vector generator could never finish.**
   0 of 20,000 uniformly random applicants scored above the cutoff,
   because risk depends on combinations of features. It was replaced by
   a bounded search: greedy coordinate ascent to find high-risk anchors,
   then a vectorised bisection. It now fails loudly if anchors cannot be
   found.

The first two would have shipped a broken site, and the browser check
caught both. That is why deploys are gated on it.

**Deploys are gated.** `.github/workflows/pages.yml` runs
`website/tests/browser_check.py` in headless Chrome before every
deploy. Any failure blocks publication. The checks cover:
- both parity checks
- an assessment matching the Python champion to six decimal places
  (0.031834)
- invalid input being rejected
- the simulator matching the saved evaluation
- each comparator landing on its threshold
- no horizontal overflow at 1,440 px or 390 px

All 12 checks pass locally.

**Still open:**
- Firefox and Safari have not been checked; CI runs Chromium.
- The Pages source must be set to "GitHub Actions" once, in the
  repository settings, by the owner.

## Decision 30 — Plain-English Reasons for the Applicant, One Source for API, Dashboard and Site

### Challenge
The point of Sentinel Credit, set out in Decision 7, is that an
applicant is told in plain English why the model flagged them, so
they can spot and challenge a wrong input. Audit (15th October 2026)
found this had never been built. The API returned a probability and a
decision only. The dashboard showed raw SHAP margins, which a risk
officer can read but an applicant cannot. The public site showed bars
against a typical applicant, with no sentences.

### Decision
Build the **customer output** of Decision 7 for new applicants. One
method and one set of wording are shared by all three surfaces.

**Method: one input at a time against a typical applicant.**
- For each input, the effect is the calibrated PD of the applicant
  minus the PD with **only that input** set to the typical applicant's
  value, everything else unchanged. It uses the calibrated champion,
  so the effects are in the same units as the reported probability
  (unlike SHAP, which is on the raw margin; Decision 14).
- The **typical applicant** is the median of each profile field over
  the champion's canonical training partition (33,283 customers), plus
  the median order value. Values are rounded to whole units, so the
  number shown is exactly the number used. It also has 3 instalments
  and the most common merchant category (Fashion & Clothing). The
  values are stored in `serving/typical_applicant.json`, generated by
  `website/tools/export_feature_examples.py`.
- **Referred** applications list up to three inputs that raised the PD
  most, then the inputs that counted in their favour. **Not referred**
  applications list up to two inputs in their favour, then those that
  counted against. Inputs equal to the typical value, or with an
  effect within 1e-6, are not mentioned.

**Wording rules** (Belton et al. 2025, Decision 7):
- Every sentence names the applicant's own value and the typical
  value, so a wrong input is easy to see. This is the features-based
  genre Belton found most effective for spotting data-input errors.
- The **decision rule is never shown**: no cutoff and no points. Belton
  found that showing it cut error detection by 18 percentage points.
- The wording never claims cause. It reads "the model links … to",
  not "because you …".
- A referral is described as review by a person, not a decline.
- Next steps tell the applicant to check the details and have wrong
  ones corrected, and say this is not financial advice.

**One source of truth.** The wording lives in
`serving/reason_templates.json` and the method in
`serving/reasons.py`. They are used by:
- the API: `/score` returns `explanation` for new applicants
- the dashboard: `dashboard/scoring.explain_new`
- the site: `website/public/assets/reasons.js`, a JavaScript port that
  reads the same two JSON files

### Evidence
| Check | Result |
|---|---|
| Site (JavaScript) vs Python, word for word: 240 generated applications, 15 referred, 687 sentences | 0 differences |
| Dashboard vs API, same applicant | identical explanation |
| API, new applicant (PD 0.033) | explanation returned, PD unchanged |
| API, existing customer | no explanation (by design, see below) |
| Headless browser checks (deploy gate) | 14 of 14 pass |

The word-for-word check runs on the public Verification page and in
the deploy gate (`website/tests/browser_check.py`). The site cannot
drift from the Python wording without blocking its own deployment.

### Observed while testing (not changed)
The reasons describe the model faithfully, including where it may
surprise a reader. In one referred test applicant, a £1,299 order was
listed as counting **in its favour** against the typical £583: the
champion links larger orders to slightly lower PD. That is the model's
learned behaviour on synthetic data, not a wording error. It is
recorded here rather than hidden, and is a candidate for the next
model review.

### Not built (remaining Decision 7 scope; built in Decision 31 the same day)
- **Error-type detection and genre selection.** Every applicant gets
  the features-based genre.
- **Existing customers.** Their features are lifetime behaviour
  aggregates (Decision 18), and no typical-customer wording exists for
  them yet.
- **The regulatory output**: approver ID and decision lineage per
  decision.
- The risk-officer output is partly met by the dashboard's SHAP
  chart and model version.
- **Interactions.** One-at-a-time effects ignore them, which the method
  line under every explanation states.

## Decision 31 — Decision 7 Completed: Error-Type Checks, Genre Selection, Three Outputs

### Challenge
Decision 30 built the applicant's plain-English reasons. It left the rest
of Decision 7 unbuilt:
- detect the error type and choose the explanation genre to match;
- a risk officer output;
- a regulatory record;
- explanations for existing customers.

The owner set the terms on 15th October 2026:
- **No routing.** When the model needs a person, the output states
  "Risk officer review required". Nothing is sent anywhere.
- **Choices are allowed if declared.** Where evidence cannot fix a
  number, we choose it, put it in config and say so, as with the
  review-capacity cutoff (Decision 19).

### Decision
One module (`serving/reasons.py`) returns three outputs for every
assessment, for both new applicants and existing customers:
- **explanation**, for the applicant, in the genre matched to the error
  type, with the review flag when a check fires;
- **risk_officer**: error type, genre and why, the three checks with
  details, and divergence (the model passes it but a check needs a
  person);
- **decision_record**: model identity, cutoff, methods, rules version,
  lineage. The approver is `null`, with the note that the approval gate
  records no approver identity (Decision 23).

**Three checks, in priority order** (Belton et al. 2025's scenarios):

| Check | Fires when | Limit and source | Genre |
|---|---|---|---|
| 1. Data input | A model feature is outside its training range, or a combination never seen in training occurs: a limited file with more than 23 months of history or more than 1 account; a full file with under 12 months or 0 accounts; commitments above 59.57% of income | All computed from the training partition | Features-based: the applicant's own values, so a wrong input is easy to spot |
| 2. Missing information (known blind spot) | Full credit file, and PD at or above the group's 95th percentile of training out-of-fold scores (line 2.23%) | Percentile: **declared choice**. Line: training OOF | Data-centric: how similar past customers fared; no per-detail attributions |
| 3. Over-reliance | One detail decides the recommendation on its own (setting it alone to typical flips it) **and** carries at least 60% of the total absolute effect | Share: **declared choice** | Data-centric |
| None | — | — | Combination: reasons plus one "past customers like you" line |

Any check firing means **risk officer review required**. Decision 7
named human review only for missing information. Extending it to the
other two error types is our choice, made because both are cases where
the model's output should not stand alone.

**Why the blind spot is a real "missing information" case** (evidence):
- Full-file customers never reach the cutoff. The highest training
  out-of-fold score is 5.11% against a 6.57% cutoff; the highest
  held-out score is 3.85%.
- So the model's false-negative rate on this group is 100% at the
  cutoff (Decision 27's stated limitation).
- Within the group the model still ranks risk:

| Full credit file | Customers | Defaults per 1,000 |
|---|---:|---:|
| Training, whole group | 22,982 | 11.1 |
| Training, top 5% (OOF) | 1,150 | 18.3 |
| Held-out, whole group | 5,648 | 10.8 |
| Held-out, top 5% (same line) | 224 | 31.2 |

  The top 5% capture 21 of 255 training defaults in the group and 7 of
  61 held-out defaults. A person reviewing these applicants is the
  practical response to a blind spot that the cutoff alone cannot see.

**"Past customers like you" figures** are default rates per band of each
feature in the training partition:
- quintile bands, or exact values for discrete features;
- groups of at least 500 (a declared choice);
- edges rounded to the precision shown, so the number shown is the
  number used;
- text pre-rendered in `serving/explanation_rules.json`.
They are aggregates; no customer row is published.

**£0 savings and £0 commitments are not data errors** (declared choice).
The training minimums are £7.92 and £0.99, so a strict range check would
flag the most ordinary answer. These two fields are checked against
their training maximum only.

**Existing customers** are explained from their 13 history-based
features against a typical existing customer: training medians, rounded
to display precision. Wording was added for the history-based features.

**Files.** `models/build_explanation_rules.py` computes every limit from
the training partition and writes:
- `serving/explanation_rules.json`, stamped with the model version;
- `serving/typical_applicant.json`;
- copies of both for the site.
The API refuses to serve explanations if the rules were built for a
different champion. Declared choices live in `config.yaml` under
`explainability.decision_review`.

### Evidence
| Check | Result |
|---|---|
| Held-out existing customers (8,321) | 7,986 no issue; 224 blind spot (2.7%); 108 over-reliance (1.3%); 3 data input. Review required for 335 (4.0%) |
| Site (JavaScript) vs Python, all three outputs word for word: 300 generated applications, every error type covered | 0 differences, 822 sentences |
| Dashboard vs API, new applicant | three outputs identical |
| API smoke test: data error, blind spot, typical applicant, two existing customers | expected error type in each case |
| Deploy gate (headless browser checks) | 14 of 14 pass |

### Finding: over-reliance cannot fire for a new applicant at the live share
- For a new applicant to reach the cutoff, several details must be
  risky. The largest single detail then carries at most 42% of the
  total effect (130 referred generated applicants), below the 60% share.
- No generated profile within three changes of the typical applicant
  reached the cutoff at all.
- The check does fire for existing customers (108 held-out).
- The share was **not** lowered to make it fire for new applicants;
  that would be choosing a number to produce a result.
- To still test the wording, every fifth parity example runs with a
  test-only share of 0.3. The examples file and the Verification page
  say so.

### Limits (stated, not hidden)
- **Genre selection follows Belton et al.'s findings but has not been
  tested with users here.** Their result came from a randomised trial
  with 8,860 UK adults; this project has no users to test with.
- **The thresholds are choices.** Changing them changes who is flagged;
  re-run the builder after any change.
- **The checks describe this model on synthetic data.** The blind-spot
  rule targets this champion's known weakness and must be revisited
  after any retraining.
- **Nothing is logged.** The decision record is returned, not stored,
  and the public site neither stores nor sends it. Persisting records is
  a separate decision.

### Addendum — Which Details the Decisions Rest On, Published on the Demo Site (15th October 2026)
The owner asked for this evidence to be shown in the app.
`models/build_explanation_rules.py` now runs the shipped checks on every
held-out test customer. It writes the aggregates to
`website/public/data/review_evidence.json`: counts, rates, and medians
over groups of at least 20, with no customer row. The demo site shows
them under "When the model asks for a person", and the deploy gate
checks that the page matches the file.

Only two details ever decide a held-out recommendation on their own:

| Detail | Recommendation | Customers | Median PD | Median PD if the detail were typical | Median share of effect |
|---|---|---:|---:|---:|---:|
| Average order value | Pass | 71 | 2.0% | 7.6% | 70% |
| Thin credit file | Refer | 37 | 7.3% | 2.3% | 62% |

- **71 customers pass only because their orders are small.** At a
  typical order size (£583) they would cross the cutoff. The model
  leans heavily on order size, which matches the training data: the
  smallest-order band defaulted at 12 per 1,000, the larger bands at
  28–34 per 1,000. This adds weight to A31; review it at the next model
  review.
- **37 customers are referred only because their credit file is thin.**
  This is the mirror image of the established-file blind spot: file
  depth moves the score more than any other single detail.

## Decision 32 — Public Streamlit Demo: Snapshot Mode, Aggregates and Model Files Only

### Challenge
The owner wants people to be able to see the Streamlit dashboard. It could
not be hosted as it was:
- four of its seven pages query the local PostgreSQL database;
- the model package (`models/artifacts_v1_1/`) and `reports/` are
  git-ignored, so a hosted copy would have no model and no evidence;
- some of that evidence is per-customer: the SHAP case studies, the
  held-out prediction file, and the existing-customer lookup.

The owner compared three options: a snapshot demo, a hosted database, and
a full Docker image. They chose the snapshot demo on 15th October 2026
and approved committing the model package and aggregate reports.

### Decision
**A demo mode** (`SENTINEL_DEMO=1`, set as a Streamlit Cloud secret)
reads everything from `dashboard/demo_data/` and never touches a
database:
- **Query snapshots.** `dashboard/tools/export_demo_data.py` runs every
  page headless against the local platform in recording mode and saves
  each query result, keyed by its exact SQL. Demo mode replays them.
- **Reports.** The report files the pages actually read are copied, and
  only those.
- **Model.** The champion package's model files are copied. These are
  the same model the public site already ships as ONNX, and contain no
  data rows.
- **Held-out scores.** The per-customer held-out prediction file is
  replaced by 1,001 score quantiles, used for "higher than X% of
  customers".
- **Training ranges.** Min, median and max per feature.

**Gates in the exporter** (it stops rather than publish):
- A query whose result has a `customer_id` column is refused. Only
  existing-customer lookup was refused.
- A report with a `customer_id` column or key stops the export. The SHAP
  case studies are never read through the recorder.
- Absolute local paths are rewritten as project-relative. Any remaining
  home-folder path stops the export: the promotion record contained one.

**What the demo shows:**
- **Live:** new-applicant scoring with all three outputs (Decision 31),
  model performance, explainability and the held-out review evidence,
  portfolio, monitoring and governance as of the snapshot.
- **Not shown:** existing-customer lookup and per-customer case
  studies. They say so on the page, and every page carries a "Public
  demo" banner with the snapshot date.

`dashboard/requirements.txt` pins the dashboard's own dependencies to
the versions the champion was pickled with. Streamlit Cloud uses this
file instead of the root one, which carries Airflow and TensorFlow.

### Evidence
| Check | Result |
|---|---|
| Clean copy of exactly the files git would push: no `reports/`, no model package, database host unreachable, demo mode | All 7 pages render with no errors. New-applicant score 0.009456, same as the API (0.0095) |
| Local mode after the change | All pages pass. Dashboard and API outputs identical. Existing-customer lookup works |
| Export manifest (`dashboard/demo_data/MANIFEST.json`) | 12 aggregate query snapshots, 20 report files, 5 model files (2.2 MB). 1 query refused (customer rows) |

### Limits
- **It is a snapshot.** It must be re-exported after any promotion,
  monitoring run or report refresh; the banner shows its date.
- **Publication is irreversible.** The model files in the repository
  are public once pushed (as the ONNX model already is).

## Decision 33 — The Applicant Sees an Outcome, Their Own Reasons and What Could Change It

### Challenge
The owner reviewed the applicant view (22nd October 2026) and found it did
not do the job:
- It said "Why this application was not referred". That is a workflow
  status, not a credit decision, and an applicant cannot tell whether
  they were given credit.
- For a typical applicant it said only "No single detail stood out" and
  showed a training-data statistic ("about 27 in every 1,000 past
  customers…"). That is model background, not a reason.
- Under Decision 31 the applicant could also be shown the model's own
  issues (the blind-spot wording). The owner ruled that the applicant
  should not see model issues.

A proposal pasted by the owner also suggested a "declined" outcome, a 35%
affordability limit, and showing the applicant their PD against the 6.57%
cutoff. None of these exists in the project's evidence or policy, so they
were put to the owner as choices.

### Decision (owner's choices, 22nd October 2026)
1. **Two outcomes, matching what the model decides.** "Your application
   has been approved", or "Your application needs a manual review… It has
   not been declined." There is **no decline**: the project has no
   evidence-based decline rule, and inventing one was rejected.
   - The outcome is **manual review** if the model refers the application
     **or** any Decision 31 check fires. A person is needed in either case.
   - Otherwise the application is approved.
   - *Corrected in Decision 35:* the project does have a decline rule.
     The FCA affordability threshold in the data generator is now
     applied at 45%.
2. **Own details only; no thresholds.** The applicant sees their own
   values against the typical applicant's, and never their PD or the
   cutoff (Belton et al. 2025, Decision 7). Model issues (blind spot,
   over-reliance) and the "past customers" statistics move to the risk
   officer output.
3. **Model-tested guidance.** For a manual review that is not a
   data-input case, the model tests single changes to what an applicant
   can act on:
   - a smaller order (in 5% steps, down to £20);
   - a different plan length (3, 6 or 12 instalments);
   - lower existing commitments (in 10% steps, down to £0);
   - more savings (in 5% steps of the gap to the training maximum).

   Each change is scored with everything else unchanged. A change
   qualifies if the model would then approve automatically: below the
   cutoff, and outside the blind spot for a full credit file. The smallest
   qualifying change per detail is shown, up to three, ranked by size
   relative to that detail's training range. Each is followed by "These
   changes do not guarantee approval…".

   **Not tested (our choice):** over-reliance on the changed application,
   and details the applicant cannot change quickly (credit score,
   history, defaults, file depth, category).

**Applicant view, in order:**
- outcome and summary;
- why (up to three of their own details, against or in favour);
- details to check (data-input cases, worded as the applicant's own
  details, not as model limits);
- what could change the outcome;
- what happens next;
- the method line.

The score card's pill is relabelled **"Model recommendation"**, so the
model's view cannot be mistaken for the outcome.

### Evidence
| Check | Result |
|---|---|
| Held-out customers (8,321) at the live rules | Approved 6,700 (80.5%). Manual review 1,621 (19.5%): 1,323 referred by the model, plus 298 where only a check fired |
| Site (JavaScript) vs Python, all three outputs word for word: 300 generated applications | 0 differences, 975 sentences (reasons, details to check, changes) |
| Site, the test applicant (full file, PD 3.2%, blind spot) | "Your application needs a manual review", 3 own-detail reasons, 3 model-tested changes. None of "blind spot", "less reliable", "cutoff", "threshold", "6.57" or "past customers" in the applicant panel (checked by the deploy gate) |
| Dashboard vs API (new and existing customers) | identical outputs |
| Streamlit demo mode, database unreachable | all pages render |

### Supersedes
This supersedes Decision 31's applicant-facing genre selection. The
applicant now always gets the features-based genre, with no data-centric
statistics. The error-type checks, the review flag, and the officer and
record outputs are unchanged (record version 2 adds `customer_outcome`).

### Limits
- **The guidance describes the model, not a promise.** It is one change
  at a time, everything else held, on synthetic data.
- **"Approved" is a demonstration label.** The champion is a
  portfolio-risk model, not a validated application-time underwriting
  model (Decision 18). The page keeps that scope statement.

## Decision 34 — Thin-File Status Worked Out, Plain-Language Risk Officer View, Material Effects Only

### Challenge
The owner reviewed the outputs on 22nd October 2026 and found three
problems:
1. **The applicant was asked about their credit file.** They had to
   tick "thin credit file", and was then told to check details when the
   tick contradicted their history. An applicant should not have to know
   or check this; it should be worked out once they enter their details.
2. **The risk officer view used modelling language.** "Combination
   explanation (Decision 7)", "training ranges", "95th-percentile line"
   and "error type" are not how a credit risk officer reads a case.
   They need risk factors, strengths, and why the case is in front of
   them.
3. **The held-out evidence sat on the applicant page.** "When the model
   asks for a person" is evidence for reviewers, not for applicants.

### Decision
**1. Thin-file status is worked out, never asked.**
- In the data, thin-file status is set first and the credit file is
  drawn to match. Thin files have scores up to 579, 0–23 months of
  history and 0–1 accounts.
- **Rule:** an applicant is thin-file if their credit score is at most
  579, their history at most 23 months, and they have at most 1 account.
  The limits are computed from the training partition by
  `models/build_explanation_rules.py`.

| Check | Result |
|---|---|
| Training partition (33,283) | Agrees with the recorded flag for 33,252 (99.91%). 31 full-file customers whose figures fall inside the thin-file limits are treated as thin; no thin file is ever missed |
| All customers (50,000) | 99.90% agreement |

- The site and dashboard forms no longer ask. The site shows the
  worked-out status as the applicant types.
- The API works it out when `thin_file_flag` is left out. If a caller
  supplies it (as a bureau would), the supplied value is used.
- The decision record says which: `thin_file_source` is `derived` or
  `supplied`.
- An application whose figures still contradict each other (for example
  a credit score of 720 with 5 months of history) goes to a risk officer
  automatically. The applicant is no longer told to check anything.

**2. The risk officer view is in plain language.** It now shows, in
order:
- **Status:** "Review needed" or "No concerns", with one plain sentence.
- **Why a person needs to look:** for example "Estimated chance of
  falling 90 or more days behind: 8.5%, at or above the 6.57% review
  line", or "Established credit file, but riskier than 95% of similar
  applicants…".
- **Risk factors and strengths:** the applicant's own details, for
  example "Low savings buffer: £150 (typical £2,165)" or "Has recent
  defaults: 2 defaults in the last 24 months".
- **Checks:** plain names, each with a "Yes" or "No" sentence.
- **Track record of similar past customers.**

The modelling terms stay in the decision record and the decision log.

**3. Only material effects are given as reasons** (declared choice:
`material_effect` = 0.1 percentage points of estimated risk). On 1,500
held-out customers:

| | All details | Material details (0.1 points or more) |
|---|---:|---:|
| Share of details kept | 100% | 35% |
| Share where the model's effect runs against conventional credit-risk expectation | 22.8% | 3.0% |

The median one-at-a-time effect was 0.04 points: model noise from a
tree model's step functions. The floor applies to the applicant's
reasons as well as the officer's lists. It also stops trivial details,
such as "Small order", being given as reasons (cf. A31).

**4. Surprising directions are flagged to the officer.**
- Each detail carries its conventional expectation (`risk_when`; a
  declared assumption, for example more defaults means more risk).
- When a material effect runs the other way, the officer reads
  "Unusually, the model counts this in the application's favour".

**5. The held-out evidence moved to the Verification page**, reworded:
"How often a person is asked to look" and "When one detail decides the
outcome".

### Evidence
| Check | Result |
|---|---|
| Site (JavaScript) vs Python, word for word: 300 applications (294 with derived thin-file status, 6 supplied), all three outputs | 0 differences, 2,339 sentences. Thin-file status matches on all 294 |
| Deploy gate (headless browser checks) | 17 of 17 pass. New checks cover the worked-out status line, no modelling jargon in the officer view, and the evidence on the Verification page |
| Dashboard vs API | identical outputs (both derive thin-file status) |
| Streamlit demo mode, database unreachable | all pages render |

### Answer recorded for the owner: data features vs model features
- **What the feature store holds.** `feature_store.customer_features`
  has 61 columns besides `customer_id`, `target_default` and
  `has_outcome`. They are built from `raw.customers`,
  `raw.transactions` and `raw.repayments`.
- **What the model uses.** 13 features (`models/model_features.py`
  MODEL_FEATURES):
  - eight profile fields: income, existing commitments, savings,
    credit score, months of history, accounts, defaults in 24 months,
    thin-file flag;
  - five behaviour features: maximum affordability ratio, average order
    value, average instalments, and the shares of high- and
    medium-risk purchases.
- **Why the rest are excluded** (all recorded in `model_features.py`):
  - exact duplicates or near-duplicates;
  - outcome leakage (`pct_late_payments`; days-late aggregates);
  - exposure-time contamination;
  - resolution-tied fields (DPD, SICR, arrears);
  - generator internals;
  - identifiers and protected or sensitive characteristics (age,
    vulnerability, employment status, income band).
- **Global importance**, mean |SHAP| share (Decision 14):

| Feature | Share |
|---|---:|
| thin_file_flag | 29.0% |
| savings_balance_gbp | 18.4% |
| avg_n_instalments | 12.1% |
| avg_order_value_gbp | 11.3% |
| pct_high_risk_category | 10.4% |
| pct_medium_risk_category | 6.6% |
| max_affordability_ratio | 3.0% |
| credit_score | 3.0% |
| The remaining five | 1.8% or less each |

## Decision 35 — Affordability Limit (45%), a Declined Outcome, and the Repayment Plan

### Challenge
The owner reviewed the approved view on 22nd October 2026.

**What was wrong:**
- "Your details were close to those of a typical applicant" does not
  explain an approval.
- The applicant saw no affordability figure.
- There was no repayment plan.
- The method line belonged to the risk officer, not the applicant.

**What the pasted proposal asked for:** a 35% affordability limit,
showing the PD against the 6.57% cutoff, and a plan table with
placeholder fields (deposit, frequency, dates, fees).

**A correction to Decision 33.** Decision 33 said "the project has no
evidence-based decline rule". That was wrong. The data generator applies
the FCA CONC 5.2A affordability ratio, (existing monthly commitments +
this instalment) ÷ monthly income, and **declines** an application above
the threshold (`application_approved = 0` when `fca_affordability_flag =
1`). The A/B test compared two thresholds: 45% (strict, group A) and
60% (relaxed, group B). The live app never applied that rule.

### Decision (owner's choices, 22nd October 2026)
1. **Affordability limit 45%** (`config.yaml` `model.affordability_limit`).
   This is the strict threshold, whose customers defaulted at 0.65%
   against 1.15% under 60% (Decisions 15–16). The pasted 35% is not a
   project limit: it is only a risk multiplier inside the generator.
2. **Over the limit, the application is declined**, as the project's
   own application process does. The applicant sees:
   - "We cannot approve this application";
   - the affordability figure against the limit, in their own numbers;
   - arithmetic changes that would bring it within the limit: a
     smaller order, a longer plan, or lower commitments.
3. **The PD and the cutoff stay hidden from the applicant** (Decision 33
   kept). The affordability limit is shown because it is a rule about
   the applicant's own figures, which they can check and act on.

**Outcome order:**
- affordability over the limit → **declined**;
- otherwise, the model refers or a check fires → **manual review**;
- otherwise → **approved**.

Model-tested changes for a manual review must also keep the repayments
within the limit.

**Approved view:**
- **What supported the decision**, each line from real figures and checks:
  - the affordability figure against the limit;
  - the credit file: enough history, or a thin file noted as such;
  - "overall position, not a single detail" (true because the
    over-reliance check did not fire);
  - the applicant's material favourable details.
- **The repayment plan**, filled only from the product as the data
  defines it:
  - monthly plans;
  - the first instalment taken at checkout, the rest every 30 days;
  - instalment = order ÷ number of instalments, any rounding penny on
    the final repayment;
  - interest and fees £0, total = order value.
  The site and dashboard show real calendar dates. There are no
  placeholders, and nothing the project lacks (deposit, payment method,
  fees) is shown.
- **The method line** moved to the risk officer view.

**Risk officer view:**
- an "Affordable" check, first in the list;
- a "Declined" status with the affordability reason.

**Decision record (version 4):** adds `affordability_limit`,
`affordability_ratio` and `customer_outcome`.

**Existing customers:** affordability is checked when the API is given
an order. The dashboard's lookup has no order, so it shows "not
assessed".

**The demo page's score card**, which shows the PD and the cutoff, is now
labelled "Model view · for the lender, not shown to the applicant".

### Evidence
| Check | Result |
|---|---|
| Applications in the project's data above 45% | 35.1%. The strict group's actual decline rate was 34.95%, so the rule reproduces the project's own process |
| Site (JavaScript) vs Python, all three outputs word for word: 300 generated applications (83 approved, 142 manual review, 75 declined) | 0 differences, 2,562 sentences, including plans and decline changes |
| Arithmetic of the decline changes (worked example: £2,442 income, £900 commitments, £900 order over 3) | A £596 order gives 44.99%; commitments of £798 give 44.99%; a 6-instalment plan gives 44.99%. All within the limit |
| Deploy gate (headless browser checks) | 19 of 19 pass, including an approved run (affordability vs limit, 7-row plan, no method line) |
| Dashboard vs API | identical outputs |
| Streamlit demo mode, database unreachable | all pages render |

### Limits
- **35% of applications would be declined.** That is the policy the
  project's data was generated under, and it is reported, not tuned.
- **The 45% limit is a demonstration policy.** It is not an FCA-set
  number. CONC 5.2A requires a creditworthiness assessment, not a
  specific ratio.

---

## Summary Table

| # | Decision | Challenge | Key Evidence | Component |
|---|---|---|---|---|
| 1 | Synthetic data | No UK BNPL dataset | Khalili 2026, Kiran 2024 | bnpl_data_generator.py |
| 2 | Validation | Unvalidated data untrustworthy | FCA/ICO/Turing 2023 | validate_synthetic.py |
| 3 | Feature store | Training-serving skew | Immaneni 2022 | sql/ (local PostgreSQL; RDS not built) |
| 4 | Three models | Performance-interpretability-fairness tradeoff | Demajo 2020 | models/ |
| 5 | Drift monitoring | Models degrade post-deployment | Kurian 2024 | pipeline/dags/ |
| 6 | Closed loop | Disconnected tools create gaps | Kurshan 2020, Immaneni 2022, Brahmandam 2025 | Full platform |
| 7 | Explainability | Wrong genre harms consumers | Belton 2025, Demajo 2020 | explainability/ |
| 8 | Model registry | Version traceability required | Kurshan 2020, Immaneni 2022 | MLflow |
| 9 | Approval gate | Automated deployment insufficient | Adari 2021, PRA SS1/23 | approval_gate.py |
| 10 | Hazard-based default | Default assigned before repayment behaviour inverted causality, making repayment features leakage | CRR/EBA default definition, IFRS 9, Klarna Wikipink 2024, Woolard Review 2021 | bnpl_data_generator.py, raw.repayments, validate_synthetic.py, config.yaml |
| 11 | Automated audit-trail enforcement | Decision log accuracy depended on human memory | FCA CRA Review May 2026, PRA SS1/23 Principle 6 | .git/hooks/pre-commit, .git/hooks/post-commit, docs/audit_trail.log |
| 12 | Feature store Stage 3 | Repayment features were proxy leakage until default became emergent | Decision 10 (hazard rewrite) | sql/02_feature_store.sql |
| 13 | Training, evaluation, cutoff as policy | Undefendable hand-picked models and thresholds | PRA SS1/23, SR 11-7 | train_baseline_models.py, evaluate_models.py, params.yaml |
| 14 | SHAP and exposure-time exclusion | Model leaned on observation-time-confounded features | Demajo 2020; repeated-CV ablation | explain_models.py, compare_feature_sets.py |
| 15 | A/B analysis, maturity collapse | has_outcome is post-treatment; underpowered test | Causal selection reasoning | analyse_ab_test.py, sql/diagnostics/ |
| 16 | Extended observation horizon | Right-censoring | Decision 15 diagnostic | config.yaml data_end_date |
| 17 | Feature store inventory | Allowlist never derived from a full inventory | — | sql/diagnostics/check_feature_store_inventory.sql |
| 18 | v1 scope and 14-feature freeze | Portfolio-risk vs application-time conflated | Point-in-time diagnostic, Equality Act 2010 | models/model_features.py |
| 19 | Business cutoff 0.0671 | Configured cutoff non-functional | Training-only OOF, review capacity | select_business_cutoff.py, segment_threshold_diagnostics.py |
| 20 | Drift monitoring | No repeatable drift check | Kurian and Allali 2024 | monitor_drift.py, monitor_drift_dag.py |
| 21 | Champion/challenger workflow | No registry, no governed comparison | PRA SS1/23 Principle 1 | promote_champion.py |
| 22 | Portfolio forecasting | Forecasting capability, no macro data | Prophet backtest | forecasting/ |
| 23 | Automated approval gate (retrospective) | Promotion not tied to monitoring evidence | PRA SS1/23 Principle 4 | approval_gate.py |
| 24 | Drift logging fix, read-time gate status | Crashing drift log; frozen PASS badge | PRA SS1/23 Principle 4 | monitor_drift.py, sql/reporting/, DAG |
| 25 | Drift overlap removed, cohort time axis | Self-overlapping comparison; identical runs | PRA SS1/23 Principle 4 | monitor_drift.py, feature_drift_by_cohort |
| 26 | pct_late_payments removal evidence | Feature mixes cure events and post-default rows | Validity-first ablation, bootstrap CI | evaluate_feature_removal.py, models/challengers/ |
| 27 | Remove pct_late_payments; v1.1 via validity path | Gini-win rule cannot approve a leakage fix | Owner decision; paired-bootstrap CIs | promote_champion.py, train_baseline_models.py, config artifacts_dir |
| 28 | One scoring path; evidence-only dashboard | Second feature-construction path would drift; dashboard partly broken | Training-serving skew (D3) | serving/features.py, dashboard/ |
| 29 | Public demo scores in the browser (ONNX) | Public demo without servers, without drifting from the champion | Two parity checks, fixed pass rule | website/tools/export_model.py, website/public/ |
| 30 | Plain-English reasons for the applicant | Decision 7's customer output never built | One-at-a-time effects vs typical applicant; word-for-word JS/Python check | serving/reasons.py, serving/reason_templates.json, website/public/assets/reasons.js |
| 31 | Decision 7 completed: error-type checks, genre, three outputs | Genre selection, risk officer and regulatory outputs, existing customers not built | Blind spot: full-file FNR 100%, top 5% default 2.9x (held-out); limits from training data; declared choices in config | models/build_explanation_rules.py, serving/reasons.py, serving/explanation_rules.json |
| 32 | Public Streamlit demo, snapshot mode | Dashboard needs a database and git-ignored files; some evidence is per-customer | Exporter refuses customer rows and local paths; clean-copy test with no database | dashboard/tools/export_demo_data.py, dashboard/demo_data/, dashboard/utils.py |
| 33 | Applicant sees outcome, own reasons, model-tested changes | "Not referred" read as a workflow status; model issues and statistics shown to applicants | Owner's choices: two outcomes (no invented decline), no thresholds (Belton), model-tested changes; held-out 80.5% approved / 19.5% manual review | serving/reasons.py, serving/reason_templates.json, website/public/assets/reasons.js |
| 34 | Thin-file status worked out; plain-language officer view; material effects only | Applicant asked about their credit file; officer view in modelling language; noisy factors | Thin-file rule 99.91% agreement; materiality floor cuts 'wrong-way' factors 22.8% to 3.0% | serving/reasons.py, models/build_explanation_rules.py, website/public/ |
| 35 | Affordability limit 45%, declined outcome, repayment plan | Approval not explained; affordability never applied; Decision 33 wrongly said no decline rule exists | Project's FCA CONC 5.2A ratio and A/B thresholds (45% vs 60%: 0.65% vs 1.15% default); 35.1% of applications over 45% | config.yaml, serving/reasons.py, website/public/ |

---

## Reference Material

Reference material that supports the decisions above, but is not itself a decision, lives with the documents it belongs to (moved 15th October 2026, verbatim):

| Former section | Now in |
|---|---|
| Appendix A — Synthetic data calibration assumptions | [Project Rationale → Synthetic Data Calibration Assumptions](project_rationale.md#synthetic-data-calibration-assumptions) |
| Appendix B — Validation methodology | [Validation Results → Validation Methodology](validation_results.md#validation-methodology) |
| Appendix C — Metric definitions | [Model Evaluation Report → Metric Definitions](model_evaluation_report.md#metric-definitions) |
| Appendix D — Regulatory source mapping | [Project Rationale → Regulatory Source Mapping](project_rationale.md#regulatory-source-mapping) |
| Regulatory reference summary | [Project Rationale → Regulatory Reference Summary](project_rationale.md#regulatory-reference-summary) |

---

## Document Version History

| Version | Date | Change |
|---|---|---|
| 1.0 | July 2026 | Initial version |
| 2.0 | July 2026 | Full rewrite — unverified claims removed, fairness added, narrative flow added |
| 2.1 | July 2026 | Technical review corrections — three drift types, additional metrics, gate scope clarified |
| 2.2 | July 2026 | Targeted wording refinements per reviewer feedback |
| 3.0 | July 2026 | Restructured for audience — detail moved to appendices, main body condensed to 12-15 pages |
| 3.1 | 30th July 2026 | Added Decision 10 (hazard-based default modelling, censoring, `run_utility` completion, `\copy` load-order lesson). Updated Summary Table, Regulatory Reference Summary, and Appendix A/B/D cross-references. Reject inference flagged as open pending methodology confirmation. |
| 3.2 | 30th July 2026 | Full rewrite of Decision 10 against the actual generator code and config.yaml. Added: the state machine (CURRENT/DELINQUENT/DEFAULT), the two-hazard split (miss vs resilience), the post-schedule collections fix (42% final-instalment finding), the two-stage calibration sweep methodology and landed parameters, the coarse-delinquency-spectrum finding, the A/B mechanism repair, reject inference (resolved — deliberately not implemented, reasoning documented), right-censoring, config structural cleanup, and the from-scratch database rebuild decision. Summary Table row 10 updated to match. |
| 3.3 | 30th July 2026 | Corrected Decision 2 and Appendix B: CTGAN benchmark was stated as implemented but does not exist in `validate_synthetic.py`. Marked explicitly as planned, not done — descoped intentionally until the core pipeline is complete. Found during a direct check of the log against the actual validation code. |
| 3.4 | 30th July 2026 | Added Decision 11: pre-commit and post-commit git hooks enforcing that audit-relevant file changes are always accompanied by a decision log entry, plus a mechanical, timestamped `docs/audit_trail.log`. Installed and verified working (test commit to `config.yaml` alone correctly blocked). Summary Table row 11 added. |
| 3.5 | 30th July 2026 | Added Decision 12: feature store Stage 3 (repayment and delinquency features), documenting why it was capped at ~10 features under the old causal flaw and how Decision 10 unblocked it past the 30+ target. Added a Decision 2 addendum documenting the validation config restructuring (privacy thresholds split into two explicit keys; `gini_tolerance` wired in as reported context; `n_runs`/`ks_threshold` confirmed still unimplemented). Both found missing when a commit shipped the underlying code/SQL changes with only Decision 11 (the hooks) as coverage. Summary Table row 12 added. |
| 3.6 | 30th July 2026 | Added a Decision 12 addendum documenting temporal leakage in resolution-tied features (`max_dpd_ever`, terminal-state counts, `cycles_delinquent`) found before model training began — a different failure mode from the causal-generation flaw Decision 10 fixed. Resolved by excluding these at feature selection (`MODEL_FEATURES` allowlist) rather than deleting them, following the precedent already set for `default_probability`/`merchant_risk_multiplier`, with the distinction between the two exclusion reasons stated explicitly. A fixed observation-window snapshot logged as planned, not yet built. |
| 3.7 | 30th July 2026 | Added a second Decision 12 addendum confirming `vulnerability_score` and `is_vulnerable` are excluded from `MODEL_FEATURES` on fairness/regulatory grounds (Equality Act 2010 via the composite's health driver; FCA Consumer Duty vulnerable-customer provisions requiring vulnerability to inform treatment, not scoring) — distinct from the temporal-leakage exclusions in the prior addendum. `models/model_features.py` created. |
| 3.8 | 30th July 2026 | Added a Decision 4 addendum stating the platform's scope is PD (Probability of Default) only. LGD and EAD, the other two components of IFRS 9 Expected Credit Loss, are explicitly out of scope pending a post-default recovery model the generator does not currently have. |
| 3.9 | 30th July 2026 | Added a second Decision 12 addendum documenting the segment-aware thresholding experiment: the fairness disparity found by `thin_file_flag` at the F1-optimal cutoff, confirmed independent of `thin_file_flag` as a model input (identical gap after removal and re-run), and the deliberate decision not to implement per-segment thresholds — reasoned as a governance/Consumer Duty policy question rather than a modelling optimisation, with the mitigation logged as future work requiring explicit policy approval. |
| 3.10 | 30th July 2026 | Added a Decision 10 addendum documenting the out-of-time split's tenor-maturity bias: the temporal split date is derived from the data's own distribution, not the generator's maturity requirements, causing longer-tenor loans to be structurally absent from the modelable population near the data boundary. A maturity-curated split date was considered and deliberately not implemented — shrinks the training population, and only relocates rather than eliminates the bias, which is a structural limit of a fixed dataset snapshot. Retained as-is, documented as a known limitation. Future work: recompute the split date using the maturity formula, or the more complete fix of a fixed observation-horizon default definition decoupled from product tenor. |
| 3.11 | 30th July 2026 | Added two Decision 12 addenda: (1) the has_outcome correction — the feature store never checked outcome_observed, silently labelling censored loans as confirmed good outcomes; corrected, modelable population 41,604→18,694, bad rate 1.91%→4.26%; (2) unequal observation opportunity — the corrected has_outcome's asymmetric qualifying paths (defaulters bypass maturity, non-defaulters don't) contaminate the entire cumulative-exposure feature family, not isolated features. n_approved and days_since_last_transaction confirmed near-total artifacts and excluded (EXPOSURE_TIME_EXCLUDED). total_order_value_gbp, transaction_span_days, max_order_value_gbp, n_merchant_categories flagged suspected but retained pending a deferred Feature Set A/B sensitivity comparison (EXPOSURE_TIME_SUSPECTED), to be resolved after SHAP/MLflow/drift monitoring are built, before the feature set is called final. |
| 3.12 | 30th July 2026 | Added Decision 13 covering the baseline training and evaluation build: GridSearchCV with StratifiedKFold, average_precision as the selection metric (roc_auc is diluted under imbalance) with Gini/AUC retained as reported metrics, mandatory rather than searched class balancing, Platt-scaling calibration (ECE 0.3552 to 0.0026), and found-not-guessed threshold selection via F1-optimal and Youden's J. Central decision: business_cutoff is a configurable policy parameter belonging to the decision engine, not a property of any algorithm — calibrated against the champion (XGBoost, 0.145) for this benchmark, with recalibration required as part of any future champion-challenger promotion rather than assuming transferability across models. |
| 3.13 | 3rd August 2026 | Added Decision 14: SHAP explainability foundation and resolution of the exposure-time features. Training now persists a governed model package (calibrated champion + uncalibrated explanation base + metadata + split IDs) so downstream scripts load rather than retrain. SHAP validated (additivity error 2.4e-06) and showed the model gave 32.1% of total importance to the four suspected features. A repeated-CV ablation (5x3 folds) found Gini/AUC/KS effectively tied, PR-AUC favouring the full set (13/15 folds), and the train/test gap favouring the conservative set (14/15). The four features were excluded on FEATURE VALIDITY grounds — they are structurally confounded with unequal exposure time and do not measure what they were intended to — accepting a modest PR-AUC cost. MODEL_FEATURES 22 to 18. Notably, an earlier single-split result suggested the conservative set was better on KS; repeated CV did not support that, and is recorded as the check that prevented a premature conclusion. |
| 3.14 | 3rd August 2026 | Extended Decision 14 with the end-to-end retraining outcome and the resulting engineering lesson. After the feature exclusions, re-running the full pipeline (including grid search, which re-selected learning_rate 0.1 to 0.05) improved the champion on Gini 0.6493 to 0.6550, KS 0.5061 to 0.5178, recall 45.91% to 47.80%, and top-decile lift 5.03 to 5.16, with PR-AUC the only reduction. Highest VIF fell 32.23 to 18.24; exposure-time SHAP share fell 32.1% to 0%. Conclusion revised: the decision still rests primarily on feature validity, with performance evidence now supportive rather than counter-indicating, though the end-to-end comparison is a single split and is stated with matching caution. Lesson recorded: a feature ablation is not the final decision criterion for a production pipeline — an ablation answers how much information a feature family contains with everything else held fixed, while a full retraining answers how the pipeline behaves without it, including hyperparameter re-selection and threshold recalibration. |
| 3.15 | 3rd August 2026 | Added Decision 15: the A/B test analysis (Decision 1's affordability-threshold experiment, previously never formally analysed) and the maturity collapse investigation it surfaced. Established that has_outcome is a post-treatment variable — observability differs 34.08% vs 40.77% between arms — so the primary test runs on a fully mature cohort anchored on last_transaction_date instead. Result: +0.52% absolute difference, 95% CI [-0.11%, +1.15%], p = 0.110 at only 35.85% achieved power against an 80% target. Diagnostic established the cause as the simulation horizon, not the design: 85.6% of last transactions fall in 2024 and the maturity cutoff precedes them. Extending data_end_date by 12 months raises the eligible cohort from 4,871 to 41,604 (a computed sufficiency point, plateauing thereafter) with no change to the causal design. Decision: extend and re-run, deliberately deferred to a separate session because the longer horizon will raise the observed default rate and may require re-sweeping the generator calibration. |
| 3.16 | 3rd August 2026 | Added Decision 16: extended data_end_date from 2024-12-31 to 2025-12-31 (origination_end_date unchanged), eliminating right-censoring entirely. Modelable population 18,694 to 41,604; right-censored transactions ~33% to 0.00%; customer default rate 4.26% to 2.72%; A/B power recomputed at 99.85% with MDE 0.233pp. Validation re-run clean (fidelity 8/8 with identical scores, utility PASS, privacy 2/2); TSTR retention moved 0.7865 to 0.7398, recorded. The substantive outcome is architectural: an entire pipeline stage (maturity determination and cohort construction) is removed, along with the post-treatment selection risk that motivated it. Diagnostics retained as safeguards; build_mature_cohort archived rather than made conditional, since the platform is not intended to support incomplete observation windows. Decision 15 marked superseded in part. Open: whether the 0.40% transaction-level reference still applies under full observation — no recalibration performed. |
| 3.17 | 3rd August 2026 | Added two Decision 16 addenda. (1) Signal recovery: EDA on the regenerated data shows exposure-related IV collapsing (n_applications -60%, bnpl_exposure_ratio -54%, late-payment features -70%) while day-zero customer attributes rose (savings_balance +18%, thin_file_flag +9%, credit_score +10%). This is independent confirmation of the exposure-time diagnosis obtained by removing the cause rather than measuring around it. Five features now exceed the 0.5 threshold, down from seven, all day-zero attributes. EPV improved to ~63 from 44. Opened, not closed: existing_monthly_obligations_gbp (IV 0.0018) and age (0.0085) now sit below the drop threshold — weak but uncontaminated, flagged for deliberate decision. (2) Artifact persistence: EDA and validation now save full result sets to reports/eda/ and reports/validation/ with metadata recording the feature list, thresholds and population, so results can be re-read and traced without re-execution. |
| 3.18 | 3rd August 2026 | Added Decision 17: full feature store inventory. All 64 columns profiled and classified by role — 18 candidates (the current MODEL_FEATURES), 12 resolution-tied, 11 redundant or exact identities, 6 exposure-time confounded, 4 regulatory/segmentation, 2 monitoring-only, 5 structural, 1 constant (n_terminal_delinquent, confirming the horizon extension), 1 direct leakage. The allowlist was correct but had never been derived from a complete inventory. Four columns found never screened at all (n_instalments_total, n_transactions_with_repayments, total_scheduled_repayment_gbp, avg_instalment_amount_gbp) — three expected to fail structurally, avg_instalment_amount_gbp a genuine open question, none excluded on suspicion. age (IV 0.0085) and existing_monthly_obligations_gbp (IV 0.0018) flagged but retained pending the governance review, on the grounds that IV is a screening tool not a decision rule and has already misled twice in this project. Standing principle recorded: keep broadly in the feature store, exclude deliberately at the model boundary — deleting a column destroys the evidence for its own exclusion. |
| 3.19 | 4th August 2026 | Added Decision 18: model scope resolved as customer-level portfolio-risk (not application-time underwriting) for v1, closing the conditional_as_of_logic question left open across Decisions 17-18. `sql/diagnostics/check_point_in_time.sql` correlated the flagged features against exposure-window length: n_applications (r=0.47), bnpl_exposure_to_monthly_income_ratio (r=0.34) and n_late_payments (r=0.22) confirmed exposure-time contaminated and removed; age removed separately on Equality Act 2010 grounds, correcting the "weak IV" framing proposed in Decision 17 once existing_monthly_obligations_gbp was noted to score lower on the same metric while being retained. Six features (pct_late_payments, pct_medium_risk_category, pct_high_risk_category, avg_n_instalments, max_affordability_ratio, avg_order_value_gbp) retained — the diagnostic found no material association with exposure length, stated explicitly as supporting evidence of scope-fitness, not proof of point-in-time validity, since near-zero linear correlation cannot rule out weak, nonlinear or segment-specific contamination. MODEL_FEATURES 18 to 14. The 285-fit governance review (Decision 17) retained as pre-freeze evidence for the 18-feature baseline, not treated as the review of the 14-feature v1 model — a fresh governance pass on 14 features queued before v1 is called final. Application-time underwriting recorded as future v2 work, not current scope. |
| 3.20 | 7th August 2026 | Added Decision 19: business_cutoff selected as 0.0671 via models/select_business_cutoff.py, using calibrated training-only out-of-fold predictions on the canonical stratified partition (frozen champion hyperparameters, no held-out data touched). Three candidates generated from the same OOF table — F1-optimal (0.0868, 7.82% flagged, 31.68% recall), Youden's J (0.0376, 24.02% flagged, 68.32% recall), review-capacity-constrained at 15% (0.0671, 15.00% flagged, 51.55% recall) — with review-capacity selected as the only candidate answering the operational question directly given a stated 15% review capacity; F1 and Youden retained as comparators, not discarded. Expected-cost-minimizing candidate configured but left unset and skipped — no defensible cost ratio exists, and inventing one would reopen the cost-matrix approach evaluate_models.py already declined (Decision 4 addendum, PD-only scope). Explicitly flagged as conditional on 15% being a real operational figure, not the illustrative placeholder currently in config.yaml. recalibrate_threshold.py (an earlier, duplicate-purpose draft) retired and deleted in favour of extending the existing select_business_cutoff.py. Precision/recall/fairness at 0.0671 on the held-out stratified and temporal splits remain unmeasured pending an evaluate_models.py rerun — the next step, not yet complete. |
| 3.21 | 7th August 2026 | Resolved Decision 19's operating-policy review (held-out confirmation on temporal only so far — stratified evaluate_models.py output not yet captured; temporal recall 38.71%, within range of the 51.55% OOF estimate; cutoff-independent metrics unchanged; stratified confirmation remains an open action). Surfaced and investigated a material finding: thin_file_flag=0 had 100% FNR on both splits. models/segment_threshold_diagnostics.py (new script, three checks) established this is a policy-scale problem, not a model-signal failure — thin_file_flag=0 defaulters' scores sit entirely below any threshold on the table (max 0.0344/0.0367, 0% above cutoff on either split, ruling out borderline cases), no single global threshold recovers thin_file_flag=0 recall without collapsing thin_file_flag=1's FPR to 75-86%+ (ruling out the threshold choice as cause), and within-segment AUC with bootstrap 95% CI shows the model still discriminates risk within thin_file_flag=0 (0.5661-0.7220 stratified, 0.6148-0.7613 temporal, both CIs clear of 0.5 and consistent across splits, overlapping substantially with thin_file_flag=1's CIs) — ruling out near-random within-segment ranking. Corroborates, does not duplicate, the earlier finding that thin_file_flag removal as a model input did not close the same gap since credit_score reconstructs it (r=-0.75). business_cutoff 0.0671 confirmed for the global-decision question; a separate governance item opened, not resolved here, on whether segment-aware review capacity is warranted — requires explicit Decision 12 policy sign-off before any segment-differentiated treatment is built. |
| 3.23 | 11th August 2026 | Added Decision 20: drift monitoring implemented and verified working end-to-end, closing the Decision 5 item queued since project inception. pipeline/monitor_drift.py compares the canonical training partition against the temporal test split (PSI per feature reusing govern_model_features.py's exact thresholds, null-rate/out-of-range/mean-shift data-quality checks, portfolio bad-rate drift), all logic unit-verified against synthetic data with known shift magnitudes before running against real data. Scheduled daily via infra/dags/monitor_drift_dag.py under local Astro/Airflow (Docker) — Airflow chosen over GitHub Actions after explicit reconsideration, since a real drift pipeline needs DAG-structured task dependencies GitHub Actions handles awkwardly, and apache-airflow was already pinned in requirements.txt before this decision. DAG runs the script as a subprocess (no separate DAG-version logic to drift out of sync) with two deliberately separated tasks: compute_drift fails only on infrastructure problems, evaluate_alerts fails only on genuine drift findings, keeping "pipeline broke" and "pipeline found something" visibly distinct in Airflow's UI. Resolved a cluster of real environment issues along the way: exact package pins conflicting with the Astro Runtime's Airflow 3.3 core and Python 3.14 (loosened to >= bounds, safe given the stable APIs used); an invalid docker-compose.override.yml referencing a nonexistent webserver service; container-to-host DB networking requiring a new SENTINEL_DB_HOST override (added symmetrically with the existing SENTINEL_DB_PASSWORD pattern); a newer MLflow version requiring MLFLOW_ALLOW_FILE_STORE=true to keep the existing file-based mlruns/ store; and a stale Windows-path artifact baked into the sentinel_credit_monitoring MLflow experiment's meta.yaml from an earlier manual Windows-native run, causing a misleading PermissionError inside the Linux container until that one experiment folder was deleted and recreated container-native. Verified via a real triggered DAG run: both tasks succeeded, monitoring_run_metadata.json shows 0 of 14 features alerting, freshly generated. Feeding this evidence into an automated Approval Gate or Champion/Challenger promotion remains future work, not resolved here. |
| 3.24 | 11th August 2026 | Added Decision 21: champion/challenger promotion workflow implemented (pipeline/promote_champion.py), the first use of MLflow's Model Registry in this project (alias-based, not the deprecated stage API). Champion auto-discovered via the existing is_champion tag; challenger requires explicit run IDs (no safe auto-discovery default exists for "which run is the challenger"). Promotion requires a material Gini win on BOTH splits, matching the temporal-stability standard from Decision 16/18. min_gini_delta left unset deliberately (same reasoning as other illustrative thresholds this project has declined to invent) — verified the script never recommends promotion with no threshold configured, regardless of delta size. --apply refuses outright if criteria aren't met and requires typing PROMOTE verbatim even when they are; config.yaml and champion_metadata.json are explicitly not touched even on a successful apply. Self-tested (no real challenger exists yet) by comparing the champion against itself: delta_gini 0.0000 both splits, correctly recommended against promotion, and successfully created this project's first registered model version. Added Decision 22: portfolio forecasting implemented (forecasting/build_portfolio_timeseries.py, forecasting/forecast_portfolio.py), Prophet only — LSTM explicitly deferred, not silently dropped, given ~34 monthly points is too little history to train a neural sequence model without overfitting to noise. Database checked directly first (four tables: raw.transactions, raw.repayments, raw.customers, feature_store.customer_features) confirming no macro/economic data exists anywhere in the project, splitting "Forecasting" into buildable-now (portfolio time series) versus blocked-on-a-real-data-decision (macro stress testing, IFRS 9 overlay) rather than fabricating either. Default-rate cohort attribution verified against synthetic data with a known unresolved case before running on real data (late-surfacing defaults correctly attributed to origination month, not recording month; unresolved outcomes correctly excluded, not counted as non-defaults). Real run: 34 monthly cohorts Jan 2022-Oct 2024, 0 unresolved (consistent with Decision 16's confirmed 0% right-censoring). Origination-volume forecast carries a caveat verified empirically, not just asserted: growth reflects the synthetic population's fixed onboarding schedule and origination_end_date, and the backtest visibly fails to track the accelerating actual values in the held-out months (MAPE 24.45%), directly evidencing the caveat rather than merely stating it. Default-rate forecast (MAE 0.0016, MAPE ~18.4%) is a more legitimate target, reflecting actual portfolio risk evolution. Reproduced identically across two separate runs (container-relayed and native PowerShell) on real data. |
| 3.22 | 7th August 2026 | Completed Decision 19's held-out confirmation with the stratified evaluate_models.py run (previously missing in v3.21 — every full output reviewed at that point was confirmed via MLflow tag to be temporal only). Stratified results land close to the OOF selection estimate (recall 54.87% vs 51.55% OOF, flagged rate 15.51% vs 15.00% target, precision 9.60% vs 9.35% OOF) — expected, since stratified is drawn from the same population distribution the cutoff was selected against; temporal's lower recall (38.71%) reflects genuine out-of-time drift, not a selection failure. thin_file_flag=0's 100% FNR confirmed via full evaluate_models.py on both splits (stratified FPR 0.00%, temporal FPR 0.03%), not only via the targeted segment diagnostics — thin_file_flag=1 FPR corrected from an approximate "~33%" to the precise stratified figure (46.53%) alongside temporal's (32.93%). business_cutoff 0.0671 status moved from "confirmed on temporal, stratified outstanding" to fully confirmed on both splits. The separate governance item (segment-aware review capacity, Decision 12) remains open, unaffected by this completion. |
| 3.25 | 13th October 2026 | Added Decision 24, following the 12th October audit (`docs/audit_findings_2026-10-12.md`). (1) Fixed `log_drift_history()` in `pipeline/monitor_drift.py`: leftover `to_sql`/`create_engine` code made the script crash right after its psycopg2 insert. It now inserts once, writes a per-run `run_id` (also in the metadata JSON and MLflow tags), and the table DDL moved to `sql/reporting/create_feature_drift_log.sql`. The 14 pre-existing rows keep `run_id = NULL`, with no id invented after the fact. (2) `reporting.governance_status_latest` now derives status at read time: a PASS whose evidence age, plus time since evaluation, exceeds the row's own `max_monitoring_age_hours` reads STALE. No new threshold was added, and BLOCK is unaffected. The badge moved from a two-month-old PASS to STALE. (3) `segment_key` folded into `sql/reporting/create_reporting_views.sql`, which now re-applies cleanly against the live views. Added a Decision 20 addendum correcting "scheduled daily" to "scheduled while the local stack is running": Airflow run history shows no runs from 14th August to 11th October because Docker Desktop is not auto-started. The DAG is unpaused, and its Variables are set. October catch-up runs succeeded and logged under the fixed code. Noted that no Decision 23 entry exists despite commit `e59a042` naming one. Open: container/host timestamp skew (UTC vs London) in both `evaluated_at` and `feature_drift_log.logged_at`; concurrent catch-up runs; the gate is not scheduled; five logged runs are value-identical, confirming drift history is not a trend on fixed snapshots. Same day, Decision 24 Addendum: the approval gate is now a DAG task (`run_approval_gate`), and `approval_gate.py` exit codes are unambiguous (0 PASS, 1 BLOCK, 2 could not complete or record; `--require-db-write`). The DAG sets `max_active_runs=1` and a 20-minute `execution_timeout`. `evaluated_at`, `created_at` and `logged_at` are migrated to `timestamptz`, with each row converted from its evidenced zone (`sql/reporting/migrations/2026_10_13_timestamptz.sql`), and writers send offset-aware times. Verified end-to-end in Airflow (all three tasks succeeded; the gate row was written from the container with a correct 0.03h evidence age), after restarting a hung Airflow API server and riding out a 9.5-minute WSL2 9P I/O stall on the bind mount, both recorded as audit findings A20/A21. |
| 3.26 | 14th October 2026 | Audit clean-up and two new decisions. Added Decision 23 retrospectively (automated approval gate, built 11th August in `e59a042` but never logged; scope stated as an evidence gate, not Decision 9's human approval). Added dated Implementation Status notes to Decisions 3 (local PostgreSQL, not RDS), 5, 6, 7, 8 and 9, marking what is built, partly built, and planned-not-built, plus a filename note on Decision 2. Extended the Summary Table to Decision 26. Added Decision 25: the drift comparison now excludes customers in the training reference (previously 79.3% overlap), and a new informational origination-quarter cohort view (`reporting.feature_drift_by_cohort`) gives drift a real time axis with a per-cohort noise floor. Added Decision 26: `pct_late_payments` removal evidence via `models/evaluate_feature_removal.py`. The feature mixes cure events and post-default rows. Removal costs −0.0054 repeated-CV Gini, with a held-out difference of −0.0038 (95% CI includes 0), and calibration is unchanged. The challenger is registered as v2; the promotion rule returned "not supported", exposing a conflict between Decision 21's Gini-win rule and validity-driven removals. The decision is left open. The champion's artifacts were verified unchanged by checksum. `config.yaml` comments only: the temporal-split note was corrected, and the identity values are documented as load-bearing (they were not edited). |
| 3.27 | 14th October 2026 | Added Decision 27: the owner ruled that `pct_late_payments` is removed because it leaks outcome information. Superseding Decision 18's frozen set is pending the owner's typed promotion. Added a validity promotion path to `promote_champion.py` (`--validity-override`): it is allowed only when the cited decision exists, the evidence matches the exact challenger runs, and on both splits the 95% paired-bootstrap CI of the Gini difference reaches zero (stratified −0.0038 [−0.0148, +0.0079]; temporal −0.0015 [−0.0164, +0.0150]). The dry run returned "supported"; both refusal paths were tested. Package location is now `config.yaml model_governance.artifacts_dir`, read by all scripts and the API. Training refuses to overwrite a package with a different identity and gains `--pin-xgboost-params` and `--no-prompt`. Added a Decision 21 addendum: champion run selection with rollback (same Gini → latest; latest worse → previous). Decision 26 marked decided. |
| 3.28 | 14th October 2026 | Decision 27 carried out: v1.1 is the live champion. The owner typed `PROMOTE` (registry v3, tagged `validity_override` / Decision 27). `MODEL_FEATURES` 14 → 13. Config identity switched to v1.1, with the package in `models/artifacts_v1_1/`; v1 is untouched and its reports are archived. The pinned retrain reproduces the promoted challenger exactly (0.5618 / 0.5611). Cutoff recalibrated by the Decision 19 method: 0.0657 (was 0.0671). Stratified recall 57.96% vs 54.87%, precision 9.90% vs 9.60%. SHAP, the governance review (13 features, pending owner sign-off) and segment diagnostics regenerated. The API serves v1.1 without the hardcoded `pct_late_payments` default. Monitoring and the gate pass on 13 features. The thin-file FNR finding is unchanged. |
| 3.29 | 15th October 2026 | Documentation separated from the decision record. Appendices A–D and the Regulatory Reference Summary moved verbatim: A and D plus the summary to `project_rationale.md` (Research and Regulatory Foundations), B to `validation_results.md`, C to `model_evaluation_report.md`. A Reference Material pointer table replaces them, and the in-text references are now links. Added a Decision 10 addendum on identifier design: pseudonymised customer IDs (owner's account, corroborated by the generator docstring and the README's record of a collision caught by the database), natural composite keys below them, and uniqueness verified on all three tables. Added Decision 28: one feature-construction module shared by the FastAPI service and the dashboard, and the Streamlit dashboard rebuilt as seven evidence-only sections with a central design system; verified headless and rendered. |
| 3.30 | 15th October 2026 | Added Decision 29: the public demo scores in the visitor's browser via ONNX (owner's choice, route 2), gated on parity with the Python champion under a pass rule fixed in advance (max \|Δ\| < 1e-5, zero decision flips). Results: Python onnxruntime on 41,604 customers, max \|Δ\| 5.96e-08, 0 flips; edge cases 8.94e-09, 0 flips; browser onnxruntime-web (WASM) on 1,988 vectors including the 200 nearest the cutoff, 2.53e-08, 0 flips. Conversion lesson recorded (XGBoost feature names stripped on a copy, parity measured against the original). Hosting and cross-browser checks remain open. |
| 3.31 | 15th October 2026 | Decision 29 addendum. The public demo site is built (live assessment with a typical-applicant comparison and a what-if search, a cutoff and fairness simulator, and a verification page) and hosted on GitHub Pages, chosen by the owner. No third-party requests: onnxruntime-web is bundled after an integrity check, and the fonts are self-hosted. No customer rows are published: test inputs are generated, and the simulator uses aggregated counts that exactly reproduce the saved evaluation. The JavaScript feature port matches `serving/features.py` on 240 generated applications. Three bugs were found and fixed while verifying: a relative wasm path that hung the page, rounding that sent comparator buttons to 0%, and a boundary generator that could never finish. Deploys are gated on 12 headless browser checks, all passing. |
| 3.32 | 15th October 2026 | Added Decision 30: plain-English reasons for new applicants. Each input is compared one at a time with a typical applicant (training medians), using the calibrated champion. Wording follows Belton et al. 2025: the applicant's own values are shown, the decision rule is never shown and nothing is worded as a cause. It is shared by the API (`explanation` in `/score`), the dashboard and the public site from one template file. JavaScript matches Python word for word on 240 applications (687 sentences, 0 differences), and that check gates deployment (14 of 14 browser checks pass). Decision 7 status changed to partly built. One surprising model behaviour is recorded: larger orders lower the PD slightly. |
| 3.33 | 15th October 2026 | Added Decision 31, completing Decision 7. Three error-type checks (data input, known blind spot, over-reliance), with limits computed from the training partition by `models/build_explanation_rules.py`. Declared choices in config: 95th-percentile blind-spot line, 60% single-detail share, groups of at least 500, £0 savings and commitments allowed. Explanation genre per Belton et al. 2025. "Risk officer review required" is stated, not routed (owner's direction). Three outputs (applicant, risk officer, decision record) for new and existing customers, in the API, the dashboard and the site. Evidence: full-file customers never reach the cutoff, and their top 5% default 2.9x as often on held-out data. Review rate on held-out customers 4.0%. JavaScript matches Python word for word on 300 applications covering every error type. Finding recorded: over-reliance cannot fire for a new applicant at the live share. |
| 3.34 | 15th October 2026 | Decision 31 addendum (owner's request). Held-out check evidence is published on the demo site as aggregates (`website/public/data/review_evidence.json`), verified by the deploy gate. Only two details decide a recommendation on their own: average order value (71 customers pass only because their orders are small) and thin credit file (37 referred only because of it). Mobile layout fix: grid items may shrink so wide tables scroll inside their panel. |
| 3.35 | 15th October 2026 | Added Decision 32: a public Streamlit demo in snapshot mode (owner's choice of option A; approved committing the model package and aggregate reports). Demo mode replays recorded aggregate query results and allowlisted reports from `dashboard/demo_data/`. The exporter refuses customer rows and local paths. Existing-customer lookup and per-customer case studies stay local. Verified on a clean copy with the database unreachable: all 7 pages render, and new-applicant scoring matches the API. |
| 3.36 | 22nd October 2026 | Added Decision 33 (owner's review of the applicant view). Two outcomes: approved, or needs a manual review (never declined; no evidence-based decline rule exists). Manual review if the model refers or any check fires. The applicant sees only their own details, no PD or cutoff and no model issues. Model-tested single changes to order, plan, commitments or savings show what could change the outcome. Supersedes Decision 31's applicant-facing genre selection. Held-out: 80.5% approved, 19.5% manual review. JavaScript matches Python on 975 sentences. |
| 3.37 | 22nd October 2026 | Added Decision 34 (owner's review). Thin-file status is worked out from the score, history and accounts (training limits 579 / 23 months / 1 account; 99.91% agreement), never asked; the record says derived or supplied. The applicant is no longer asked to check details. Plain-language risk officer view: status, why a person needs to look, risk factors, strengths, plain checks, track record. Materiality floor of 0.1 points for reasons and factors, plus a flag when a material effect runs against convention. Held-out evidence moved to the Verification page. The owner's question on data features vs model features is answered in the decision. |
| 3.38 | 22nd October 2026 | Added Decision 35 (owner's choices): affordability limit 45% (FCA CONC 5.2A ratio, the project's strict A/B threshold), declined above it. The approved view shows the affordability figure, credit-file and check-based reasons, and the repayment plan as the data defines it: monthly, first at checkout, every 30 days, £0 fees. The declined view shows arithmetic changes that bring it within the limit. The method line moved to the officer view. PD and cutoff stay hidden from applicants. Corrects Decision 33's claim that no decline rule existed. |
| 3.39 | 23rd October 2026 | Demo site restyled to a light editorial theme (owner's brief: warm ivory, navy ink, Sentinel teal, one meaning per colour). Styling only: no wording, figures or logic changed. DM Serif Display and DM Sans replace Cormorant Garamond and Inter, still self-hosted (owner approved the download), so Decision 29's no-third-party-requests promise holds. Chart colours keep their gold and grey captions true. Applied to the current site, not the mock-up's older layout (no thin-file checkbox; the held-out evidence stays on Verification). 19 of 19 browser checks pass. |
| 3.40 | 24th October 2026 | Demo site restyled to Midnight navy (owner's design brief, revision 2), on top of the committed light theme. Styling only, where the brief and existing decisions disagreed the decisions won. Kept: every page title's words, the model recommendation wording and the full model-view label (Decisions 33, 35), all seven repayment-plan rows, all four 'How this is built' cards, and the synthetic-data notice on the Assessment page only. Orange marks risk and is never text; a decline is neutral ink and white. Bricolage Grotesque and JetBrains Mono replace the DM fonts, still self-hosted (owner approved the download). Owner-approved new copy: the bridge strip and the footer link to koredefolarin.com. The only wording change is 'Gold' to 'Navy' in two chart captions. The deploy gate gains checks for same-origin requests, the mobile menu, neutral declines and no orange text: 25 of 25 pass. |

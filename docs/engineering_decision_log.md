# Sentinel Credit
## Engineering Decision Log

**Project:** Sentinel Credit — Research-Informed CreditOps
Architecture for BNPL Lending
**Author:** Sentinel Credit Project
**Version:** 3.10
**Last Updated:** 30th July 2026
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
and building a connected CreditOps architecture.

To avoid repeating that material, this document
begins with the individual engineering decisions.
Each entry focuses on the implementation challenge,
the evidence consulted, the engineering implication,
the resulting design decision, and its regulatory
context.

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
are provided in Appendix A.

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

Detailed methodology provided in Appendix B.

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
provided in Appendix C.

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
implemented in Decision 2 and Appendix B for some time before
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

---

## Summary Table

| # | Decision | Challenge | Key Evidence | Component |
|---|---|---|---|---|
| 1 | Synthetic data | No UK BNPL dataset | Khalili 2026, Kiran 2024 | bnpl_data_generator.py |
| 2 | Validation | Unvalidated data untrustworthy | FCA/ICO/Turing 2023 | validate_synthetic.py |
| 3 | Feature store | Training-serving skew | Immaneni 2022 | sql/ AWS RDS |
| 4 | Three models | Performance-interpretability-fairness tradeoff | Demajo 2020 | models/ |
| 5 | Drift monitoring | Models degrade post-deployment | Kurian 2024 | pipeline/dags/ |
| 6 | Closed loop | Disconnected tools create gaps | Kurshan 2020, Immaneni 2022, Brahmandam 2025 | Full platform |
| 7 | Explainability | Wrong genre harms consumers | Belton 2025, Demajo 2020 | explainability/ |
| 8 | Model registry | Version traceability required | Kurshan 2020, Immaneni 2022 | MLflow |
| 9 | Approval gate | Automated deployment insufficient | Adari 2021, PRA SS1/23 | approval_gate.py |
| 10 | Hazard-based default | Default assigned before repayment behaviour inverted causality, making repayment features leakage | CRR/EBA default definition, IFRS 9, Klarna Wikipink 2024, Woolard Review 2021 | bnpl_data_generator.py, raw.repayments, validate_synthetic.py, config.yaml |
| 11 | Automated audit-trail enforcement | Decision log accuracy depended on human memory | FCA CRA Review May 2026, PRA SS1/23 Principle 6 | .git/hooks/pre-commit, .git/hooks/post-commit, docs/audit_trail.log |
| 12 | Feature store Stage 3 | Repayment features were proxy leakage until default became emergent | Decision 10 (hazard rewrite) | sql/02_feature_store.sql |

---

## Regulatory Reference Summary

| Regulation | Scope | Decisions |
|---|---|---|
| FCA Consumer Duty PS22/9 2023 | Good outcomes for retail customers | 1, 7 |
| FCA CONC 5.2A | Creditworthiness assessment | 1, 3 |
| PRA SS1/23 April 2023 | Model risk management UK banks | 3, 4, 5, 8, 9 |
| SR 11-7 Federal Reserve 2011 | Model risk management global | 9 |
| FCA Mills Review July 2026 | AI governance retail financial services | 5, 6 |
| FCA CRA Review May 2026 | Governance weakness examples | 5, 6 |
| FCA/ICO/Turing Synthetic Data 2023 | Synthetic data validation | 2 |
| Woolard Review FCA February 2021 | BNPL consumer harm | 1 |
| CRR/EBA default definition | 90-DPD regulatory default threshold | 10 |
| IFRS 9 (SICR) | Significant increase in credit risk threshold | 10 |

---

## Appendix A — Synthetic Data Calibration
## Assumptions and Derivations

Detailed calibration parameters for the synthetic
data generator. These are documented design
assumptions informed by published statistics.

**Base default rate**
base_default_rate in config.yaml = 0.004 (0.4%)
Source: Klarna Wikipink UK 2024 published UK default rate.

This is the RAW Klarna rate. The population uplift for
higher-risk segments is applied by the default probability
engine multipliers — it is NOT baked into the base rate.

The default probability engine applies these multipliers
per transaction (mean multiplier across the population
shown in brackets):

  merchant_risk_multiplier   range 0.50 to 2.50   (mean 1.363) (varies by product)
  thin file       x 2.0      applied to 31%        (mean 1.312)  (only 31% affected)
  vulnerable      x 1.5      applied to ~59%*       (mean 1.285)   (only 59% affected)
  affordability   x 1.3      when ratio > 0.35      (mean 1.103)   (only stretched customers)
  credit score    x 1.4      when score < 550       (mean 1.140)   (only poor credit)



  Compound mean multiplier = 2.89x
  A/B uplift for Group B = +0.004 flat (added last, post-cap)
  Cap = 0.15 maximum per transaction

Realised portfolio default rate = 1.47% per approved
transaction. This emergent rate is the OUTPUT of the engine,
not a target. It is defensible: above Klarna's published
0.4% because our population is 31% thin-file and 24%
low-resilience, and below the Woolard Review FCA 2021
finding of roughly 1 in 10 BNPL customers in arrears at a
major bank.

*Note: is_vulnerable fires if ANY of the four Consumer Duty
drivers is present, so composite prevalence (~59%) is higher
than the single resilience-driver figure of 24%. This is
correct and intended — 24% calibrates the resilience driver
only, not overall vulnerability.

**Note on Decision 10 (v3.1):** The multiplier structure
above still governs which customers are exposed to higher
arrears risk. What changed is that `default_probability`
is no longer stored or drawn directly — these multipliers
now feed into the hazard/arrears process described in
Decision 10, and default emerges from that process at
90 DPD rather than being assigned in advance.

**Age distribution**
Read directly from Klarna Wikipink UK 2024
published age bracket chart for UK Pay in 3:
18-25: 19%, 26-35: 33%, 36-45: 23%,
46-55: 15%, 56+: 10%.

**Vulnerability prevalence 24%**
FCA Financial Lives Survey May 2022.
Measures resilience driver only — one of four
Consumer Duty vulnerability drivers.
Used as conservative lower-bound estimate.

**Thin-file prevalence 31%**
PwC and TotallyMoney YouGov February 2022.
1 in 3 UK adults financially under-served with
thin file as qualifying characteristic.
Represents limited or near-prime credit history.
LexisNexis UK 2021: 1.7m adults with no credit
activity in past 24 months for additional context.

**Mental health prevalence 17%**
NHS Digital Adult Psychiatric Morbidity Survey.
1 in 6 UK adults experience common mental health
problem in any given week.
Used only for FCA Consumer Duty vulnerability
flagging (Driver 1: Health).
Must never be used as direct model input —
would violate Equality Act 2010.

**Merchant categories**
Informed by Klarna UK Store Directory July 2026
and Klarna press release March 2026.
Luxury excluded — price point not purchase type.
Marketplaces excluded — platform not risk category.
Food and Grocery retained — Woolard Review FCA
2021 flagged BNPL for essential spending as
vulnerability indicator.
Percentages are design assumptions. No published
source provides UK BNPL category breakdown.

**Merchant risk multiplier range 0.50 to 2.50**
Klarna Q2 2025 Investor Report:
Standard BNPL delinquency: 0.88%
Fair Financing delinquency: 2.18%
Ratio: 2.18 / 0.88 = 2.47x
Range anchored to this empirically grounded spread.
Individual multipliers derived from two-dimension
scoring framework: order value and discretionary
nature of spend. Design assumptions documented
in config.yaml.

**Income distribution**
ONS Household Income Inequality UK FYE 2023
quintile boundaries as baseline.
Shifted downward — Woolard Review FCA 2021 and
PwC/TotallyMoney 2022 confirm BNPL users skew
lower income than general population.
CFPB 2024 US data used as directional indicator.
Shift magnitude is a design assumption.

**A/B test uplift 0.004 (0.40pp)**
Calculated for 80% statistical power at 5%
significance with 25,000 customers per group
at 1.0% base default rate.
Represents 40% relative increase.
Design assumption tested in validate_synthetic.py.

---

## Appendix B — Validation Methodology

**Fidelity measures**
KS statistic per feature — tests whether synthetic
and reference distributions are drawn from the
same distribution.
Hellinger distance — bounded 0 to 1, measures
distributional similarity.
Jensen-Shannon divergence — symmetric variant of
KL divergence, more stable for distributions with
zero probability mass.

**Utility measure**
Train-on-Synthetic Test-on-Real (TSTR).
Model trained on synthetic data. Scored on Home
Credit Default Risk as reference.
Primary metric: Gini coefficient.
Tolerance: within 5 percentage points of
equivalent real-data trained model.
Limitation: Home Credit is PCA-transformed and
from a different market. Explicitly documented.
Where no real reference file is present, an
internal held-out check runs instead and the
result records that real TSTR was not performed
(see Decision 10).

**Privacy measures**
Singling-out test — checks whether any synthetic
record is unique enough to correspond to a single
real individual.
Nearest-neighbour distance analysis — checks
average distance between synthetic records and
their nearest real-world equivalent. Low distance
indicates re-identification risk.

**Internal consistency**
Five generation runs with different random seeds.
KS statistic across runs should be below 0.05
for stable generator output.

**CTGAN benchmark — planned, not yet implemented**
Intent: train CTGAN on Home Credit Default Risk and
compare shared distributional properties against the
parametric generator's output, as supporting evidence
of realistic statistical structure where they
converge. Descoped until the core pipeline is
complete. Not present in `validate_synthetic.py`.

---

## Appendix C — Metric Definitions

**Gini coefficient**
Measures discriminatory power of the model.
Derived from AUC-ROC: Gini = 2 x AUC - 1.
Range 0 to 1. Higher is better.
Industry benchmark for credit scoring: above 0.40
considered acceptable, above 0.60 strong.

**KS statistic**
Kolmogorov-Smirnov statistic.
Maximum difference between cumulative default and
non-default score distributions.
Range 0 to 1. Higher is better.

**AUC-ROC**
Area Under the Receiver Operating Characteristic
curve. Measures ability to distinguish defaulters
from non-defaulters across all thresholds.

**Precision-Recall AUC**
More informative than ROC-AUC on imbalanced
datasets. Credit default datasets are heavily
imbalanced — non-defaulters are the majority
class. Precision-Recall AUC penalises false
positives more appropriately.

**Expected Calibration Error (ECE)**
Measures whether predicted default probabilities
match observed default rates. A model predicting
10% default probability should default 10% of the
time. Low ECE indicates well-calibrated model.

**Brier Score**
Mean squared error of probability predictions.
Lower is better. Combines calibration and
discrimination into one metric.

**Population Stability Index (PSI)**
Measures distribution shift between training and
scoring populations.
Below 0.10: stable.
0.10 to 0.25: moderate shift, monitor.
Above 0.25: significant shift, model review
required.
Source: Kurian and Allali 2024.

**Recall at operating threshold**
Proportion of actual defaulters correctly
identified at the business lending cutoff.
More operationally relevant than overall AUC
because the business operates at one threshold
not all thresholds.

---

## Appendix D — Regulatory Source Mapping

Full mapping of regulatory documents to specific
design decisions and implementation components.

**FCA Consumer Duty PS22/9 July 2023**
Obligation: deliver good outcomes for retail
customers including understandable information.
Implementation: adaptive explainability engine
produces plain English customer output matched
to error type. Vulnerability flags trigger
enhanced treatment in scoring and explanation.

**FCA CONC 5.2A**
Obligation: creditworthiness assessment before
extending credit. Forward-looking for BNPL as
regulation phases in.
Implementation: affordability_ratio field
calculated as (obligations + instalment) /
monthly income. fca_affordability_flag fires
when ratio exceeds 0.45.

**PRA SS1/23 April 2023**
Principle 1 — model inventory: MLflow registry.
Principle 2 — independent governance: human
approval gate, second line sign-off.
Principle 3 — documented development: feature
store, three-model comparison with metrics.
Principle 4 — independent validation: automated
monthly revalidation, champion/challenger.
Principle 5 — risk mitigants: human gate as
compensating control on drift detection.
Principle 6 — internal audit evidence: MLflow
logs plus audit trail PDF export.

**SR 11-7 Federal Reserve 2011**
Requirement: effective challenge by independent
qualified individuals.
Implementation: human approval gate with second
line reviewer. Rejection logged with rationale.

**FCA Mills Review July 2026**
Finding: AI governance in retail financial
services is still maturing. Consumer Duty is
the primary governance framework.
Implementation: full platform designed around
Consumer Duty obligations. Continuous monitoring
addresses maturity gaps.

**FCA CRA Multi-Firm Review May 2026**
Finding: examples of governance weaknesses
including surveillance gaps and disconnected
processes.
Implementation: closed-loop architecture
connects monitoring to governance automatically.
Used as inspiration not direct regulatory
obligation — CRA review covers rating agencies
not BNPL lenders.

**FCA/ICO/Alan Turing Synthetic Data June 2023**
Framework: utility, fidelity, privacy.
Implementation: validate_synthetic.py covers
all three dimensions. See Appendix B.

**Woolard Review FCA February 2021**
Finding: more than one in ten BNPL customers
at a major bank were already in arrears.
BNPL for essential spending flagged as
vulnerability indicator.
Implementation: Food and Grocery retained as
merchant category. Vulnerability scoring
reflects BNPL population risk profile.

**CRR/EBA Default Definition**
Requirement: default is defined at 90 days past
due on a credit obligation.
Implementation: `days_past_due` accumulates
against the oldest outstanding instalment;
default is reached when it crosses 90 days. See
Decision 10.

**IFRS 9 — Significant Increase in Credit Risk**
Requirement: a distinct, earlier threshold (30
DPD is the commonly applied backstop) for
recognising deteriorating credit risk ahead of
default.
Implementation: `reached_sicr` flags this
threshold separately from `default_flag`. See
Decision 10.

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

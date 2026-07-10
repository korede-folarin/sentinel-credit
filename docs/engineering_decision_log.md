# Sentinel Credit
## Engineering Decision Log

**Project:** Sentinel Credit — Research-Informed CreditOps
Architecture for BNPL Lending
**Author:** Sentinel Credit Project
**Version:** 3.0
**Last Updated:** 7th July 2026
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

---

## Origin Story

The project began with a question: what does a
production-grade credit risk system actually look
like in regulated environments?

Reading the FCA's Credit Rating Agencies multi-firm
review (May 2026) provided a detailed picture of
where production credit model governance breaks down
in practic,disconnected tool and surverlance gaps
exceeding a year. First line teams validating their
own work.

That raised an engineering question:

If continuous monitoring, governance, documentation,
and oversight are all necessary, what does a system
that connects them automatically look like?

That question became Sentinel Credit.

BNPL was chosen as the application context for two
reasons. First, BNPL disproportionately serves
thin-file customers who cannot be scored with
traditional bureau data, making it a genuinely hard
credit scoring problem. Second, errors in thin-file
scoring fall hardest on people who are already
financially excluded, making fairness a core concern
alongside accuracy. BNPL is where the governance
question and the fairness question meet.

Synthetic data was chosen because public credit
datasets have significant limitations for this use
case. Many are PCA-transformed, which destroys
feature correlations. None represent UK BNPL
customers specifically. Generating synthetic data
introduced its own challenges around privacy,
fidelity, and utility — requiring a structured
validation approach informed by published guidance.

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
deliberate methodological choice — usin`=    zg a learned
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

Fidelity — KS tests, Hellinger distance, and
Jensen-Shannon divergence against published
regulatory statistics used as calibration ground
truth.

Utility — Train-on-Synthetic Test-on-Real (TSTR)
Gini comparison using Home Credit Default Risk as
an imperfect proxy. Limitation explicitly
documented: PCA-transformed, different market.
Tests directional signal preservation not exact
fidelity.

Privacy — singling-out and nearest-neighbour
distance analysis for spontaneous re-identification
risk. No differential privacy applied — generator
uses aggregate published statistics not
individual-level data.

CTGAN benchmark — parametric generator output
compared against CTGAN trained on Home Credit.
Where distributions converge, provides supporting
evidence of realistic statistical structure.

Internal consistency — five generation runs with
different seeds. KS tests confirm stable output.

Detailed methodology provided in Appendix B.

### Regulatory Context
Directly implements the three dimensions defined
in the FCA/ICO/Alan Turing Institute guidance.
Privacy assessment follows the FCA risk-based
model — acknowledging that zero risk equals zero
utility and documenting the trade-off explicitly.

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

*With governance records in place, the final
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

---

## Appendix A — Synthetic Data Calibration
## Assumptions and Derivations

Detailed calibration parameters for the synthetic
data generator. These are documented design
assumptions informed by published statistics.

**Base default rate 1.0%**
Klarna Wikipink UK 2024: UK default rate 0.4%.
Population uplift applied for higher-risk
composition:
thin-file (31%) x 2.0 + vulnerable (24%) x 1.5
+ standard (45%) x 1.0 = 1.43x weighted uplift.
0.4% x 1.43 = 0.57%, rounded to 1.0%.
Uplift factors are design assumptions tested in
validate_synthetic.py sensitivity analysis.

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

**CTGAN benchmark**
CTGAN trained on Home Credit Default Risk.
Shared distributional properties compared between
parametric and learned generator.
Convergence on shared features provides supporting
evidence of realistic statistical structure.

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

---

## Document Version History

| Version | Date | Change |
|---|---|---|
| 1.0 | July 2026 | Initial version |
| 2.0 | July 2026 | Full rewrite — unverified claims removed, fairness added, narrative flow added |
| 2.1 | July 2026 | Technical review corrections — three drift types, additional metrics, gate scope clarified |
| 2.2 | July 2026 | Targeted wording refinements per reviewer feedback |
| 3.0 | July 2026 | Restructured for audience — detail moved to appendices, main body condensed to 12-15 pages |
# Project Rationale

Why Sentinel Credit exists, the engineering questions it explores, and the research that informed its design.

---

## The Engineering Question

Reading FCA publications on model governance and AI explainability in financial services raised a practical engineering question:

> *If continuous monitoring, governance, documentation, and human oversight are all necessary in regulated lending, what does a system that connects them automatically look like?*

Most machine learning projects focus primarily on predictive performance. In regulated lending, prediction is only one part of a much larger system.

A production credit model exists within a lifecycle that includes data validation, monitoring, explainability, governance, documentation, deployment, and human oversight.

Sentinel Credit explores how those activities can be engineered as a connected workflow rather than a collection of independent tools.

---

## Why BNPL

Buy Now, Pay Later (BNPL) was chosen as the application context for two reasons.

First, BNPL frequently serves customers with limited traditional credit histories, making credit assessment more challenging than in conventional lending.

Second, errors in assessing thin-file customers can disproportionately affect people who already face barriers to accessing financial services. This makes governance, explainability, and fairness important engineering considerations alongside predictive performance.

---

## Why Synthetic Data

The project required a representative UK BNPL population that is not publicly available.

Existing public credit datasets have important limitations for this use case. Many publicly available fraud and credit datasets are PCA-transformed, replacing original variables with anonymous components. While useful for benchmarking classification models, this transformation obscures the original relationships between variables such as income, repayment behaviour, and credit history. Those relationships are exactly what a synthetic credit population must model explicitly. It also leaves the resulting variables difficult to interpret, making the assumptions behind them harder to document, validate, and defend.

Public datasets from other lending markets also differ in customer behaviour, product structure, and regulatory context.

Building the dataset was therefore a deliberate engineering decision rather than a fallback. It enabled the project to explore reproducibility, traceability, validation, and governance from the earliest stages of the pipeline.

Generating synthetic data also introduced its own engineering challenges around fidelity, utility, and privacy, requiring a structured validation approach informed by published guidance.

---

## Research Problems

### Problem 1 — Governance activities are disconnected

The FCA Credit Rating Agencies Multi-Firm Review (May 2026) highlights that governance weaknesses can persist even where firms possess sophisticated tooling, particularly when monitoring, governance, and audit activities are insufficiently integrated.

**Design Response**

Monitoring events trigger governance workflows automatically through a closed-loop architecture rather than relying on manual hand-offs between independent processes.

---

### Problem 2 — Explainability is often treated as one-size-fits-all

The FCA AI Explainability Research Note (Belton et al., February 2025) found that increasing explanation detail reduced participants' ability to identify model errors by up to twenty percentage points.

The findings suggest that explanation quality depends not only on content but also on matching the explanation style to the intended audience and error type.

**Design Response**

An adaptive explanation engine first classifies the error before selecting an explanation strategy appropriate for the audience.

---

### Problem 3 — Validated synthetic UK BNPL data is scarce

No widely available public dataset was identified that adequately represents UK BNPL customer behaviour.

Although synthetic data is increasingly discussed within financial services, comparatively few publicly documented examples demonstrate a structured validation process covering utility, fidelity, and privacy.

**Design Response**

Synthetic data generation and validation follow the FCA, ICO, and Alan Turing Institute framework covering fidelity, utility, and privacy.

---

## Design Principles

Several principles guided engineering decisions throughout the project.

- **Traceability over convenience** — every derived feature can be traced back to immutable source data.
- **Reproducibility over randomness** — datasets regenerate deterministically from a fixed random seed.
- **Governance by design** — governance is built into the architecture rather than added afterwards.
- **Production realism over benchmark optimisation** — engineering decisions prioritise deployment, auditability, and maintainability alongside predictive performance.

These principles influenced decisions across the synthetic data generator, database design, feature engineering, monitoring, explainability, deployment, and governance workflows.

---

## Scope and Positioning

Sentinel Credit is a personal portfolio project demonstrating production-oriented credit risk engineering concepts and their implementation.

It is not intended to replicate or replace a production system used by a regulated lender.

References to the FCA, PRA, ICO, Alan Turing Institute, and other organisations indicate publications that informed engineering decisions. They should not be interpreted as evidence of endorsement, approval, certification, or regulatory compliance.

---

Further technical reasoning is documented in the [Engineering Decision Log](engineering_decision_log.md).

---

## Research and Regulatory Foundations

Reference material behind the design. The decisions it informed are in the [Engineering Decision Log](engineering_decision_log.md).

### Synthetic Data Calibration Assumptions

> Moved verbatim from the Engineering Decision Log (Appendix A) on 15th October 2026, when reference material was separated from the decision record. Content is unchanged; it describes the design as documented at the time.

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

### Regulatory Source Mapping

> Moved verbatim from the Engineering Decision Log (Appendix D) on 15th October 2026, when reference material was separated from the decision record. Content is unchanged; it describes the design as documented at the time.

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
all three dimensions. See [Validation Methodology](validation_results.md#validation-methodology).

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

### Regulatory Reference Summary

> Moved verbatim from the Engineering Decision Log (Regulatory Reference Summary) on 15th October 2026, when reference material was separated from the decision record. Content is unchanged; it describes the design as documented at the time.

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

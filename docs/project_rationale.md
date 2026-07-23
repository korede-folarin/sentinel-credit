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
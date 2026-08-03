# Sentinel Credit

**A research-informed production architecture demonstrating how synthetic data, explainability, monitoring, and governance can be connected into a closed-loop credit risk workflow for Buy Now, Pay Later (BNPL) lending.**

Every major engineering decision is linked to published research or regulatory guidance and documented in the accompanying Engineering Decision Log.

---

## Architecture

![Sentinel Credit Architecture](docs/sentinel_architecture.png)

---

## Why

Sentinel Credit explores how a modern credit risk system can be designed for regulated lending environments.

Rather than treating machine learning as a modelling exercise alone, the project treats **governance, explainability, monitoring, auditability, and deployment** as first-class engineering problems.

BNPL was chosen as the application context because it commonly serves customers with limited traditional credit histories, making **governance, explainability, and responsible model deployment** particularly important.

Read more in the [Project Rationale](docs/project_rationale.md).

---

## Repository Philosophy

Sentinel Credit is built as a production-oriented engineering project rather than a collection of disconnected machine learning experiments.

Each completed component solves a specific engineering problem, with the reasoning, implementation, and lessons documented alongside the code.

The goal is not simply to build accurate models, but to demonstrate how modern credit risk systems can be designed, governed, monitored, explained, and deployed as connected workflows.

---

## Implementation Progress

| Component | Description | Status |
|---|---|---|
| Synthetic data generator | 50k customers · 174k transactions · 484k repayments · hazard-based, emergent default | ✅ Complete |
| Validation module | Fidelity 8/8 · Utility PASS · Privacy 2/2 — re-run against rebuilt data | ✅ Complete |
| Database schema | Constrained raw schema · Foreign keys · ERD · Third Normal Form | ✅ Complete |
| Raw data load | Three source tables loaded and integrity verified | ✅ Complete |
| Feature store: engineered features | 30+ SQL features across customer, application, exposure, and repayment/delinquency behaviour | ✅ Complete |
| Feature store: AWS RDS migration | Local PostgreSQL → Amazon RDS | 🔄 In Progress |
| Three-model benchmarking | Logistic Regression · Decision Tree · XGBoost | 🔄 In Progress |
| Champion / Challenger workflow | MLflow model registry | 🔄 In Progress |
| A/B testing & causal inference | Z-test · Chi-squared · Difference-in-Differences | 🔄 In Progress |
| Drift monitoring | PSI-based Airflow monitoring | 🔄 In Progress |
| Adaptive explanation engine | Audience-aware LLM explanations | 🔄 In Progress |
| Governance workflow | Human approval gate | 🔄 In Progress |
| Time-series forecasting | Prophet · LSTM · IFRS 9 scenarios | 🔄 In Progress |
| Streamlit dashboard | Risk officer interface | 🔄 In Progress |
| Deployment | BentoML · Docker · AWS | 🔄 In Progress |
| Power BI dashboard | Portfolio and model monitoring | 🔄 In Progress |

---

## Data Foundation

The synthetic dataset was built specifically for this project because no publicly available dataset was identified that adequately represents UK BNPL customer behaviour.

Default is modelled as an emergent outcome of a repayment hazard process, reaching the CRR/EBA 90-days-past-due definition through actual delinquency behaviour, rather than being assigned to a transaction in advance. This keeps repayment history usable as a genuine predictive signal rather than a byproduct of the label.

Generation separates three distinct concepts:

- Published evidence
- Calibrated values
- Documented engineering assumptions

so the origin and justification of every generated value remain explicit.

Validation follows the **FCA, ICO and Alan Turing Institute** framework across:

- Fidelity
- Utility
- Privacy

Loading the data into PostgreSQL exposed an identifier collision that Python validation had not detected, leading to a redesign of the identifier strategy and reinforcing the importance of database-level integrity constraints.

Generator-only variables remain in the raw layer for auditability but are excluded from the feature store to prevent label leakage.

Read more in [Database Design](docs/database_design.md) and [Engineering Decision Log](docs/engineering_decision_log.md).

---

## Engineering Lessons

Several engineering decisions proved more valuable than expected during the project.

- Generating synthetic data is easier than demonstrating that it is fit for purpose.
- Well-designed database constraints uncovered an integrity issue that notebook validation missed.
- Feature engineering is more reliable when performed close to the data.
- A label assigned before the behaviour that should predict it produces features that cannot be trusted, however plausible the individual values look.
- Governance is easier to maintain when it is designed into the architecture from the beginning rather than added afterwards.

---

## Research Foundation

The architecture is informed by published research including:

- FCA Credit Rating Agencies Multi-Firm Review (May 2026)
- Belton et al., FCA AI Explainability Research Note (February 2025)
- FCA / ICO / Alan Turing Institute Synthetic Data Framework (2023)
- PRA SS1/23 Model Risk Management
- CRR/EBA Default Definition and IFRS 9 Significant Increase in Credit Risk

Further documentation:

- [Project Rationale](docs/project_rationale.md)
- [Engineering Decision Log](docs/engineering_decision_log.md)
- [Database Design](docs/database_design.md)
- [References](docs/references.md)

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Data | Python · PostgreSQL · AWS S3 · AWS RDS |
| Machine Learning | Scikit-learn · XGBoost · SHAP |
| MLOps | MLflow · Apache Airflow · GitHub Actions |
| Serving | BentoML · Docker · AWS Lambda · EC2 |
| Visualisation | Streamlit · Plotly · Power BI |
| AI | Anthropic Claude API |

---

## Getting Started

Generate the synthetic dataset.

```bash
python data_generator/bnpl_data_generator.py
```

Validate the generated dataset.

```bash
python validation/validation.py
```

Create the PostgreSQL schema.

```bash
psql -U postgres -d sentinel_credit -f sql/01_create_schema.sql
```

Project configuration is centralised in:

```text
config.yaml
```

---

## Documentation

| Document | Purpose |
|---|---|
| [Project Rationale](docs/project_rationale.md) | Why the project exists and the engineering questions it explores |
| [Engineering Decision Log](docs/engineering_decision_log.md) | Engineering decisions and the evidence supporting them |
| [Database Design](docs/database_design.md) | Schema, integrity constraints, and feature engineering |
| [References](docs/references.md) | Regulatory publications, academic research, and calibration sources |

---

## Current Status

The data foundation is rebuilt, validated, and feature-engineered: hazard-based emergent default, reloaded into PostgreSQL, revalidated with recorded results (fidelity 8/8, utility PASS, privacy 2/2), and 30+ features spanning customer, application, exposure, and repayment/delinquency behaviour. Next up is model benchmarking, followed by monitoring, explainability, governance, deployment, and forecasting.

Follow the build on LinkedIn:

**Folarin Korede**

---

## Disclaimer

Sentinel Credit is an independent portfolio project informed by publicly available research and regulatory publications.

References to the FCA, PRA, ICO, Alan Turing Institute, and other organisations indicate publications that informed engineering decisions. They should not be interpreted as evidence of regulatory approval, certification, endorsement, or affiliation with the project.

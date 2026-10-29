# Sentinel Credit

**A research-informed production architecture demonstrating how synthetic data, explainability, monitoring, and governance can be connected into a closed-loop credit risk workflow for Buy Now, Pay Later (BNPL) lending.**

**[▶ Try the live demo](https://korede-folarin.github.io/sentinel-credit/)** · [Read the case study](https://koredefolarin.com/sentinel-credit.html) · [Engineering Decision Log](docs/engineering_decision_log.md) · [Model results](docs/model_results_summary.md)

[![CI](https://github.com/korede-folarin/sentinel-credit/actions/workflows/ci.yml/badge.svg)](https://github.com/korede-folarin/sentinel-credit/actions/workflows/ci.yml) ![Python 3.11](https://img.shields.io/badge/Python-3.11-17233a) ![XGBoost](https://img.shields.io/badge/model-XGBoost%20v1.1-17233a) ![Decisions](https://img.shields.io/badge/engineering%20decisions-35%20documented-17233a) ![Data](https://img.shields.io/badge/data-100%25%20synthetic-565a62)

---

## In 30 seconds

- **What it is:** an end-to-end credit risk platform for BNPL lending. It covers synthetic data, a feature store, model training and promotion, explanations, drift monitoring and audit evidence, all wired into one loop.
- **What you can do right now:** [score an application in your browser](https://korede-folarin.github.io/sentinel-credit/). The governed champion model runs on the page and nothing you type leaves it. You get the decision an applicant would see, in plain English, and what would change it.
- **Why it's different:** every consequential choice is written up with its evidence in an **[Engineering Decision Log](docs/engineering_decision_log.md)** of 35 decisions. That includes the mistakes I found in my own work and how I fixed them.
- **Headline result:** a calibrated XGBoost with **Gini 0.56** that holds out of time (0.5618 → 0.5611). The browser copy of the model was checked against the Python original on **41,604 customers with zero decisions changed**.

---

## See it working

<p align="center">
  <img src="docs/images/demo-assessment.png" alt="Assessment result: the lender's model view with the probability of default, recommendation and risk scale, above the applicant's plain-English explanation and what could change the outcome" width="760">
</p>

**Above:** one assessment. The lender sees the probability of default against the 6.57% review cutoff. The applicant sees a plain-English outcome, the reasons in their own figures, and model-tested changes that would alter it. They never see the score or the model's internals.

<p align="center">
  <img src="docs/images/demo-simulator.png" alt="Cutoff and fairness simulator: a slider for the review cutoff, headline metrics, and charts of workload against defaults caught and defaulters missed by credit-file depth" width="760">
</p>

**Above:** the [cutoff and fairness simulator](https://korede-folarin.github.io/sentinel-credit/simulator.html). Move the cutoff and watch the workload, the defaults caught, and *who bears the errors* change, using the champion's real held-out results.

---

## Results

Champion **v1.1**: XGBoost, Platt-calibrated, 13 features. Trained on 33,283 customers, tested on 8,321 held out, plus an out-of-time split. Full detail: [Model Results Summary](docs/model_results_summary.md).

| Held-out | Stratified | Out-of-time |
|---|---:|---:|
| Gini | **0.5618** | **0.5611** |
| AUC | 0.7809 | 0.7806 |
| KS | 0.4712 | 0.4390 |
| Brier score | 0.0254 | 0.0241 |
| Defaults caught at the cutoff (recall) | 57.96% | 40.32% |
| Applications sent to review | 15.9% | 11.9% |

- **Three models were benchmarked:** XGBoost (Gini 0.5618) against a decision tree (0.5486) and logistic regression (0.4896).
- **The riskiest 10% of customers hold 39% of all defaults** (lift 3.93×).

---

## The decisions I'm proudest of

Accuracy is the easy part. These are the judgement calls a model risk reviewer would ask about. Each links to its write-up.

1. **I removed a strong feature because it leaked the answer.**
   - `pct_late_payments` partly encoded the outcome: "late" also recorded arrears pay-offs, and post-default rows sat in its denominator.
   - Removing it cost 0.004 Gini, with a 95% confidence interval that includes zero.
   - The promotion rule only allowed a new champion that beat the old one on Gini, so a fix like this could never pass. I added a validity override, with evidence, instead of bending the numbers. *(Decisions 26–27)*
2. **I found the model's blind spot and routed it to a person.**
   - Customers with an established credit file *never* reach the review cutoff, so at the cutoff the model misses every defaulter in that group.
   - Within the group it still ranks risk: the top 5% default about **three times as often** (31 against 11 per 1,000 on held-out data).
   - Those applicants now get "risk officer review required", stated rather than silently passed. *(Decision 31)*
3. **I don't trust a converted model: I prove it.**
   - The public demo runs an ONNX copy of the champion in the browser.
   - Before every deploy, 25 automated browser checks confirm three things: it scores like Python (max difference 6e-8 across 41,604 customers, 0 flips), the explanations match the Python service word for word, and no request leaves the site. *(Decision 29)*
4. **Explanations are for the person reading them.**
   - The applicant gets an outcome (approved, declined over the 45% affordability limit, or manual review), reasons in their own figures, and changes the model has actually tested.
   - The risk officer gets risk factors, strengths and checks in plain language.
   - The decision rule is deliberately hidden from applicants. FCA research (Belton et al., 2025) found that showing it cut error-spotting by 18 points. *(Decisions 30–35)*
5. **Synthetic data, validated before use.**
   - No public UK BNPL dataset exists, so I generated 50,000 customers, 174,000 transactions and 484,000 repayments.
   - Default is *emergent*: it comes from repayment behaviour and reaches the 90-days-past-due definition, rather than being stamped on in advance.
   - The data passes the FCA / ICO / Alan Turing Institute fidelity, utility and privacy checks. *(Decisions 1, 2, 10)*
6. **I audit my own documentation.** In October 2026 I checked every claim in the docs against the code. I logged what was wrong, fixed it and kept the record: [Audit Findings](docs/audit_findings_2026-10-12.md).

---

## How it works

![Sentinel Credit architecture](docs/sentinel_architecture.png)

```text
Synthetic data ─► Validation ─► PostgreSQL feature store ─► Train & calibrate (MLflow)
      ▲                                                            │
      │                                              Champion / challenger + evidence gate
      │                                                            │
 Drift monitoring (Airflow, PSI) ◄── Scoring: FastAPI · Streamlit · in-browser ONNX
                                                                   │
                                   Explanations: applicant · risk officer · decision record
```

| Layer | Built with |
|---|---|
| Data and features | Python · PostgreSQL (constrained raw schema, 64-column feature store) |
| Modelling | scikit-learn · XGBoost · SHAP · Prophet (portfolio forecasting) |
| MLOps | MLflow tracking and Model Registry · Apache Airflow (Astro, Docker) · evidence-gated promotion |
| Serving | FastAPI · Streamlit dashboard · Power BI (DirectQuery) · ONNX Runtime Web (public demo) |
| Quality | Headless-browser deploy gate (GitHub Actions) · parity checks · word-for-word explanation tests |

---

## Run it yourself

```bash
python data_generator/bnpl_data_generator.py          # generate the synthetic data
python validation/validation.py                       # fidelity, utility and privacy checks
psql -U postgres -d sentinel_credit -f sql/01_create_schema.sql
uvicorn serving.app:app --reload                      # scoring API
streamlit run dashboard/Home.py                       # risk officer dashboard
```

- **Settings:** everything is configured in `config.yaml`.
- **Database password:** set `SENTINEL_DB_PASSWORD` in the environment. It is never written into code.
- **Shared scoring:** the API, the dashboard and the demo site share one scoring path (`serving/features.py`, `serving/reasons.py`), so they give identical answers.
- **Tests:** `pytest`. What is and isn't covered, honestly: [TESTING.md](TESTING.md).
- **No database?** The Streamlit dashboard also runs from a published snapshot: see *Public Streamlit demo* below.

<details>
<summary><b>Repository map</b></summary>

| Folder | What's in it |
|---|---|
| `data_generator/` | Hazard-based synthetic BNPL data generator |
| `validation/` | Synthetic-data fidelity, utility and privacy checks |
| `sql/` | Raw schema, feature store, reporting views |
| `models/` | Training, evaluation, SHAP, cutoff selection, explanation rules |
| `pipeline/` | Drift monitoring, approval gate, champion promotion (Airflow DAGs) |
| `serving/` | FastAPI scoring service, shared features and explanations |
| `dashboard/` | Streamlit risk officer dashboard, Power BI setup |
| `forecasting/` | Prophet portfolio forecasts |
| `website/` | Public demo site, its export tools and browser checks |
| `docs/` | Decision log, results, audit findings, design and references |

</details>

<details>
<summary><b>Implementation status</b> (October 2026, checked against the code)</summary>

✅ built and verified · 🟡 partly built · ⬜ planned. "D" numbers refer to the [Engineering Decision Log](docs/engineering_decision_log.md).

| Component | What exists | Status |
|---|---|---|
| Synthetic data generator | 50k customers · 174k transactions · 484k repayments · hazard-based, emergent default (D1, D10) | ✅ |
| Validation module | Fidelity 8/8 · Utility PASS · Privacy 2/2 (D2). CTGAN benchmark planned | ✅ |
| Database and feature store | Local PostgreSQL, constrained raw schema, 64-column feature store (D3, D12, D17) | ✅ |
| Three-model benchmarking | Logistic regression · decision tree · XGBoost, calibrated, MLflow-tracked (D4, D13) | ✅ |
| Champion model | **v1.1**: XGBoost, 13 features, cutoff 0.0657 (D18, D19, D27) | ✅ |
| Explainability (SHAP) | Global and local SHAP on the champion (D14) | ✅ |
| A/B test analysis | Two-proportion test, power analysis, covariate balance (D15, D16) | ✅ |
| Champion / challenger workflow | MLflow Model Registry aliases, validity-override path, rollback rule (D21, D27) | ✅ |
| Drift monitoring | PSI + data quality, origination-cohort view; Airflow, runs while the local stack is up (D20, D24, D25) | ✅ |
| Approval gate | Automated evidence gate on promotion + typed human confirmation; no approver identity recorded (D23) | 🟡 |
| Portfolio forecasting | Prophet, backtested. LSTM and IFRS 9 macro scenarios planned (D22) | 🟡 |
| Scoring API | FastAPI, runs locally (`serving/app.py`) | ✅ |
| Streamlit dashboard | Risk officer interface; public demo mode from a published snapshot (D32) | ✅ |
| Power BI report | DirectQuery against local PostgreSQL (`dashboard/powerbi/`) | ✅ |
| Plain-English reasons | Outcome (approved, declined over the 45% affordability limit, or manual review), reasons in the applicant's own figures, repayment plan, model-tested changes (D30, D31, D33–D35) | ✅ |
| Adaptive explanation engine | Error-type checks, "risk officer review required" flag; applicant, risk officer and audit-record outputs (D7, D31). Approver identity not recorded (D23) | ✅ |
| Public demo site | In-browser ONNX champion on GitHub Pages, gated on 25 browser checks (D29–D31) | ✅ |
| Cloud deployment / AWS | BentoML · Docker · AWS RDS. The API, database and dashboard run locally (D3, D6) | ⬜ |

</details>

<details>
<summary><b>Public Streamlit demo</b></summary>

The dashboard has a demo mode for Streamlit Community Cloud (Decision 32):
- **What it reads:** a published snapshot in `dashboard/demo_data/` (aggregate query results, aggregate reports and the model files). It never needs a database.
- **What's live:** new-applicant scoring.
- **What stays local:** existing-customer lookup and per-customer case studies, because customer records are never published.

To deploy it once:
1. At [share.streamlit.io](https://share.streamlit.io), choose **Create app**.
2. Pick this repository, branch `main`, main file `dashboard/Home.py`.
3. Under **Advanced settings**, choose Python 3.11 and add the secret `SENTINEL_DEMO = "1"`.

Every push to `main` then redeploys it. Refresh the snapshot after a promotion or report run with `python dashboard/tools/export_demo_data.py`.

</details>

---

## Honest limitations

A model is only as trustworthy as what it admits about itself:

- **All data is synthetic.** It is calibrated to published UK evidence, but it is not real lending data.
- **The champion is a customer-level portfolio-risk model,** not a validated application-time underwriting model. The demo's approvals illustrate the mechanism (Decision 18).
- **The 15% review capacity behind the cutoff is a placeholder.** It's an assumption awaiting a business owner.
- **The model leans on order size in ways a lender should question.** For 71 held-out customers, small orders alone keep them below the review line. This is flagged for the next model review (audit findings A31, A35).
- **Approver identity is not yet recorded** in promotion or decision records (Decision 23).
- **The platform runs locally.** Only the demo site and the dashboard snapshot are public.

---

## Documentation

| Document | Purpose |
|---|---|
| [Engineering Decision Log](docs/engineering_decision_log.md) | Every consequential decision, its evidence and its outcome |
| [Model Results Summary](docs/model_results_summary.md) | One-page results for the champion |
| [Project Rationale](docs/project_rationale.md) | Why the project exists, plus its research and regulatory foundations |
| [Database Design](docs/database_design.md) | Schema, integrity constraints, feature engineering |
| [Validation Results](docs/validation_results.md) | Synthetic-data validation results and method |
| [Model Evaluation Report](docs/model_evaluation_report.md) | Evaluation results and metric definitions |
| [Forecasting Summary](docs/forecasting_summary.md) | Prophet portfolio forecasts, backtests and caveats |
| [Audit Findings](docs/audit_findings_2026-10-12.md) | My audit of the documentation against the code |
| [Power BI Setup](dashboard/powerbi/README.md) | Connecting the Power BI report to the reporting views |
| [References](docs/references.md) | Regulatory publications, research and calibration sources |

**Research foundation:**
- FCA Credit Rating Agencies Multi-Firm Review (2026)
- Belton et al., FCA AI Explainability Research Note (2025)
- FCA / ICO / Alan Turing Institute Synthetic Data Framework (2023)
- PRA SS1/23 Model Risk Management
- CRR/EBA default definition and IFRS 9

---

## About

Built by **Korede Folarin**: [portfolio](https://koredefolarin.com/) · [the Sentinel Credit case study](https://koredefolarin.com/sentinel-credit.html).

<sub>**Disclaimer.** Sentinel Credit is an independent portfolio project informed by publicly available research and regulatory publications. References to the FCA, PRA, ICO, Alan Turing Institute and other organisations indicate publications that informed engineering decisions. They do not imply regulatory approval, certification, endorsement or affiliation.</sub>

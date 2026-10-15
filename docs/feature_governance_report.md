# Sentinel Credit - Feature Governance Report

> **Current champion: v1.1 (14th October 2026).** This report documents the **August 2026 v1** champion (14 features, cutoff 0.0671) and is kept as the historical record of it. v1.1 removed `pct_late_payments` for leaking outcome information and is otherwise the same model (identical hyperparameters, split and calibration). See [Decision 27](engineering_decision_log.md). Live v1.1 evidence is regenerated under `reports/`; v1's is archived in `reports/_archive_v1_14f_2026-08/`.
>
> v1.1 governance review regenerated for the 13-feature set (`reports/feature_governance/feature_governance_review.md`). All 13 features are PENDING_MANUAL_REVIEW, awaiting the owner's sign-off (audit finding A13).


**Report status:** Historical: describes the v1 champion, superseded by v1.1 on 14 October 2026 (see the box above).
**Dataset version:** `full_horizon_2025_v1`  
**Feature-set label:** `18_features_pre_governance_review`  
**Governed model:** XGBoost `v1`  
**Review date:** 4 August 2026  
**Primary script:** `models/govern_model_features.py`

---

## 1. Purpose

This report consolidates the feature-governance evidence for the current 18-feature candidate set. It combines:

```text
decision-time availability
information value
variance inflation factor
pairwise correlation
SHAP importance
temporal PSI
repeated cross-validation ablation
exposure-window diagnostics
business and regulatory relevance
```

The automated file `feature_governance_review.md` remains the machine-generated evidence table. This document provides the human interpretation, limitations, decisions that can already be made, and questions that still block a final feature freeze.

---

## 2. Governed package identity

| Item | Value |
|---|---|
| Model | XGBoost |
| Model version | `v1` |
| Dataset | `full_horizon_2025_v1` |
| Feature-set label | `18_features_pre_governance_review` |
| Modelable customers | 41,604 |
| Canonical training customers | 33,283 |
| Defaults in canonical training set | 906 |
| Training default rate | 2.72% |
| Temporal test customers | 4,857 |
| Candidate features | 18 |

The script verified the saved model package, split IDs, feature list, dataset identity and model version before running the review.

---

## 3. Review design

Repeated leave-one-feature-out validation was run on the canonical training partition.

```text
5 folds × 3 repeats
1 all-feature baseline
18 leave-one-feature-out scenarios
19 scenarios × 15 folds
285 fitted XGBoost models
```

The held-out stratified and temporal test labels were not used to make feature decisions.

For each feature, the review measured the change in validation Gini and PR-AUC when the feature was removed. In the table below:

```text
negative delta = removal made performance worse
positive delta = removal improved performance
```

Small differences must be interpreted alongside fold consistency, SHAP, validity and business meaning rather than used as automatic retain-or-remove rules.

---

## 4. Consolidated evidence

| Feature | Decision-time status | IV | VIF | SHAP share | Temporal PSI | Δ Gini if removed | Δ PR-AUC if removed |
|---|---|---:|---:|---:|---:|---:|---:|
| `thin_file_flag` | Available | 0.7533 | 5.0095 | 27.77% | 0.0007 | -0.0007 | -0.0001 |
| `bnpl_exposure_to_monthly_income_ratio` | Conditional as-of | 0.2952 | 2.6280 | 17.21% | 0.1147 | -0.0020 | -0.0016 |
| `savings_balance_gbp` | Available | 0.7470 | 1.7586 | 17.01% | 0.0030 | -0.0068 | -0.0060 |
| `avg_n_instalments` | Conditional as-of | 0.1213 | 1.1736 | 9.16% | 0.0195 | -0.0109 | -0.0028 |
| `n_applications` | Conditional as-of | 0.2394 | 2.1957 | 6.05% | 0.1449 | +0.0007 | +0.0000 |
| `pct_high_risk_category` | Conditional as-of | 0.1127 | 1.5012 | 5.74% | 0.0793 | -0.0037 | -0.0012 |
| `credit_score` | Available | 0.6282 | 2.4315 | 2.71% | 0.0013 | +0.0012 | -0.0004 |
| `avg_order_value_gbp` | Conditional as-of | 0.1294 | 1.5912 | 2.50% | 0.0422 | -0.0000 | +0.0003 |
| `pct_late_payments` | Conditional as-of | 0.1565 | 2.5160 | 2.30% | 0.0277 | -0.0029 | +0.0014 |
| `months_credit_history` | Available | 0.6199 | 1.9320 | 1.85% | 0.0015 | -0.0007 | +0.0002 |
| `pct_medium_risk_category` | Conditional as-of | 0.1170 | 1.4540 | 1.55% | 0.0724 | +0.0002 | -0.0001 |
| `num_defaults_24m` | Available | 0.3022 | 1.5020 | 1.34% | 0.0005 | -0.0004 | +0.0005 |
| `age` | Available | 0.0130 | 1.0000 | 1.22% | 0.0019 | +0.0006 | +0.0000 |
| `monthly_income_gbp` | Available | 0.0250 | 5.0175 | 1.14% | 0.0039 | +0.0002 | -0.0007 |
| `existing_monthly_obligations_gbp` | Available | 0.0032 | 4.0090 | 1.09% | 0.0026 | +0.0009 | -0.0001 |
| `max_affordability_ratio` | Conditional as-of | 0.0388 | 2.1220 | 0.73% | 0.0073 | +0.0012 | +0.0002 |
| `num_credit_accounts` | Available | 0.6244 | 1.9320 | 0.65% | 0.0020 | -0.0000 | -0.0003 |
| `n_late_payments` | Conditional as-of | 0.0733 | 3.0950 | 0.00% | 0.0329 | -0.0000 | -0.0001 |

No VIF exceeded a level that would independently require automatic exclusion. The largest PSI values were observed for `n_applications` and `bnpl_exposure_to_monthly_income_ratio`, but both remained below the project's severe-drift threshold of 0.25. PSI does not resolve whether a feature is valid at decision time.

---

## 5. Findings from repeated ablation

### 5.1 Strong contribution evidence

**`savings_balance_gbp`**

Removing savings produced the largest consistent deterioration among day-zero attributes:

```text
Δ validation Gini:   -0.0068
Δ validation PR-AUC: -0.0060
baseline PR-AUC wins: 12 of 15 folds
```

It is available at decision time and is a strong provisional retain candidate.

**`avg_n_instalments`**

Removing the feature produced the largest mean Gini deterioration:

```text
Δ validation Gini:   -0.0109
Δ validation PR-AUC: -0.0028
baseline PR-AUC wins: 11 of 15 folds
```

This is strong predictive evidence but not a final retention decision for underwriting. The current lifetime average must first be reconstructed as a prior-history feature.

### 5.2 Zero-use feature

**`n_late_payments`**

The feature had:

```text
SHAP share: 0.00%
baseline PR-AUC wins: 4
removal PR-AUC wins: 2
ties: 9
```

The zero SHAP value means the saved XGBoost ensemble did not split on the feature. Frequent identical fold results are therefore expected. This is not evidence of a broken review.

Combined with exposure-time dependence, the current lifetime count has no defensible reason to remain in an application-time model.

### 5.3 Conflicting metrics

**`pct_late_payments`**

Removal worsened Gini but improved PR-AUC:

```text
Δ Gini:   -0.0029
Δ PR-AUC: +0.0014
```

The feature cannot be decided from one metric. Its point-in-time construction must be resolved first.

**`credit_score`**

Removal slightly improved Gini while slightly reducing PR-AUC. Credit score also sits in a correlated credit-file cluster with `thin_file_flag`, `months_credit_history` and `num_credit_accounts`. The result indicates substitution rather than proving that credit score is irrelevant.

### 5.4 Weak univariate features

`age` and `existing_monthly_obligations_gbp` had low IV:

```text
age:                                 0.0130
existing_monthly_obligations_gbp:    0.0032
```

IV is a screening measure, not a complete decision rule. Applying a weak-IV rule consistently would challenge both features, not age alone. Any final decision should be based on predictive contribution, intended use, explainability and regulatory defensibility.

---

## 6. Point-in-time investigation

Nine features were initially labelled `conditional_as_of_logic`. A diagnostic then tested correlation with the amount of post-first-transaction activity available to each customer.

| Feature | Correlation with exposure-window length |
|---|---:|
| `n_applications` | 0.4705 |
| `bnpl_exposure_to_monthly_income_ratio` | 0.3394 |
| `n_late_payments` | 0.2172 |
| `avg_order_value_gbp` | 0.0822 |
| `pct_late_payments` | -0.0037 |
| `pct_medium_risk_category` | -0.0039 |
| `pct_high_risk_category` | -0.0055 |
| `avg_n_instalments` | -0.0309 |
| `max_affordability_ratio` | -0.0337 |

Exposure-bucket means provided additional evidence:

| Exposure bucket | Avg applications | Avg order value | Avg BNPL-to-income ratio | Avg instalments |
|---:|---:|---:|---:|---:|
| 1 | 2.614 | £573.25 | 0.561 | 3.803 |
| 2 | 3.913 | £618.67 | 0.988 | 3.692 |
| 3 | 4.139 | £624.99 | 1.053 | 3.727 |
| 4 | 4.255 | £616.93 | 1.080 | 3.741 |
| 5 | 4.437 | £622.68 | 1.127 | 3.715 |
| 6 | 4.527 | £638.48 | 1.184 | 3.683 |
| 7 | 4.673 | £643.75 | 1.215 | 3.677 |
| 8 | 4.839 | £643.78 | 1.280 | 3.662 |

### Confirmed exposure-time dependence

```text
n_applications
bnpl_exposure_to_monthly_income_ratio
n_late_payments
```

The current customer-level values accumulate or change materially as more transactions become available. They must not be used as lifetime totals in an application-time underwriting model.

### Borderline exposure-time dependence

```text
avg_order_value_gbp
```

Its relationship is much weaker but the upward drift warrants caution. It also remains vulnerable to future-record inclusion even where the mean is not strongly correlated with window length.

### Not cleared by this test

```text
avg_n_instalments
pct_high_risk_category
pct_medium_risk_category
pct_late_payments
max_affordability_ratio
```

Near-zero correlation shows that the values do not mechanically rise with longer exposure. It does not establish that the value at a historical scoring date excluded future transactions. `max_affordability_ratio` is the clearest example: a lifetime maximum can contain a later application even when its correlation with total exposure length is near zero.

---

## 7. Modelling-scope decision

The governance review cannot produce one final feature list until the model purpose is fixed.

### Option A: Retrospective portfolio-risk model

The unit remains one row per customer and features describe behaviour across a defined observation period.

Under this scope:

- lifetime averages and proportions may be coherent;
- exposure opportunity must be standardised or explicitly controlled;
- the report must state that the model evaluates customers using already-observed portfolio behaviour;
- the model must not be presented as an application-time underwriting model.

### Option B: Application-time underwriting model

The training grain becomes one row per application:

```text
application_id
customer_id
scoring_timestamp
features available strictly before scoring_timestamp
target observed after scoring_timestamp
```

Current lifetime features should be redesigned as:

```text
n_prior_applications
prior_avg_order_value_gbp
prior_avg_n_instalments
prior_pct_high_risk_category
prior_pct_medium_risk_category
prior_n_late_payments
prior_pct_late_payments
max_prior_affordability_ratio
outstanding_bnpl_exposure_to_income_at_scoring
```

The current leave-one-feature-out results cannot be transferred mechanically to these reconstructed variables because their distributions and predictive relationships will change.

---

## 8. Decisions supported now

The evidence supports the following conclusions without waiting for a final model-scope decision:

1. `n_late_payments` should not remain in its current form. It has zero SHAP use and is exposure-time dependent.
2. `n_applications` should not be used as a lifetime count for application-time underwriting.
3. `bnpl_exposure_to_monthly_income_ratio` should not be used as a lifetime aggregate for application-time underwriting.
4. `avg_order_value_gbp` requires strict as-of reconstruction or exclusion before application-time use.
5. Near-zero exposure correlation does not clear the remaining behavioural aggregates.
6. `savings_balance_gbp` has strong predictive and decision-time evidence.
7. `avg_n_instalments` has strong predictive evidence but unresolved timing validity.
8. No final approved feature set exists yet.

---

## 9. Governance controls

The final feature decision record should contain, for every retained feature:

```text
business definition
source tables and SQL lineage
training grain
decision-time availability
missingness treatment
expected direction or behaviour
valid range
drift thresholds
reason for retention
known limitations
owner and approval date
```

For every excluded feature, it should record:

```text
reason for exclusion
supporting evidence
whether retained in the feature store for monitoring
whether a point-in-time replacement is planned
```

The feature store should remain broad. Model eligibility should be controlled by the explicit allowlist rather than by deleting evidence-bearing columns from storage.

---

## 10. Evidence locations

Machine-generated governance evidence:

```text
reports/feature_governance/feature_governance_review.csv
reports/feature_governance/feature_governance_review.md
reports/feature_governance/leave_one_feature_out_cv.csv
reports/feature_governance/leave_one_feature_out_cv_folds.csv
reports/feature_governance/feature_decisions.json
reports/feature_governance/governance_metadata.json
reports/feature_governance/missingness.csv
reports/feature_governance/information_value.csv
reports/feature_governance/vif.csv
reports/feature_governance/correlation_matrix.csv
reports/feature_governance/correlation_evidence.csv
reports/feature_governance/temporal_psi.csv
```

The folder is attached to the canonical XGBoost MLflow run:

```text
2da086f12a544699abcb808d1ec982bf
```

---

## 11. Required sign-off sequence

1. Confirm the model purpose and scoring grain.
2. Rebuild conditional features where application-time underwriting is selected.
3. Re-run EDA, IV, VIF, correlations and point-in-time checks.
4. Re-run repeated feature governance on the reconstructed candidate set.
5. Record final retain, remove or monitoring-only decisions.
6. Update `models/model_features.py`.
7. Retrain all three models.
8. Re-run evaluation and SHAP.
9. Select the operating cutoff using training-only out-of-fold predictions.
10. Record human approval and update the engineering decision log with the conclusion and links to these reports.

---

## 12. Status statement

The quantitative governance review is complete for the current 18-feature baseline. The review has identified valid day-zero attributes, features with strong predictive contribution, a zero-use feature, metric conflicts and material point-in-time defects. Final feature approval remains blocked by the unresolved choice between retrospective portfolio risk and application-time underwriting.

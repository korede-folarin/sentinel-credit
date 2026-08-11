# Sentinel Credit - SHAP Explainability Report

**Report status:** Pre-governance explainability review  
**Dataset version:** `full_horizon_2025_v1`  
**Feature-set label:** `18_features_pre_governance_review`  
**Explained model:** XGBoost `v1`  
**Explained population:** 8,321 stratified holdout customers  
**Report date:** 4 August 2026  
**Primary script:** `models/explain_models.py`

---

## 1. Purpose

This report documents the numerical SHAP explanation layer for the current XGBoost baseline. It records the explained model, validation checks, global importance, interpretation boundaries, local-explanation outputs and governance issues raised by the results.

The report does not generate customer-facing narratives and does not claim that SHAP values identify causal effects.

---

## 2. Model being explained

Training persists two related artifacts:

```text
champion_calibrated.joblib
champion_explanation_base.joblib
```

The calibrated model produces the probabilities used by the decision layer. The explanation base is an uncalibrated XGBoost model fitted on the same canonical training partition with the same selected hyperparameters.

SHAP is calculated against `champion_explanation_base.joblib`.

This distinction is required because the calibrated wrapper contains fold-level estimators and does not provide one unambiguous tree model for direct SHAP reconstruction.

---

## 3. Interpretation boundary

SHAP values in this report explain the XGBoost raw margin or underlying ranking score.

They do **not** represent changes in the calibrated probability of default. A local contribution must therefore be described as:

> The feature pushed the underlying model score higher or lower.

It must not be described as:

> The feature increased the customer's calibrated probability of default by a stated number of percentage points.

Calibration is monotonic, so global ranking patterns remain informative. Local SHAP magnitudes and calibrated PD movements are nevertheless different scales.

---

## 4. Validation checks

The explanation run was validated before the results were used.

| Check | Result |
|---|---|
| Dataset version | `full_horizon_2025_v1` |
| Model version | `v1` |
| Split | Stratified |
| Explained rows | 8,321 |
| Features | 18 |
| Feature list matched saved model package | Pass |
| Current allowlist matched saved package | Pass |
| Excluded fields present | None |
| Null values in explained matrix | None |
| Maximum absolute additivity error | `2.384e-06` |
| Additivity tolerance | `1e-04` |
| Additivity result | Pass |

The additivity error was substantially below the tolerance, confirming that the SHAP contributions reconstructed the raw model output to numerical precision.

---

## 5. Global feature importance

Importance is measured as mean absolute SHAP value and expressed as a percentage of total model dependence.

| Rank | Feature | Share of total importance |
|---:|---|---:|
| 1 | `thin_file_flag` | 27.77% |
| 2 | `bnpl_exposure_to_monthly_income_ratio` | 17.21% |
| 3 | `savings_balance_gbp` | 17.01% |
| 4 | `avg_n_instalments` | 9.16% |
| 5 | `n_applications` | 6.05% |
| 6 | `pct_high_risk_category` | 5.74% |
| 7 | `credit_score` | 2.71% |
| 8 | `avg_order_value_gbp` | 2.50% |
| 9 | `pct_late_payments` | 2.30% |
| 10 | `months_credit_history` | 1.85% |
| 11 | `pct_medium_risk_category` | 1.55% |
| 12 | `num_defaults_24m` | 1.34% |
| 13 | `age` | 1.22% |
| 14 | `monthly_income_gbp` | 1.14% |
| 15 | `existing_monthly_obligations_gbp` | 1.09% |
| 16 | `max_affordability_ratio` | 0.73% |
| 17 | `num_credit_accounts` | 0.65% |
| 18 | `n_late_payments` | 0.00% |

The top three features accounted for 61.99% of total importance. The top five accounted for 77.20%. This concentration means the validity and stability of a small number of features materially affects the defensibility of the entire model.

---

## 6. Governance interpretation of the ranking

### 6.1 Point-in-time review

Nine behavioural features were marked as requiring conditional as-of logic:

```text
bnpl_exposure_to_monthly_income_ratio
avg_n_instalments
n_applications
pct_high_risk_category
avg_order_value_gbp
pct_late_payments
pct_medium_risk_category
max_affordability_ratio
n_late_payments
```

Together they account for 45.24% of total SHAP importance.

The later exposure-window diagnostic confirmed that three current lifetime versions were materially associated with post-first-transaction opportunity:

| Feature | Correlation with exposure-window length | SHAP share |
|---|---:|---:|
| `n_applications` | 0.4705 | 6.05% |
| `bnpl_exposure_to_monthly_income_ratio` | 0.3394 | 17.21% |
| `n_late_payments` | 0.2172 | 0.00% |

The two features on which the model materially depends account for 23.26% of total SHAP importance. This is not a minor documentation issue. It means a substantial part of the current ranking logic depends on values that are unsuitable for application-time underwriting in their lifetime form.

`avg_order_value_gbp` showed weaker but non-zero exposure-window association, with correlation 0.0822 and a gradual increase across longer windows. Its 2.50% SHAP share is therefore treated as borderline until rebuilt as a strict prior-history feature.

### 6.2 Low correlation is not point-in-time proof

The following features had near-zero correlation with exposure-window length:

```text
avg_n_instalments
pct_high_risk_category
pct_medium_risk_category
pct_late_payments
max_affordability_ratio
```

This shows low mechanical sensitivity to observation length. It does not prove that their current values were available at every historical lending decision. A lifetime average, proportion or maximum may still include transactions that occurred after the application being scored.

The model's heavy dependence on `avg_n_instalments` and `pct_high_risk_category` therefore cannot be approved for application-time use solely from the correlation result.

---

## 7. Feature-level observations

### `thin_file_flag`

This is the dominant model feature at 27.77%. Its importance is consistent with the generated portfolio design and the materially higher observed default rate among thin-file customers. Because it is also a segmentation and fairness-relevant attribute, its use requires monitoring for disparate outcomes rather than an assumption that high importance is automatically desirable.

### `savings_balance_gbp`

This feature contributes 17.01% of total importance and is available at decision time. The leave-one-feature-out governance review found the largest consistent deterioration when it was removed, supporting its value as a core risk attribute.

### `bnpl_exposure_to_monthly_income_ratio`

This feature contributes 17.21%, but its current lifetime construction rises strongly with exposure-window length. High SHAP importance demonstrates model dependence; it does not make the construction valid. For application-time use it should be rebuilt as outstanding BNPL exposure divided by income at the scoring timestamp.

### `n_applications`

This feature contributes 6.05% and increases monotonically with observation length. The current lifetime count should be replaced by `n_prior_applications` calculated strictly before the scoring timestamp.

### `n_late_payments`

This feature has zero SHAP importance. XGBoost did not use it in the fitted tree ensemble. Its zero contribution, together with exposure-time dependence, supports removal or reconstruction rather than retention in its current form.

---

## 8. Local explanations

The explanation pipeline produces local waterfall plots and machine-readable JSON records for selected cases. Each record contains:

```text
customer_id
case_type
actual outcome
calibrated probability
configured cutoff
raw-margin SHAP base value
raw-margin SHAP contributions
ordered feature contributions
scope note
```

Local explanations based on the configured cutoff of `0.137` are illustrative only. That cutoff produced almost no high-risk classifications and is scheduled for recalibration after final feature governance.

Local explanations must also avoid causal language. SHAP identifies the features the model relied on; it does not prove that changing one feature would cause the customer's outcome or model decision to change in isolation.

---

## 9. Generated evidence

Expected outputs under `reports/shap/` include:

```text
global_feature_importance.csv
summary or beeswarm plot
dependence plots
local waterfall plots
local contribution JSON files
shap_metadata.json
validation evidence
```

The SHAP evidence is attached to the canonical stratified XGBoost MLflow run:

```text
2da086f12a544699abcb808d1ec982bf
```

---

## 10. Use restrictions

The current SHAP outputs are valid for:

- explaining the raw XGBoost ranking logic;
- identifying global model dependence;
- supporting feature-governance review;
- examining selected local model-score contributions;
- checking whether model reliance is concentrated.

They are not valid for:

- expressing SHAP values as calibrated PD changes;
- claiming causal drivers of default;
- providing final customer adverse-action reasons;
- approving lifetime aggregates for application-time underwriting;
- supporting final cutoff-dependent case classifications;
- substituting for fairness or feature-validity review.

---

## 11. Required follow-up

1. Decide whether the final model is retrospective portfolio risk or application-time underwriting.
2. For application-time underwriting, rebuild all behavioural features using strict as-of application logic.
3. Retrain the champion on the frozen feature set.
4. Re-run SHAP against the new saved explanation base.
5. Recalculate global and local importance.
6. Build audience-specific explanations only after the numerical layer and reason-code policy are approved.
7. Review whether feature importance remains overly concentrated after point-in-time reconstruction.

---

## 12. Status statement

The SHAP implementation is numerically valid for the saved pre-governance XGBoost model. The explanations reveal a material governance issue: 45.24% of model dependence lies in behavioural features whose application-time availability is not yet established, including 23.26% in features confirmed to be exposure-time contaminated in their current form. The explanation layer is therefore complete as diagnostic evidence but not final for customer-facing underwriting use.

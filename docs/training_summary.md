# Sentinel Credit - Baseline Model Training Summary

> **Current champion: v1.1 (14th October 2026).** This report documents the **August 2026 v1** champion (14 features, cutoff 0.0671) and is kept as the historical record of it. v1.1 removed `pct_late_payments` for leaking outcome information and is otherwise the same model (identical hyperparameters, split and calibration). See [Decision 27](engineering_decision_log.md). Live v1.1 evidence is regenerated under `reports/`; v1's is archived in `reports/_archive_v1_14f_2026-08/`.
>
> v1.1 training: `train_baseline_models.py --pin-xgboost-params` from v1's metadata (learning rate 0.01, depth 3, 300 trees), 13 features. XGBoost Gini 0.5618 stratified, 0.5611 temporal; Logistic Regression 0.4896, Decision Tree 0.5486 (stratified).


**Report status:** Historical: describes the v1 champion, superseded by v1.1 on 14 October 2026 (see the box above).
**Dataset version:** `full_horizon_2025_v1`
**Feature-set label:** `14_features_v1_frozen`
**Model version:** `v1`
**Training date:** 6 August 2026
**Decision log version:** 3.19
**Primary script:** `models/train_baseline_models.py`

---

## 1. Purpose

This report records the model-training process and outputs for the current Sentinel Credit baseline, retrained after Decision 18 resolved the model scope and froze the feature set.

This is not a final approved model-development report. The champion model is confirmed and stable, but the operating cutoff has not yet been recalibrated for the frozen feature set — see Section 9.

---

## 2. Modelling scope

**Decision 18** resolved the modelling scope explicitly: Sentinel Credit v1 is a **customer-level portfolio-risk model**, not an application-time underwriting model. The training grain remains:

```text
one row per modelable customer
```

Under this scope, customer-level behavioural aggregates calculated across the observation period are legitimate model inputs, provided they are not confirmed to grow with exposure length (see Section 4). Application-time underwriting — scoring a specific application using only prior history — is recorded as a future v2 redesign, not the current scope.

The model covers Probability of Default only. Loss Given Default and Exposure at Default remain outside the present scope.

---

## 3. Dataset identity

| Item | Value |
|---|---:|
| Generated customers | 50,000 |
| Customers with at least one modelable transaction | 41,604 |
| Customer default rate | 2.72% |
| Outcome observation | 100% observed |
| Right-censored transactions | 0% |
| Data end date | 2025-12-31 |
| Origination end date | 2024-10-31 |

Unchanged from the pre-governance baseline — Decision 18's feature-set freeze removed columns, not rows. The modelable population, its size, and its default rate are identical to the prior report.

---

## 4. Feature allowlist

**MODEL_FEATURES** was frozen at 14 features by Decision 18, down from the prior 18-feature pre-governance set.

| # | Feature |
|---:|---|
| 1 | `monthly_income_gbp` |
| 2 | `existing_monthly_obligations_gbp` |
| 3 | `savings_balance_gbp` |
| 4 | `credit_score` |
| 5 | `months_credit_history` |
| 6 | `num_credit_accounts` |
| 7 | `num_defaults_24m` |
| 8 | `thin_file_flag` |
| 9 | `max_affordability_ratio` |
| 10 | `avg_order_value_gbp` |
| 11 | `avg_n_instalments` |
| 12 | `pct_high_risk_category` |
| 13 | `pct_medium_risk_category` |
| 14 | `pct_late_payments` |

### 4.1 Removed features

Four features were removed from the prior 18-feature set:

| Feature | Reason |
|---|---|
| `age` | Regulatory defensibility (Equality Act 2010 proxy risk), not IV. Corrects Decision 17's original "weak IV" framing once `existing_monthly_obligations_gbp` — scoring lower on the same metric — was noted to be retained. |
| `n_applications` | Confirmed exposure-time contaminated: correlates with exposure-window length at r=0.47. |
| `bnpl_exposure_to_monthly_income_ratio` | Confirmed exposure-time contaminated: r=0.34. |
| `n_late_payments` | Confirmed exposure-time contaminated: r=0.22. Corroborated, not driven, by an exact 0.000000 SHAP contribution in the prior 18-feature governance review. |

### 4.2 Retained features carrying the same prior flag

Six features previously flagged `conditional_as_of_logic` (`max_affordability_ratio`, `avg_order_value_gbp`, `avg_n_instalments`, `pct_high_risk_category`, `pct_medium_risk_category`, `pct_late_payments`) are **retained**. The point-in-time diagnostic (`sql/diagnostics/check_point_in_time.sql`, correlation against exposure-window length) found no material association for these six. This is stated deliberately as supporting evidence for scope-fitness under the customer-level portfolio-risk scope, **not** as proof the features are point-in-time safe — near-zero linear correlation cannot rule out weak, nonlinear, or segment-specific contamination. That distinction only matters for an application-time scope, which v1 is not.

All 14 features had zero nulls in the modelable population. Median filling is still applied defensively in the training pipeline.

---

## 5. Training methodology

Unchanged from the prior baseline — the retraining exercised the same pipeline against the new feature set, not a new methodology.

### 5.1 Models

Three models were benchmarked: Logistic Regression (linear, auditable baseline), Decision Tree (interpretable rule-based challenger), XGBoost (non-linear performance challenger).

### 5.2 Hyperparameter selection

`GridSearchCV` with five-fold `StratifiedKFold` on the training partition only, average precision as the selection metric. Class balancing remained mandatory: `class_weight="balanced"` for Logistic Regression and Decision Tree, `scale_pos_weight` computed per training partition for XGBoost.

### 5.3 Preprocessing and calibration

Logistic Regression used `RobustScaler`. Tree models were fitted unscaled. All three models' predicted probabilities were calibrated via sigmoid (Platt) scaling.

### 5.4 Validation designs

**Stratified holdout:** 33,283 training customers, 8,321 test customers, 2.72% default rate in both partitions. Canonical split for the governed model package.

**Temporal holdout**, cutoff 2024-09-01: 36,747 training customers, 4,857 test customers, 2.74% training default rate, 2.55% test default rate.

Both splits are unchanged in population from the 18-feature baseline — only the feature columns differ.

---

## 6. Training results

### 6.1 Stratified split

| Model | CV average precision | Test Gini | Test AUC |
|---|---:|---:|---:|
| XGBoost | **0.0987** | **0.5656** | **0.7828** |
| Decision Tree | 0.0728 | 0.5140 | 0.7570 |
| Logistic Regression | 0.0729 | 0.4898 | 0.7449 |

Selected parameters:

| Model | Selected parameters |
|---|---|
| Logistic Regression | `C=0.01`, `penalty=l2`, `class_weight=balanced`, `max_iter=1000` |
| Decision Tree | `max_depth=8`, `min_samples_leaf=100`, `class_weight=balanced` |
| XGBoost | `learning_rate=0.01`, `max_depth=3`, `n_estimators=300`, `scale_pos_weight=35.7362` |

### 6.2 Temporal split

| Model | CV average precision | Test Gini | Test AUC |
|---|---:|---:|---:|
| XGBoost | **0.0997** | **0.5675** | **0.7837** |
| Decision Tree | 0.0762 | 0.5059 | 0.7530 |
| Logistic Regression | 0.0729 | 0.5018 | 0.7509 |

Selected parameters:

| Model | Selected parameters |
|---|---|
| Logistic Regression | `C=0.01`, `penalty=l2`, `class_weight=balanced`, `max_iter=1000` |
| Decision Tree | `max_depth=4`, `min_samples_leaf=100`, `class_weight=balanced` |
| XGBoost | `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `scale_pos_weight=35.4554` |

### 6.3 Split stability

| Model | Stratified Gini | Temporal Gini | Stratified minus temporal |
|---|---:|---:|---:|
| XGBoost | 0.5656 | 0.5675 | -0.0019 |
| Logistic Regression | 0.4898 | 0.5018 | -0.0120 |
| Decision Tree | 0.5140 | 0.5059 | +0.0081 |

XGBoost's temporal generalization improved relative to the 18-feature baseline (gap tightened from -0.0257 to -0.0019, essentially flat). No evidence of the model relying on patterns that fail to hold over time.

### 6.4 Effect of the feature-set reduction

Logistic Regression and Decision Tree lost more discrimination than XGBoost when the four features were removed (LR stratified Gini 0.5603 → 0.4898; DT 0.5451 → 0.5140), while XGBoost held nearly flat (0.5636 → 0.5656). This is consistent with the four removed features carrying more of the linear, easily-split signal the simpler models depended on, while XGBoost's depth-3 boosted ensemble was already drawing more of its signal from interactions among the retained features. The relative gap between XGBoost and the other two models widened as a result — the champion choice is, if anything, more clear-cut on the 14-feature set than on the 18-feature baseline.

---

## 7. Champion candidate

XGBoost remains the baseline champion, on the same grounds as the prior report and more decisively:

- highest cross-validation average precision on both splits;
- highest Gini and AUC on both holdouts, by a wider margin than under the 18-feature set;
- near-zero stratified/temporal Gini gap, the best temporal stability of the three models.

Logistic Regression, previously a close discrimination challenger, fell furthest from XGBoost on this feature set — it is no longer as strong an interpretability counter-argument to the champion choice as it was in the 18-feature report.

---

## 8. Reproducibility and saved artifacts

MLflow experiment: `sentinel_credit_baseline`.

Stratified training runs:

| Model | Run ID |
|---|---|
| Logistic Regression | `e5e3f907d59e45ed96d95cf3433cbea2` |
| Decision Tree | `45b4fd7697894a37aa69544a48b75442` |
| XGBoost (champion) | `a20e01b73c0d4ca3aff1c78ebff542ab` |

Temporal training runs:

| Model | Run ID |
|---|---|
| Logistic Regression | `d165ddc9b2ea4b0a89114725f6307564` |
| Decision Tree | `46b281d4fd2144bfb2d2c8b9d64f1aa3` |
| XGBoost | `d3e336f360314788ac7a5219da8f26ca` |

The canonical stratified package (`models/artifacts/`) — `champion_calibrated.joblib`, `champion_explanation_base.joblib`, `champion_metadata.json`, `feature_list.json`, `split_ids_train.csv`, `split_ids_test.csv`, `test_predictions_stratified.csv`, `test_predictions_temporal.csv` — was regenerated in this run and now records `feature_set_label: 14_features_v1_frozen` and `decision_log_version: 3.19`.

---

## 9. Known limitations and open actions

1. **Business cutoff is confirmed on both splits.** `business_cutoff: 0.0671`, selected from training-only out-of-fold predictions under a 15% review-capacity constraint (Decision 19), confirmed against held-out evaluation on both stratified and temporal — see `business_cutoff_selection_report.md` and the evaluation report, Section 7.
2. **Business cutoff does not affect discrimination or calibration metrics in this report.** Gini, AUC, KS, PR-AUC, ECE, Brier, and lift/capture are rank- or decile-based and were already final for this feature set before the cutoff was chosen. Only confusion-matrix-derived figures and cutoff-dependent fairness rows depend on it.
3. **A segment-level operating-policy item is open, not resolved.** `thin_file_flag=0` shows 100% false-negative rate at 0.0671 on both splits — investigated in detail (`models/segment_threshold_diagnostics.py`, Decision 19 addendum) and found to be a policy-scale problem, not a model-signal failure: the model discriminates risk within `thin_file_flag=0` about as well as within `thin_file_flag=1` (within-segment AUC 0.64-0.69, both CIs clear of 0.5 on both splits), but a rare event within an already low-base-rate segment gets compressed toward zero under one shared calibration. No global threshold change fixes this. Whether segment-aware review capacity is warranted is a governance decision under Decision 12, not resolved here.
4. **Application-time underwriting remains a future redesign,** not current scope (Decision 18). None of today's evidence is a claim about point-in-time application safety.
5. **Synthetic-data scope applies,** as before — these results demonstrate the engineering and model-development process, not performance on a live UK BNPL portfolio.
6. **External validation is not complete;** temporal validation remains internal to the generated dataset.
7. **No final promotion has occurred.** XGBoost is the frozen-feature-set, frozen-cutoff champion candidate, not an approved deployed model — pending the Section 9.3 governance item.

---

## 10. Status statement

The v1 feature set is frozen (Decision 18) and `business_cutoff` is confirmed at 0.0671 on both stratified and temporal held-out evidence (Decision 19). The baseline is technically complete and reproducible for `full_horizon_2025_v1` at `14_features_v1_frozen`. One item remains before v1 is unconditionally final: a governance decision on segment-aware review capacity for `thin_file_flag`, surfaced during cutoff confirmation and documented but not resolved (Decision 19 addendum, Decision 12).

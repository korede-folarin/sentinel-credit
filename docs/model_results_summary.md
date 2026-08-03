# Sentinel Credit — Model Results Summary

**Population:** 18,694 modelable customers · **Bad rate:** 4.26% (796 defaults) · **Features:** 22
**Split:** stratified (14,955 train / 3,739 test) · **Business cutoff:** 0.145
**Source:** `models/train_baseline_models.py`, `models/evaluate_models.py --split stratified`, `models/generate_charts.py`
**Full detail:** `docs/eda_results.md`, `docs/engineering_decision_log.md` (Decision 13)

> **Run against the corrected population.** An earlier version reported results on 41,604 customers at a 1.91% bad rate — a population that included unmatured loans mislabelled as confirmed non-defaults. See Decision 12 addendum ("has_outcome Correction").

---

## Population and Target

![EDA target summary](charts/eda_target_summary.png)

*Source: `reports/eda_target_summary.csv`*

| Check | Result |
|---|---|
| Total customers | 50,000 |
| Modelable (`has_outcome=1`) | 18,694 |
| Bad rate | 4.26% (796 defaults) |
| Transactions | 58,611 |
| Repayment instalments | 172,968 |
| Missing values in training population | 0 |
| Final feature count | 22 |

---

## Model Comparison

![Model comparison](charts/model_comparison_stratified.png)

*Source: `reports/model_comparison_stratified.csv`*

| Model | Gini | AUC-ROC | KS | PR-AUC | ECE | Brier |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.6493** | **0.8247** | **0.5061** | **0.2052** | 0.0026 | **0.0368** |
| Logistic Regression | 0.6241 | 0.8120 | 0.4849 | 0.1929 | 0.0067 | 0.0373 |
| Decision Tree | 0.6083 | 0.8041 | 0.4651 | 0.1756 | 0.0031 | 0.0377 |

All three clear the Gini 0.40 "acceptable" benchmark comfortably. XGBoost's KS of 0.5061 is strong by conventional credit scoring standards. All three are well calibrated (ECE under 0.007) following the Platt-scaling fix — ECE was 0.3552 before calibration was added.

**Best hyperparameters (5-fold CV, `average_precision` scoring):**

| Model | Best hyperparameters |
|---|---|
| Logistic Regression | C=0.1, class_weight=balanced, penalty=l2 |
| Decision Tree | max_depth=4, min_samples_leaf=50, class_weight=balanced |
| XGBoost | max_depth=3, n_estimators=100, learning_rate=0.1, scale_pos_weight=22.48 |

---

## Lift by Risk Decile — All Models vs Random

![Lift comparison](charts/lift_comparison_stratified.png)

*Source: `reports/lift_tables_stratified.csv`*

| Decile | XGBoost | Logistic Regression | Decision Tree | Random |
|---|---|---|---|---|
| 0 (highest risk) | **5.03** | 4.78 | 4.59 | 1.00 |
| 1 | 1.63 | 1.51 | 1.76 | 1.00 |
| 2 | 1.13 | 1.32 | 0.69 | 1.00 |
| 3 | 0.63 | 0.57 | 0.82 | 1.00 |
| 4 | 0.57 | 0.63 | 1.01 | 1.00 |
| 5 | 0.44 | 0.63 | 0.50 | 1.00 |
| 6 | 0.25 | 0.19 | 0.31 | 1.00 |
| 7 | 0.06 | 0.13 | 0.13 | 1.00 |
| 8 | 0.06 | 0.13 | 0.19 | 1.00 |
| 9 (lowest risk) | 0.19 | 0.13 | 0.00 | 1.00 |

Reviewing the top 10% highest-risk customers under XGBoost captures **50.3% of all defaults** at 5.03x the rate of random selection.

---

## XGBoost — Calibration

| Decile | Predicted PD | Actual default rate | A/E ratio |
|---|---|---|---|
| 0 | 0.63% | 0.80% | 1.271 |
| 1 | 0.73% | 0.27% | 0.366 |
| 2 | 0.85% | 0.27% | 0.313 |
| 3 | 1.02% | 1.07% | 1.045 |
| 4 | 1.31% | 1.87% | 1.425 |
| 5 | 1.83% | 2.41% | 1.319 |
| 6 | 2.78% | 2.67% | 0.962 |
| 7 | 4.70% | 4.81% | 1.025 |
| 8 | 9.00% | 6.95% | 0.772 |
| 9 | 19.61% | 21.39% | 1.091 |

A/E ratios cluster near 1.0 across the upper deciles where risk concentration matters most. The lower deciles show larger relative swings, but on very small absolute numbers (1-3 defaults per decile).

---

## Performance at the Business Cutoff (0.145)

| Model | TP | FP | FN | TN | Precision | Recall | Flagged | Bad rate among approved |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | 73 | 248 | 86 | 3,332 | **22.74%** | **45.91%** | 321 | **2.52%** |
| Logistic Regression | 57 | 197 | 102 | 3,383 | 22.44% | 35.85% | 254 | 2.93% |
| Decision Tree | 53 | 203 | 106 | 3,377 | 20.70% | 33.33% | 256 | 3.04% |

Precision of ~22% against a 4.26% base rate is roughly 5x better than random selection. Low absolute precision is structurally expected in rare-event classification and is correctly judged against the base rate, not against 100%.

**On the cutoff itself:** `business_cutoff` is a configurable policy parameter belonging to the decision engine, not a property of any algorithm. It is calibrated here against the champion model (XGBoost, F1-optimal 0.1449). Future champion-challenger promotion requires threshold recalibration rather than assuming transferability across models. See Decision 13.

---

## Model Selection: Performance vs Explainability

XGBoost leads on every discrimination metric. That does not automatically make it the deployed model.

| Requirement | Logistic Regression | XGBoost |
|---|---|---|
| Coefficient-level interpretability | Direct | None — requires SHAP |
| Adverse action explanation | Generated from the model | Generated from an approximation |
| Regulatory precedent | Long-standing (PRA SS1/23, SR 11-7) | Accepted, but requires the SHAP layer validated separately |
| Explanation stability | Coefficients fixed once trained | SHAP values shift with model updates unless version-controlled |

**Practical read:** the gap between XGBoost and Logistic Regression is 0.0252 Gini — larger than the previous run's 0.0087, but still modest. XGBoost is the stronger candidate for internal risk-ranking and challenger comparison, provided SHAP (Decision 7) is built before customer-facing use. Logistic Regression remains the faster-to-defend choice for the actual approve/decline decision until that layer exists.

---

## Fairness at the Business Cutoff — XGBoost

| Segment | n | Actual default rate | Avg predicted PD | FNR | FPR |
|---|---|---|---|---|---|
| `thin_file_flag=0` | 2,567 | 1.95% | 1.95% | 100.00% | 0.36% |
| `thin_file_flag=1` | 1,172 | 9.30% | 9.29% | 33.03% | 22.48% |
| `is_vulnerable=0` | 1,552 | 3.22% | 3.96% | 54.00% | 6.32% |
| `is_vulnerable=1` | 2,187 | 4.98% | 4.45% | 54.13% | 7.36% |

Average predicted PD tracks actual default rate closely within each `thin_file_flag` segment (1.95% vs 1.95%; 9.29% vs 9.30%) — the model is well calibrated per segment. The disparity is a property of applying one global threshold to segments with genuinely different score distributions, confirmed by experiment: removing `thin_file_flag` as a model input produced an identical gap. Segment-aware thresholding was considered and deliberately not implemented — a governance and Consumer Duty policy decision, not a modelling one. See Decision 12 addendum.

`is_vulnerable` (never a model input, comparison label only) shows near-identical FNR between segments, in contrast to the thin-file gap.

---

## What's Not Yet Built

| Item | Status |
|---|---|
| SHAP explainability (Decision 7) | Not built — prerequisite for XGBoost in customer-facing decisions |
| Feature Set A/B sensitivity comparison | Not run — required before `MODEL_FEATURES` is finalised |
| Rate-normalised exposure features | Not built — the production-realistic fix for observation-time contamination |
| Temporal split | Currently unusable (degenerate single-class test set); fixed-observation-horizon redesign is future work |
| MLflow champion-challenger promotion + human approval gate (Decision 8/9) | Logging exists; promotion workflow does not |
| Drift monitoring (Decision 5) | Not built |
| Segment-aware thresholding | Considered, deliberately not built — governance decision |

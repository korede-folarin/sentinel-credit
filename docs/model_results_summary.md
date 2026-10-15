# Sentinel Credit — Model Results Summary

**Champion:** XGBoost **v1.1**, Platt-calibrated (5-fold `CalibratedClassifierCV`) · **Features:** 13
**Population:** 41,604 customers with an observed outcome · **Default rate:** 2.72% · **Right-censoring:** 0%
**Split:** stratified 33,283 train / 8,321 test, plus an out-of-time split (4,857 test)
**Decision cutoff:** probability of default ≥ **0.0657** → refer for manual review
**Updated:** 15th October 2026 · Decisions 18, 19, 27 in the [Engineering Decision Log](engineering_decision_log.md)

## Headline results (held-out)

| | Stratified | Out-of-time |
|---|---:|---:|
| Gini | 0.5618 | 0.5611 |
| AUC | 0.7809 | 0.7806 |
| KS | 0.4712 | 0.4390 |
| PR-AUC | 0.0909 | 0.0782 |
| Brier | 0.0254 | 0.0241 |
| Sent to review at cutoff | 15.9% | 11.9% |
| Defaults caught (recall) | 57.96% | 40.32% |
| Precision | 9.90% | 8.65% |
| Bad rate among those not reviewed | 1.36% | 1.73% |

The riskiest 10% of customers hold 39% of all defaults (lift 3.93×). Gini holds almost exactly
out of time (0.5618 → 0.5611).

## Three-model benchmark (stratified)

| Model | Gini | AUC | KS | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| **XGBoost (champion)** | **0.5618** | **0.7809** | 0.4712 | **0.0909** | **0.0254** |
| Decision tree | 0.5486 | 0.7743 | 0.4718 | 0.0869 | 0.0255 |
| Logistic regression | 0.4896 | 0.7448 | 0.4509 | 0.0668 | 0.0258 |

## What changed from v1, and why

v1 (August 2026) had 14 features and cutoff 0.0671 (Gini 0.5656 stratified). The October audit
found that `pct_late_payments` partly encodes the outcome: "late" also records arrears pay-offs,
and post-default instalments sit in its denominator. It was removed on validity grounds; the Gini
cost (−0.004 held-out, 95% CI including zero) is the leak disappearing. v1.1 was promoted through
an evidence-bound validity override and the cutoff recalibrated (Decision 27).

## Known limitation, stated not hidden

At the cutoff, defaulters with an established credit file (`thin_file_flag = 0`) are all scored
below it (false-negative rate 100% on both splits), while thin-file customers carry most false
positives. The model still separates risk within the established-file segment; the shared
probability scale is the problem. Segment-specific treatment is an open governance decision
(Decisions 12, 19). Explore it in the public demo's cutoff simulator.

## Where the evidence lives

- Evaluation: `reports/model_comparison_*.csv`, `reports/evaluation/<split>/xgboost/`
- Cutoff selection: `reports/threshold_selection/`
- SHAP: `reports/shap/` · Feature governance: `reports/feature_governance/`
- Removal experiment: `reports/feature_removal/pct_late_payments/`

---

## Superseded: results as of 3rd August 2026 (18-feature model, before the horizon extension)

> Kept as the historical record. These figures predate Decisions 16–19 and 27 (censored
> population, 18 features, cutoff 0.137) and do not describe any current model.

**Population:** 18,694 modelable · **Bad rate:** 4.26% (796 defaults) · **Features:** 18
**Split:** stratified (14,955 train / 3,739 test) · **Business cutoff:** 0.137
**Champion:** XGBoost (`xgboost_v1_20260803`)
**Source:** `train_baseline_models.py`, `evaluate_models.py --split stratified`, `explain_models.py`, `generate_charts.py`
**Full detail:** `docs/eda_results.md`, `docs/engineering_decision_log.md` (Decisions 13, 14)

---

## Population and Target

![EDA target summary](charts/eda_target_summary.png)

| Check | Result |
|---|---|
| Total customers | 50,000 |
| Modelable (`has_outcome=1`) | 18,694 |
| Bad rate | 4.26% (796 defaults) |
| Transactions / instalments | 58,611 / 172,968 |
| Missing values in training population | 0 |
| Final feature count | 18 |

---

## Model Comparison

![Model comparison](charts/model_comparison_stratified.png)

| Model | Gini | AUC-ROC | KS | PR-AUC | ECE | Brier |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.6550** | **0.8275** | **0.5178** | **0.1956** | **0.0026** | **0.0369** |
| Logistic Regression | 0.6237 | 0.8118 | 0.4876 | 0.1933 | 0.0037 | 0.0373 |
| Decision Tree | 0.6146 | 0.8073 | 0.4921 | 0.1496 | 0.0098 | 0.0380 |

All three clear the Gini 0.40 benchmark comfortably. XGBoost's KS of 0.5178 is strong by conventional credit scoring standards. Calibration is excellent across all three (ECE under 0.01), following the Platt-scaling fix that reduced ECE from 0.3552.

**Best hyperparameters (5-fold CV, `average_precision` scoring):**

| Model | Best hyperparameters |
|---|---|
| Logistic Regression | C=1.0, class_weight=balanced, penalty=l2 |
| Decision Tree | max_depth=6, min_samples_leaf=50, class_weight=balanced |
| XGBoost | max_depth=3, n_estimators=100, learning_rate=0.05, scale_pos_weight=22.48 |

### Effect of the feature-validity exclusions

Removing the four exposure-time features (22 → 18) improved the champion:

| Metric | 22 features | 18 features |
|---|---|---|
| Gini | 0.6493 | **0.6550** |
| AUC | 0.8247 | **0.8275** |
| KS | 0.5061 | **0.5178** |
| PR-AUC | 0.2052 | 0.1956 |
| Highest VIF | 32.23 | **18.24** |

The small PR-AUC reduction was the anticipated, accepted cost — the repeated-CV ablation predicted it (Set B won PR-AUC in 13 of 15 folds). Gini, AUC and KS all improved in the full pipeline. See Decision 14.

---

## Lift by Risk Decile

![Lift comparison](charts/lift_comparison_stratified.png)

| Decile | XGBoost | Logistic Regression | Decision Tree | Random |
|---|---|---|---|---|
| 0 (highest risk) | **5.16** | 4.72 | 4.53 | 1.00 |
| 1 | 1.13 | 1.51 | 1.95 | 1.00 |
| 2 | 1.45 | 1.32 | 0.94 | 1.00 |
| 3 | 0.82 | 0.88 | 0.82 | 1.00 |
| 4 | 0.50 | 0.57 | 0.82 | 1.00 |
| 5 | 0.50 | 0.38 | 0.19 | 1.00 |
| 6 | 0.19 | 0.25 | 0.31 | 1.00 |
| 7 | 0.00 | 0.13 | 0.31 | 1.00 |
| 8 | 0.06 | 0.13 | 0.13 | 1.00 |
| 9 (lowest risk) | 0.19 | 0.13 | 0.00 | 1.00 |

Reviewing the top 10% highest-risk customers under XGBoost captures **51.6% of all defaults** at 5.16x the rate of random selection.

---

## XGBoost — Calibration

| Decile | Predicted PD | Actual default rate | A/E ratio |
|---|---|---|---|
| 0 | 0.56% | 0.80% | 1.432 |
| 1 | 0.66% | 0.27% | 0.404 |
| 2 | 0.79% | 0.00% | 0.000 |
| 3 | 0.99% | 0.80% | 0.808 |
| 4 | 1.33% | 2.14% | 1.611 |
| 5 | 1.90% | 2.14% | 1.129 |
| 6 | 2.83% | 3.48% | 1.228 |
| 7 | 4.68% | 6.15% | 1.314 |
| 8 | 9.15% | 4.81% | 0.526 |
| 9 | 19.71% | 21.93% | 1.112 |

The top decile — where risk concentration matters most — sits at A/E 1.11. Mid-range deciles swing more, but on small absolute counts (single-digit defaults per decile).

---

## Performance at the Business Cutoff (0.137)

| Model | TP | FP | FN | TN | Precision | Recall | Flagged | Bad rate among approved |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | 76 | 255 | 83 | 3,325 | **22.96%** | **47.80%** | 331 | **2.44%** |
| Logistic Regression | 55 | 186 | 104 | 3,394 | 22.82% | 34.59% | 241 | 2.97% |
| Decision Tree | 40 | 166 | 119 | 3,414 | 19.42% | 25.16% | 206 | 3.37% |

*Confusion matrices shown at the previous 0.145 cutoff; recalibrated to XGBoost's F1-optimal 0.1367 in config.*

Precision of ~23% against a 4.26% base rate is roughly 5x better than random. Low absolute precision is structurally expected in rare-event classification and is judged against the base rate, not against 100%.

**On the cutoff:** `business_cutoff` is a configurable policy parameter belonging to the decision engine, not a property of any algorithm. It is calibrated against the champion (XGBoost F1-optimal). Champion-challenger promotion requires threshold recalibration rather than assuming transferability. See Decision 13.

---

## Explainability (SHAP)

Validated before use: additivity max absolute error 1.669e-06 against a 1e-4 tolerance, feature list matching both the saved model and current code, zero nulls, no excluded fields.

**Top features by mean |SHAP|:**

| Rank | Feature | % of total |
|---|---|---|
| 1 | `bnpl_exposure_to_monthly_income_ratio` | 15.12% |
| 2 | `thin_file_flag` | 14.53% |
| 3 | `avg_n_instalments` | 10.57% |
| 4 | `avg_order_value_gbp` | 9.76% |
| 5 | `n_applications` | 9.24% |
| 6 | `savings_balance_gbp` | 9.21% |

Importance is now spread far more evenly than on the previous feature set, where `total_order_value_gbp` alone accounted for 20.83% and the exposure-time family for 32.1% combined. That family now contributes 0%.

**Scope:** SHAP explains `champion_explanation_base.joblib` — the uncalibrated model fitted in the same training run — not the calibrated probability that drives the decision. Global rankings stay directionally consistent (calibration is monotonic); local magnitudes must not be described as changes to the calibrated PD. Valid for importance ranking, dependence inspection, and error-case comparison. Not valid for customer-facing adverse-action wording or claims about calibrated probability changes. See Decision 14.

---

## Model Selection: Performance vs Explainability

XGBoost leads on every discrimination metric. That does not automatically make it the deployed model.

| Requirement | Logistic Regression | XGBoost |
|---|---|---|
| Coefficient-level interpretability | Direct | None — requires SHAP |
| Adverse action explanation | Generated from the model | Generated from an approximation |
| Regulatory precedent | Long-standing (PRA SS1/23, SR 11-7) | Accepted, but requires the SHAP layer validated separately |
| Explanation stability | Coefficients fixed once trained | SHAP values shift with model updates unless version-controlled |

The gap has widened to 0.0313 Gini (0.6550 vs 0.6237), larger than the 0.0087 on the previous feature set. With the SHAP foundation now built and validated, XGBoost is a materially stronger candidate than before — but the audience-specific explanation layer (customer / risk officer / regulatory) remains unbuilt, so Logistic Regression stays the faster-to-defend choice for customer-facing decisions.

---

## Fairness at the Business Cutoff — XGBoost

| Segment | n | Actual default rate | Avg predicted PD | FNR | FPR |
|---|---|---|---|---|---|
| `thin_file_flag=0` | 2,567 | 1.95% | 1.87% | 100.00% | 0.04% |
| `thin_file_flag=1` | 1,172 | 9.30% | 9.50% | 30.28% | 23.89% |
| `is_vulnerable=0` | 1,552 | 3.22% | 4.00% | 50.00% | 6.46% |
| `is_vulnerable=1` | 2,187 | 4.98% | 4.44% | 53.21% | 7.60% |

Predicted PD tracks actual default rate closely within each segment — the model is well calibrated per segment. The disparity comes from applying one global threshold to segments with genuinely different score distributions, confirmed by experiment (removing `thin_file_flag` as an input produced an identical gap). Segment-aware thresholding was considered and deliberately not implemented — a governance and Consumer Duty policy decision, not a modelling one. See Decision 12 addendum.

---

## What's Not Yet Built

> **Historical (3rd August 2026).** Most of these have since been built: champion/challenger promotion and the approval gate (Decisions 21, 23), drift monitoring (Decisions 20, 24, 25) and the audience-specific explanation layer: plain-English reasons, error-type checks, risk officer flag and decision record (Decisions 30, 31). See the progress table in the README for current status.

| Item | Status |
|---|---|
| Audience-specific explanation layer (Decision 7) | SHAP foundation built; customer / risk officer / regulatory narratives not started |
| Rate-normalised exposure features | The production-realistic replacement for the excluded features — not built |
| Temporal split | Unusable at current date; fixed-observation-horizon redesign is future work |
| MLflow champion-challenger promotion + human approval gate (Decision 8/9) | Logging and model persistence exist; promotion workflow does not |
| Drift monitoring (Decision 5) | Not built |
| Segment-aware thresholding | Considered, deliberately not built — governance decision |

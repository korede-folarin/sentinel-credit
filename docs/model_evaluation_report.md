# Sentinel Credit - Baseline Model Evaluation Report

> **Current champion: v1.1 (14th October 2026).** This report documents the **August 2026 v1** champion (14 features, cutoff 0.0671) and is kept as the historical record of it. v1.1 removed `pct_late_payments` for leaking outcome information and is otherwise the same model (identical hyperparameters, split and calibration). See [Decision 27](engineering_decision_log.md). Live v1.1 evidence is regenerated under `reports/`; v1's is archived in `reports/_archive_v1_14f_2026-08/`.
>
> v1.1 held-out results at cutoff 0.0657 (stratified / temporal): Gini 0.5618 / 0.5611, AUC 0.7809 / 0.7806, KS 0.4712 / 0.4390, PR-AUC 0.0909 / 0.0782, Brier 0.0254 / 0.0241. Flagged 15.9% / 11.9%, recall 57.96% / 40.32%, precision 9.90% / 8.65%. Top-decile lift 3.93 (stratified). Established-file (thin_file_flag=0) defaulters are still 100% missed on both splits (Decision 19 item, open).


**Report status:** Historical: describes the v1 champion, superseded by v1.1 on 14 October 2026 (see the box above).
**Dataset version:** `full_horizon_2025_v1`
**Feature-set label:** `14_features_v1_frozen`
**Model version:** `v1`
**Evaluation date:** 6 August 2026
**Decision log version:** 3.19
**Primary script:** `models/evaluate_models.py`

---

## 1. Purpose

This report evaluates the three baseline models retrained on the frozen 14-feature v1 set (Decision 18):

```text
Logistic Regression
Decision Tree
XGBoost
```

It uses the exact saved test predictions from the retrained models and attaches detailed evidence to the matching MLflow runs.

This report does not approve a business cutoff.

---

## 2. Evaluation populations

| Split | Train population | Test population | Train default rate | Test default rate |
|---|---:|---:|---:|---:|
| Stratified | 33,283 | 8,321 | 2.72% | 2.72% |
| Temporal, cutoff 2024-09-01 | 36,747 | 4,857 | 2.74% | 2.55% |

Unchanged from the prior 18-feature evaluation — the feature-set freeze removed columns, not rows or splits.

---

## 3. Metric framework

Unchanged from the prior report.

| Metric | Interpretation |
|---|---|
| AUC | Probability that a randomly selected defaulter is ranked above a randomly selected non-defaulter. |
| Gini | `2 × AUC - 1`; a credit-risk convention for discrimination. |
| KS | Maximum separation between cumulative score distributions for defaulters and non-defaulters. |
| PR-AUC | Precision-recall performance, emphasising the rare default class. |
| ECE | Difference between predicted probability and observed frequency across probability bins. Lower is better. |
| Brier score | Mean squared probability error. Lower is better. |
| Lift | Default concentration in a risk band relative to the portfolio default rate. |
| Capture rate | Cumulative share of all defaults contained in the highest-risk population. |

**None of the metrics in Sections 4–6 depend on `business_cutoff`.** They are rank- or decile-based and are already final for this feature set. Only Sections 7 and 8's cutoff-dependent rows will change once a threshold is approved.

---

## 4. Full model comparison

### 4.1 Stratified holdout

| Model | Gini | AUC | KS | PR-AUC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4898 | 0.7449 | 0.4414 | 0.0684 | 0.0009 | 0.0258 |
| Decision Tree | 0.5140 | 0.7570 | 0.4401 | 0.0864 | **0.0003** | 0.0256 |
| XGBoost | **0.5656** | **0.7828** | **0.4686** | **0.0897** | 0.0008 | **0.0254** |

XGBoost now leads on every metric except ECE, where the Decision Tree is marginally lower. XGBoost's lead over both challengers widened substantially relative to the 18-feature baseline, where the three models were closer together (see Section 9).

### 4.2 Temporal holdout

| Model | Gini | AUC | KS | PR-AUC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5018 | 0.7509 | **0.4554** | 0.0677 | 0.0027 | 0.0244 |
| Decision Tree | 0.5059 | 0.7530 | 0.4284 | 0.0682 | **0.0009** | 0.0242 |
| XGBoost | **0.5675** | **0.7837** | 0.4332 | **0.0876** | 0.0024 | **0.0241** |

XGBoost again leads on discrimination and rare-event ranking. Logistic Regression retains the highest temporal KS, as it did under the 18-feature set, though its overall discrimination is now well behind XGBoost's.

---

## 5. Lift and capture

### 5.1 Highest-risk decile

| Split | Model | Top-decile lift | Defaults captured in top 10% |
|---|---|---:|---:|
| Stratified | XGBoost | **4.02** | **40.27%** |
| Temporal | XGBoost | **3.71** | **37.10%** |

Top-decile concentration is somewhat lower than the 18-feature baseline (stratified 4.24 → 4.02; temporal 4.35 → 3.71) — consistent with the removal of features that carried real, if exposure-time-confounded, signal. This is the expected and accepted cost of the Decision 18 removals, not a modelling regression.

### 5.2 XGBoost cumulative capture

Stratified holdout:

| Reviewed population | Cumulative defaults captured |
|---:|---:|
| Top 10% | 40.27% |
| Top 20% | 65.04% |
| Top 30% | 74.78% |
| Top 40% | 79.65% |
| Top 50% | 86.28% |

Temporal holdout:

| Reviewed population | Cumulative defaults captured |
|---:|---:|
| Top 10% | 37.10% |
| Top 20% | 59.68% |
| Top 30% | 69.35% |
| Top 40% | 79.84% |
| Top 50% | 85.48% |

---

## 6. Calibration

| Split | ECE | Brier |
|---|---:|---:|
| Stratified | 0.0008 | 0.0254 |
| Temporal | 0.0024 | 0.0241 |

Portfolio-level calibration remains strong and is essentially unchanged from the 18-feature baseline (stratified ECE 0.0017 → 0.0008; temporal 0.0038 → 0.0024, both improved slightly). As before, decile-level actual-to-expected ratios are noisier in the smaller temporal test population.

---

## 7. Threshold assessment

**Resolved via Decision 19 — confirmed on both splits.**

### 7.1 Confirmed cutoff

`business_cutoff` is now:

```text
0.0671
```

Selected from training-only out-of-fold predictions under a 15% review-capacity constraint (Decision 19; see `business_cutoff_selection_report.md`). Observed performance at this cutoff:

| Split | Model | TP | FP | FN | TN | Precision | Recall | Flagged | Flagged rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stratified | Logistic Regression | 60 | 744 | 166 | 7,351 | 7.46% | 26.55% | 804 | 9.66% |
| Stratified | Decision Tree | 74 | 636 | 152 | 7,459 | 10.42% | 32.74% | 710 | 8.53% |
| Stratified | XGBoost | 124 | 1,167 | 102 | 6,928 | 9.60% | 54.87% | 1,291 | 15.51% |
| Temporal | Logistic Regression | 35 | 447 | 89 | 4,286 | 7.26% | 28.23% | 482 | 9.92% |
| Temporal | Decision Tree | 50 | 528 | 74 | 4,205 | 8.65% | 40.32% | 578 | 11.90% |
| Temporal | XGBoost | 48 | 486 | 76 | 4,247 | 8.99% | 38.71% | 534 | 10.99% |

XGBoost's stratified recall (54.87%) and flagged rate (15.51%) land close to the OOF selection evidence (51.55% recall, 15.00% flagged) — stratified is drawn from the same population distribution the cutoff was selected against, so this is the expected result. Temporal's lower recall (38.71%) reflects a genuinely out-of-time population with a different score distribution, not a failure of the selection.

### 7.2 Diagnostic thresholds (superseded)

The F1/Youden diagnostics previously shown here were computed on the evaluation set itself and were always explicitly diagnostic-only, never eligible for direct promotion to `config.yaml`. They are superseded by Decision 19's training-only OOF candidates — see `business_cutoff_selection_report.md` Section 5 for the full three-way comparison (F1-optimal, Youden's J, review-capacity-constrained) `business_cutoff` was actually selected from.

---

## 8. Segment-level performance

### 8.1 XGBoost at confirmed cutoff 0.0671, both splits

| Split | Segment | n | Actual default rate | Average predicted PD | FNR | FPR |
|---|---|---:|---:|---:|---:|---:|
| Stratified | Thin-file = 0 | 5,648 | 1.08% | 1.11% | **100.00%** | 0.00% |
| Stratified | Thin-file = 1 | 2,673 | 6.17% | 6.31% | 24.85% | 46.53% |
| Stratified | Vulnerable = 0 | 3,494 | 2.09% | 2.69% | 43.84% | 13.65% |
| Stratified | Vulnerable = 1 | 4,827 | 3.17% | 2.85% | 45.75% | 14.98% |
| Temporal | Thin-file = 0 | 3,294 | 1.03% | 1.00% | **100.00%** | 0.03% |
| Temporal | Thin-file = 1 | 1,563 | 5.76% | 5.13% | 46.67% | 32.93% |
| Temporal | Vulnerable = 0 | 2,066 | 2.13% | 2.35% | 61.36% | 10.98% |
| Temporal | Vulnerable = 1 | 2,791 | 2.87% | 2.30% | 61.25% | 9.74% |

**Confirmed stable on both splits, not a temporal-only artifact:** `thin_file_flag=0`'s FNR is 100.00% on stratified as well as temporal — every defaulted customer in that segment is missed regardless of split, while `thin_file_flag=1` carries the model's recall alongside a substantial false-positive rate (46.53% stratified, 32.93% temporal).

Three targeted checks (`models/segment_threshold_diagnostics.py`, run on both splits; full detail in `business_cutoff_selection_report.md` Section 8 and Decision 19's addendum) established this is a **policy-scale problem, not a model-signal failure**: `thin_file_flag=0` defaulters' calibrated scores sit entirely below any threshold on the table on both splits (max 0.0344 stratified / 0.0367 temporal, 0% above cutoff either way — a hard ceiling, not borderline cases); no single global threshold recovers their recall without collapsing `thin_file_flag=1`'s FPR past 75%; and within-segment AUC (0.6409 stratified [95% CI 0.5661-0.7220], 0.6885 temporal [95% CI 0.6148-0.7613]) shows the model still discriminates risk within the segment on both splits — it is the *absolute* probability scale, compressed by the segment's ~1% base rate under one shared calibration, that suppresses recall to zero in practice. This echoes the earlier finding that `thin_file_flag` removal didn't close the same gap because `credit_score` reconstructs it (r=-0.75).

This does not license segment-specific thresholds, and ranking surviving within the segment doesn't reduce it to "just calibration" — both remain governance questions under Decision 12, opened but not resolved by this evaluation.

---

## 9. Champion conclusion

XGBoost is the confirmed champion on the frozen feature set, more decisively than under the 18-feature baseline:

- highest Gini, AUC, PR-AUC, and Brier on both holdouts;
- best temporal stability (Gini gap -0.0019, versus -0.0257 under 18 features);
- Logistic Regression and Decision Tree both lost more discrimination than XGBoost when the four Decision 18 features were removed, widening XGBoost's lead rather than narrowing it.

Logistic Regression retains the highest KS on both splits, as before, but is no longer close to XGBoost on overall discrimination — its standing as a credible challenger is weaker on this feature set than it was pre-governance.

---

## 10. Evaluation limitations

1. **A `thin_file_flag` operating-policy item is open, not resolved** — Section 8.1, confirmed stable on both splits. Ranking survives within the segment; the shared probability scale does not carry that ranking through to positive decisions. Requires Decision 12 governance sign-off, not a modelling fix.
2. Results come from synthetic UK BNPL-like data rather than a live lender portfolio.
3. No external out-of-sample lender dataset has been used for model validation.
4. Calibration by decile remains noisy where event counts are small, particularly within `thin_file_flag=0` (61 stratified / 34 temporal defaulted customers).
5. Application-time underwriting safety was not assessed — out of scope for v1 (Decision 18).

---

## 11. Evidence locations

Summary outputs:

```text
reports/model_comparison_stratified.csv
reports/model_comparison_temporal.csv
reports/lift_tables_stratified.csv
reports/lift_tables_temporal.csv
```

Detailed model evidence:

```text
reports/evaluation/stratified/<model>/
reports/evaluation/temporal/<model>/
```

Evaluation artifacts are attached to their exact MLflow training runs (see the training summary, Section 8, for run IDs).

---

## 12. Status statement

Evaluation of the frozen 14-feature v1 champion is complete on both splits. `business_cutoff` is confirmed at 0.0671 (Decision 19) — held-out results on stratified and temporal are both consistent with the training-only OOF selection evidence. XGBoost remains the clear champion. One item remains open, not blocking the champion or cutoff conclusions: a Decision 12 governance decision on segment-aware review capacity for `thin_file_flag`, whose ranking survives within-segment on both splits but whose recall is structurally suppressed to zero by the shared probability scale at any global cutoff.

---

## Metric Definitions

> Moved verbatim from the Engineering Decision Log (Appendix C) on 15th October 2026, when reference material was separated from the decision record. Content is unchanged; it describes the design as documented at the time.

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

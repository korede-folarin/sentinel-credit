# Sentinel Credit - SHAP Explainability Report

> **Current champion: v1.1 (14th October 2026).** This report documents the **August 2026 v1** champion (14 features, cutoff 0.0671) and is kept as the historical record of it. v1.1 removed `pct_late_payments` for leaking outcome information and is otherwise the same model (identical hyperparameters, split and calibration). See [Decision 27](engineering_decision_log.md). Live v1.1 evidence is regenerated under `reports/`; v1's is archived in `reports/_archive_v1_14f_2026-08/`.
>
> v1.1 SHAP (additivity max error 1.8e-06): thin_file_flag 29.0%, savings_balance_gbp 18.4%, avg_n_instalments 12.1%, avg_order_value_gbp 11.3%, pct_high_risk_category 10.4%, pct_medium_risk_category 6.6%, max_affordability_ratio 3.0%, credit_score 3.0%, then under 2% each; existing_monthly_obligations_gbp 0.33%. Global importance and local cases are on the dashboard's Explainability page.


**Report status:** Historical: describes the v1 champion, superseded by v1.1 on 14 October 2026 (see the box above).
**Dataset version:** `full_horizon_2025_v1`
**Feature-set label:** `14_features_v1_frozen`
**Model version:** `v1`
**Explanation date:** 7 August 2026
**Decision log version:** 3.22
**Primary script:** `models/explain_models.py`

---

## 1. Purpose and scope

This report explains the frozen 14-feature XGBoost champion's risk-ranking
logic using SHAP (SHapley Additive exPlanations), computed on the exact
uncalibrated explanation artifact persisted alongside the calibrated
production model.

**What this is:** global feature-importance ranking, feature-direction and
interaction inspection, and representative case studies against the raw
risk margin the tree model produces before calibration.

**What this is not:** customer-facing adverse-action wording, a claim
about changes in calibrated probability, or a formal end-to-end
reconstruction of the calibrated decision. SHAP values describe movement
in the model's raw margin; calibration is monotonic, so global rankings
transfer, but magnitudes do not directly translate to probability points.

---

## 2. Validation

| Check | Result |
|---|---|
| Feature names match saved champion package | Pass |
| Feature list matches current `MODEL_FEATURES` | Pass |
| No excluded/leaked features present | Pass |
| Nulls in explained data | 0 |
| Additivity (max absolute error) | 2.384e-06 (tolerance 1e-04) |
| Additivity (mean absolute error) | 3.515e-07 |
| All validation checks pass | **True** |

8,321 stratified test customers explained with all 14 frozen features.
Additivity error is five orders of magnitude below tolerance — SHAP
values reconstruct the model's raw margin essentially exactly, confirming
the explanation is faithful to what the tree actually computed rather
than an approximation.

---

## 3. Global feature importance

| Rank | Feature | Mean \|SHAP\| | % of total |
|---:|---|---:|---:|
| 1 | `thin_file_flag` | 0.4575 | 28.55% |
| 2 | `savings_balance_gbp` | 0.3016 | 18.82% |
| 3 | `avg_n_instalments` | 0.1827 | 11.40% |
| 4 | `avg_order_value_gbp` | 0.1605 | 10.01% |
| 5 | `pct_high_risk_category` | 0.1512 | 9.43% |
| 6 | `pct_medium_risk_category` | 0.0873 | 5.44% |
| 7 | `pct_late_payments` | 0.0866 | 5.40% |
| 8 | `credit_score` | 0.0472 | 2.95% |
| 9 | `max_affordability_ratio` | 0.0375 | 2.34% |
| 10 | `months_credit_history` | 0.0261 | 1.63% |
| 11 | `num_defaults_24m` | 0.0259 | 1.62% |
| 12 | `monthly_income_gbp` | 0.0196 | 1.22% |
| 13 | `num_credit_accounts` | 0.0180 | 1.12% |
| 14 | `existing_monthly_obligations_gbp` | 0.0011 | 0.07% |

The top two features alone account for 47.4% of total attribution.
`thin_file_flag` is the single largest driver of the model's risk
ranking — a fact directly relevant to the `thin_file_flag` operating-policy
finding documented in Decision 19: the model does not merely correlate
with this segmentation, it is organized around it more than around any
other single input. `existing_monthly_obligations_gbp` sits at
essentially zero importance (0.07%) — retained per Decision 17/18's
evidence-based-retention standard, but a candidate for future governance
attention if the feature set is revisited.

---

## 4. Dependence plots — patterns and interactions

Six features were plotted: the four highest-ranked by SHAP
(`thin_file_flag`, `savings_balance_gbp`, `avg_n_instalments`,
`avg_order_value_gbp`) plus the two configured strategic features
(`credit_score`, `max_affordability_ratio`).

### 4.1 `thin_file_flag` — clean bimodal separation

The cleanest pattern of the six: `thin_file_flag=0` clusters tightly
between -0.35 and -0.40 (risk-reducing); `thin_file_flag=1` clusters
between +0.20 and +0.55 (risk-increasing), with no overlap between the
two clusters. Coloring by `avg_order_value_gbp` shows no gradient within
either cluster — the effect of `thin_file_flag` is essentially constant
regardless of order value. This is a feature the model treats as a
near-binary switch, not a continuum, which is consistent with it being
the top-ranked feature by a wide margin.

### 4.2 `savings_balance_gbp` — strong monotonic threshold

Near-zero savings carries a strong positive (risk-increasing)
contribution, up to +1.0 at the extreme. The effect falls sharply and
monotonically as savings rise, crossing to protective (negative) by
roughly £700–800, and plateauing around -0.3 to -0.5 beyond £3,000–4,000.
Coloring by `avg_order_value_gbp` shows no clear interaction — the
savings effect is stable across order-value levels.

### 4.3 `avg_order_value_gbp` — sharp threshold, with a `thin_file_flag` interaction

A pronounced cliff: below roughly £250–300, strongly risk-increasing
(down to -1.15 in the plot's sign convention, i.e. this is the SHAP
value *for* `avg_order_value_gbp`, meaning low order value drives risk
up sharply). Above ~£500, the effect plateaus near zero to slightly
protective. **The interaction with `thin_file_flag` is visible and
worth noting**: in the low-order-value region, `thin_file_flag=1`
(pink) points sit consistently lower (more risk-increasing) than
`thin_file_flag=0` (blue) points at the same order value — low order
value is a worse signal specifically for thin-file customers. Above
~£750, the pattern partially inverts: `thin_file_flag=1` sits at the
upper (more protective) edge of the band while `thin_file_flag=0`
spreads wider, including some higher-order-value points that remain
negative-to-flat.

### 4.4 `avg_n_instalments` — threshold with a `thin_file_flag` interaction in the plateau

Sharp rise from strongly negative (-0.30 at 3 instalments, the
minimum) to positive by 4–5 instalments, then a wide, noisy plateau
extending to 12 instalments. **In the plateau region, `thin_file_flag=0`
customers consistently reach higher SHAP values (up to +0.40) than
`thin_file_flag=1` customers at the same instalment count (typically
capped near +0.20)** — more instalments is a stronger positive signal
for non-thin-file customers than for thin-file customers, even though
both groups benefit directionally.

### 4.5 `credit_score` — threshold with a `pct_late_payments` interaction

A clear step function centered around 550–650: below this range,
positive (risk-increasing) contribution up to +0.08; above it, negative
(protective) down to -0.15. **The interaction with `pct_late_payments`
is the most striking pattern among the six plots**: within any
credit-score band, customers with high `pct_late_payments` (pink, ~0.20–0.25)
sit near zero regardless of credit score, while customers with low
`pct_late_payments` (blue) show the full range of the credit-score
effect in both directions. In other words, **`credit_score`'s
contribution is substantially dampened once `pct_late_payments` is
high** — a customer with a poor recent payment record gets little
additional protective credit for a high bureau score, and little
additional penalty for a low one. This is a genuine, non-additive
interaction the global importance ranking alone would not surface.

### 4.6 `max_affordability_ratio` — threshold with a `thin_file_flag` interaction at the low end

Below ~0.30–0.35, negative (protective) contribution; above that,
a positive plateau around +0.02 to +0.06. **In the protective region,
`thin_file_flag=1` customers reach substantially more negative values
(down to -0.47) than `thin_file_flag=0` customers (capped near -0.15)**
at the same affordability ratio — a low peak financial-stretch ratio is
a stronger protective signal for thin-file customers than for
non-thin-file customers.

### 4.7 Reading across the six plots

Four of the six dependence plots (4.3, 4.4, 4.5, 4.6) show a visible
interaction with either `thin_file_flag` or `pct_late_payments`, not a
clean independent marginal effect. This is consistent with, and adds
detail to, the Decision 19 finding: the model's behavior is not simply
"the same relationship, offset by segment" — several individual
features change *how strongly* they discriminate depending on segment
membership, which is part of why a single global threshold interacts
awkwardly with the `thin_file_flag=0` population (Decision 19's
operating-policy item).

---

## 5. Local case studies

Six representative customers, selected under the current cutoff
(0.0671) and therefore partially provisional where marked.

| Case | Customer | Calibrated PD | Decision | Actual | Dominant driver(s) |
|---|---|---:|---|---:|---|
| Low risk | `CFD4B625B4EA9FC1E` | 0.0039 | pass | 0 | `avg_order_value_gbp` (-0.56), `thin_file_flag=0` (-0.36) |
| High risk | `CA1B4465238F53DE5` | 0.1284 | flag_high_risk | 0 | `savings_balance_gbp` (+0.46), `thin_file_flag=1` (+0.42) |
| Near threshold | `C6ACB96E5469A06CF` | 0.0671 | pass | 0 | `thin_file_flag=1` (+0.48), `savings_balance_gbp` (+0.46), `pct_high_risk_category=0` (-0.22) |
| False positive | `CA1B4465238F53DE5` | 0.1284 | flag_high_risk | 0 | (same as high-risk case — same customer) |
| **False negative** | `C8E270BA275EBC784` | **0.0042** | **pass** | **1** | Every feature contributes negatively; largest: `pct_late_payments` (-0.48), `avg_order_value_gbp` (-0.46), `savings_balance_gbp` (-0.38) |
| Single-feature dominant | `C3EE432C84CC72561` | 0.0069 | pass | 0 | `avg_order_value_gbp` (-1.16), dominating all 13 other features combined |

**The false-negative case is the most operationally important of the
six.** Customer `C8E270BA275EBC784` is `thin_file_flag=0`, genuinely
defaulted, and every one of the 14 features pushed the model *toward*
"pass" — nothing in this customer's profile was flagged as risky by any
individual feature, including `thin_file_flag` itself (-0.29, i.e.
being non-thin-file was read as protective here). This is a concrete,
named instance of the pattern `segment_threshold_diagnostics.py`
established statistically: `thin_file_flag=0` defaulters' scores sit
entirely below the operating range, not just near it. Nothing in this
single case contradicts the Decision 19 conclusion that this is a
policy-scale problem rather than a ranking failure — the customer's
features are genuinely unremarkable by the model's own logic, which is
exactly what "the segment's base rate compresses even higher-risk
members toward zero" looks like at the individual level.

---

## 6. Scope reminders

Carried forward from `shap_metadata.json`, restated here for anyone
reading this report without the underlying JSON:

**Valid uses:** global feature-importance ranking; feature-direction and
dependence inspection; identifying suspicious model dependencies;
comparing low-risk, high-risk, and error cases.

**Not valid for:** customer-facing adverse-action wording; claims about
calibrated-probability changes; formal reconstruction of the calibrated
decision; regulatory output presented as an exact end-to-end explanation.

---

## 7. Status statement

SHAP explainability is complete and validated for the frozen 14-feature
v1 champion. Global importance confirms `thin_file_flag` as the
dominant driver of the model's risk ranking, corroborating the Decision
19 segment finding independently of the threshold diagnostics that
first surfaced it. Four of six inspected dependence relationships show
a genuine interaction with `thin_file_flag` or `pct_late_payments`
rather than a clean marginal effect — evidence relevant to, but not by
itself resolving, the open Decision 12 governance item on segment-aware
review capacity. No new modelling action is implied by this report; it
documents and corroborates what Decision 19 already established.

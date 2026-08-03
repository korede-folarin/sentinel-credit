# Sentinel Credit — Pre-Modelling EDA Results

**Population:** 50,000 customers · 18,694 modelable (`has_outcome = 1`)
**Bad rate:** 4.26% (796 defaults)
**Scale:** 58,611 transactions · 172,968 repayment instalments
**Feature set:** 18 (final — see Decision 14)
**Script:** `EDA/eda_baseline.py` · **Chart:** `reports/charts/eda_target_summary.png`

---

## 1. Target imbalance

| | Count | Rate |
|---|---|---|
| Total customers | 50,000 | — |
| Modelable (`has_outcome = 1`) | 18,694 | 37.4% of total |
| Defaults | 796 | 4.26% of modelable |

| Segment | Bad rate | n |
|---|---|---|
| `thin_file_flag = 0` | 1.85% | 12,735 |
| `thin_file_flag = 1` | 9.41% | 5,959 |
| `is_vulnerable = 0` | 3.38% | 8,000 |
| `is_vulnerable = 1` | 4.92% | 10,694 |
| `ab_test_group = A_strict` | 2.89% | 8,623 |
| `ab_test_group = B_relaxed` | 5.43% | 10,071 |

The thin-file gap (roughly 5x) and the A/B split confirm the generator's calibration propagated correctly. Class weighting (mandatory) and Platt calibration are both required downstream because of the ~96:4 imbalance.

**Statistical adequacy:** 796 defaults across 18 features gives roughly 44 events per variable, comfortably above the conventional 10-per-variable minimum (Peduzzi et al.).

---

## 2. Missingness

16.79% of the full 50,000-row table is missing in exposure-derived fields (`avg_order_value_gbp`, `avg_n_instalments`, the risk-category percentages, `pct_late_payments`), matching the population who never had an approved transaction. **Zero missing values in the modelable training population.**

---

## 3. Univariate screening (Information Value)

| Feature | IV |
|---|---|
| `thin_file_flag` | 0.6884 |
| `savings_balance_gbp` | 0.6538 |
| `num_credit_accounts` | 0.6227 |
| `bnpl_exposure_to_monthly_income_ratio` | 0.6224 |
| `n_applications` | 0.5829 |
| `months_credit_history` | 0.5811 |
| `credit_score` | 0.5718 |
| `avg_order_value_gbp` | 0.3691 |
| `num_defaults_24m` | 0.2951 |
| `n_late_payments` | 0.2493 |
| `pct_late_payments` | 0.2045 |
| `avg_n_instalments` | 0.1506 |
| `pct_medium_risk_category` | 0.1042 |
| `pct_high_risk_category` | 0.0863 |
| `max_affordability_ratio` | 0.0816 |
| `monthly_income_gbp` | 0.0216 |
| `age` | 0.0173 |
| `existing_monthly_obligations_gbp` | 0.0154 |

Seven features sit above the 0.5 leakage-investigation threshold, down from twelve on the previous feature set. All seven are day-zero customer attributes wired directly into the generator's `miss_hazard()` and `financial_resilience()` functions — legitimate predictive power, causally upstream of the outcome by construction, not leakage.

`thin_file_flag`'s IV was previously misreported as 0.0000 by a binning bug: `pd.qcut()` was applied unconditionally, collapsing a binary column into a single bucket. Fixed by bucketing low-cardinality features by exact value. Verified by hand against the segment bad rates — the corrected 0.6884 is mathematically right.

---

## 4. Multicollinearity

Highly correlated pairs (|r| > 0.7) reduced to three:

| Pair | r |
|---|---|
| `monthly_income_gbp` / `existing_monthly_obligations_gbp` | 0.757 |
| `credit_score` / `thin_file_flag` | −0.750 |
| `n_late_payments` / `pct_late_payments` | 0.704 |

VIF above the 10.0 threshold, down to five features:

| Feature | VIF |
|---|---|
| `credit_score` | 18.24 |
| `max_affordability_ratio` | 12.30 |
| `monthly_income_gbp` | 12.22 |
| `n_applications` | 10.34 |
| `avg_n_instalments` | 10.10 |

Highest VIF fell from 63.53 (24-feature set) to 32.23 (22) to **18.24** (18) — the exposure-time exclusions removed the bulk of the remaining multicollinearity, not just the features themselves. Remaining VIFs are handled via L2 regularisation in Logistic Regression.

---

## 5. Exclusions

38 candidate features reduced to 18. Every exclusion is documented in `models/model_features.py` with a specific reason:

| Category | Reason | Examples |
|---|---|---|
| `MULTICOLLINEARITY_EXCLUDED` | Exact identity or near-duplicate | `n_transactions` (= `n_approved`), `annual_income_gbp` (= monthly × 12), `pct_low_risk_category` (sums to 1), `total_instalment_exposure_gbp` (r=0.974) |
| `RESOLUTION_ADJACENT_EXCLUDED` | Conflates independent misses with cure-event arrears depth | `avg_days_late`, `max_days_late`, `total_days_late` |
| `EXPOSURE_TIME_EXCLUDED` | Structurally confounded with unequal observation time | `n_approved`, `days_since_last_transaction`, `total_order_value_gbp`, `transaction_span_days`, `max_order_value_gbp`, `n_merchant_categories` |
| `NON_MODEL_EXCLUDED` | Identifiers and protected/sensitive characteristics | `is_vulnerable`, `vulnerability_score`, `age_bracket`, `ab_test_group` |
| `GENERATOR_INTERNAL_EXCLUDED` | Never real-world observable | `merchant_risk_multiplier` |

The exposure-time exclusions were resolved via SHAP plus a repeated-CV ablation — see Decision 14.

---

## 6. Outliers

IQR-flagged rates consistent with genuine long-tail behaviour. `num_defaults_24m` at 23.42% is a measurement artefact (near-categorical, values 0/1/2), not real skew.

---

## 7. Split stability

Train/test bad rate difference: **0.01%** (14,955 at 4.26% / 3,739 at 4.25%).

**Temporal split is unusable** at the current date — the corrected `has_outcome` maturity requirement leaves 16 customers in the test set, all defaulters, making discrimination metrics undefined. Skipped automatically with an explanatory message. See Decision 10 addendum.

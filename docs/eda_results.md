# Sentinel Credit — Pre-Modelling EDA Results

> **Note (15th October 2026).** This EDA describes the data and the 18 candidate features screened at the time; the figures remain valid for the data. The model feature set has since narrowed to 13 (Decision 18 removed four, Decision 27 removed `pct_late_payments` for leakage). See the [Model Results Summary](model_results_summary.md).

**Population:** 50,000 customers · **41,604 modelable** (`has_outcome = 1`)
**Bad rate:** 2.72% (1,132 defaults)
**Right-censoring:** 0.00%
**Feature set:** 18
**Script:** `EDA/eda_baseline.py` · **Artifacts:** `reports/eda/` (9 files)

> **Run against the extended-horizon dataset.** `data_end_date` moved to 2025-12-31, eliminating right-censoring. Modelable population rose from 18,694 to 41,604; bad rate moved from 4.26% to 2.72%. See Decision 16.

---

## 1. Target imbalance

| | Count | Rate |
|---|---|---|
| Total customers | 50,000 | — |
| Modelable | 41,604 | 83.2% of total |
| Defaults | 1,132 | 2.72% of modelable |

| Segment | Bad rate | n |
|---|---|---|
| `thin_file_flag = 0` | 1.10% | 28,630 |
| `thin_file_flag = 1` | 6.29% | 12,974 |
| `is_vulnerable = 0` | 2.13% | 17,858 |
| `is_vulnerable = 1` | 3.17% | 23,746 |
| `ab_test_group = A_strict` | 1.99% | 19,013 |
| `ab_test_group = B_relaxed` | 3.33% | 22,591 |

The thin-file gap is now **5.7x** (6.29% vs 1.10%), sharper than the 5.1x measured under censoring — the generator's ×2.0 miss-hazard multiplier compounding across a fully observed lifetime.

**Statistical adequacy:** 1,132 defaults across 18 features gives roughly **63 events per variable**, far above the conventional 10-per-variable minimum (Peduzzi et al.) and up from 44 on the previous dataset.

---

## 2. Missingness

16.79% of the full 50,000-row table is missing in exposure-derived fields, matching the population who never had an approved transaction. **Zero missing values in the modelable training population.**

---

## 3. Univariate screening (Information Value)

| Feature | IV | Previous (censored) | Change |
|---|---|---|---|
| `savings_balance_gbp` | 0.7692 | 0.6538 | +0.115 |
| `thin_file_flag` | 0.7535 | 0.6884 | +0.065 |
| `credit_score` | 0.6294 | 0.5718 | +0.058 |
| `num_credit_accounts` | 0.6190 | 0.6227 | −0.004 |
| `months_credit_history` | 0.6075 | 0.5811 | +0.026 |
| `bnpl_exposure_to_monthly_income_ratio` | 0.2839 | 0.6224 | **−0.339** |
| `num_defaults_24m` | 0.2794 | 0.2951 | −0.016 |
| `n_applications` | 0.2344 | 0.5829 | **−0.349** |
| `avg_order_value_gbp` | 0.1414 | 0.3691 | −0.228 |
| `pct_high_risk_category` | 0.1016 | 0.0863 | +0.015 |
| `pct_medium_risk_category` | 0.0994 | 0.1042 | −0.005 |
| `n_late_payments` | 0.0743 | 0.2493 | −0.175 |
| `avg_n_instalments` | 0.0699 | 0.1506 | −0.081 |
| `pct_late_payments` | 0.0567 | 0.2045 | −0.148 |
| `max_affordability_ratio` | 0.0332 | 0.0816 | −0.048 |
| `monthly_income_gbp` | 0.0228 | 0.0216 | +0.001 |
| `age` | 0.0085 | 0.0173 | −0.009 |
| `existing_monthly_obligations_gbp` | 0.0018 | 0.0154 | −0.014 |

### The pattern: exposure-related signal collapsed, customer-attribute signal rose

This is the most informative result in the run, and it splits cleanly in two directions.

**Activity and exposure features fell sharply.** `n_applications` dropped 60% (0.5829 → 0.2344), `bnpl_exposure_to_monthly_income_ratio` dropped 54%, `avg_order_value_gbp` 62%, the late-payment features 70%+. Under censoring, more activity meant more chance of having an observable outcome, so activity-related features were inflated by a selection mechanism rather than by predictive power. With right-censoring at zero, that inflation is gone and they fall back to what they are actually worth.

**Day-zero customer attributes rose.** `savings_balance_gbp`, `thin_file_flag`, `credit_score` and `months_credit_history` all increased. These are generated before any repayment simulation and feed directly into `miss_hazard()` and `financial_resilience()` — they were always genuine, and they are now more visible relative to the noise that was crowding them.

**This retrospectively confirms the exposure-time diagnosis** made under Decision 14. Those features were not weakly predictive; they were being inflated by a selection mechanism that no longer exists. The four excluded then (`total_order_value_gbp`, `transaction_span_days`, `max_order_value_gbp`, `n_merchant_categories`) are no longer in the feature set, but the same collapse is visible in their surviving relatives.

**Five features exceed the 0.5 investigation threshold, down from seven.** All five are day-zero customer attributes wired into the hazard functions — causally upstream by construction, not leakage. No behavioural or exposure feature crosses the line anymore.

---

## 4. Multicollinearity

Correlated pairs above 0.7 — three, unchanged from the previous run:

| Pair | r |
|---|---|
| `monthly_income_gbp` / `existing_monthly_obligations_gbp` | 0.765 |
| `credit_score` / `thin_file_flag` | −0.746 |
| `n_late_payments` / `pct_late_payments` | 0.701 |

VIF above the 10.0 threshold — four features, down from five:

| Feature | VIF |
|---|---|
| `credit_score` | 18.53 |
| `monthly_income_gbp` | 13.11 |
| `max_affordability_ratio` | 12.43 |
| `n_applications` | 11.68 |

Highest VIF has fallen across successive corrections: 63.53 (24 features) → 32.23 (22) → 18.24 (18, censored) → **18.53** (18, full observation). Essentially stable at the current feature set — the remaining multicollinearity is structural (income/obligations, credit score/thin file), not artifact.

---

## 5. Weak features — an open question, not a decision

Two features now sit below the 0.02 drop threshold:

| Feature | IV |
|---|---|
| `age` | 0.0085 |
| `existing_monthly_obligations_gbp` | 0.0018 |

`existing_monthly_obligations_gbp` is effectively zero — an order of magnitude below the threshold. Both are currently retained.

This is a **different question** from the exposure-time investigation: there is no contamination here, just weak signal. Weak-but-clean features cost little, and both are conventional credit attributes a reviewer would expect to see considered rather than silently dropped. Flagged for a deliberate decision, not removed reflexively.

---

## 6. Outliers

IQR-flagged rates consistent with genuine long-tail behaviour. `num_defaults_24m` at 23.15% is a measurement artefact (near-categorical, values 0/1/2), not real skew.

---

## 7. Split stability

Train/test bad rate difference: **0.00%** (33,283 train / 8,321 test, both at 2.72%).

The temporal split's tenor-maturity problem no longer applies — with zero censoring, no customer is excluded for lack of a matured outcome. See Decision 16.

---

## Artifacts

All results are persisted to `reports/eda/`: `target_summary.csv`, `missingness.csv`, `iv_screening.csv`, `vif.csv`, `correlation_matrix.csv`, `correlated_pairs.csv`, `outliers.csv`, `split_stability.csv`, `eda_metadata.json`.

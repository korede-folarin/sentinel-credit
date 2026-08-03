# Sentinel Credit — Pre-Modelling EDA Results

**Population:** 50,000 customers · 18,694 modelable (`has_outcome = 1`)
**Bad rate:** 4.26% (796 defaults)
**Scale:** 58,611 transactions · 172,968 repayment instalments
**Script:** `EDA/eda_baseline.py`
**Chart:** `reports/charts/eda_target_summary.png` (from `reports/eda_target_summary.csv`)

> **Run against the corrected population.** An earlier version of this document reported 41,604 modelable customers at a 1.91% bad rate. That population included customers whose loans had not yet matured, silently labelled as confirmed non-defaulters. See `docs/engineering_decision_log.md` Decision 12 addendum ("has_outcome Correction") for the full finding.

---

## 1. Target imbalance

| | Count | Rate |
|---|---|---|
| Total customers | 50,000 | — |
| Modelable (`has_outcome = 1`) | 18,694 | 37.4% of total |
| Defaults | 796 | 4.26% of modelable |

By segment:

| Segment | Bad rate | n |
|---|---|---|
| `thin_file_flag = 0` | 1.85% | 12,735 |
| `thin_file_flag = 1` | 9.41% | 5,959 |
| `is_vulnerable = 0` | 3.38% | 8,000 |
| `is_vulnerable = 1` | 4.92% | 10,694 |
| `ab_test_group = A_strict` | 2.89% | 8,623 |
| `ab_test_group = B_relaxed` | 5.43% | 10,071 |

The thin-file ratio (9.41% vs 1.85%, roughly 5x) and the A/B split (5.43% vs 2.89%) both confirm the generator's calibration propagated correctly through to the corrected population.

**Statistical adequacy:** 796 defaults across ~22 candidate features gives roughly 33 events per variable, comfortably above the conventional 10-per-variable minimum (Peduzzi et al.). The smaller corrected population remains adequate for modelling.

---

## 2. Missingness

16.79% of the full 50,000-customer table is missing values in exposure-derived fields (`avg_order_value_gbp`, `transaction_span_days`, etc.). This matches the population who never had an approved transaction. **Zero missing values in the modelable training population.**

---

## 3. Univariate screening (Information Value)

| Feature | IV | Note |
|---|---|---|
| `days_since_last_transaction` | 1.1099 | **EXCLUDED** — exposure-time artifact |
| `total_order_value_gbp` | 0.8864 | Suspected — retained, see below |
| `n_approved` | 0.8400 | **EXCLUDED** — exposure-time artifact |
| `thin_file_flag` | 0.6884 | Genuine — matches 5x segment bad-rate gap |
| `savings_balance_gbp` | 0.6538 | Genuine — wired into `financial_resilience()` |
| `num_credit_accounts` | 0.6227 | Genuine |
| `bnpl_exposure_to_monthly_income_ratio` | 0.6224 | |
| `n_merchant_categories` | 0.5971 | Suspected — retained |
| `n_applications` | 0.5829 | |
| `months_credit_history` | 0.5811 | Genuine |
| `max_order_value_gbp` | 0.5732 | Suspected — retained |
| `credit_score` | 0.5718 | Genuine — wired into `miss_hazard()` |
| `transaction_span_days` | 0.4120 | Suspected — retained |
| `avg_order_value_gbp` | 0.3691 | |
| `num_defaults_24m` | 0.2951 | |
| `n_late_payments` | 0.2493 | |
| `pct_late_payments` | 0.2045 | |
| `avg_n_instalments` | 0.1506 | |
| `pct_medium_risk_category` | 0.1042 | |
| `pct_high_risk_category` | 0.0863 | |
| `max_affordability_ratio` | 0.0816 | |
| `monthly_income_gbp` | 0.0216 | |
| `age` | 0.0173 | Below drop threshold |
| `existing_monthly_obligations_gbp` | 0.0154 | Below drop threshold |

### Unequal observation opportunity — the reason 12 features crossed the 0.5 threshold

This is not 12 independently leaky features. It is one upstream cause expressing itself across an entire family.

The corrected `has_outcome` has two asymmetric qualifying paths: a defaulted customer qualifies regardless of timing, but a non-defaulted customer only qualifies if every repayment has fully matured — which requires having originated early enough in the window. Verified: `target_default = 1` customers average 313.5 days of `transaction_span_days` against 204.1 for `target_default = 0`. Every cumulative exposure feature inflates simply from having more calendar time available, independent of risk.

**Confirmed and excluded** (per-day-of-exposure normalisation):
- `n_approved` — raw gap 2.59 vs 4.04; normalised, 0.012654 vs 0.012874 (near-identical, ~100% artifact)
- `days_since_last_transaction` — IV 1.11, highest in the dataset, mechanically implausible as genuine signal

**Suspected, retained for this round, not cleared**: `total_order_value_gbp` (only ~15 of its 77-point raw gap survives normalisation), `transaction_span_days` (the exposure-time variable itself), `max_order_value_gbp`, `n_merchant_categories`. A Feature Set A/B sensitivity comparison is required before the feature set is finalised. See Decision 12 addendum ("Unequal Observation Opportunity").

---

## 4. Multicollinearity — correlation and VIF

**Exact identities and near-duplicates removed** (see `models/model_features.py` `MULTICOLLINEARITY_EXCLUDED`): `n_transactions` (= `n_approved`), `n_declined`, `n_affordability_flags`, `approval_rate`, `pct_affordability_flagged`, `annual_income_gbp` (= `monthly_income_gbp` × 12), `pct_low_risk_category`, `total_instalment_exposure_gbp`, `n_transactions_with_late_payment`, `avg_affordability_ratio`, `debt_to_income_ratio`.

**Resolution-adjacent exclusions**: `avg_days_late`, `max_days_late`, `total_days_late` — `payment_status='Late'` conflates an independent random miss with cure-event arrears depth.

**Remaining high VIF, retained deliberately** (handled via L2 regularisation, `C=0.1`):

| Feature | VIF |
|---|---|
| `n_approved`* | 63.53 |
| `max_order_value_gbp` | 33.30 |
| `n_applications` | 29.56 |
| `avg_order_value_gbp` | 28.42 |
| `total_order_value_gbp` | 28.20 |
| `n_merchant_categories` | 21.35 |
| `credit_score` | 20.70 |
| `max_affordability_ratio` | 19.08 |
| `monthly_income_gbp` | 12.35 |
| `avg_n_instalments` | 10.81 |

*`n_approved` was subsequently excluded on exposure-time grounds; VIF shown as measured during this EDA run.

---

## 5. Outliers

IQR-flagged rates consistent with genuine long-tail behaviour. `num_defaults_24m` at 23.42% is a measurement artefact (near-categorical, values 0/1/2 only), not real skew.

---

## 6. Split stability

Train/test bad rate difference: **0.01%** (14,955 train at 4.26% / 3,739 test at 4.25%). The stratified split preserves class balance correctly.

**Temporal split is currently unusable** — at `2024-09-01`, the corrected `has_outcome` maturity requirement leaves only 16 customers in the test set, all defaulters, making discrimination metrics undefined. Skipped automatically with an explanatory message. See Decision 10 addendum (tenor-maturity bias).

---

## Final feature set

**22 features** in `models/model_features.py`. Not finalised — the four `EXPOSURE_TIME_SUSPECTED` features require the Feature Set A/B comparison before this list is locked.

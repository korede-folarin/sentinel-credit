"""
Sentinel Credit model feature allowlist.

The feature store retains the full analytical record.
MODEL_FEATURES defines the fields permitted for model training.

Exclusion decisions and supporting evidence are documented in:
- Decision 12: leakage and resolution-tied features
- Decision 14: exposure-time feature investigation
- Decision 18: v1 scope and 14-feature freeze
- Decision 27: pct_late_payments removed for leakage (v1.1, 13 features)
"""

# ── TARGET AND FILTER ──────────────────────────────────────
TARGET_COLUMN = "target_default"

# Train only where an outcome is genuinely observable.
# See Decision 10 (right-censoring) / Decision 12 (reject
# inference — has_outcome flag rather than imputed labels).
FILTER_COLUMN = "has_outcome"
FILTER_VALUE = 1


# ── MODEL_FEATURES — safe to train on ──────────────────────
# v1 scope: customer-level portfolio-risk model (Decision 18).
MODEL_FEATURES = [
    # Customer attributes (bureau-style, day-zero observable)
    "monthly_income_gbp",
    "existing_monthly_obligations_gbp",
    "savings_balance_gbp",
    "credit_score",
    "months_credit_history",
    "num_credit_accounts",
    "num_defaults_24m",
    "thin_file_flag",

    # Application behaviour and exposure
    "max_affordability_ratio",
    "avg_order_value_gbp",
    "avg_n_instalments",
    "pct_high_risk_category",
    "pct_medium_risk_category",
]
# v1.1: 13 features. pct_late_payments removed for leakage
# (Decision 27); see RESOLUTION_ADJACENT_EXCLUDED below. The
# August 2026 v1 champion (14 features, models/artifacts/) was
# trained with it; its feature list is saved in that package.
# Remaining features retained for the v1 customer-level
# portfolio-risk scope (Decision 18).


# ── EXCLUDED — kept in the table, never fed to the model ───
# Reasons differ; see Decision 12 addendum for why they
# should not be cited interchangeably.

# Exact linear identities (VIF = inf) or near-duplicates
# (r > 0.9) of features already in MODEL_FEATURES. Feeding
# both sides of an identity to Logistic Regression makes its
# coefficients unstable and meaningless.
MULTICOLLINEARITY_EXCLUDED = [
    "n_declined",                       # = n_applications - n_approved
    "n_transactions",                   # = n_approved
    "n_affordability_flags",            # = n_declined
    "approval_rate",                    # = n_approved / n_applications
    "pct_affordability_flagged",        # = n_declined / n_applications
    "annual_income_gbp",                # = monthly_income_gbp * 12
    "pct_low_risk_category",            # pct_high + pct_medium + pct_low = 1.0
    "avg_affordability_ratio",          # r=0.93 with max_affordability_ratio
    "debt_to_income_ratio",             # r=0.91 with avg_affordability_ratio
    "total_instalment_exposure_gbp",    # r=0.974 with total_order_value_gbp
    "n_transactions_with_late_payment", # r=0.949 with n_late_payments
]

# Aggregates of days_late, which mixes two different generator
# events (independent miss vs. cure-event depth) under one
# label — the same category of leakage as max_dpd_ever.
# See Decision 12 addendum.
RESOLUTION_ADJACENT_EXCLUDED = [
    "avg_days_late",
    "max_days_late",
    "total_days_late",
    # 'Late' also records full/partial arrears-cure payments (the
    # process default emerges from), and post-default 'Defaulted'
    # rows sit in its denominator. Decisions 26-27.
    "pct_late_payments",
]

# Confirmed exposure-time contaminated: value grows with how
# long a customer has been observed, rather than with genuine
# risk. See Decision 12 addendum (n_approved,
# days_since_last_transaction), Decision 14 (the SHAP/ablation
# four), and Decision 18 (the three added after the
# point-in-time correlation check).
EXPOSURE_TIME_EXCLUDED = [
    "n_approved",
    "days_since_last_transaction",
    "total_order_value_gbp",
    "transaction_span_days",
    "max_order_value_gbp",
    "n_merchant_categories",
    "n_applications",
    "bnpl_exposure_to_monthly_income_ratio",
    "n_late_payments",
]

# Exposure-time investigation resolved in Decision 14.
# Retained as an empty list for compatibility with
# compare_feature_sets.py.
EXPOSURE_TIME_SUSPECTED = []

# Generator-internal fields. Never meant to represent
# real-world observable data. Retained in raw for auditing
# and post-hoc validation only. See Decision 3.
GENERATOR_INTERNAL_EXCLUDED = [
    "default_probability",       # removed from schema entirely, listed for completeness
    "merchant_risk_multiplier",
]

# Resolution-tied features. Real, meaningful outcome data, but
# only knowable after the outcome they would be predicting.
# Retained for explainability, drift monitoring, and portfolio
# reporting. See Decision 12 addendum.
RESOLUTION_TIED_EXCLUDED = [
    "max_dpd_ever",
    "avg_max_dpd",
    "n_reached_sicr",
    "pct_reached_sicr",
    "max_cycles_delinquent",
    "avg_cycles_delinquent",
    "max_arrears_depth",
    "avg_arrears_depth_when_behind",
    "n_terminal_current",
    "n_terminal_delinquent",
    "n_terminal_default",
]

# Identifiers, protected/sensitive characteristics, and
# comparison-group labels — excluded regardless of leakage
# status. thin_file_flag was tested for removal and restored;
# see Decision 12 addendum. age removed on regulatory
# defensibility grounds (Equality Act 2010 proxy risk), not
# IV; see Decision 18.
NON_MODEL_EXCLUDED = [
    "customer_id",
    "vulnerability_score",
    "is_vulnerable",
    "age",
    "age_bracket",
    "income_band",
    "employment_status",
    "ab_test_group",
    "first_transaction_date",
    "last_transaction_date",
]

"""
Sentinel Credit — model feature allowlist.

Mirrors the naming and boundary pattern already established
in validate_synthetic.py (SYNTHETIC_MODEL_FEATURES,
CROSS_MARKET_FEATURES). The feature store table
(feature_store.customer_features) is the complete record —
used for auditing, explainability, drift monitoring, and
portfolio reporting. MODEL_FEATURES defines the subset the
model is actually allowed to see. Everything excluded here
stays in the table; nothing is deleted at the SQL layer.

See docs/engineering_decision_log.md — Decision 12 addendum
(temporal leakage), Decision 3 (original leakage boundary
precedent).
"""

# ── TARGET AND FILTER ──────────────────────────────────────
TARGET_COLUMN = "target_default"

# Train only where an outcome is genuinely observable.
# See Decision 10 (right-censoring) / Decision 12 (reject
# inference — has_outcome flag rather than imputed labels).
FILTER_COLUMN = "has_outcome"
FILTER_VALUE = 1


# ── MODEL_FEATURES — safe to train on ──────────────────────
MODEL_FEATURES = [
    # Customer attributes (bureau-style, day-zero observable)
    "age",
    "monthly_income_gbp",
    "existing_monthly_obligations_gbp",
    "savings_balance_gbp",
    "credit_score",
    "months_credit_history",
    "num_credit_accounts",
    "num_defaults_24m",
    "thin_file_flag",

    # Application behaviour (all applications — decline
    # patterns are a legitimate distress signal, independent
    # of default outcome). n_declined, n_affordability_flags,
    # approval_rate, and pct_affordability_flagged removed —
    # see MULTICOLLINEARITY_EXCLUDED below, they are exact
    # functions of n_applications and n_approved.
    "n_applications",
    "max_affordability_ratio",

    # Exposure (approved transactions only). n_transactions
    # removed — exact duplicate of n_approved (VIF = inf).
    # total_order_value_gbp, transaction_span_days,
    # max_order_value_gbp, n_merchant_categories: SUSPECTED
    # contamination from unequal observation opportunity —
    # see EXPOSURE_TIME_SUSPECTED below. Retained for this
    # training round pending the Feature Set A/B sensitivity
    # comparison, deferred until after the rest of the
    # architecture is built.
    "avg_order_value_gbp",
    "avg_n_instalments",
    "pct_high_risk_category",
    "pct_medium_risk_category",

    # Late-payment path — count-based fields only. avg/max/
    # total_days_late excluded (see RESOLUTION_ADJACENT_EXCLUDED
    # below); n_transactions_with_late_payment excluded as a
    # near-duplicate of n_late_payments (r=0.949).
    "n_late_payments",
    "pct_late_payments",

    # Derived ratios
    "bnpl_exposure_to_monthly_income_ratio",

    # Timing (structural, not outcome-derived).
    # transaction_span_days: SUSPECTED — see above.
]


# ── EXCLUDED — kept in the table, never fed to the model ───
# Reasons differ; see Decision 12 addendum for why they
# should not be cited interchangeably.

# Exact linear identities, found via VIF = inf during EDA
# (not mere high correlation — one feature is a deterministic
# function of others already in the set). Kept in the feature
# store for analysis/reporting; excluded here because feeding
# both sides of an identity to Logistic Regression makes its
# coefficients unstable and meaningless, undermining the
# interpretability Decision 4 chose LR for.
MULTICOLLINEARITY_EXCLUDED = [
    "n_declined",               # = n_applications - n_approved
    "n_transactions",           # = n_approved (duplicate CTE, IV identical to 4dp)
    "n_affordability_flags",    # = n_declined (a decline IS the affordability flag firing)
    "approval_rate",            # = n_approved / n_applications
    "pct_affordability_flagged",# = n_declined / n_applications
    "annual_income_gbp",        # = monthly_income_gbp * 12, exact
    "pct_low_risk_category",    # pct_high + pct_medium + pct_low = 1.0 always
    "avg_affordability_ratio",  # r=0.93 with max_affordability_ratio, r=0.91 with
                                 # debt_to_income_ratio; all three IV<0.03 (weak) —
                                 # consolidated to max_affordability_ratio (highest
                                 # individual IV, captures peak financial stretch)
    "debt_to_income_ratio",     # structurally near-contained within affordability_ratio
                                 # (obligations/income is its dominant term); r=0.91
                                 # with avg_affordability_ratio
    "total_instalment_exposure_gbp",  # r=0.974 with total_order_value_gbp — near
                                       # identity (instalment amounts are order value
                                       # divided across payments, summed back). Kept
                                       # total_order_value_gbp, higher IV.
    "n_transactions_with_late_payment",  # r=0.949 with n_late_payments — near
                                          # duplicate count. Kept n_late_payments.
]

# Resolution-adjacent contamination, found via a corrected
# EDA run (30th July 2026) after feature_store.customer_features
# was rebuilt against final calibration data. payment_status =
# 'Late' fires from TWO different events in the generator: an
# independent early-miss draw, AND a cure event (full/partial),
# where days_late = how deep in arrears the customer was at the
# moment they recovered. Both get the same 'Late' label, so any
# feature built from the days_late VALUE (not just counting
# 'Late' occurrences) mixes independent noise with genuine
# delinquency-severity signal from the same hazard process that
# produces default — the same category of problem Decision 12
# addressed for max_dpd_ever, found here in a different field.
# avg_days_late crossed the 0.5 leakage-investigation threshold
# (IV 0.61); max_days_late and total_days_late are the same
# underlying field aggregated differently and share the issue
# at lower magnitude (IV 0.45 and 0.22 respectively).
RESOLUTION_ADJACENT_EXCLUDED = [
    "avg_days_late",
    "max_days_late",
    "total_days_late",
]

# Found after the has_outcome fix (30th July 2026 — see
# Decision 12 addendum "Unequal Observation Opportunity").
# has_outcome's two qualifying paths are asymmetric: a
# defaulted customer qualifies regardless of timing
# (default_flag=1 alone), but a non-defaulted customer only
# qualifies if ALL their repayments have fully matured —
# which requires having originated early enough in the window.
# Verified: target_default=1 customers average 313.5 days of
# transaction_span_days vs 204.1 for target_default=0 — a
# structural artifact of which qualifying path a customer
# took, not a risk signal. n_approved's per-day-of-exposure
# rate is nearly identical between groups (0.012654 vs
# 0.012874) confirming its raw gap is ~100% artifact.
# days_since_last_transaction shows the same pattern even more
# severely (IV 1.11, the highest in the dataset — mechanically
# implausible as genuine signal).
EXPOSURE_TIME_EXCLUDED = [
    "n_approved",
    "days_since_last_transaction",
    # Added 3rd August 2026 after the SHAP + ablation
    # investigation (Decision 14). SHAP showed the model gave
    # 32.1% of total mean |SHAP| importance to these four
    # features. A repeated-CV ablation (5 folds x 3 repeats)
    # found Gini/AUC/KS effectively tied between feature sets,
    # PR-AUC favouring their inclusion (13/15 folds), and the
    # train/test gap favouring their exclusion (14/15 folds).
    # Excluded on FEATURE VALIDITY grounds, not performance:
    # the observation-window analysis established they are
    # structurally confounded with unequal exposure time and
    # do not measure the phenomenon they were intended to.
    # A modest PR-AUC cost is accepted deliberately. Removing
    # them also changes 9.04% of individual decisions at the
    # business cutoff, so they were not present-but-harmless.
    "total_order_value_gbp",
    "transaction_span_days",
    "max_order_value_gbp",
    "n_merchant_categories",
]

# SUSPECTED of the same contamination, NOT yet confirmed or
# excluded. total_order_value_gbp's raw group gap is 77%, but
# only ~15 points of that survive per-day-of-exposure
# normalisation (7.75 vs 8.93) — meaning most, not all, of its
# strength is artifact; a real behavioural signal may exist
# underneath. transaction_span_days is not merely another
# suspected feature — it IS the exposure-time variable driving
# the contamination in the others, and risks the model
# learning "who had enough calendar time to mature into the
# selected population" rather than genuine risk.
# max_order_value_gbp and n_merchant_categories are mechanically
# different (a maximum and a distinct-count inflate simply from
# having more transactions/draws, not from calendar time per
# se) and were not conclusively shown to be contaminated in the
# per-transaction-normalised check, but are flagged for the
# same reason: retained for now, not cleared.
#
# Retained in MODEL_FEATURES for this training round. NOT
# finalised. Before this feature set is called final, a
# sensitivity comparison is required: Feature Set A
# (conservative — all four excluded) vs Feature Set B
# (exploratory — retained) compared on Gini, KS, PR-AUC,
# calibration, feature importance, and train/test stability.
# If Set B performs dramatically better than Set A, that is
# evidence the observation-window asymmetry is contributing to
# apparent performance, not evidence the features are
# genuinely predictive. Deferred until after SHAP (Decision 7),
# the MLflow promotion workflow (Decision 8/9), and drift
# monitoring (Decision 5) are built — the feature set should be
# locked once, deliberately, before those layers are built on
# top of it, not revised mid-build.
# RESOLVED 3rd August 2026 — all four moved to
# EXPOSURE_TIME_EXCLUDED above following the SHAP and ablation
# investigation. Kept as an empty list rather than deleted so
# compare_feature_sets.py still imports cleanly and the
# investigation's structure remains visible in the code.
EXPOSURE_TIME_SUSPECTED = []

# Generator-internal fields. Never meant to represent
# real-world observable data. Retained in raw for auditing
# and post-hoc validation only. See Decision 3.
GENERATOR_INTERNAL_EXCLUDED = [
    "default_probability",       # removed from schema entirely, listed for completeness
    "merchant_risk_multiplier",
]

# Resolution-tied features. Real, meaningful outcome data,
# but aggregated across the customer's full lifetime
# including the point of default itself — using them as
# predictive input restates the target rather than
# predicting it. Retained for explainability (Decision 7),
# drift monitoring (Decision 5), and portfolio reporting.
# See Decision 12 addendum.
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

# Deliberately excluded regardless of leakage status —
# identifiers, protected/sensitive characteristics, and
# fields flagged in the SQL itself as "analysis, not model
# input."
#
# thin_file_flag was tested as an exclusion (30th July 2026)
# after producing a 100% false-negative-rate gap for
# thin_file_flag=0 customers as a model input. Removing it
# and re-running produced an IDENTICAL gap — credit_score
# (r=-0.75 with thin_file_flag) reconstructs the same
# segmentation on its own. Restored to MODEL_FEATURES: the
# gap is a real property of a single global cutoff applied to
# a segment with genuinely low intrinsic risk (0.85% actual
# default rate, well-calibrated predicted PD), not something
# caused by this one input. See Decision 12 addendum for the
# full experiment and the segment-aware thresholding
# consideration that follows from it.
NON_MODEL_EXCLUDED = [
    "customer_id",
    "vulnerability_score",       # composite includes health driver — see generator's
    "is_vulnerable",             # mental_health_flag exclusion rule, Equality Act 2010
    "age_bracket",
    "income_band",
    "employment_status",
    "ab_test_group",
    "first_transaction_date",
    "last_transaction_date",
]

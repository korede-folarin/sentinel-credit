"""
Sentinel Credit, First-Application Feature Construction

Single source of truth for turning a new customer's profile plus one
order into the champion's model features. Used by the FastAPI service
(serving/app.py) and the Streamlit Credit Risk Assessment page, so the
two can never score the same applicant differently.

Every value is computed from the application itself, never an invented
average: avg_order_value_gbp and avg_n_instalments equal this single
order, the category shares reflect this order's own category, and
max_affordability_ratio is (obligations + this instalment) / income.

The merchant risk tiers mirror sql/02_feature_store.sql exactly.
"""

PROFILE_FIELDS = [
    "monthly_income_gbp",
    "existing_monthly_obligations_gbp",
    "savings_balance_gbp",
    "credit_score",
    "months_credit_history",
    "num_credit_accounts",
    "num_defaults_24m",
    "thin_file_flag",
]

MERCHANT_CATEGORIES = [
    "Fashion & Clothing", "Electronics", "Home", "Health & Beauty",
    "Sports", "Travel", "Food & Grocery", "Other",
]

# Same tiers as sql/02_feature_store.sql (pct_*_risk_category CASE lists).
HIGH_RISK_CATEGORIES = {"Travel", "Electronics"}
MEDIUM_RISK_CATEGORIES = {"Fashion & Clothing", "Home", "Other"}


def build_new_customer_features(profile: dict, order_value_gbp: float,
                                n_instalments: int, merchant_category: str) -> dict:
    """
    profile must contain every PROFILE_FIELDS key (raises KeyError naming
    the missing ones otherwise). Returns the model feature values.
    """
    missing = [f for f in PROFILE_FIELDS if profile.get(f) is None]
    if missing:
        raise KeyError(f"New-customer profile fields are required: {missing}")
    if order_value_gbp <= 0 or n_instalments <= 0 or profile["monthly_income_gbp"] <= 0:
        raise ValueError("order value, instalments and monthly income must be positive")

    instalment_amount = order_value_gbp / n_instalments
    affordability_ratio = (
        profile["existing_monthly_obligations_gbp"] + instalment_amount
    ) / profile["monthly_income_gbp"]

    return {
        **{f: profile[f] for f in PROFILE_FIELDS},
        "max_affordability_ratio": round(affordability_ratio, 4),
        "avg_order_value_gbp": order_value_gbp,
        "avg_n_instalments": n_instalments,
        "pct_high_risk_category": 1.0 if merchant_category in HIGH_RISK_CATEGORIES else 0.0,
        "pct_medium_risk_category": 1.0 if merchant_category in MEDIUM_RISK_CATEGORIES else 0.0,
    }

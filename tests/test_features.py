"""
Feature engineering: the champion's frozen feature set and no leakage.

Plain English: the v1.1 champion uses exactly 13 features (v1 Decision 27
removed pct_late_payments because it leaked the outcome). These tests check
that the code, the shipped model and the governance lists all agree, that no
feature known to leak or to grow with time can reach the model, and that a
new applicant's features are built correctly.
"""

import json
import os

import pytest

from conftest import REPO_MODEL_DIR, ROOT
from models import model_features as mf
from serving.features import PROFILE_FIELDS, build_new_customer_features

EXPECTED_13 = [
    "monthly_income_gbp", "existing_monthly_obligations_gbp", "savings_balance_gbp",
    "credit_score", "months_credit_history", "num_credit_accounts", "num_defaults_24m",
    "thin_file_flag", "max_affordability_ratio", "avg_order_value_gbp", "avg_n_instalments",
    "pct_high_risk_category", "pct_medium_risk_category",
]


def test_model_uses_exactly_the_13_frozen_features():
    assert mf.MODEL_FEATURES == EXPECTED_13


def test_shipped_champion_was_trained_on_the_same_13():
    with open(os.path.join(REPO_MODEL_DIR, "feature_list.json"), encoding="utf-8") as f:
        assert json.load(f)["features"] == EXPECTED_13


def test_leaking_feature_is_excluded():
    assert "pct_late_payments" not in mf.MODEL_FEATURES
    assert "pct_late_payments" in mf.RESOLUTION_ADJACENT_EXCLUDED


@pytest.mark.parametrize("group", [
    "RESOLUTION_TIED_EXCLUDED", "RESOLUTION_ADJACENT_EXCLUDED", "EXPOSURE_TIME_EXCLUDED",
    "GENERATOR_INTERNAL_EXCLUDED", "NON_MODEL_EXCLUDED", "MULTICOLLINEARITY_EXCLUDED",
])
def test_no_excluded_feature_reaches_the_model(group):
    # Outcome-tied, exposure-time, generator-internal and protected fields
    # (age, vulnerability, employment, income band) must never be inputs.
    assert not set(getattr(mf, group)) & set(mf.MODEL_FEATURES)


def test_target_is_not_a_feature():
    assert mf.TARGET_COLUMN not in mf.MODEL_FEATURES


def test_new_applicant_features():
    profile = {f: 0 for f in PROFILE_FIELDS}
    profile.update(monthly_income_gbp=2000.0, existing_monthly_obligations_gbp=300.0)
    f = build_new_customer_features(profile, 600.0, 3, "Electronics")
    assert set(f) == set(EXPECTED_13)
    assert f["max_affordability_ratio"] == pytest.approx((300 + 200) / 2000, abs=1e-4)
    assert f["avg_order_value_gbp"] == 600.0 and f["avg_n_instalments"] == 3
    assert (f["pct_high_risk_category"], f["pct_medium_risk_category"]) == (1.0, 0.0)
    g = build_new_customer_features(profile, 600.0, 3, "Home")
    assert (g["pct_high_risk_category"], g["pct_medium_risk_category"]) == (0.0, 1.0)


def test_new_applicant_input_validation():
    profile = {f: 1 for f in PROFILE_FIELDS}
    with pytest.raises(KeyError):
        build_new_customer_features({k: v for k, v in profile.items() if k != "credit_score"},
                                    100.0, 3, "Home")
    with pytest.raises(ValueError):
        build_new_customer_features(profile, 0.0, 3, "Home")


def test_feature_store_sql_defines_every_model_feature():
    with open(os.path.join(ROOT, "sql", "02_feature_store.sql"), encoding="utf-8") as f:
        sql = f.read()
    missing = [c for c in EXPECTED_13 if c not in sql]
    assert not missing, f"feature store SQL does not define {missing}"

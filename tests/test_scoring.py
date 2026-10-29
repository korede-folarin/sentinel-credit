"""
The champion scoring path.

Plain English: scoring a new applicant must give the same probability of
default every time, the same as the published demo and API (known reference
applicants), and the decision must follow the business cutoff (0.0657,
v1 Decision 27): at or above it, refer for review.
"""

import json
import os

import joblib
import pandas as pd
import pytest
import yaml

from conftest import LOCAL_MODEL_DIR, REPO_MODEL_DIR, ROOT, needs_local_package
from serving.reasons import explain

with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
    CUTOFF = yaml.safe_load(f)["model"]["business_cutoff"]

TYPICAL = dict(monthly_income_gbp=2442.0, existing_monthly_obligations_gbp=512.0, savings_balance_gbp=2165.0,
               credit_score=638, months_credit_history=58, num_credit_accounts=2, num_defaults_24m=0,
               order_value_gbp=583.0, n_instalments=3, merchant_category="Fashion & Clothing")
BROWSER_TEST = dict(monthly_income_gbp=2200.0, existing_monthly_obligations_gbp=300.0, savings_balance_gbp=150.0,
                    credit_score=610, months_credit_history=18, num_credit_accounts=2, num_defaults_24m=0,
                    order_value_gbp=500.0, n_instalments=6, merchant_category="Electronics")


def scorer(model_dir):
    model = joblib.load(os.path.join(model_dir, "champion_calibrated.joblib"))
    with open(os.path.join(model_dir, "feature_list.json"), encoding="utf-8") as f:
        feats = json.load(f)["features"]
    return lambda ds: model.predict_proba(pd.DataFrame([{k: d[k] for k in feats} for d in ds]))[:, 1].tolist()


def test_business_cutoff_is_the_governed_value():
    assert CUTOFF == 0.0657


@pytest.mark.parametrize("applicant,expected", [(TYPICAL, 0.009456), (BROWSER_TEST, 0.031834)])
def test_reference_applicants_score_as_published(applicant, expected):
    # The same values the demo site and API produce (v1 Decisions 28-29).
    out = explain({**applicant, "thin_file_flag": None}, scorer(REPO_MODEL_DIR), CUTOFF)
    assert out["probability"] == pytest.approx(expected, abs=5e-6)


def test_scoring_is_deterministic():
    s = scorer(REPO_MODEL_DIR)
    a = explain({**BROWSER_TEST, "thin_file_flag": None}, s, CUTOFF)
    b = explain({**BROWSER_TEST, "thin_file_flag": None}, s, CUTOFF)
    assert a["probability"] == b["probability"] and a["explanation"] == b["explanation"]


def test_decision_follows_the_cutoff():
    out = explain({**BROWSER_TEST, "thin_file_flag": None}, scorer(REPO_MODEL_DIR), CUTOFF)
    expected = "refer" if out["probability"] >= CUTOFF else "pass"
    assert out["risk_officer"]["model_recommendation"] == expected
    risky = dict(BROWSER_TEST, credit_score=420, months_credit_history=6, num_credit_accounts=1,
                 num_defaults_24m=2, savings_balance_gbp=0.0, n_instalments=12)
    r = explain({**risky, "thin_file_flag": None}, scorer(REPO_MODEL_DIR), CUTOFF)
    assert r["probability"] >= CUTOFF and r["risk_officer"]["model_recommendation"] == "refer"


def test_probabilities_are_valid():
    probs = scorer(REPO_MODEL_DIR)([{**TYPICAL, **{"thin_file_flag": 0, "max_affordability_ratio": 0.3,
                                                 "avg_order_value_gbp": 583.0, "avg_n_instalments": 3,
                                                 "pct_high_risk_category": 0.0,
                                                 "pct_medium_risk_category": 1.0}}])
    assert all(0.0 <= p <= 1.0 for p in probs)


@needs_local_package
def test_committed_copy_is_the_governed_champion():
    # The committed model (used by CI and the demo) must be byte-identical to
    # the governed, git-ignored package the API loads by default.
    for name in ("champion_calibrated.joblib", "feature_list.json"):
        with open(os.path.join(REPO_MODEL_DIR, name), "rb") as a, open(os.path.join(LOCAL_MODEL_DIR, name), "rb") as b:
            assert a.read() == b.read(), name

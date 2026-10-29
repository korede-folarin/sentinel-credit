"""
The FastAPI scoring service (serving/app.py).

Plain English: the API's endpoints answer correctly, reject bad input, and
give exactly the same answers whether it loads the champion from its default
(git-ignored) package or from the committed copy via SENTINEL_MODEL_DIR, the
optional setting added for CI and containers (v2 V2-01).
"""

import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_MODEL_DIR, needs_db, needs_local_package

NEW_APPLICANT = dict(order_value_gbp=500.0, n_instalments=6, merchant_category="Electronics",
                     monthly_income_gbp=2200.0, existing_monthly_obligations_gbp=300.0,
                     savings_balance_gbp=150.0, credit_score=610, months_credit_history=18,
                     num_credit_accounts=2, num_defaults_24m=0)


def load_app(model_dir):
    """Import (or re-import) serving.app with or without the override."""
    if model_dir is None:
        os.environ.pop("SENTINEL_MODEL_DIR", None)
    else:
        os.environ["SENTINEL_MODEL_DIR"] = model_dir
    module = importlib.reload(sys.modules["serving.app"]) if "serving.app" in sys.modules \
        else importlib.import_module("serving.app")
    return module


@pytest.fixture(scope="module")
def client():
    module = load_app(REPO_MODEL_DIR)
    yield TestClient(module.app)
    os.environ.pop("SENTINEL_MODEL_DIR", None)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_version"] == "v1.1" and info["n_features"] == 13
    assert info["business_cutoff"] == 0.0657


def test_score_new_applicant(client):
    r = client.post("/score", json=NEW_APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_type"] == "new"
    assert body["calibrated_probability"] == pytest.approx(0.0318, abs=1e-4)
    assert body["decision"] == "pass"
    assert body["explanation"]["outcome"] in {"approved", "manual_review", "declined"}
    assert body["decision_record"]["thin_file_source"] == "derived"
    assert set(body["feature_values_used"]) and len(body["feature_values_used"]) == 13


def test_missing_profile_fields_are_rejected(client):
    bad = {k: v for k, v in NEW_APPLICANT.items() if k != "credit_score"}
    assert client.post("/score", json=bad).status_code == 422


def test_invalid_types_are_rejected(client):
    assert client.post("/score", json=dict(NEW_APPLICANT, order_value_gbp="lots")).status_code == 422


def test_scoring_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and "Sentinel" in r.text


@needs_db
def test_existing_customer_is_scored_from_the_feature_store(client):
    import pandas as pd

    from conftest import LOCAL_MODEL_DIR
    ids = pd.read_csv(os.path.join(LOCAL_MODEL_DIR, "split_ids_test.csv"))["customer_id"].head(1)
    r = client.post("/score", json={"customer_id": ids.iloc[0], "order_value_gbp": 100.0,
                                    "n_instalments": 3, "merchant_category": "Home"})
    assert r.status_code == 200 and r.json()["customer_type"] == "existing"


@needs_local_package
def test_override_gives_identical_answers_to_the_default():
    """v1 behaviour is unchanged: default package vs the override copy."""
    def answers(model_dir):
        c = TestClient(load_app(model_dir).app)
        out = c.post("/score", json=NEW_APPLICANT).json()
        out["decision_record"].pop("generated_at", None)
        return c.get("/model/info").json(), out
    try:
        assert answers(None) == answers(REPO_MODEL_DIR)
    finally:
        os.environ.pop("SENTINEL_MODEL_DIR", None)

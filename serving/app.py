"""
Sentinel Credit, Scoring API

Purpose
-------
Live scoring endpoint for a single application, combining a live
database lookup for existing customer history with the current
application's own details, exactly the design this project
settled on directly, not a lookup only or payload only shortcut.

Existing customer: their 14 governed feature values are read
directly from feature_store.customer_features, the same table
the model was actually trained against, rather than this API
recomputing aggregates separately and risking drift from the
real feature engineering logic already governed elsewhere in
this project.

New customer, no row in feature_store.customer_features yet:
the caller must supply the eight profile fields directly
(income, savings, credit score, and so on, exactly as a real
merchant integration or bureau pull would provide at the point
of application). The five behavioral aggregate features default
to first-application values computed from this order, not an
invented average: avg_order_value_gbp and avg_n_instalments equal
this single order, pct_high_risk_category and
pct_medium_risk_category reflect this order's own category, and
max_affordability_ratio is computed from this application itself.
(v1 also needed pct_late_payments, which the API had to fill with
an assumed 0.0; v1.1 removed that feature for leakage, Decision 27.)

Known scope limitation, stated here and in every response, not
hidden
------------------------------------------------------------
The champion model was trained as a customer-level portfolio-risk
model on lifetime aggregates (Decision 18), not as an
application-time underwriting model. True application-time
underwriting, with features properly reconstructed and a model
retrained for that grain, was explicitly scoped as future v2
work, not resolved here. This API demonstrates live feature
reconstruction and scoring end to end using the existing v1
model as is. Treat calibrated_probability from a new customer
scored here as illustrative of the mechanism, not as a validated
application-time decision the way the v1 model's real evaluation
evidence supports for portfolio-level scoring.

See docs/engineering_decision_log.md, Decision 18.
"""

import json
import os
import warnings
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datetime import datetime, timezone

import joblib
import pandas as pd
import psycopg2
import yaml
from pydantic import BaseModel, Field

from serving.features import PROFILE_FIELDS
from serving.features import build_new_customer_features as shared_build_features
from serving.reasons import (
    derive_thin_file, explain as explain_reasons, explain_existing, load_rules,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    CONFIG["model_governance"].get("artifacts_dir", "models/artifacts"),
)  # the governed champion package; see config.yaml model_governance
# Optional override for CI and containers (v2 V2-01): point at a copy of the
# same champion package, e.g. dashboard/demo_data/model. Unset = the default
# above, unchanged; tests/test_api.py proves both give identical answers.
if os.environ.get("SENTINEL_MODEL_DIR"):
    ARTIFACTS_DIR = os.path.abspath(os.environ["SENTINEL_MODEL_DIR"])

DB_CFG = CONFIG["database"]
GOV_CFG = CONFIG["model_governance"]
BUSINESS_CUTOFF = CONFIG["model"]["business_cutoff"]

with open(os.path.join(ARTIFACTS_DIR, "feature_list.json"), "r", encoding="utf-8") as f:
    MODEL_FEATURES = json.load(f)["features"]

with open(os.path.join(ARTIFACTS_DIR, "champion_metadata.json"), "r", encoding="utf-8") as f:
    CHAMPION_METADATA = json.load(f)

MODEL = joblib.load(os.path.join(ARTIFACTS_DIR, "champion_calibrated.joblib"))

# The explanation rules are built for one champion (Decision 31); after a
# promotion or rollback they must be rebuilt before explanations are served.
EXPLANATION_RULES = load_rules()
RULES_MODEL_VERSION = EXPLANATION_RULES["model"]["model_version"]

app = FastAPI(

    
    title="Sentinel Credit Scoring API",
    description=(
        "Live BNPL credit-risk scoring. Known scope limitation: v1 "
        "customer-level portfolio-risk model applied to application-time "
        "scoring, application-time underwriting is deferred future work "
        "(Decision 18). See module docstring."
    ),
    version=GOV_CFG["model_version"],
    docs_url=None,
    redoc_url=None,
    
    openapi_tags=[
        {
            "name": "Status",
            "description": "Health check and model metadata — no scoring, safe to call anytime.",
        },
        {
            "name": "Scoring",
            "description": (
                "The real credit-risk decision endpoint. Existing customers use their "
                "governed feature_store history; new customers require profile fields "
                "and get first-application defaults for behavioral features."
            ),
        },
    ],
)

TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "serving", "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "serving", "static")

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

def get_db_connection():
    host = os.environ.get("SENTINEL_DB_HOST", DB_CFG.get("host", "localhost"))
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    return psycopg2.connect(
        host=host, port=DB_CFG.get("port", 5432),
        dbname=DB_CFG.get("name", "sentinel_credit"),
        user="postgres", password=password,
    )


class ScoreRequest(BaseModel):
    customer_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing customer: their real feature_store.customer_features "
            "row is used directly, the profile fields below are ignored. "
            "New or unrecognized customer: leave blank or use an ID not "
            "in the database, then the profile fields below become required."
        ),
    )
    order_value_gbp: float = Field(..., gt=0, description="This application's order value.")
    n_instalments: int = Field(..., gt=0, description="This application's instalment count.")
    merchant_category: Literal[
        "Fashion & Clothing",
        "Electronics",
        "Home",
        "Health & Beauty",
        "Sports",
        "Travel",
        "Food & Grocery",
        "Other",
    ] = Field(
        ...,
        description="Merchant category for this application."
    )

    # Required only if customer_id is missing or not found in
    # feature_store.customer_features.
    monthly_income_gbp: Optional[float] = Field(
        default=None, description="New-customer only. Required if customer_id is unknown."
    )
    existing_monthly_obligations_gbp: Optional[float] = Field(
        default=None, description="New-customer only."
    )
    savings_balance_gbp: Optional[float] = Field(default=None, description="New-customer only.")
    credit_score: Optional[float] = Field(default=None, description="New-customer only.")
    months_credit_history: Optional[float] = Field(default=None, description="New-customer only.")
    num_credit_accounts: Optional[float] = Field(default=None, description="New-customer only.")
    num_defaults_24m: Optional[float] = Field(default=None, description="New-customer only.")
    thin_file_flag: Optional[int] = Field(
        default=None,
        description=("New-customer only. Leave out to have it worked out from the credit "
                     "score, months of history and number of accounts (Decision 34); supply "
                     "it only if it comes from a bureau. 1 = limited history."),
    )

    class Config:
        json_schema_extra = {
            "examples": [
        {
            "summary": "New customer, first application",
            "value": {
                "customer_id": None,
                "order_value_gbp": 500.0,
                "n_instalments": 4,
                "merchant_category": "Electronics",
                "monthly_income_gbp": 2200.0,
                "existing_monthly_obligations_gbp": 300.0,
                "savings_balance_gbp": 150.0,
                "credit_score": 610,
                "months_credit_history": 18,
                "num_credit_accounts": 2,
                "num_defaults_24m": 0,
            },
        },
                {
                    "summary": "Existing customer (profile fields ignored)",
                    "value": {
                        "customer_id": "C0001",
                        "order_value_gbp": 250.0,
                        "n_instalments": 3,
                        "merchant_category": "Electronics",
                    },
                },
            ]
        }


class ScoreResponse(BaseModel):
    customer_id: Optional[str]
    customer_type: Literal["existing", "new"]
    calibrated_probability: float
    business_cutoff: float
    decision: Literal["flag_high_risk", "pass"]
    model_version: str
    feature_set_label: str
    feature_values_used: dict
    scope_warning: str
    risk_officer_review_required: Optional[bool] = Field(
        default=None,
        description=("True when a Decision 31 check needs a person: a likely data-input "
                     "error, the model's known blind spot, or a recommendation resting on "
                     "one detail. A statement, not a routing: nothing is sent anywhere."),
    )
    explanation: Optional[dict] = Field(
        default=None,
        description=(
            "For the applicant (Decisions 33-35): the outcome (approved, declined when "
            "over the 45% affordability limit, or needs a manual review), what supported "
            "or counted against it in the applicant's own figures, the repayment plan when "
            "approved, and what could change the outcome. "
            "Model issues and the cutoff are not shown here. Each detail is compared with a "
            "typical applicant by changing that one detail and re-scoring; this "
            "describes how the model responds, not what causes default."),
    )
    risk_officer: Optional[dict] = Field(
        default=None,
        description="For a risk officer: error type, the three checks, divergence, context.")
    decision_record: Optional[dict] = Field(
        default=None,
        description=("Structured record for audit: model identity, cutoff, methods, rules "
                     "version, lineage. Approver identity is not recorded (Decision 23)."))
    effects_vs_typical: Optional[dict] = Field(
        default=None,
        description="Change in probability when each detail alone is set to typical (Decision 30).")
    explanation_note: Optional[str] = None


SCOPE_WARNING = (
    "Champion model is trained as a customer-level portfolio-risk model "
    "(Decision 18), not a validated application-time underwriting model. "
    "This score demonstrates live feature reconstruction and scoring; "
    "treat it as illustrative, not as evidence-backed application-time "
    "risk assessment. True application-time underwriting is deferred "
    "future work."
)


def lookup_existing_customer(customer_id: str) -> Optional[dict]:
    """
    Reads the customer's real, governed feature values directly
    from feature_store.customer_features rather than recomputing
    aggregates in this API, avoiding any risk of this endpoint's
    feature logic drifting from what the model was actually
    trained against.
    """
    conn = get_db_connection()
    try:
        query = (
            f"SELECT {', '.join(MODEL_FEATURES)} "
            f"FROM feature_store.customer_features WHERE customer_id = %s"
        )
        df = pd.read_sql(query, conn, params=(customer_id,))
    finally:
        conn.close()

    if df.empty:
        return None
    return df.iloc[0].to_dict()


def build_new_customer_features(request: ScoreRequest) -> dict:
    """
    First-application feature construction, delegated to the shared
    serving/features.py so this API and the Streamlit assessment page
    build features identically. Missing profile fields -> HTTP 422.
    """
    profile = {f: getattr(request, f) for f in PROFILE_FIELDS}
    if profile["thin_file_flag"] is None and None not in (
            profile["credit_score"], profile["months_credit_history"],
            profile["num_credit_accounts"]):
        profile["thin_file_flag"] = derive_thin_file(   # Decision 34: never asked
            profile["credit_score"], profile["months_credit_history"],
            profile["num_credit_accounts"], EXPLANATION_RULES)
    try:
        return shared_build_features(
            profile, request.order_value_gbp, request.n_instalments,
            request.merchant_category,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=422,
            detail=f"customer_id not found or not provided — {error.args[0]}",
        )

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def scoring_interface(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="score.html",
        context={
            "model_version": GOV_CFG["model_version"],
            "business_cutoff": BUSINESS_CUTOFF,
        },
    )


@app.get("/health", tags=["Status"], summary="Liveness check")
def health():
    return {"status": "ok"}


@app.get("/model/info", tags=["Status"], summary="Champion model identity and governance metadata")
def model_info():
    return {
        "model_version": GOV_CFG["model_version"],
        "champion_model": GOV_CFG["champion_model"],
        "feature_set_label": GOV_CFG["feature_set_label"],
        "decision_log_version": GOV_CFG["decision_log_version"],
        "business_cutoff": BUSINESS_CUTOFF,
        "n_features": len(MODEL_FEATURES),
        "trained_at": CHAMPION_METADATA.get("trained_at"),
        "scope_warning": SCOPE_WARNING,
    }


@app.post(
    "/score",
    response_model=ScoreResponse,
    tags=["Scoring"],
    summary="Score one application: existing customer history or new-customer profile",
)
def score(request: ScoreRequest):
    customer_type = "new"
    feature_values = None

    if request.customer_id:
        existing = lookup_existing_customer(request.customer_id)
        if existing is not None:
            feature_values = existing
            customer_type = "existing"

    if feature_values is None:
        feature_values = build_new_customer_features(request)
        customer_type = "new"

    missing = [f for f in MODEL_FEATURES if f not in feature_values]
    if missing:
        # The loaded champion needs a feature this API cannot supply
        # (e.g. after a rollback to a package with a different feature
        # list). Fail clearly instead of a bare KeyError.
        raise HTTPException(
            status_code=500,
            detail=f"Champion package expects features this API does not build: {missing}",
        )
    X = pd.DataFrame([{f: feature_values[f] for f in MODEL_FEATURES}])
    calibrated_probability = float(MODEL.predict_proba(X)[0, 1])
    decision = "flag_high_risk" if calibrated_probability >= BUSINESS_CUTOFF else "pass"

    # Decisions 30-31: three outputs (applicant, risk officer, audit record).
    outputs, note = {}, None
    if RULES_MODEL_VERSION != GOV_CFG["model_version"]:
        note = (f"Explanations unavailable: serving/explanation_rules.json was built for "
                f"{RULES_MODEL_VERSION}, the champion is {GOV_CFG['model_version']}. "
                "Re-run models/build_explanation_rules.py.")
    else:
        def score_many(feature_dicts):
            frame = pd.DataFrame([{f: d[f] for f in MODEL_FEATURES} for d in feature_dicts])
            return MODEL.predict_proba(frame)[:, 1].tolist()

        if customer_type == "new":
            raw = {f: getattr(request, f) for f in PROFILE_FIELDS}
            raw.update(order_value_gbp=request.order_value_gbp,
                       n_instalments=request.n_instalments,
                       merchant_category=request.merchant_category)
            outputs = explain_reasons(raw, score_many, BUSINESS_CUTOFF)
        else:
            outputs = explain_existing(
                feature_values, score_many, BUSINESS_CUTOFF,
                order={"order_value_gbp": request.order_value_gbp,
                       "n_instalments": request.n_instalments})
        outputs["decision_record"].update(
            customer_id=request.customer_id,
            calibrated_probability=round(calibrated_probability, 6),
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    return ScoreResponse(
        customer_id=request.customer_id,
        customer_type=customer_type,
        calibrated_probability=round(calibrated_probability, 4),
        business_cutoff=BUSINESS_CUTOFF,
        decision=decision,
        model_version=GOV_CFG["model_version"],
        feature_set_label=GOV_CFG["feature_set_label"],
        feature_values_used={f: feature_values[f] for f in MODEL_FEATURES},
        scope_warning=SCOPE_WARNING,
        risk_officer_review_required=(outputs["risk_officer"]["review_required"]
                                      if outputs else None),
        explanation=outputs.get("explanation"),
        risk_officer=outputs.get("risk_officer"),
        decision_record=outputs.get("decision_record"),
        effects_vs_typical=({k: round(v, 6) for k, v in outputs["effects"].items()}
                            if outputs else None),
        explanation_note=note,
    )
from serving.docs_ui import add_custom_docs

add_custom_docs(app)
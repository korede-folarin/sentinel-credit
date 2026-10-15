"""
Sentinel Credit, Dashboard Scoring Service

Scores one application with the governed champion, for the Credit Risk
Assessment page. Business logic only; no Streamlit layout here.

- The champion is loaded from config.yaml model_governance.artifacts_dir,
  the same package the FastAPI service serves.
- New-customer features come from serving/features.py, shared with the
  API, so both score an applicant identically.
- The decision uses config.yaml model.business_cutoff (>=, as everywhere).

Context added to a score is drawn only from real evidence:
- percentile: share of the champion's held-out test predictions below
  this score (models/<package>/test_predictions_stratified.csv);
- drivers: SHAP values from the champion's explanation model (raw risk
  margin, not the calibrated probability; stated wherever shown);
- range checks: inputs outside the training population's observed range;
- outputs (new applicants and existing customers): the applicant's
  explanation, the risk officer checks and the decision record, from
  serving/reasons.py, the same module and wording the API and the public
  site use (Decisions 30, 31).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from serving.features import (  # noqa: E402
    MERCHANT_CATEGORIES, PROFILE_FIELDS, build_new_customer_features,
)
from serving.reasons import (  # noqa: E402
    explain as _explain_new, explain_existing as _explain_existing, load_typical,
)
from utils import load_config, query_db  # noqa: E402

__all__ = ["MERCHANT_CATEGORIES", "PROFILE_FIELDS", "champion", "score",
           "lookup_customer", "training_ranges", "explain_new", "explain_existing",
           "typical_applicant"]


def _package_dir() -> str:
    cfg = load_config()
    return os.path.join(PROJECT_ROOT, cfg["model_governance"].get("artifacts_dir",
                                                                    "models/artifacts"))


@st.cache_resource(show_spinner=False)
def champion() -> dict:
    """Loads the governed champion package once per server process."""
    pkg = _package_dir()
    with open(os.path.join(pkg, "feature_list.json"), encoding="utf-8") as f:
        features = json.load(f)["features"]
    with open(os.path.join(pkg, "champion_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    preds = pd.read_csv(os.path.join(pkg, "test_predictions_stratified.csv"),
                        usecols=["proba_xgboost"])
    return {
        "features": features,
        "metadata": metadata,
        "model": joblib.load(os.path.join(pkg, "champion_calibrated.joblib")),
        "explainer_model": joblib.load(os.path.join(pkg, "champion_explanation_base.joblib")),
        "heldout_scores": np.sort(preds["proba_xgboost"].to_numpy()),
        "package_dir": os.path.relpath(pkg, PROJECT_ROOT),
    }


@st.cache_resource(show_spinner=False)
def _explainer():
    import shap  # heavy import, only when an explanation is requested
    return shap.TreeExplainer(champion()["explainer_model"])


@st.cache_data(ttl=3600, show_spinner=False)
def training_ranges() -> pd.DataFrame:
    """Observed min / median / max of each model feature in the champion's
    canonical training partition, read live from the feature store."""
    feats = champion()["features"]
    ids = pd.read_csv(os.path.join(_package_dir(), "split_ids_train.csv"))["customer_id"]
    frame = query_db(
        f"SELECT customer_id, {', '.join(feats)} FROM feature_store.customer_features "
        "WHERE has_outcome = 1"
    )
    frame = frame[frame["customer_id"].astype(str).isin(set(ids.astype(str)))]
    return frame[feats].agg(["min", "median", "max"]).T


def lookup_customer(customer_id: str):
    """Existing customer's governed feature row, or None if not found."""
    feats = champion()["features"]
    frame = query_db(
        f"SELECT {', '.join(feats)} FROM feature_store.customer_features "
        "WHERE customer_id = %s",
        (customer_id.strip().upper(),),
    )
    return None if frame.empty else frame.iloc[0].to_dict()


def score(feature_values: dict, explain: bool = True) -> dict:
    """Calibrated PD, decision at the configured cutoff, and evidence context."""
    champ = champion()
    feats = champ["features"]
    missing = [f for f in feats if f not in feature_values]
    if missing:
        raise KeyError(f"The champion expects features this page cannot build: {missing}")

    X = pd.DataFrame([{f: feature_values[f] for f in feats}])
    pd_value = float(champ["model"].predict_proba(X)[0, 1])
    cutoff = float(load_config()["model"]["business_cutoff"])
    heldout = champ["heldout_scores"]

    result = {
        "probability": pd_value,
        "cutoff": cutoff,
        "decision": "flag_high_risk" if pd_value >= cutoff else "pass",
        "percentile": float(np.searchsorted(heldout, pd_value, side="right") / len(heldout)),
        "heldout_n": int(len(heldout)),
        "features": {f: feature_values[f] for f in feats},
        "drivers": None,
    }
    if explain:
        values = _explainer().shap_values(X)
        values = values[1] if isinstance(values, list) else values
        result["drivers"] = (
            pd.DataFrame({"feature": feats, "value": X.iloc[0].values,
                          "shap": np.asarray(values)[0]})
            .assign(abs_shap=lambda d: d["shap"].abs())
            .sort_values("abs_shap", ascending=False)
            .drop(columns="abs_shap")
        )
    return result


def new_customer(profile: dict, order_value_gbp: float, n_instalments: int,
                 merchant_category: str) -> dict:
    return build_new_customer_features(profile, order_value_gbp, n_instalments,
                                       merchant_category)


def typical_applicant() -> dict:
    """The typical applicant the reasons compare against (serving/typical_applicant.json)."""
    return load_typical()


def _score_many():
    champ = champion()
    feats = champ["features"]

    def score_many(feature_dicts):
        frame = pd.DataFrame([{f: d[f] for f in feats} for d in feature_dicts])
        return champ["model"].predict_proba(frame)[:, 1].tolist()
    return score_many


def explain_new(inputs: dict) -> dict:
    """Three outputs for a new applicant (Decisions 30, 31).

    `inputs` holds the PROFILE_FIELDS plus order_value_gbp, n_instalments and
    merchant_category. Each input is compared with the typical applicant by
    changing that one input and re-scoring with the champion.
    """
    cutoff = float(load_config()["model"]["business_cutoff"])
    return _explain_new(inputs, _score_many(), cutoff)


def explain_existing(features: dict) -> dict:
    """Three outputs for an existing customer, from their 13 model features."""
    cutoff = float(load_config()["model"]["business_cutoff"])
    return _explain_existing(features, _score_many(), cutoff)

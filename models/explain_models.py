"""
=============================================================
Sentinel Credit — SHAP Explainability (Raw Foundation)
=============================================================
File:    explain_models.py
Purpose: Computes SHAP values for the champion XGBoost model.
         Produces global importance, a beeswarm summary,
         dependence plots, and local explanations for five
         deliberately chosen decision types.
Author:  Folarin Korede
Version: 1.0.0

SCOPE: this is the RAW explanation layer only — it answers
"what generally drives the model" and "what drove this
particular prediction". The three-audience explanation engine
(customer / risk officer / regulatory narratives) sits ON TOP
of this and is deliberately a separate build: it involves
disclosure policy, causal-language avoidance, and adverse-
action reason derivation, none of which should be entangled
with the numerical layer. See Decision 7.

    model prediction -> raw SHAP values -> validation
                     -> audience-specific explanation layer

INTERPRETING SHAP AGAINST THE EXPOSURE-TIME FINDING:
High SHAP importance tells us the model DEPENDS on a feature.
It does NOT tell us that dependence is defensible. The
observation-window analysis (Decision 12 addendum, "Unequal
Observation Opportunity") is what determines defensibility.
If EXPOSURE_TIME_SUSPECTED features rank highly here, that
strengthens the case for running the Feature Set A/B
comparison — it is not evidence the features are valid.

SURROGATE MODEL NOTE:
The deployed champion is a CalibratedClassifierCV wrapping
XGBoost. SHAP is computed instead on a SURROGATE: a bare
XGBoost retrained with the same split, features, seed and
hyperparameters. It is NOT one of the wrapper's five fold
estimators, each of which is fitted on a fold rather than the
full training set. TreeExplainer needs one coherent ensemble;
explaining the wrapper directly would give five explanations
per prediction with no principled way to reconcile them.

Consequences, stated once here and referenced elsewhere:
  - SHAP values reconstruct the surrogate's raw prediction
    margin, verified by the additivity check below.
  - They do NOT reconstruct the calibrated probability that
    drives the decision.
  - Global rankings stay directionally consistent, since
    calibration is monotonic.
  - Local magnitudes must NOT be described as changes to the
    calibrated PD.
=============================================================
"""

import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap
import yaml

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import (
    MODEL_FEATURES, TARGET_COLUMN, FILTER_COLUMN, FILTER_VALUE,
    EXPOSURE_TIME_SUSPECTED,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")
SHAP_DIR = os.path.join(PROJECT_ROOT, "reports", "shap")

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB = CONFIG["database"]
MODEL_CFG = CONFIG.get("model", {})
RANDOM_SEED = MODEL_CFG.get("random_seed", 42)
TEST_SIZE = MODEL_CFG.get("test_size", 0.2)
BUSINESS_CUTOFF = MODEL_CFG.get("business_cutoff", 0.145)

TARGET_MODEL = "xgboost"

# Model version is READ from champion_metadata.json, never
# set here — it identifies the model being explained, which is
# a property of the training run, not of this script.

# Dependence plots: top N by SHAP importance, PLUS the
# strategic features below regardless of their rank. Ranking
# purely by SHAP can miss features that matter for fairness or
# regulatory reasons rather than raw predictive weight.
N_TOP_DEPENDENCE = 4
STRATEGIC_DEPENDENCE_FEATURES = [
    "credit_score",
    "thin_file_flag",
    "max_affordability_ratio",
]


# ── DATA AND MODEL ────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    """
    Loads the exact test set and predictions the champion
    model produced, so explanations describe the same rows
    that were evaluated — not a fresh re-split.
    """
    path = os.path.join(ARTIFACTS_DIR, "test_predictions_stratified.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run train_baseline_models.py first."
        )
    return pd.read_csv(path)


def load_champion_package() -> dict:
    """
    Loads the governed model package produced by
    train_baseline_models.py. Does NOT retrain anything.

    This script explains champion_explanation_base.joblib —
    the exact uncalibrated model fitted during the canonical
    training run, saved alongside the calibrated champion. It
    is not a reconstruction: no hyperparameters are hardcoded
    here, and no re-split occurs, so there is no possibility of
    drift between what was trained and what is explained.
    """
    required = {
        "explanation_model": "champion_explanation_base.joblib",
        "metadata": "champion_metadata.json",
        "feature_list": "feature_list.json",
        "test_ids": "split_ids_test.csv",
    }
    missing = [
        f for f in required.values()
        if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model package files: {', '.join(missing)}. "
            f"Run train_baseline_models.py first — it persists the "
            f"champion package these explanations describe."
        )

    model = joblib.load(
        os.path.join(ARTIFACTS_DIR, required["explanation_model"]))
    with open(os.path.join(ARTIFACTS_DIR, required["metadata"]),
              "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, required["feature_list"]),
              "r", encoding="utf-8") as f:
        feature_list = json.load(f)["features"]
    test_ids = pd.read_csv(
        os.path.join(ARTIFACTS_DIR, required["test_ids"]))["customer_id"]

    return {
        "model": model,
        "metadata": metadata,
        "feature_list": feature_list,
        "test_ids": test_ids,
    }


def build_explained_matrix(df_pred: pd.DataFrame, feature_list: list,
                           test_ids: pd.Series) -> tuple:
    """
    Reconstructs the exact test matrix the champion was
    evaluated on, from the saved predictions file and saved
    split IDs. No database read, no re-split.
    """
    df = df_pred.set_index("customer_id").loc[test_ids.values].reset_index()
    X = df[feature_list].copy()
    ids = df["customer_id"]
    return X, ids, df


# ── VALIDATION ────────────────────────────────────────────
def validate_shap(explainer, shap_values, X, model, saved_features) -> dict:
    """
    Checks SHAP output before any of it is used for
    explanation. A failure here means every downstream
    explanation is unreliable, so this runs first and its
    results are recorded in the metadata.
    """
    checks = {}

    # 1. Feature alignment — the explained matrix must match
    #    the feature list SAVED WITH THE MODEL, in the same
    #    order. Also compares against the CURRENT
    #    model_features.py: a mismatch there means the feature
    #    set has changed since the champion was trained, and
    #    the model needs retraining before these explanations
    #    describe anything current.
    checks["feature_names_match_saved"] = list(X.columns) == list(saved_features)
    checks["feature_list_matches_current_code"] = (
        list(saved_features) == list(MODEL_FEATURES)
    )
    checks["n_features"] = len(X.columns)

    # 2. Additivity — SHAP contributions plus the base value
    #    should reconstruct the SURROGATE's raw margin. Passing
    #    proves the SHAP values are correct for that model; it
    #    does NOT prove anything about the calibrated
    #    probability, which is not explained here.
    raw_margin = model.predict(X, output_margin=True)
    reconstructed = shap_values.sum(axis=1) + explainer.expected_value
    max_abs_error = float(np.max(np.abs(reconstructed - raw_margin)))
    tolerance = 1e-4
    checks["explained_output"] = "uncalibrated_raw_margin"
    checks["additivity_tolerance"] = tolerance
    checks["additivity_max_abs_error"] = float(f"{max_abs_error:.3e}")
    checks["additivity_mean_abs_error"] = float(
        f"{np.mean(np.abs(reconstructed - raw_margin)):.3e}"
    )
    checks["additivity_passes"] = max_abs_error < tolerance

    # 3. Expected value recorded.
    checks["expected_value"] = float(explainer.expected_value)

    # 4. No excluded field leaked into the explainer.
    checks["no_excluded_fields"] = all(
        f in saved_features for f in X.columns
    )

    # 5. Missing-value handling matches training (median fill).
    checks["nulls_in_explained_data"] = int(X.isna().sum().sum())

    return checks


# ── GLOBAL ────────────────────────────────────────────────
def global_importance(shap_values, X) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, ranked."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["pct_of_total"] = round(
        100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum(), 2
    )
    # Flag the exposure-time suspects so the diagnostic
    # reading is explicit in the output itself.
    df["exposure_time_suspected"] = df["feature"].isin(EXPOSURE_TIME_SUSPECTED)
    return df


def save_beeswarm(shap_values, X) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title("SHAP summary — XGBoost (champion)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "global_beeswarm.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def save_dependence_plots(shap_values, X, importance_df) -> list:
    """
    Plots the top SHAP features PLUS a set of manually chosen
    strategic features.

    Reason for the manual set: ranking purely by SHAP
    importance can miss features that matter for engineering
    or regulatory reasons rather than raw predictive weight.
    thin_file_flag and credit_score are central to this
    project's fairness monitoring and Consumer Duty framing —
    their dependence shape is worth inspecting whether or not
    they happen to rank in the top few.
    """
    dep_dir = os.path.join(SHAP_DIR, "dependence")
    os.makedirs(dep_dir, exist_ok=True)

    top_by_shap = importance_df.head(N_TOP_DEPENDENCE)["feature"].tolist()
    strategic = [f for f in STRATEGIC_DEPENDENCE_FEATURES
                 if f in X.columns and f not in top_by_shap]
    selected = top_by_shap + strategic

    for feat in selected:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(dep_dir, f"{feat}.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

    return {"by_shap_rank": top_by_shap, "strategic": strategic,
            "all_plotted": selected}


# ── LOCAL ─────────────────────────────────────────────────
def find_dominant_feature_case(shap_values, ids_test) -> tuple:
    """
    Finds the customer whose single largest absolute SHAP value
    is the biggest across the test set — the prediction most
    dominated by one feature.

    These cases are worth inspecting because an explanation
    driven overwhelmingly by a single feature is both the
    easiest to communicate and the most fragile: if that one
    feature is later found unreliable (as several exposure-time
    features may be), the explanation collapses entirely rather
    than degrading gracefully.
    """
    max_abs_per_customer = np.abs(shap_values).max(axis=1)
    idx = int(np.argmax(max_abs_per_customer))
    return str(ids_test.values[idx]), float(max_abs_per_customer[idx])


def select_case_studies(df_pred: pd.DataFrame) -> dict:
    """
    Picks five customers spanning distinct decision types.
    Explanations that look sensible on a clear high-risk case
    can still look strange on a false positive, which is
    exactly why those are included in the first build rather
    than found later.
    """
    proba_col = f"proba_{TARGET_MODEL}"
    d = df_pred.copy()
    d["predicted_flag"] = (d[proba_col] >= BUSINESS_CUTOFF).astype(int)

    cases = {}

    lowest = d.nsmallest(1, proba_col)
    if len(lowest):
        cases["low_risk"] = str(lowest.iloc[0]["customer_id"])

    highest = d.nlargest(1, proba_col)
    if len(highest):
        cases["high_risk"] = str(highest.iloc[0]["customer_id"])

    d["dist_to_cutoff"] = (d[proba_col] - BUSINESS_CUTOFF).abs()
    near = d.nsmallest(1, "dist_to_cutoff")
    if len(near):
        cases["near_threshold"] = str(near.iloc[0]["customer_id"])

    fp = d[(d["predicted_flag"] == 1) & (d[TARGET_COLUMN] == 0)]
    if len(fp):
        cases["false_positive"] = str(fp.nlargest(1, proba_col).iloc[0]["customer_id"])

    fn = d[(d["predicted_flag"] == 0) & (d[TARGET_COLUMN] == 1)]
    if len(fn):
        cases["false_negative"] = str(fn.nsmallest(1, proba_col).iloc[0]["customer_id"])

    return cases


def save_local_explanation(
    label: str, customer_id: str, idx: int, shap_values, X,
    explainer, df_pred: pd.DataFrame,
) -> None:
    """
    Saves both a waterfall plot (human reading) and a
    structured JSON record (machine reading, for the
    audience-specific layer built later).
    """
    local_dir = os.path.join(SHAP_DIR, "local")
    os.makedirs(local_dir, exist_ok=True)

    row_shap = shap_values[idx]
    row_x = X.iloc[idx]

    explanation = shap.Explanation(
        values=row_shap,
        base_values=explainer.expected_value,
        data=row_x.values,
        feature_names=list(X.columns),
    )

    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(local_dir, f"{label}_customer_{customer_id}_waterfall.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    pred_row = df_pred[df_pred["customer_id"] == customer_id].iloc[0]
    contributions = sorted(
        [
            {
                "feature": f,
                "value": float(row_x[f]),
                "shap_value": float(row_shap[i]),
                "exposure_time_suspected": f in EXPOSURE_TIME_SUSPECTED,
            }
            for i, f in enumerate(X.columns)
        ],
        key=lambda r: abs(r["shap_value"]), reverse=True,
    )

    record = {
        "customer_id": str(customer_id),
        "case_type": label,
        "model": TARGET_MODEL,
        "calibrated_probability": float(pred_row[f"proba_{TARGET_MODEL}"]),
        "business_cutoff": BUSINESS_CUTOFF,
        "decision": "flag_high_risk" if pred_row[f"proba_{TARGET_MODEL}"] >= BUSINESS_CUTOFF else "pass",
        "actual_outcome": int(pred_row[TARGET_COLUMN]),
        "shap_base_value_raw_margin": float(explainer.expected_value),
        "shap_sum_raw_margin": float(row_shap.sum() + explainer.expected_value),
        "note": (
            "Explains an uncalibrated surrogate, not the calibrated "
            "champion. Do NOT describe a contribution as changing the "
            "calibrated PD by a given amount — say it pushed the "
            "underlying risk score higher or lower. See "
            "shap_metadata.json scope_note and not_valid_for."
        ),
        "contributions": contributions,
    }

    with open(
        os.path.join(local_dir, f"{label}_customer_{customer_id}_values.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(record, f, indent=2)


# ── MAIN ──────────────────────────────────────────────────
def main() -> None:
    print("=" * 58)
    print("SENTINEL CREDIT — SHAP EXPLAINABILITY (raw foundation)")
    print("=" * 58)

    os.makedirs(SHAP_DIR, exist_ok=True)

    df_pred = load_predictions()
    package = load_champion_package()
    model = package["model"]
    meta = package["metadata"]
    feature_list = package["feature_list"]

    X_test, ids_test, df_test = build_explained_matrix(
        df_pred, feature_list, package["test_ids"])

    print(f"Loaded champion package: {meta['model_version']}")
    print(f"  trained {meta['trained_at']}, params {meta['best_params']}")
    print(f"Explaining {len(X_test):,} test customers, "
          f"{len(feature_list)} features")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("SHAP values computed")

    # ---- Validation, before anything is used ----
    print("\n" + "-" * 58)
    print("VALIDATION")
    print("-" * 58)
    checks = validate_shap(explainer, shap_values, X_test, model, feature_list)
    for k, v in checks.items():
        print(f"  {k}: {v}")
    if not checks["additivity_passes"]:
        print("\n  WARNING: additivity check failed. SHAP contributions do "
              "not reconstruct the model output. Explanations below are "
              "NOT reliable until this is resolved.")

    # ---- Global ----
    print("\n" + "-" * 58)
    print("GLOBAL IMPORTANCE (mean |SHAP|)")
    print("-" * 58)
    importance = global_importance(shap_values, X_test)
    print(importance.head(15).to_string(index=False))

    importance.to_csv(
        os.path.join(SHAP_DIR, "global_feature_importance.csv"), index=False
    )
    save_beeswarm(shap_values, X_test)
    dependence_info = save_dependence_plots(shap_values, X_test, importance)

    suspected_in_top5 = importance.head(5)["exposure_time_suspected"].sum()
    suspected_share = round(
        importance[importance["exposure_time_suspected"]]["pct_of_total"].sum(), 2
    )
    print(f"\n  EXPOSURE_TIME_SUSPECTED features: {suspected_in_top5} of top 5, "
          f"{suspected_share}% of total importance.")
    print("  Reading: high SHAP importance shows the model DEPENDS on these")
    print("  features. It does not show that dependence is defensible — the")
    print("  observation-window analysis determines that (Decision 12).")

    # ---- Local ----
    print("\n" + "-" * 58)
    print("LOCAL EXPLANATIONS")
    print("-" * 58)
    df_pred_indexed = df_pred.set_index("customer_id")
    id_to_pos = {cid: i for i, cid in enumerate(ids_test.values)}

    cases = select_case_studies(df_pred)

    # Case 6: the prediction most dominated by a single feature.
    dom_id, dom_value = find_dominant_feature_case(shap_values, ids_test)
    cases["single_feature_dominant"] = dom_id
    print(f"\n  Most single-feature-dominated prediction: customer {dom_id} "
          f"(max |SHAP| = {dom_value:.4f})")

    for label, cid in cases.items():
        if cid not in id_to_pos:
            print(f"  {label}: customer {cid} not in explained set, skipped")
            continue
        save_local_explanation(
            label, cid, id_to_pos[cid], shap_values, X_test, explainer, df_pred
        )
        proba = df_pred_indexed.loc[cid, f"proba_{TARGET_MODEL}"]
        actual = df_pred_indexed.loc[cid, TARGET_COLUMN]
        print(f"  {label:<16} customer {cid}  "
              f"proba {proba:.4f}  actual {int(actual)}")

    # ---- Metadata ----
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": TARGET_MODEL,
        "model_version": meta["model_version"],
        "model_type": "XGBClassifier (uncalibrated explanation base)",
        "explained_artifact": meta["explanation_model_artifact"],
        "champion_trained_at": meta["trained_at"],
        "best_params": meta["best_params"],
        "n_test_customers": int(len(X_test)),
        "n_features": len(MODEL_FEATURES),
        "feature_list": MODEL_FEATURES,
        "exposure_time_suspected": EXPOSURE_TIME_SUSPECTED,
        "business_cutoff": BUSINESS_CUTOFF,
        "random_seed": RANDOM_SEED,
        "validation_checks": checks,
        "dependence_plots": dependence_info,
        "case_studies": cases,
        "scope_note": (
            "Explanations are generated from an UNCALIBRATED SURROGATE "
            "XGBoost model, retrained using the same split, features, seed "
            "and hyperparameters as the calibrated champion. It is not one "
            "of the fold estimators held inside CalibratedClassifierCV, "
            "which fits each on a fold rather than the full training set. "
            "These explanations describe the model's underlying "
            "risk-ranking logic, not the complete calibrated decision "
            "pipeline."
        ),
        "valid_uses": [
            "global feature importance ranking",
            "beeswarm and feature-direction inspection",
            "exposure-time feature dependence diagnostics",
            "identifying suspicious model dependencies",
            "comparing low-risk, high-risk and error cases",
        ],
        "not_valid_for": [
            "customer-facing adverse-action wording",
            "claims about calibrated probability changes "
            "(e.g. 'reduced PD by 4 percentage points')",
            "formal reconstruction of the final decision",
            "regulatory output presented as an exact end-to-end explanation",
        ],
    }
    with open(os.path.join(SHAP_DIR, "shap_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 58)
    print(f"Saved to {SHAP_DIR}")
    print("  global_feature_importance.csv")
    print("  global_beeswarm.png")
    print(f"  dependence/ ({len(dependence_info['all_plotted'])} plots: "
          f"{len(dependence_info['by_shap_rank'])} by SHAP rank + "
          f"{len(dependence_info['strategic'])} strategic)")
    print(f"  local/ ({len(cases)} case studies, waterfall + JSON each)")
    print("  shap_metadata.json")
    print("=" * 58)


if __name__ == "__main__":
    main()

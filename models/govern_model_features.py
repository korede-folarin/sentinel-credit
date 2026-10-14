"""
=============================================================
Sentinel Credit — Feature Governance Review
=============================================================
File:    govern_model_features.py
Purpose: Builds a complete evidence pack for every feature
         currently in MODEL_FEATURES before the final feature
         set is frozen for this version.

Evidence produced for every current MODEL_FEATURES feature:
  - definition, source, decision-time availability and rationale
  - missingness and cardinality
  - Information Value (training partition only)
  - VIF and maximum pairwise correlation
  - SHAP rank and share from the governed champion explanation
  - temporal PSI against the out-of-time test population
  - leave-one-feature-out repeated CV: 5 folds x 3 repeats
  - paired performance deltas against the all-feature baseline
  - attention flags and a provisional evidence status

Important:
  - Final retain/remove decisions are NOT automated.
  - Hyperparameters are frozen from champion_metadata.json.
  - Ablation uses the exact canonical stratified training IDs.
  - Held-out stratified and temporal labels are not used to
    decide whether a feature improves repeated-CV performance.
  - Ablation calibration metrics describe the raw XGBoost
    model comparatively; the deployed champion is calibrated.
  - FEATURE_CATALOGUE may contain entries for features no
    longer in MODEL_FEATURES (retained as historical
    documentation of prior decisions, e.g. Decision 18's
    removals). Only entries for CURRENT MODEL_FEATURES are
    required and used — see require_catalogue_complete().
=============================================================
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import MODEL_FEATURES, TARGET_COLUMN, FILTER_COLUMN, FILTER_VALUE  # noqa: E402


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_DIR = os.path.join(PROJECT_ROOT, "reports", "shap")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "feature_governance")

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    CONFIG["model_governance"].get("artifacts_dir", "models/artifacts"),
)  # the governed champion package; see config.yaml model_governance

with open(PARAMS_PATH, "r", encoding="utf-8") as file:
    PARAMS = yaml.safe_load(file)

DB = CONFIG["database"]
MODEL_CFG = CONFIG["model"]
GOV_CFG = CONFIG["model_governance"]
MLFLOW_CFG = CONFIG["mlflow"]
FG_CFG = CONFIG["feature_governance"]

RANDOM_SEED = MODEL_CFG["random_seed"]
CHAMPION_MODEL = GOV_CFG["champion_model"]
MODEL_VERSION = GOV_CFG["model_version"]
FEATURE_SET_LABEL = GOV_CFG["feature_set_label"]
DATASET_VERSION = GOV_CFG["dataset_version"]
PIPELINE_STAGE = GOV_CFG["pipeline_stage"]

N_SPLITS = FG_CFG["n_splits"]
N_REPEATS = FG_CFG["n_repeats"]
IV_BINS = FG_CFG["iv_bins"]
PSI_BINS = FG_CFG["psi_bins"]
ECE_BINS = FG_CFG["ece_bins"]
XGBOOST_N_JOBS = FG_CFG["xgboost_n_jobs"]
MLFLOW_ARTIFACT_PATH = FG_CFG["mlflow_artifact_path"]

THRESHOLDS = FG_CFG["thresholds"]
IV_WEAK = THRESHOLDS["iv_weak"]
VIF_HIGH = THRESHOLDS["vif_high"]
CORRELATION_HIGH = THRESHOLDS["correlation_high"]
PSI_MODERATE = THRESHOLDS["psi_moderate"]
PSI_HIGH = THRESHOLDS["psi_high"]
SHAP_LOW_SHARE_PCT = THRESHOLDS["shap_low_share_pct"]
CV_PR_AUC_MATERIAL_DELTA = THRESHOLDS["cv_pr_auc_material_delta"]

MLFLOW_TRACKING_URI = MLFLOW_CFG["tracking_uri"]
MLFLOW_EXPERIMENT = MLFLOW_CFG["experiment_name"]

# Scenario name for the all-features baseline, derived from the
# CURRENT feature count rather than hardcoded — a literal
# "all_18_features" string here would go stale on every future
# freeze the same way the four-feature removal just made it
# stale (Decision 18: 18 -> 14). Every lookup of the baseline
# scenario below uses this constant, not a literal string.
BASELINE_SCENARIO_NAME = f"all_{len(MODEL_FEATURES)}_features"


FEATURE_CATALOGUE = {
    # NOTE: entries may exist here for features no longer in
    # MODEL_FEATURES (e.g. age, n_applications, n_late_payments,
    # bnpl_exposure_to_monthly_income_ratio — removed by Decision
    # 18). They are retained as historical documentation and are
    # simply not read, since every lookup below iterates over
    # MODEL_FEATURES, not over this dict's keys. Only a MISSING
    # entry for a current feature is an error — see
    # require_catalogue_complete().
    "age": {
        "source": "customer",
        "definition": "Customer age at the relevant scoring date.",
        "decision_time_status": "available",
        "business_rationale": "May capture lifecycle differences in credit capacity and behaviour.",
        "key_governance_risk": "Sensitive-characteristic review required; check necessity, policy basis and segment outcomes.",
        "fairness_consideration": "Review performance and outcomes across age bands before final retention.",
    },
    "monthly_income_gbp": {
        "source": "customer",
        "definition": "Declared or verified monthly income in GBP.",
        "decision_time_status": "available",
        "business_rationale": "Core affordability and repayment-capacity measure.",
        "key_governance_risk": "Income quality, verification status and reporting error may vary across customers.",
        "fairness_consideration": "Monitor whether income measurement quality differs systematically across groups.",
    },
    "existing_monthly_obligations_gbp": {
        "source": "customer",
        "definition": "Existing monthly debt or financial obligations before the new decision.",
        "decision_time_status": "available",
        "business_rationale": "Represents committed expenditure that reduces repayment capacity.",
        "key_governance_risk": "Requires complete and current obligations; stale values can understate affordability pressure.",
        "fairness_consideration": "Check data completeness across thin-file and vulnerability segments.",
    },
    "savings_balance_gbp": {
        "source": "customer",
        "definition": "Observed or declared savings balance in GBP.",
        "decision_time_status": "available",
        "business_rationale": "Represents financial resilience and capacity to absorb shocks.",
        "key_governance_risk": "May be volatile, incomplete or unavailable depending on account visibility.",
        "fairness_consideration": "Review whether data availability creates unequal treatment.",
    },
    "credit_score": {
        "source": "bureau",
        "definition": "External or simulated bureau-style credit score available at application.",
        "decision_time_status": "available",
        "business_rationale": "Summarises prior credit performance and bureau risk.",
        "key_governance_risk": "Strong overlap with thin-file and credit-history features; bureau coverage may vary.",
        "fairness_consideration": "Check performance and calibration for thin-file customers.",
    },
    "months_credit_history": {
        "source": "bureau",
        "definition": "Length of recorded credit history in months.",
        "decision_time_status": "available",
        "business_rationale": "Longer histories generally provide more evidence about repayment behaviour.",
        "key_governance_risk": "May proxy age or market access and overlap with thin_file_flag.",
        "fairness_consideration": "Review effects on younger and thin-file customers.",
    },
    "num_credit_accounts": {
        "source": "bureau",
        "definition": "Number of credit accounts visible at decision time.",
        "decision_time_status": "available",
        "business_rationale": "Represents breadth of prior credit use and possible leverage.",
        "key_governance_risk": "Relationship may be non-monotonic and depends on account type and status.",
        "fairness_consideration": "Check whether sparse bureau files are penalised simply for limited history.",
    },
    "num_defaults_24m": {
        "source": "bureau",
        "definition": "Count of prior defaults in the preceding 24 months.",
        "decision_time_status": "available",
        "business_rationale": "Direct measure of recent adverse credit history.",
        "key_governance_risk": "Must contain only defaults known before the scoring date.",
        "fairness_consideration": "Check data quality and reporting consistency across bureau coverage groups.",
    },
    "thin_file_flag": {
        "source": "bureau-derived",
        "definition": "Binary indicator that the customer has limited recorded credit history.",
        "decision_time_status": "available",
        "business_rationale": "Identifies limited evidence and uncertainty in conventional bureau assessment.",
        "key_governance_risk": "Largest SHAP driver in the baseline and correlated with credit_score/history; necessity must be demonstrated.",
        "fairness_consideration": "Explicitly compare error rates, calibration and outcomes for thin-file versus non-thin-file customers.",
    },
    "n_applications": {
        "source": "application-history",
        "definition": "Count of prior applications available before the scored decision.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Repeated application activity may indicate credit demand or financial stress.",
        "key_governance_risk": "Must exclude the future and ensure the current application is handled consistently.",
        "fairness_consideration": "Check whether channel or access differences drive application frequency.",
    },
    "max_affordability_ratio": {
        "source": "application-derived",
        "definition": "Maximum historical affordability ratio observable before the scored decision.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Captures peak financial stretch rather than only average affordability.",
        "key_governance_risk": "Must be computed from information available before decision and not from later applications.",
        "fairness_consideration": "Review sensitivity to income and obligations measurement quality.",
    },
    "avg_order_value_gbp": {
        "source": "approved-transaction-history",
        "definition": "Average value of prior approved BNPL orders.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Represents typical exposure size and purchasing behaviour.",
        "key_governance_risk": "Requires strict as-of aggregation; product and merchant mix may confound borrower risk.",
        "fairness_consideration": "Check whether merchant access or product mix creates segment differences.",
    },
    "avg_n_instalments": {
        "source": "approved-transaction-history",
        "definition": "Average number of instalments across prior approved transactions.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "May capture repayment horizon and product structure associated with risk.",
        "key_governance_risk": "Could represent product design or merchant policy more than intrinsic customer risk.",
        "fairness_consideration": "Review outcomes by product and merchant mix.",
    },
    "pct_high_risk_category": {
        "source": "merchant-category-history",
        "definition": "Share of prior approved transactions in categories classified as high risk.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Captures concentration in merchant categories associated with higher observed risk.",
        "key_governance_risk": "Category labels require governance and may drift; strict as-of aggregation is required.",
        "fairness_consideration": "Check for indirect socioeconomic or demographic proxy effects.",
    },
    "pct_medium_risk_category": {
        "source": "merchant-category-history",
        "definition": "Share of prior approved transactions in categories classified as medium risk.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Complements high-risk-category exposure while avoiding the exact low-risk identity.",
        "key_governance_risk": "Category labels require governance and may overlap with high-risk share.",
        "fairness_consideration": "Check for indirect socioeconomic or demographic proxy effects.",
    },
    "n_late_payments": {
        "source": "repayment-history",
        "definition": "Count of prior late-payment events observable before the scored decision.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Historical payment difficulty is a direct behavioural risk signal.",
        "key_governance_risk": "Must exclude repayment events occurring after the scoring date; zero SHAP in this baseline requires redundancy review.",
        "fairness_consideration": "Check whether differing observation history causes unequal opportunity to accumulate late events.",
    },
    "pct_late_payments": {
        "source": "repayment-history",
        "definition": "Share of prior observable repayments recorded as late.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Normalises late-payment behaviour for differing repayment volumes.",
        "key_governance_risk": "Denominator maturity and as-of treatment must be consistent.",
        "fairness_consideration": "Review minimum-history requirements and thin-file stability.",
    },
    "bnpl_exposure_to_monthly_income_ratio": {
        "source": "derived customer and transaction",
        "definition": "Prior BNPL exposure divided by monthly income.",
        "decision_time_status": "conditional_as_of_logic",
        "business_rationale": "Measures exposure relative to repayment capacity.",
        "key_governance_risk": "Numerator must be decision-time exposure only; overlaps with income and transaction-value components.",
        "fairness_consideration": "Check sensitivity to income verification and missing transaction history.",
    },
}


def require_catalogue_complete() -> None:
    """
    Only MISSING entries for CURRENT MODEL_FEATURES are an
    error. Extra catalogue entries for features that used to be
    in MODEL_FEATURES (e.g. Decision 18's four removals) are
    left in place deliberately as historical documentation —
    every consumer of FEATURE_CATALOGUE below iterates over
    MODEL_FEATURES and looks entries up by key, so unused extras
    are simply never read. This means a feature removal no
    longer requires a matching catalogue edit before this script
    will run.
    """
    missing = sorted(set(MODEL_FEATURES) - set(FEATURE_CATALOGUE))
    if missing:
        raise RuntimeError(
            "FEATURE_CATALOGUE is missing entries required for current "
            f"MODEL_FEATURES: {missing}\n"
            "Add a catalogue entry (source, definition, "
            "decision_time_status, business_rationale, "
            "key_governance_risk, fairness_consideration) for each "
            "before running governance."
        )


def load_modelable_data() -> pd.DataFrame:
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    connection = psycopg2.connect(
        host=DB["host"],
        port=DB["port"],
        dbname=DB["name"],
        user="postgres",
        password=password,
    )
    try:
        frame = pd.read_sql(
            f"SELECT * FROM {DB['schema']}.customer_features",
            connection,
        )
    finally:
        connection.close()

    frame["customer_id"] = frame["customer_id"].astype(str)
    return frame[frame[FILTER_COLUMN] == FILTER_VALUE].copy()


def load_governed_artifacts() -> dict:
    paths = {
        "metadata": os.path.join(ARTIFACTS_DIR, "champion_metadata.json"),
        "feature_list": os.path.join(ARTIFACTS_DIR, "feature_list.json"),
        "train_ids": os.path.join(ARTIFACTS_DIR, "split_ids_train.csv"),
        "temporal_predictions": os.path.join(
            ARTIFACTS_DIR, "test_predictions_temporal.csv"
        ),
        "shap_importance": os.path.join(
            SHAP_DIR, "global_feature_importance.csv"
        ),
    }

    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required governance inputs:\n  "
            + "\n  ".join(missing)
            + "\nComplete training, temporal evaluation and SHAP first."
        )

    with open(paths["metadata"], "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(paths["feature_list"], "r", encoding="utf-8") as file:
        saved_features = json.load(file)["features"]

    train_ids = pd.read_csv(paths["train_ids"])["customer_id"].astype(str)
    temporal = pd.read_csv(paths["temporal_predictions"])
    temporal["customer_id"] = temporal["customer_id"].astype(str)
    shap_importance = pd.read_csv(paths["shap_importance"])

    return {
        "metadata": metadata,
        "saved_features": saved_features,
        "train_ids": train_ids,
        "temporal": temporal,
        "shap_importance": shap_importance,
    }


def nested_value(mapping: dict, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def verify_identity(frame: pd.DataFrame, artifacts: dict) -> dict:
    metadata = artifacts["metadata"]
    dataset = metadata.get("dataset", {})
    saved_features = artifacts["saved_features"]
    train_ids = artifacts["train_ids"]
    temporal = artifacts["temporal"]
    shap_importance = artifacts["shap_importance"]

    mismatches = []

    comparisons = {
        "model_name": (metadata.get("model_name"), CHAMPION_MODEL),
        "model_version": (metadata.get("model_version"), MODEL_VERSION),
        "split_type": (metadata.get("split_type"), "stratified"),
        "dataset_version": (
            dataset.get("dataset_version", metadata.get("dataset_version")),
            DATASET_VERSION,
        ),
        "feature_set_label": (
            dataset.get("feature_set_label", metadata.get("feature_set_label")),
            FEATURE_SET_LABEL,
        ),
        "pipeline_stage": (
            dataset.get("pipeline_stage", metadata.get("pipeline_stage")),
            PIPELINE_STAGE,
        ),
        "data_end_date": (
            dataset.get("data_end_date"),
            CONFIG["data_generator"]["data_end_date"],
        ),
        "origination_end_date": (
            dataset.get("origination_end_date"),
            CONFIG["data_generator"]["origination_end_date"],
        ),
    }

    for name, (observed, expected) in comparisons.items():
        if observed is not None and str(observed) != str(expected):
            mismatches.append(
                f"{name}: saved={observed!r}, current={expected!r}"
            )

    if list(saved_features) != list(MODEL_FEATURES):
        mismatches.append(
            "Saved feature list does not match current MODEL_FEATURES."
        )

    if int(metadata.get("n_features", -1)) != len(MODEL_FEATURES):
        mismatches.append(
            f"n_features: saved={metadata.get('n_features')}, "
            f"current={len(MODEL_FEATURES)}"
        )

    saved_n_modelable = dataset.get("n_modelable")
    if saved_n_modelable is not None and int(saved_n_modelable) != len(frame):
        mismatches.append(
            f"n_modelable: saved={saved_n_modelable}, current={len(frame)}"
        )

    if train_ids.duplicated().any():
        mismatches.append("split_ids_train.csv contains duplicate IDs.")

    if not set(train_ids).issubset(set(frame["customer_id"])):
        mismatches.append(
            "Some saved canonical training IDs are missing from the feature store."
        )

    temporal_missing = sorted(set(MODEL_FEATURES) - set(temporal.columns))
    if temporal_missing:
        mismatches.append(
            f"Temporal prediction file is missing features: {temporal_missing}"
        )

    shap_missing = sorted(
        set(MODEL_FEATURES) - set(shap_importance["feature"])
    )
    if shap_missing:
        mismatches.append(
            f"SHAP importance file is missing features: {shap_missing}"
        )

    if mismatches:
        raise RuntimeError(
            "FEATURE GOVERNANCE IDENTITY CHECK FAILED:\n  "
            + "\n  ".join(mismatches)
        )

    return {
        "model_name": CHAMPION_MODEL,
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_set_label": FEATURE_SET_LABEL,
        "pipeline_stage": PIPELINE_STAGE,
        "n_modelable": len(frame),
        "n_train": len(train_ids),
        "n_temporal_test": len(temporal),
        "n_features": len(MODEL_FEATURES),
        "identity_check_passes": True,
    }


def reconstruct_training_matrix(
    frame: pd.DataFrame,
    train_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Mirrors training exactly: median fill is calculated on the
    full modelable population before selecting canonical train IDs.
    """
    if frame["customer_id"].duplicated().any():
        raise RuntimeError(
            "Feature store contains duplicate customer_id values."
        )

    full_features = frame[MODEL_FEATURES].copy()
    medians = full_features.median(numeric_only=True)
    full_features = full_features.fillna(medians)
    full_features.index = frame["customer_id"].values

    target = frame.set_index("customer_id")[TARGET_COLUMN].astype(int)

    X_train = full_features.loc[train_ids.values].copy()
    y_train = target.loc[train_ids.values].copy()

    if X_train.isna().any().any():
        null_columns = X_train.columns[X_train.isna().any()].tolist()
        raise RuntimeError(
            f"Nulls remain in canonical training matrix: {null_columns}"
        )

    return X_train, y_train


def ks_statistic(y_true: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(probability)
    y_sorted = y_true[order]
    total_bad = max(int(y_sorted.sum()), 1)
    total_good = max(int((1 - y_sorted).sum()), 1)
    cumulative_bad = np.cumsum(y_sorted) / total_bad
    cumulative_good = np.cumsum(1 - y_sorted) / total_good
    return float(np.max(np.abs(cumulative_bad - cumulative_good)))


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0

    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (
                (probability >= edges[index])
                & (probability <= edges[index + 1])
            )
        else:
            mask = (
                (probability >= edges[index])
                & (probability < edges[index + 1])
            )

        if not mask.any():
            continue

        ece += (
            mask.sum() / total
        ) * abs(y_true[mask].mean() - probability[mask].mean())

    return float(ece)


def score_predictions(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict:
    auc = roc_auc_score(y_true, probability)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "ks": ks_statistic(y_true, probability),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(
            y_true, probability, ECE_BINS
        ),
    }


def information_value(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int,
) -> float:
    data = pd.DataFrame({"feature": feature, "target": target}).copy()

    if data["feature"].nunique(dropna=True) <= n_bins:
        bucket = data["feature"].astype("object")
        bucket = bucket.where(data["feature"].notna(), "__MISSING__")
    else:
        ranked = data["feature"].rank(method="first")
        try:
            bucket = pd.qcut(
                ranked,
                q=n_bins,
                duplicates="drop",
            ).astype(str)
        except ValueError:
            bucket = data["feature"].astype(str)
        bucket = bucket.where(data["feature"].notna(), "__MISSING__")

    grouped = (
        pd.DataFrame({"bucket": bucket, "target": data["target"]})
        .groupby("bucket", dropna=False)["target"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "bad"})
    )
    grouped["good"] = grouped["count"] - grouped["bad"]

    smoothing = 0.5
    bad_distribution = (
        grouped["bad"] + smoothing
    ) / (
        grouped["bad"].sum() + smoothing * len(grouped)
    )
    good_distribution = (
        grouped["good"] + smoothing
    ) / (
        grouped["good"].sum() + smoothing * len(grouped)
    )

    woe = np.log(good_distribution / bad_distribution)
    iv = ((good_distribution - bad_distribution) * woe).sum()
    return float(iv)


def iv_band(iv: float) -> str:
    if iv < 0.02:
        return "very_weak"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "investigate_possible_leakage_or_instability"


def calculate_vif(X_train: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    rows = []
    for feature in scaled.columns:
        y_feature = scaled[feature].to_numpy()
        if np.isclose(np.var(y_feature), 0.0):
            vif = np.inf
        else:
            other_features = scaled.drop(columns=[feature])
            model = LinearRegression()
            model.fit(other_features, y_feature)
            r_squared = model.score(other_features, y_feature)
            vif = (
                np.inf
                if r_squared >= 0.999999
                else 1.0 / max(1.0 - r_squared, 1e-12)
            )

        rows.append({"feature": feature, "vif": float(vif)})

    return pd.DataFrame(rows)


def calculate_correlation_evidence(
    X_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = X_train.corr(method="pearson")
    rows = []

    for feature in MODEL_FEATURES:
        series = correlation[feature].drop(index=feature).abs()
        if series.empty:
            partner = None
            value = np.nan
        else:
            partner = str(series.idxmax())
            value = float(series.max())

        rows.append(
            {
                "feature": feature,
                "most_correlated_with": partner,
                "max_abs_correlation": value,
            }
        )

    return correlation, pd.DataFrame(rows)


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    n_bins: int,
) -> float:
    expected = pd.Series(expected).copy()
    actual = pd.Series(actual).copy()

    combined_unique = pd.concat([expected, actual]).nunique(dropna=True)

    if combined_unique <= n_bins:
        expected_bucket = expected.astype("object")
        actual_bucket = actual.astype("object")
        expected_bucket = expected_bucket.where(
            expected.notna(), "__MISSING__"
        )
        actual_bucket = actual_bucket.where(
            actual.notna(), "__MISSING__"
        )

        categories = sorted(
            set(expected_bucket.astype(str))
            | set(actual_bucket.astype(str))
        )
        expected_dist = (
            expected_bucket.astype(str)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0.0)
        )
        actual_dist = (
            actual_bucket.astype(str)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0.0)
        )
    else:
        non_missing_expected = expected.dropna()
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(non_missing_expected.quantile(quantiles).values)

        if len(edges) < 3:
            return 0.0

        edges[0] = -np.inf
        edges[-1] = np.inf

        expected_bucket = pd.cut(
            expected, bins=edges, include_lowest=True
        ).astype(str)
        actual_bucket = pd.cut(
            actual, bins=edges, include_lowest=True
        ).astype(str)

        expected_bucket = expected_bucket.where(
            expected.notna(), "__MISSING__"
        )
        actual_bucket = actual_bucket.where(
            actual.notna(), "__MISSING__"
        )

        categories = sorted(
            set(expected_bucket) | set(actual_bucket)
        )
        expected_dist = (
            expected_bucket.value_counts(normalize=True)
            .reindex(categories, fill_value=0.0)
        )
        actual_dist = (
            actual_bucket.value_counts(normalize=True)
            .reindex(categories, fill_value=0.0)
        )

    epsilon = 1e-6
    expected_dist = expected_dist.clip(lower=epsilon)
    actual_dist = actual_dist.clip(lower=epsilon)

    return float(
        ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()
    )


def psi_band(psi: float) -> str:
    if psi < PSI_MODERATE:
        return "stable"
    if psi < PSI_HIGH:
        return "moderate_shift"
    return "material_shift"


def build_static_evidence(
    frame: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    temporal: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    missingness = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "missing_rate": [
                float(frame[feature].isna().mean())
                for feature in MODEL_FEATURES
            ],
            "n_unique": [
                int(frame[feature].nunique(dropna=True))
                for feature in MODEL_FEATURES
            ],
            "data_type": [
                str(frame[feature].dtype)
                for feature in MODEL_FEATURES
            ],
        }
    )

    iv_rows = []
    for feature in MODEL_FEATURES:
        value = information_value(
            X_train[feature],
            y_train,
            IV_BINS,
        )
        iv_rows.append(
            {
                "feature": feature,
                "iv": value,
                "iv_band": iv_band(value),
            }
        )
    iv_frame = pd.DataFrame(iv_rows)

    vif_frame = calculate_vif(X_train)
    correlation_matrix, correlation_evidence = (
        calculate_correlation_evidence(X_train)
    )

    psi_rows = []
    for feature in MODEL_FEATURES:
        psi = population_stability_index(
            X_train[feature],
            temporal[feature],
            PSI_BINS,
        )
        psi_rows.append(
            {
                "feature": feature,
                "temporal_psi": psi,
                "psi_band": psi_band(psi),
            }
        )
    psi_frame = pd.DataFrame(psi_rows)

    shap = shap_importance[
        ["feature", "rank", "mean_abs_shap", "pct_of_total"]
    ].rename(
        columns={
            "rank": "shap_rank",
            "mean_abs_shap": "mean_abs_shap",
            "pct_of_total": "shap_pct_of_total",
        }
    )

    catalogue = pd.DataFrame(
        [
            {"feature": feature, **FEATURE_CATALOGUE[feature]}
            for feature in MODEL_FEATURES
        ]
    )

    review = catalogue
    for evidence in (
        missingness,
        iv_frame,
        vif_frame,
        correlation_evidence,
        psi_frame,
        shap,
    ):
        review = review.merge(evidence, on="feature", how="left")

    supporting = {
        "missingness": missingness,
        "iv": iv_frame,
        "vif": vif_frame,
        "correlation_matrix": correlation_matrix,
        "correlation_evidence": correlation_evidence,
        "psi": psi_frame,
        "shap": shap,
    }
    return review, supporting


def build_xgboost(best_params: dict, seed: int) -> XGBClassifier:
    params = dict(best_params)
    params.setdefault("n_jobs", XGBOOST_N_JOBS)
    params.setdefault("verbosity", 0)

    return XGBClassifier(
        random_state=seed,
        eval_metric="auc",
        **params,
    )


def run_repeated_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    fold_output_path: str,
    fresh: bool,
) -> pd.DataFrame:
    scenarios = [(BASELINE_SCENARIO_NAME, None)] + [
        (f"without__{feature}", feature)
        for feature in MODEL_FEATURES
    ]

    if fresh and os.path.exists(fold_output_path):
        os.remove(fold_output_path)

    if os.path.exists(fold_output_path):
        completed = pd.read_csv(fold_output_path)
    else:
        completed = pd.DataFrame()

    # A cached fold file from a DIFFERENT feature-set version (e.g.
    # the prior 18-feature governance run) is NOT safely identified
    # by scenario name alone: a "without__X" scenario name can be
    # identical across two feature-set versions (X still exists in
    # both) while meaning a different experiment each time (leaving
    # one feature out of 18 vs out of 14 fits a model with a
    # different n_features). Filtering on scenario name alone would
    # silently keep stale rows wearing a name that now means
    # something else. Require BOTH the scenario name and its
    # expected n_features to match this run's current scenario set.
    expected_n_features = {
        name: (len(MODEL_FEATURES) - (0 if removed is None else 1))
        for name, removed in scenarios
    }
    if not completed.empty:
        expected_series = completed["scenario"].map(expected_n_features)
        stale_mask = (
            expected_series.isna()
            | (completed["n_features"] != expected_series)
        )
        n_stale = int(stale_mask.sum())
        if n_stale:
            print(
                f"  Discarding {n_stale} cached fold rows from a prior "
                f"feature-set version (scenario name and/or n_features "
                f"don't match the current {len(expected_n_features)}-scenario "
                f"set) — {fold_output_path} will be rewritten without them."
            )
            completed = completed[~stale_mask].reset_index(drop=True)

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )
    split_indices = list(splitter.split(X_train, y_train))

    total_fits = len(scenarios) * len(split_indices)
    completed_keys = set()

    if not completed.empty:
        completed_keys = set(
            zip(
                completed["scenario"],
                completed["repeat"],
                completed["fold"],
            )
        )

    rows = completed.to_dict("records") if not completed.empty else []
    fit_counter = len(rows)

    for scenario, removed_feature in scenarios:
        feature_list = [
            feature
            for feature in MODEL_FEATURES
            if feature != removed_feature
        ]

        for split_number, (fit_index, valid_index) in enumerate(
            split_indices
        ):
            repeat = split_number // N_SPLITS + 1
            fold = split_number % N_SPLITS + 1
            key = (scenario, repeat, fold)

            if key in completed_keys:
                continue

            fit_counter += 1
            print(
                f"[{fit_counter:>3}/{total_fits}] "
                f"{scenario} — repeat {repeat}/{N_REPEATS}, "
                f"fold {fold}/{N_SPLITS}"
            )

            X_fit = X_train.iloc[fit_index][feature_list]
            y_fit = y_train.iloc[fit_index]
            X_valid = X_train.iloc[valid_index][feature_list]
            y_valid = y_train.iloc[valid_index]

            model = build_xgboost(
                best_params,
                seed=RANDOM_SEED + split_number,
            )
            model.fit(X_fit, y_fit)

            fit_probability = model.predict_proba(X_fit)[:, 1]
            valid_probability = model.predict_proba(X_valid)[:, 1]

            fit_metrics = score_predictions(
                y_fit.to_numpy(),
                fit_probability,
            )
            valid_metrics = score_predictions(
                y_valid.to_numpy(),
                valid_probability,
            )

            row = {
                "scenario": scenario,
                "removed_feature": (
                    removed_feature
                    if removed_feature is not None
                    else ""
                ),
                "repeat": repeat,
                "fold": fold,
                "n_features": len(feature_list),
                "n_fit": len(X_fit),
                "n_validation": len(X_valid),
                "fit_bad_rate": float(y_fit.mean()),
                "validation_bad_rate": float(y_valid.mean()),
                **{
                    f"train_{metric}": value
                    for metric, value in fit_metrics.items()
                },
                **{
                    f"validation_{metric}": value
                    for metric, value in valid_metrics.items()
                },
                "train_validation_auc_gap": (
                    fit_metrics["auc"] - valid_metrics["auc"]
                ),
                "train_validation_gini_gap": (
                    fit_metrics["gini"] - valid_metrics["gini"]
                ),
            }

            rows.append(row)
            pd.DataFrame(rows).to_csv(
                fold_output_path,
                index=False,
            )

    return pd.DataFrame(rows)


def summarise_ablation(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "validation_auc",
        "validation_gini",
        "validation_pr_auc",
        "validation_ks",
        "validation_brier",
        "validation_ece",
        "train_validation_auc_gap",
        "train_validation_gini_gap",
    ]

    summary = (
        fold_results.groupby(
            ["scenario", "removed_feature", "n_features"],
            dropna=False,
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            [str(part) for part in column if str(part)]
        ).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    baseline_folds = fold_results[
        fold_results["scenario"] == BASELINE_SCENARIO_NAME
    ][
        ["repeat", "fold"]
        + metric_columns
    ].rename(
        columns={
            column: f"baseline_{column}"
            for column in metric_columns
        }
    )

    paired = fold_results.merge(
        baseline_folds,
        on=["repeat", "fold"],
        how="left",
    )

    for metric in metric_columns:
        paired[f"delta_{metric}"] = (
            paired[metric] - paired[f"baseline_{metric}"]
        )

    paired_rows = []
    for scenario, group in paired.groupby("scenario"):
        removed_feature = group["removed_feature"].iloc[0]
        row = {
            "scenario": scenario,
            "removed_feature": removed_feature,
            "n_paired_folds": len(group),
        }

        for metric in metric_columns:
            row[f"delta_{metric}_mean"] = float(
                group[f"delta_{metric}"].mean()
            )
            row[f"delta_{metric}_std"] = float(
                group[f"delta_{metric}"].std(ddof=1)
            )

        if scenario == BASELINE_SCENARIO_NAME:
            row["baseline_pr_auc_win_folds"] = 0
            row["removal_pr_auc_win_folds"] = 0
            row["pr_auc_tie_folds"] = len(group)
        else:
            difference = group["delta_validation_pr_auc"]
            row["baseline_pr_auc_win_folds"] = int(
                (difference < 0).sum()
            )
            row["removal_pr_auc_win_folds"] = int(
                (difference > 0).sum()
            )
            row["pr_auc_tie_folds"] = int(
                np.isclose(difference, 0.0).sum()
            )

        paired_rows.append(row)

    paired_summary = pd.DataFrame(paired_rows)
    return summary.merge(
        paired_summary,
        on=["scenario", "removed_feature"],
        how="left",
    )


def evidence_status(row: pd.Series) -> tuple[str, str]:
    flags = []

    if row["decision_time_status"] != "available":
        flags.append("confirm_as_of_feature_logic")

    if row["iv"] < IV_WEAK:
        flags.append("very_weak_iv")

    if np.isfinite(row["vif"]) and row["vif"] >= VIF_HIGH:
        flags.append("high_vif")
    elif not np.isfinite(row["vif"]):
        flags.append("infinite_vif")

    if row["max_abs_correlation"] >= CORRELATION_HIGH:
        flags.append("high_pairwise_correlation")

    if row["temporal_psi"] >= PSI_HIGH:
        flags.append("material_temporal_shift")
    elif row["temporal_psi"] >= PSI_MODERATE:
        flags.append("moderate_temporal_shift")

    if row["shap_pct_of_total"] <= SHAP_LOW_SHARE_PCT:
        flags.append("very_low_shap_share")

    delta_pr = row.get(
        "delta_validation_pr_auc_mean",
        np.nan,
    )
    delta_gini = row.get(
        "delta_validation_gini_mean",
        np.nan,
    )

    if pd.notna(delta_pr):
        if delta_pr > CV_PR_AUC_MATERIAL_DELTA:
            flags.append("removal_improves_pr_auc")
        elif delta_pr < -CV_PR_AUC_MATERIAL_DELTA:
            flags.append("removal_harms_pr_auc")

    if pd.notna(delta_gini):
        if delta_gini > 0:
            flags.append("removal_improves_gini")
        elif delta_gini < 0:
            flags.append("removal_harms_gini")

    simplification_candidate = (
        row["iv"] < IV_WEAK
        and row["shap_pct_of_total"] <= SHAP_LOW_SHARE_PCT
        and pd.notna(delta_pr)
        and delta_pr >= -CV_PR_AUC_MATERIAL_DELTA
    )

    if simplification_candidate:
        status = "simplification_candidate"
    elif "confirm_as_of_feature_logic" in flags:
        status = "retain_or_remove_after_validity_review"
    elif any(
        flag in flags
        for flag in (
            "high_vif",
            "infinite_vif",
            "high_pairwise_correlation",
            "material_temporal_shift",
            "removal_improves_pr_auc",
        )
    ):
        status = "focused_review_required"
    else:
        status = "provisional_retain_subject_to_manual_review"

    return "; ".join(flags) if flags else "none", status


def merge_review_and_ablation(
    static_review: pd.DataFrame,
    ablation_summary: pd.DataFrame,
) -> pd.DataFrame:
    ablation = ablation_summary[
        ablation_summary["scenario"] != BASELINE_SCENARIO_NAME
    ].copy()

    merged = static_review.merge(
        ablation,
        left_on="feature",
        right_on="removed_feature",
        how="left",
    )

    flags_and_status = merged.apply(
        evidence_status,
        axis=1,
        result_type="expand",
    )
    merged["attention_flags"] = flags_and_status[0]
    merged["provisional_evidence_status"] = flags_and_status[1]

    merged["final_decision"] = "PENDING_MANUAL_REVIEW"
    merged["final_rationale"] = ""

    preferred_columns = [
        "feature",
        "source",
        "definition",
        "decision_time_status",
        "business_rationale",
        "key_governance_risk",
        "fairness_consideration",
        "data_type",
        "missing_rate",
        "n_unique",
        "iv",
        "iv_band",
        "vif",
        "most_correlated_with",
        "max_abs_correlation",
        "shap_rank",
        "mean_abs_shap",
        "shap_pct_of_total",
        "temporal_psi",
        "psi_band",
        "validation_auc_mean",
        "delta_validation_auc_mean",
        "validation_gini_mean",
        "delta_validation_gini_mean",
        "validation_pr_auc_mean",
        "delta_validation_pr_auc_mean",
        "validation_ks_mean",
        "delta_validation_ks_mean",
        "validation_brier_mean",
        "delta_validation_brier_mean",
        "validation_ece_mean",
        "delta_validation_ece_mean",
        "train_validation_auc_gap_mean",
        "delta_train_validation_auc_gap_mean",
        "baseline_pr_auc_win_folds",
        "removal_pr_auc_win_folds",
        "pr_auc_tie_folds",
        "attention_flags",
        "provisional_evidence_status",
        "final_decision",
        "final_rationale",
    ]

    return merged[
        [column for column in preferred_columns if column in merged.columns]
    ].sort_values("shap_rank")


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if np.isinf(value):
            return "inf"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def save_markdown_review(
    review: pd.DataFrame,
    path: str,
    metadata: dict,
) -> None:
    display_columns = [
        "feature",
        "decision_time_status",
        "iv",
        "vif",
        "max_abs_correlation",
        "shap_rank",
        "shap_pct_of_total",
        "temporal_psi",
        "delta_validation_gini_mean",
        "delta_validation_pr_auc_mean",
        "baseline_pr_auc_win_folds",
        "removal_pr_auc_win_folds",
        "provisional_evidence_status",
        "final_decision",
    ]

    lines = [
        "# Sentinel Credit — Feature Governance Review",
        "",
        f"- Generated: {metadata['generated_at']}",
        f"- Model: `{metadata['model_name']}`",
        f"- Model version: `{metadata['model_version']}`",
        f"- Dataset: `{metadata['dataset_version']}`",
        f"- Feature set: `{metadata['feature_set_label']}`",
        f"- Repeated CV: {N_SPLITS} folds × {N_REPEATS} repeats",
        "- Final decisions remain pending manual review.",
        "",
        "## Evidence summary",
        "",
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join(["---"] * len(display_columns)) + " |",
    ]

    for _, row in review[display_columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                format_value(row[column])
                for column in display_columns
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "No feature is removed automatically because it has low IV, "
            "low SHAP importance, high correlation or a small ablation "
            "effect. Final decisions combine feature validity, "
            "decision-time availability, business rationale, stability, "
            "fairness and repeated-CV evidence.",
            "",
            "## Calibration scope",
            "",
            "The repeated-CV ablation fits the frozen uncalibrated "
            "XGBoost configuration. Brier and ECE are included as "
            "relative sensitivity evidence only. Final calibrated-PD "
            "performance is assessed after the post-governance model is "
            "trained and calibrated.",
        ]
    )

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_decision_template(review: pd.DataFrame) -> dict:
    return {
        row["feature"]: {
            "provisional_evidence_status": row[
                "provisional_evidence_status"
            ],
            "attention_flags": row["attention_flags"],
            "final_decision": "PENDING_MANUAL_REVIEW",
            "final_rationale": "",
            "approved_by": "",
            "approved_at": "",
        }
        for _, row in review.iterrows()
    }


def find_matching_mlflow_run(metadata: dict) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT
    )
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment {MLFLOW_EXPERIMENT!r} not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError(
            f"No MLflow runs found in {MLFLOW_EXPERIMENT!r}."
        )

    dataset = metadata.get("dataset", {})
    criteria = {
        "params.model_type": CHAMPION_MODEL,
        "tags.split_strategy": "stratified",
        "tags.feature_set": FEATURE_SET_LABEL,
        "tags.model_version": MODEL_VERSION,
        "tags.data_end_date": str(
            CONFIG["data_generator"]["data_end_date"]
        ),
        "tags.origination_end_date": str(
            CONFIG["data_generator"]["origination_end_date"]
        ),
    }

    git_commit = dataset.get("git_commit")
    if git_commit and git_commit != "unknown":
        criteria["tags.git_commit"] = str(git_commit)

    matching = runs.copy()
    for column, expected in criteria.items():
        if column not in matching.columns:
            raise RuntimeError(
                f"MLflow search results do not contain {column!r}."
            )
        matching = matching[
            matching[column].astype(str) == str(expected)
        ]

    if matching.empty:
        raise RuntimeError(
            "No matching stratified XGBoost run found.\nCriteria:\n"
            + json.dumps(criteria, indent=2)
        )

    if len(matching) > 1:
        # Reruns without an intervening git commit produce multiple
        # runs carrying identical governance tags — not an error,
        # the same tolerance evaluate_models.py's find_training_run_id
        # already applies. runs was fetched with order_by start_time
        # DESC and filtering preserves order, so the first row is the
        # most recent.
        print(
            f"  MLflow: {len(matching)} matching stratified XGBoost runs "
            f"found; attaching to latest run."
        )

    return str(matching.iloc[0]["run_id"])


def log_to_mlflow(
    run_id: str,
    output_dir: str,
    ablation_summary: pd.DataFrame,
    review: pd.DataFrame,
) -> None:
    baseline = ablation_summary[
        ablation_summary["scenario"] == BASELINE_SCENARIO_NAME
    ].iloc[0]

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(
            {
                "feature_governance_status": (
                    "evidence_generated_pending_manual_decisions"
                ),
                "feature_governance_cv_design": (
                    f"{N_SPLITS}_folds_x_{N_REPEATS}_repeats"
                ),
                "feature_governance_scope": "all_model_features",
            }
        )

        mlflow.log_metrics(
            {
                "fg_n_features_reviewed": len(review),
                "fg_baseline_cv_auc": float(
                    baseline["validation_auc_mean"]
                ),
                "fg_baseline_cv_gini": float(
                    baseline["validation_gini_mean"]
                ),
                "fg_baseline_cv_pr_auc": float(
                    baseline["validation_pr_auc_mean"]
                ),
                "fg_baseline_cv_ks": float(
                    baseline["validation_ks_mean"]
                ),
                "fg_baseline_raw_brier": float(
                    baseline["validation_brier_mean"]
                ),
                "fg_baseline_raw_ece": float(
                    baseline["validation_ece_mean"]
                ),
                "fg_n_simplification_candidates": int(
                    (
                        review["provisional_evidence_status"]
                        == "simplification_candidate"
                    ).sum()
                ),
                "fg_n_focused_review_required": int(
                    (
                        review["provisional_evidence_status"]
                        == "focused_review_required"
                    ).sum()
                ),
            }
        )

        mlflow.log_artifacts(
            output_dir,
            artifact_path=MLFLOW_ARTIFACT_PATH,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete existing fold-level ablation cache and rerun "
            "all fits. Without this flag, completed folds resume."
        ),
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Generate local evidence without attaching it to MLflow.",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("SENTINEL CREDIT — ALL-FEATURE GOVERNANCE REVIEW")
    print("=" * 68)

    require_catalogue_complete()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame = load_modelable_data()
    artifacts = load_governed_artifacts()
    identity = verify_identity(frame, artifacts)

    print("Governed package identity verified:")
    for key, value in identity.items():
        if key != "identity_check_passes":
            print(f"  {key}: {value}")

    X_train, y_train = reconstruct_training_matrix(
        frame,
        artifacts["train_ids"],
    )

    print(
        f"\nCanonical training partition: {len(X_train):,} customers, "
        f"{int(y_train.sum()):,} defaults ({y_train.mean():.2%})"
    )
    print(
        f"Repeated CV design: {N_SPLITS} folds × {N_REPEATS} repeats"
    )
    print(
        f"Scenarios: 1 all-feature baseline ({BASELINE_SCENARIO_NAME}) + "
        f"{len(MODEL_FEATURES)} leave-one-out comparisons"
    )
    print(
        f"Total model fits: "
        f"{(1 + len(MODEL_FEATURES)) * N_SPLITS * N_REPEATS}"
    )

    static_review, supporting = build_static_evidence(
        frame,
        X_train,
        y_train,
        artifacts["temporal"],
        artifacts["shap_importance"],
    )

    supporting["missingness"].to_csv(
        os.path.join(OUTPUT_DIR, "missingness.csv"),
        index=False,
    )
    supporting["iv"].to_csv(
        os.path.join(OUTPUT_DIR, "information_value.csv"),
        index=False,
    )
    supporting["vif"].to_csv(
        os.path.join(OUTPUT_DIR, "vif.csv"),
        index=False,
    )
    supporting["correlation_matrix"].to_csv(
        os.path.join(OUTPUT_DIR, "correlation_matrix.csv"),
    )
    supporting["correlation_evidence"].to_csv(
        os.path.join(OUTPUT_DIR, "correlation_evidence.csv"),
        index=False,
    )
    supporting["psi"].to_csv(
        os.path.join(OUTPUT_DIR, "temporal_psi.csv"),
        index=False,
    )

    fold_path = os.path.join(
        OUTPUT_DIR,
        "leave_one_feature_out_cv_folds.csv",
    )
    fold_results = run_repeated_cv(
        X_train,
        y_train,
        artifacts["metadata"]["best_params"],
        fold_path,
        fresh=args.fresh,
    )

    ablation_summary = summarise_ablation(fold_results)
    ablation_summary_path = os.path.join(
        OUTPUT_DIR,
        "leave_one_feature_out_cv.csv",
    )
    ablation_summary.to_csv(
        ablation_summary_path,
        index=False,
    )

    review = merge_review_and_ablation(
        static_review,
        ablation_summary,
    )
    review_path = os.path.join(
        OUTPUT_DIR,
        "feature_governance_review.csv",
    )
    review.to_csv(review_path, index=False)

    metadata = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        **identity,
        "cv_design": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "baseline_scenario_name": BASELINE_SCENARIO_NAME,
            "n_scenarios": 1 + len(MODEL_FEATURES),
            "n_total_fits": (
                (1 + len(MODEL_FEATURES))
                * N_SPLITS
                * N_REPEATS
            ),
            "model": (
                "XGBClassifier with champion hyperparameters frozen"
            ),
            "calibration_scope": (
                "raw_xgboost_relative_sensitivity_only"
            ),
            "held_out_labels_used_for_feature_selection": False,
        },
        "best_params": artifacts["metadata"]["best_params"],
        "thresholds": THRESHOLDS,
        "final_decisions_automated": False,
        "final_decision_status": "pending_manual_review",
    }

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "governance_metadata.json",
    )
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    markdown_path = os.path.join(
        OUTPUT_DIR,
        "feature_governance_review.md",
    )
    save_markdown_review(
        review,
        markdown_path,
        metadata,
    )

    decision_template = build_decision_template(review)
    decisions_path = os.path.join(
        OUTPUT_DIR,
        "feature_decisions.json",
    )
    with open(decisions_path, "w", encoding="utf-8") as file:
        json.dump(decision_template, file, indent=2)

    run_id = None
    if not args.skip_mlflow:
        run_id = find_matching_mlflow_run(
            artifacts["metadata"]
        )
        log_to_mlflow(
            run_id,
            OUTPUT_DIR,
            ablation_summary,
            review,
        )

    print("\n" + "=" * 68)
    print("FEATURE GOVERNANCE EVIDENCE GENERATED")
    print("=" * 68)
    print(f"Output directory: {OUTPUT_DIR}")
    print("  feature_governance_review.csv")
    print("  feature_governance_review.md")
    print("  leave_one_feature_out_cv.csv")
    print("  leave_one_feature_out_cv_folds.csv")
    print("  feature_decisions.json")
    print("  governance_metadata.json")
    print("  missingness.csv")
    print("  information_value.csv")
    print("  vif.csv")
    print("  correlation_matrix.csv")
    print("  correlation_evidence.csv")
    print("  temporal_psi.csv")
    if run_id:
        print(f"\nAttached to MLflow run: {run_id}")
    else:
        print("\nMLflow attachment skipped.")
    print(
        "\nNo MODEL_FEATURES entry was changed. "
        "Final retain/remove decisions remain manual."
    )
    print("=" * 68)


if __name__ == "__main__":
    main()

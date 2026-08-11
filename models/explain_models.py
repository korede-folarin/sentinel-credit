"""
Sentinel Credit — SHAP Explainability (Raw Foundation)

Purpose
-------
Explains the exact uncalibrated XGBoost explanation model
persisted by train_baseline_models.py for the governed
stratified champion package.

The script:
- does not retrain or re-split;
- verifies model, feature, split and dataset identity;
- reconstructs the exact saved stratified test set;
- validates SHAP additivity;
- produces global and local explanation artifacts;
- attaches the evidence to the matching MLflow training run.

Scope
-----
This is the numerical explanation foundation only. It explains
the champion model's underlying risk-ranking logic. It does not
produce customer-facing adverse-action wording and it does not
claim that SHAP values are changes in calibrated PD.
"""

import json
import os
import shutil
import sys
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import yaml

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import (  # noqa: E402
    MODEL_FEATURES,
    TARGET_COLUMN,
    MULTICOLLINEARITY_EXCLUDED,
    RESOLUTION_ADJACENT_EXCLUDED,
    EXPOSURE_TIME_EXCLUDED,
    EXPOSURE_TIME_SUSPECTED,
    GENERATOR_INTERNAL_EXCLUDED,
    RESOLUTION_TIED_EXCLUDED,
    NON_MODEL_EXCLUDED,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")
SHAP_DIR = os.path.join(PROJECT_ROOT, "reports", "shap")

with open(
    os.path.join(PROJECT_ROOT, "config.yaml"),
    "r",
    encoding="utf-8",
) as file:
    CONFIG = yaml.safe_load(file)

MODEL_CFG = CONFIG["model"]
GOV_CFG = CONFIG["model_governance"]
MLFLOW_CFG = CONFIG["mlflow"]
EXPLAIN_CFG = CONFIG["explainability"]

RANDOM_SEED = MODEL_CFG["random_seed"]
BUSINESS_CUTOFF = MODEL_CFG["business_cutoff"]

TARGET_MODEL = GOV_CFG["champion_model"]
MODEL_VERSION = GOV_CFG["model_version"]
FEATURE_SET_LABEL = GOV_CFG["feature_set_label"]
DATASET_VERSION = GOV_CFG["dataset_version"]
PIPELINE_STAGE = GOV_CFG["pipeline_stage"]

EXPLANATION_SPLIT = EXPLAIN_CFG["split"]
N_TOP_DEPENDENCE = EXPLAIN_CFG["n_top_dependence"]
STRATEGIC_DEPENDENCE_FEATURES = EXPLAIN_CFG[
    "strategic_dependence_features"
]
ADDITIVITY_TOLERANCE = EXPLAIN_CFG["additivity_tolerance"]
MLFLOW_ARTIFACT_PATH = EXPLAIN_CFG["mlflow_artifact_path"]

MLFLOW_TRACKING_URI = MLFLOW_CFG["tracking_uri"]
MLFLOW_EXPERIMENT = MLFLOW_CFG["experiment_name"]


def load_predictions() -> pd.DataFrame:
    """Loads predictions from the governed explanation split."""
    path = os.path.join(
        ARTIFACTS_DIR,
        f"test_predictions_{EXPLANATION_SPLIT}.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run train_baseline_models.py first."
        )

    frame = pd.read_csv(path)
    frame["customer_id"] = frame["customer_id"].astype(str)
    return frame


def load_champion_package() -> dict:
    """
    Loads the exact governed champion explanation package.

    No model is retrained and no split is reconstructed from a
    random seed. The explanation model, metadata, feature list
    and test IDs all come from the canonical training run.
    """
    metadata_path = os.path.join(
        ARTIFACTS_DIR,
        "champion_metadata.json",
    )
    feature_path = os.path.join(
        ARTIFACTS_DIR,
        "feature_list.json",
    )
    test_ids_path = os.path.join(
        ARTIFACTS_DIR,
        "split_ids_test.csv",
    )

    required = [metadata_path, feature_path, test_ids_path]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing governed model-package files:\n  "
            + "\n  ".join(missing)
            + "\nRun train_baseline_models.py first."
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(feature_path, "r", encoding="utf-8") as file:
        feature_list = json.load(file)["features"]

    explanation_artifact = metadata.get(
        "explanation_model_artifact"
    )
    if not explanation_artifact:
        raise KeyError(
            "champion_metadata.json does not define "
            "'explanation_model_artifact'."
        )

    model_path = os.path.join(
        ARTIFACTS_DIR,
        explanation_artifact,
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. The champion explanation "
            "artifact is incomplete."
        )

    model = joblib.load(model_path)

    test_ids = pd.read_csv(test_ids_path)["customer_id"].astype(str)

    return {
        "model": model,
        "metadata": metadata,
        "feature_list": feature_list,
        "test_ids": test_ids,
    }


def verify_package_identity(
    predictions: pd.DataFrame,
    package: dict,
) -> dict:
    """
    Fails closed if the saved package does not match the current
    governed configuration or the saved prediction population.
    """
    metadata = package["metadata"]
    dataset = metadata.get("dataset", {})
    saved_features = package["feature_list"]
    test_ids = package["test_ids"]

    expected = {
        "model_name": TARGET_MODEL,
        "model_version": MODEL_VERSION,
        "split_type": EXPLANATION_SPLIT,
        "dataset_version": DATASET_VERSION,
        "feature_set_label": FEATURE_SET_LABEL,
        "pipeline_stage": PIPELINE_STAGE,
        "data_end_date": str(
            CONFIG["data_generator"]["data_end_date"]
        ),
        "origination_end_date": str(
            CONFIG["data_generator"]["origination_end_date"]
        ),
    }

    observed = {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "split_type": metadata.get("split_type"),
        "dataset_version": dataset.get("dataset_version"),
        "feature_set_label": dataset.get("feature_set_label"),
        "pipeline_stage": dataset.get("pipeline_stage"),
        "data_end_date": str(dataset.get("data_end_date")),
        "origination_end_date": str(
            dataset.get("origination_end_date")
        ),
    }

    mismatches = []
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if str(observed_value) != str(expected_value):
            mismatches.append(
                f"{key}: saved={observed_value!r}, "
                f"current={expected_value!r}"
            )

    if list(saved_features) != list(MODEL_FEATURES):
        mismatches.append(
            "feature list saved with the model does not match "
            "the current MODEL_FEATURES allowlist"
        )

    if int(metadata.get("n_features", -1)) != len(MODEL_FEATURES):
        mismatches.append(
            f"n_features: saved={metadata.get('n_features')}, "
            f"current={len(MODEL_FEATURES)}"
        )

    if int(metadata.get("n_test", -1)) != len(predictions):
        mismatches.append(
            f"n_test: saved={metadata.get('n_test')}, "
            f"prediction_rows={len(predictions)}"
        )

    if predictions["customer_id"].duplicated().any():
        mismatches.append(
            "prediction file contains duplicate customer_id values"
        )

    if test_ids.duplicated().any():
        mismatches.append(
            "split_ids_test.csv contains duplicate customer_id values"
        )

    prediction_ids = set(predictions["customer_id"])
    saved_test_ids = set(test_ids)
    if prediction_ids != saved_test_ids:
        missing_from_predictions = len(
            saved_test_ids - prediction_ids
        )
        unexpected_predictions = len(
            prediction_ids - saved_test_ids
        )
        mismatches.append(
            "prediction/test-ID population mismatch: "
            f"{missing_from_predictions} saved IDs missing from "
            f"predictions; {unexpected_predictions} unexpected "
            "prediction IDs"
        )

    probability_column = f"proba_{TARGET_MODEL}"
    if probability_column not in predictions.columns:
        mismatches.append(
            f"prediction file is missing {probability_column!r}"
        )

    if TARGET_COLUMN not in predictions.columns:
        mismatches.append(
            f"prediction file is missing target column "
            f"{TARGET_COLUMN!r}"
        )

    if mismatches:
        raise RuntimeError(
            "CHAMPION PACKAGE IDENTITY CHECK FAILED.\n"
            "SHAP generation has stopped because the model, data, "
            "features or split may be stale:\n  "
            + "\n  ".join(mismatches)
        )

    return {
        **expected,
        "n_features": len(MODEL_FEATURES),
        "n_test_customers": len(predictions),
        "identity_check_passes": True,
    }


def build_explained_matrix(
    predictions: pd.DataFrame,
    feature_list: list,
    test_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Reconstructs the exact ordered test matrix."""
    indexed = predictions.set_index("customer_id")
    ordered = indexed.loc[test_ids.values].reset_index()

    X_test = ordered[feature_list].copy()
    ids_test = ordered["customer_id"].astype(str)

    return X_test, ids_test, ordered


def excluded_feature_set() -> set:
    """Returns every field explicitly excluded from modelling."""
    groups = (
        MULTICOLLINEARITY_EXCLUDED,
        RESOLUTION_ADJACENT_EXCLUDED,
        EXPOSURE_TIME_EXCLUDED,
        EXPOSURE_TIME_SUSPECTED,
        GENERATOR_INTERNAL_EXCLUDED,
        RESOLUTION_TIED_EXCLUDED,
        NON_MODEL_EXCLUDED,
    )
    return {
        feature
        for group in groups
        for feature in group
    }


def validate_shap(
    expected_value: float,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    model,
    saved_features: list,
) -> dict:
    """Validates feature alignment, exclusions and additivity."""
    checks = {}

    checks["feature_names_match_saved"] = (
        list(X_test.columns) == list(saved_features)
    )
    checks["feature_list_matches_current_code"] = (
        list(saved_features) == list(MODEL_FEATURES)
    )
    checks["n_features"] = len(X_test.columns)

    leaked_features = sorted(
        set(X_test.columns).intersection(excluded_feature_set())
    )
    checks["excluded_features_present"] = leaked_features
    checks["no_excluded_fields"] = len(leaked_features) == 0

    checks["nulls_in_explained_data"] = int(
        X_test.isna().sum().sum()
    )

    raw_margin = model.predict(X_test, output_margin=True)
    reconstructed = shap_values.sum(axis=1) + expected_value
    absolute_error = np.abs(reconstructed - raw_margin)

    checks["explained_output"] = "uncalibrated_raw_margin"
    checks["additivity_tolerance"] = ADDITIVITY_TOLERANCE
    checks["additivity_max_abs_error"] = float(
        f"{absolute_error.max():.3e}"
    )
    checks["additivity_mean_abs_error"] = float(
        f"{absolute_error.mean():.3e}"
    )
    checks["additivity_passes"] = bool(
        absolute_error.max() < ADDITIVITY_TOLERANCE
    )
    checks["expected_value"] = expected_value

    required_passes = (
        checks["feature_names_match_saved"]
        and checks["feature_list_matches_current_code"]
        and checks["no_excluded_fields"]
        and checks["nulls_in_explained_data"] == 0
        and checks["additivity_passes"]
    )
    checks["all_validation_checks_pass"] = bool(required_passes)

    return checks



def normalise_shap_output(
    explainer,
    raw_shap_values,
) -> tuple[np.ndarray, float]:
    """
    Normalises SHAP output across library versions.

    Binary XGBoost commonly returns one matrix, but some SHAP
    versions return a two-element class list. In that case the
    positive-class explanation is selected explicitly.
    """
    expected = explainer.expected_value

    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) == 2:
            shap_values = np.asarray(raw_shap_values[1])
            expected_array = np.asarray(expected).reshape(-1)
            expected_value = float(expected_array[1])
        elif len(raw_shap_values) == 1:
            shap_values = np.asarray(raw_shap_values[0])
            expected_value = float(
                np.asarray(expected).reshape(-1)[0]
            )
        else:
            raise RuntimeError(
                "Unexpected multi-output SHAP result with "
                f"{len(raw_shap_values)} outputs."
            )
    else:
        shap_values = np.asarray(raw_shap_values)
        expected_array = np.asarray(expected).reshape(-1)
        if len(expected_array) != 1:
            raise RuntimeError(
                "Unexpected SHAP expected-value shape: "
                f"{expected_array.shape}."
            )
        expected_value = float(expected_array[0])

    if shap_values.ndim != 2:
        raise RuntimeError(
            f"Expected a 2D SHAP matrix, got shape "
            f"{shap_values.shape}."
        )

    return shap_values, expected_value


def global_importance(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Returns mean absolute SHAP importance for every feature."""
    mean_absolute = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": mean_absolute,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance["rank"] = importance.index + 1
    total = importance["mean_abs_shap"].sum()
    importance["pct_of_total"] = np.where(
        total > 0,
        100 * importance["mean_abs_shap"] / total,
        0.0,
    ).round(2)
    importance["exposure_time_suspected"] = importance[
        "feature"
    ].isin(EXPOSURE_TIME_SUSPECTED)

    return importance


def save_beeswarm(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
) -> None:
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
        max_display=len(MODEL_FEATURES),
    )
    plt.title(
        f"SHAP summary — {TARGET_MODEL} champion",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_DIR, "global_beeswarm.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_dependence_plots(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict:
    """Plots top-ranked and configured strategic features."""
    output_dir = os.path.join(SHAP_DIR, "dependence")
    os.makedirs(output_dir, exist_ok=True)

    top_by_shap = importance.head(
        N_TOP_DEPENDENCE
    )["feature"].tolist()

    strategic = [
        feature
        for feature in STRATEGIC_DEPENDENCE_FEATURES
        if feature in X_test.columns
        and feature not in top_by_shap
    ]

    selected = top_by_shap + strategic

    for feature in selected:
        plt.figure()
        shap.dependence_plot(
            feature,
            shap_values,
            X_test,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{feature}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    return {
        "by_shap_rank": top_by_shap,
        "strategic": strategic,
        "all_plotted": selected,
    }


def find_dominant_feature_case(
    shap_values: np.ndarray,
    ids_test: pd.Series,
) -> tuple[str, float]:
    """Finds the prediction most dominated by one feature."""
    maximum_per_customer = np.abs(shap_values).max(axis=1)
    index = int(np.argmax(maximum_per_customer))

    return (
        str(ids_test.iloc[index]),
        float(maximum_per_customer[index]),
    )


def select_case_studies(
    predictions: pd.DataFrame,
) -> dict:
    """
    Selects representative cases using the currently configured
    business cutoff. Because this is the pre-final baseline,
    cutoff-dependent local cases are provisional.
    """
    probability_column = f"proba_{TARGET_MODEL}"

    frame = predictions.copy()
    frame["predicted_flag"] = (
        frame[probability_column] >= BUSINESS_CUTOFF
    ).astype(int)

    cases = {}

    lowest = frame.nsmallest(1, probability_column)
    if len(lowest):
        cases["low_risk"] = str(
            lowest.iloc[0]["customer_id"]
        )

    highest = frame.nlargest(1, probability_column)
    if len(highest):
        cases["high_risk"] = str(
            highest.iloc[0]["customer_id"]
        )

    frame["distance_to_cutoff"] = (
        frame[probability_column] - BUSINESS_CUTOFF
    ).abs()
    nearest = frame.nsmallest(1, "distance_to_cutoff")
    if len(nearest):
        cases["near_threshold_provisional"] = str(
            nearest.iloc[0]["customer_id"]
        )

    false_positive = frame[
        (frame["predicted_flag"] == 1)
        & (frame[TARGET_COLUMN] == 0)
    ]
    if len(false_positive):
        cases["false_positive_provisional"] = str(
            false_positive.nlargest(
                1,
                probability_column,
            ).iloc[0]["customer_id"]
        )

    false_negative = frame[
        (frame["predicted_flag"] == 0)
        & (frame[TARGET_COLUMN] == 1)
    ]
    if len(false_negative):
        cases["false_negative_provisional"] = str(
            false_negative.nsmallest(
                1,
                probability_column,
            ).iloc[0]["customer_id"]
        )

    return cases


def save_local_explanation(
    label: str,
    customer_id: str,
    row_index: int,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    expected_value: float,
    ordered_predictions: pd.DataFrame,
) -> None:
    """Saves one waterfall plot and one structured JSON record."""
    output_dir = os.path.join(SHAP_DIR, "local")
    os.makedirs(output_dir, exist_ok=True)

    row_shap = shap_values[row_index]
    row_features = X_test.iloc[row_index]

    explanation = shap.Explanation(
        values=row_shap,
        base_values=expected_value,
        data=row_features.values,
        feature_names=list(X_test.columns),
    )

    plt.figure()
    shap.plots.waterfall(
        explanation,
        max_display=len(MODEL_FEATURES),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"{label}_customer_{customer_id}_waterfall.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    matching = ordered_predictions[
        ordered_predictions["customer_id"].astype(str)
        == str(customer_id)
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one prediction for customer "
            f"{customer_id}, found {len(matching)}."
        )

    prediction = matching.iloc[0]
    probability_column = f"proba_{TARGET_MODEL}"

    contributions = sorted(
        [
            {
                "feature": feature,
                "value": float(row_features[feature]),
                "shap_value": float(row_shap[position]),
                "exposure_time_suspected": (
                    feature in EXPOSURE_TIME_SUSPECTED
                ),
            }
            for position, feature in enumerate(X_test.columns)
        ],
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    record = {
        "customer_id": str(customer_id),
        "case_type": label,
        "case_status": (
            "provisional_until_business_cutoff_is_finalised"
            if "provisional" in label
            else "cutoff_independent"
        ),
        "model": TARGET_MODEL,
        "calibrated_probability": float(
            prediction[probability_column]
        ),
        "business_cutoff": BUSINESS_CUTOFF,
        "decision": (
            "flag_high_risk"
            if prediction[probability_column] >= BUSINESS_CUTOFF
            else "pass"
        ),
        "actual_outcome": int(prediction[TARGET_COLUMN]),
        "shap_base_value_raw_margin": expected_value,
        "shap_sum_raw_margin": float(
            row_shap.sum() + expected_value
        ),
        "note": (
            "This explains the exact uncalibrated explanation "
            "model persisted during champion training. SHAP values "
            "describe movement in its raw risk margin, not changes "
            "in the calibrated probability."
        ),
        "contributions": contributions,
    }

    with open(
        os.path.join(
            output_dir,
            f"{label}_customer_{customer_id}_values.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(record, file, indent=2)


def find_matching_mlflow_run(
    metadata: dict,
) -> str:
    """Finds the exact stratified champion training run."""
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
            f"No runs found in MLflow experiment "
            f"{MLFLOW_EXPERIMENT!r}."
        )

    dataset = metadata["dataset"]

    required_values = {
        "params.model_type": TARGET_MODEL,
        "tags.split_strategy": EXPLANATION_SPLIT,
        "tags.feature_set": FEATURE_SET_LABEL,
        "tags.model_version": MODEL_VERSION,
        "tags.dataset_version": DATASET_VERSION,
        "tags.data_end_date": str(
            CONFIG["data_generator"]["data_end_date"]
        ),
        "tags.origination_end_date": str(
            CONFIG["data_generator"]["origination_end_date"]
        ),
        "tags.git_commit": str(dataset["git_commit"]),
    }

    matching = runs.copy()
    for column, expected_value in required_values.items():
        if column not in matching.columns:
            raise RuntimeError(
                f"MLflow search results are missing {column!r}."
            )

        matching = matching[
            matching[column].astype(str)
            == str(expected_value)
        ]

    if matching.empty:
        raise RuntimeError(
            "No matching MLflow champion run found. Matching criteria:\n"
            + json.dumps(required_values, indent=2)
        )

    if len(matching) > 1:
        # Same tolerance as evaluate_models.py / govern_model_features.py:
        # reruns without an intervening commit share identical tags.
        # runs was fetched order_by start_time DESC, order preserved
        # through filtering, so the first row is the most recent.
        print(
            f"  MLflow: {len(matching)} matching champion runs found; "
            f"attaching to latest run."
        )

    return str(matching.iloc[0]["run_id"])


def attach_to_mlflow(
    run_id: str,
    checks: dict,
    importance: pd.DataFrame,
    metadata: dict,
) -> None:
    """Attaches SHAP evidence to the existing champion run."""
    top_feature = importance.iloc[0]

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(
            {
                "shap_status": (
                    "validated"
                    if checks["all_validation_checks_pass"]
                    else "failed_validation"
                ),
                "shap_explained_output": (
                    "uncalibrated_raw_margin"
                ),
                "shap_cutoff_status": (
                    "provisional_pre_final_model_selection"
                ),
            }
        )

        mlflow.log_metrics(
            {
                "shap_additivity_max_abs_error": checks[
                    "additivity_max_abs_error"
                ],
                "shap_additivity_mean_abs_error": checks[
                    "additivity_mean_abs_error"
                ],
                "shap_additivity_pass": int(
                    checks["additivity_passes"]
                ),
                "shap_validation_all_pass": int(
                    checks["all_validation_checks_pass"]
                ),
                "shap_n_customers": metadata[
                    "n_test_customers"
                ],
                "shap_top_feature_share": float(
                    top_feature["pct_of_total"] / 100
                ),
            }
        )

        mlflow.log_param(
            "shap_top_feature",
            str(top_feature["feature"]),
        )
        mlflow.log_param(
            "shap_n_features",
            len(MODEL_FEATURES),
        )

        mlflow.log_artifacts(
            SHAP_DIR,
            artifact_path=MLFLOW_ARTIFACT_PATH,
        )


def reset_shap_dir() -> None:
    """
    Clears stale files from SHAP_DIR without requiring the
    directory itself to be deletable.

    shutil.rmtree needs an exclusive lock on the directory entry,
    which OneDrive's sync process can hold on this project's
    OneDrive-backed working folder for longer than a few retries
    can wait out — not always transient. Removing files
    individually only needs a lock on each file, so one locked
    file (e.g. a dependence-plot PNG open in an image viewer)
    no longer blocks every other file from being cleared. Any
    file that still can't be removed is reported by name rather
    than silently left behind or treated as a hard failure.
    """
    os.makedirs(SHAP_DIR, exist_ok=True)

    removed = 0
    skipped = []
    for root, dirs, files in os.walk(SHAP_DIR, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            try:
                os.remove(path)
                removed += 1
            except (PermissionError, OSError) as error:
                skipped.append((path, str(error)))
        for name in dirs:
            path = os.path.join(root, name)
            try:
                os.rmdir(path)
            except (PermissionError, OSError):
                pass  # non-empty (a file inside was skipped) or still locked

    if skipped:
        print(
            f"  WARNING: {len(skipped)} stale file(s) in {SHAP_DIR} "
            f"could not be removed (likely open elsewhere, or locked "
            f"by OneDrive sync) and were left in place:"
        )
        for path, error in skipped:
            print(f"    {path}")
        print(
            "  These may be leftover evidence from a prior run. Close "
            "any open file in reports/shap/ and rerun if you need a "
            "guaranteed-clean rebuild; this run's fresh outputs below "
            "are still correct regardless."
        )

    print(f"  Cleared {removed} stale file(s) from {SHAP_DIR}.")


def main() -> None:
    print("=" * 64)
    print("SENTINEL CREDIT — SHAP EXPLAINABILITY")
    print("=" * 64)

    predictions = load_predictions()
    package = load_champion_package()

    identity = verify_package_identity(
        predictions,
        package,
    )

    print("Champion package identity verified:")
    for key, value in identity.items():
        if key != "identity_check_passes":
            print(f"  {key}: {value}")

    reset_shap_dir()

    X_test, ids_test, ordered_predictions = (
        build_explained_matrix(
            predictions,
            package["feature_list"],
            package["test_ids"],
        )
    )

    metadata = package["metadata"]
    model = package["model"]

    print(f"\nLoaded champion package: {metadata['model_version']}")
    print(f"  trained_at: {metadata['trained_at']}")
    print(f"  best_params: {metadata['best_params']}")
    print(
        f"Explaining {len(X_test):,} test customers "
        f"with {len(X_test.columns)} features"
    )

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_test)
    shap_values, expected_value = normalise_shap_output(
        explainer,
        raw_shap_values,
    )
    print("SHAP values computed")

    print("\n" + "-" * 64)
    print("VALIDATION")
    print("-" * 64)

    checks = validate_shap(
        expected_value,
        shap_values,
        X_test,
        model,
        package["feature_list"],
    )

    for key, value in checks.items():
        print(f"  {key}: {value}")

    if not checks["all_validation_checks_pass"]:
        raise RuntimeError(
            "SHAP validation failed. No explanation artifacts "
            "will be treated as valid."
        )

    print("\n" + "-" * 64)
    print("GLOBAL IMPORTANCE")
    print("-" * 64)

    importance = global_importance(
        shap_values,
        X_test,
    )
    print(importance.to_string(index=False))

    importance.to_csv(
        os.path.join(
            SHAP_DIR,
            "global_feature_importance.csv",
        ),
        index=False,
    )

    save_beeswarm(shap_values, X_test)

    dependence_info = save_dependence_plots(
        shap_values,
        X_test,
        importance,
    )

    print("\n" + "-" * 64)
    print("LOCAL EXPLANATIONS")
    print("-" * 64)

    id_to_position = {
        str(customer_id): position
        for position, customer_id in enumerate(ids_test)
    }

    cases = select_case_studies(ordered_predictions)

    dominant_id, dominant_value = (
        find_dominant_feature_case(
            shap_values,
            ids_test,
        )
    )
    cases["single_feature_dominant"] = dominant_id

    print(
        "Most single-feature-dominated prediction: "
        f"customer {dominant_id} "
        f"(max |SHAP|={dominant_value:.4f})"
    )

    for label, customer_id in cases.items():
        position = id_to_position.get(str(customer_id))
        if position is None:
            raise KeyError(
                f"Customer {customer_id} is not in the exact "
                "explained test population."
            )

        save_local_explanation(
            label,
            customer_id,
            position,
            shap_values,
            X_test,
            expected_value,
            ordered_predictions,
        )

        row = ordered_predictions[
            ordered_predictions["customer_id"].astype(str)
            == str(customer_id)
        ].iloc[0]

        print(
            f"  {label:<34} "
            f"customer {customer_id}  "
            f"probability "
            f"{row[f'proba_{TARGET_MODEL}']:.4f}  "
            f"actual {int(row[TARGET_COLUMN])}"
        )

    shap_metadata = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "model": TARGET_MODEL,
        "model_version": metadata["model_version"],
        "dataset_identity": identity,
        "model_type": (
            "XGBClassifier uncalibrated explanation base"
        ),
        "explained_artifact": metadata[
            "explanation_model_artifact"
        ],
        "champion_trained_at": metadata["trained_at"],
        "best_params": metadata["best_params"],
        "n_test_customers": int(len(X_test)),
        "n_features": len(MODEL_FEATURES),
        "feature_list": list(MODEL_FEATURES),
        "business_cutoff": BUSINESS_CUTOFF,
        "business_cutoff_status": (
            "provisional_pre_final_model_selection"
        ),
        "random_seed": RANDOM_SEED,
        "validation_checks": checks,
        "dependence_plots": dependence_info,
        "case_studies": cases,
        "scope_note": (
            "Explanations come from the exact uncalibrated "
            "XGBoost explanation artifact persisted during the "
            "canonical champion training run. They explain the "
            "model's raw risk margin. They do not reconstruct or "
            "attribute changes to the calibrated probability."
        ),
        "valid_uses": [
            "global feature-importance ranking",
            "feature-direction and dependence inspection",
            "identifying suspicious model dependencies",
            "comparing low-risk, high-risk and error cases",
        ],
        "not_valid_for": [
            "customer-facing adverse-action wording",
            "claims about calibrated probability changes",
            "formal reconstruction of the calibrated decision",
            "regulatory output presented as an exact end-to-end explanation",
        ],
    }

    metadata_path = os.path.join(
        SHAP_DIR,
        "shap_metadata.json",
    )
    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(shap_metadata, file, indent=2)

    run_id = find_matching_mlflow_run(metadata)

    attach_to_mlflow(
        run_id,
        checks,
        importance,
        shap_metadata,
    )

    print("\n" + "=" * 64)
    print(f"Saved SHAP evidence to: {SHAP_DIR}")
    print(f"Attached to MLflow run: {run_id}")
    print("  global_feature_importance.csv")
    print("  global_beeswarm.png")
    print(
        f"  dependence/ "
        f"({len(dependence_info['all_plotted'])} plots)"
    )
    print(f"  local/ ({len(cases)} case studies)")
    print("  shap_metadata.json")
    print("=" * 64)


if __name__ == "__main__":
    main()

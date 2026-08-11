"""
Sentinel Credit — Training-Only OOF Business Cutoff Selection

Purpose
-------
Generates candidate business cutoffs from calibrated,
training-only out-of-fold predictions, for human review.

It deliberately does not use:
- test_predictions_stratified.csv
- test_predictions_temporal.csv
- any held-out test labels

The exact canonical training population is reconstructed from
models/artifacts/split_ids_train.csv, produced by
train_baseline_models.py.

Candidates
----------
- f1_optimal and youden_j: found directly from the OOF scores,
  no assumptions required.
- review_capacity_constrained: the threshold that flags at most
  config.yaml threshold_recalibration.review_capacity.max_flag_rate
  of the population. Skipped if not configured.
- expected_cost_minimizing: uses
  threshold_recalibration.cost_assumptions (false_negative_cost,
  false_positive_cost). Skipped, not defaulted, if either is
  unset — an invented cost ratio would manufacture a threshold
  that looks governed without being defensible. See the comments
  in config.yaml.

No candidate is written to config.yaml automatically. Update
model.business_cutoff manually after review, then rerun
evaluate_models.py on both held-out splits.
"""

import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "threshold_selection")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import (  # noqa: E402
    MODEL_FEATURES,
    TARGET_COLUMN,
    FILTER_COLUMN,
    FILTER_VALUE,
)

with open(
    os.path.join(PROJECT_ROOT, "config.yaml"),
    "r",
    encoding="utf-8",
) as file:
    CONFIG = yaml.safe_load(file)

with open(
    os.path.join(PROJECT_ROOT, "params.yaml"),
    "r",
    encoding="utf-8",
) as file:
    PARAMS = yaml.safe_load(file)

DB = CONFIG["database"]
MODEL_CFG = CONFIG.get("model", {})
GOV_CFG = CONFIG["model_governance"]
RECAL_CFG = CONFIG.get("threshold_recalibration", {})

RANDOM_SEED = MODEL_CFG.get("random_seed", 42)
CV_CFG = PARAMS.get("cv", {})
N_SPLITS = CV_CFG.get("n_splits", 5)

CHAMPION_MODEL = GOV_CFG["champion_model"]
FEATURE_SET_LABEL = GOV_CFG["feature_set_label"]
DATASET_VERSION = GOV_CFG["dataset_version"]

COST_CFG = RECAL_CFG.get("cost_assumptions", {})
FALSE_NEGATIVE_COST = COST_CFG.get("false_negative_cost")
FALSE_POSITIVE_COST = COST_CFG.get("false_positive_cost")
MAX_FLAG_RATE = RECAL_CFG.get("review_capacity", {}).get("max_flag_rate")


def load_modelable_data() -> pd.DataFrame:
    """Loads the current modelable feature-store population."""
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

    return frame[frame[FILTER_COLUMN] == FILTER_VALUE].copy()


def load_training_package() -> tuple[pd.Series, dict]:
    """Loads the saved canonical training IDs and champion metadata."""
    ids_path = os.path.join(ARTIFACTS_DIR, "split_ids_train.csv")
    metadata_path = os.path.join(
        ARTIFACTS_DIR,
        "champion_metadata.json",
    )

    missing = [
        path
        for path in (ids_path, metadata_path)
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required training artifacts:\n  "
            + "\n  ".join(missing)
            + "\nRun train_baseline_models.py first."
        )

    train_ids = pd.read_csv(ids_path)["customer_id"]

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return train_ids, metadata


def verify_dataset_identity(metadata: dict, n_modelable: int) -> None:
    """Stops execution if the saved model package is stale."""
    saved = metadata.get("dataset", {})

    expected = {
        "dataset_version": str(DATASET_VERSION),
        "data_end_date": str(
            CONFIG["data_generator"]["data_end_date"]
        ),
        "origination_end_date": str(
            CONFIG["data_generator"]["origination_end_date"]
        ),
        "feature_set_label": str(FEATURE_SET_LABEL),
    }

    mismatches = []
    for key, expected_value in expected.items():
        saved_value = saved.get(key)
        if str(saved_value) != expected_value:
            mismatches.append(
                f"{key}: saved={saved_value!r}, "
                f"config={expected_value!r}"
            )

    saved_n = saved.get("n_modelable")
    if saved_n is not None and int(saved_n) != int(n_modelable):
        mismatches.append(
            f"n_modelable: saved={saved_n}, "
            f"current={n_modelable}"
        )

    if mismatches:
        raise RuntimeError(
            "Dataset identity check failed. The saved model package "
            "does not match the current feature-store/config state:\n  "
            + "\n  ".join(mismatches)
        )


def reconstruct_canonical_training_data(
    frame: pd.DataFrame,
    train_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reconstructs the exact canonical stratified training population.

    Imputation statistics are calculated from the canonical training
    population only, so held-out test information is not used for
    cutoff selection.
    """
    if frame["customer_id"].duplicated().any():
        raise ValueError(
            "customer_features contains duplicate customer_id values."
        )

    indexed = frame.set_index("customer_id")

    missing_ids = train_ids[~train_ids.isin(indexed.index)]
    if len(missing_ids):
        raise ValueError(
            f"{len(missing_ids):,} saved training customer IDs are "
            "missing from the current feature store."
        )

    train_frame = indexed.loc[train_ids.values].copy()

    X_train = train_frame[MODEL_FEATURES].copy()
    y_train = train_frame[TARGET_COLUMN].astype(int).copy()

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)

    remaining_nulls = int(X_train.isna().sum().sum())
    if remaining_nulls:
        null_columns = X_train.columns[
            X_train.isna().any()
        ].tolist()
        raise ValueError(
            f"{remaining_nulls:,} null values remain after median "
            f"imputation in: {null_columns}"
        )

    return X_train, y_train


def generate_oof_probabilities(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> np.ndarray:
    """
    Produces calibrated out-of-fold probabilities.

    Each customer's probability is generated by a model that was not
    fitted on that customer. Hyperparameters are frozen from the
    governed champion package; only the cutoff is being selected here.
    """
    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    oof_probability = np.full(len(X_train), np.nan, dtype=float)

    for fold_number, (fit_idx, valid_idx) in enumerate(
        outer_cv.split(X_train, y_train),
        start=1,
    ):
        X_fit = X_train.iloc[fit_idx]
        y_fit = y_train.iloc[fit_idx]
        X_valid = X_train.iloc[valid_idx]

        inner_cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + fold_number,
        )

        base_model = XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="auc",
            **best_params,
        )

        calibrated_model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv=inner_cv,
        )
        calibrated_model.fit(X_fit, y_fit)

        oof_probability[valid_idx] = calibrated_model.predict_proba(
            X_valid
        )[:, 1]

        print(
            f"  Fold {fold_number}/{N_SPLITS}: "
            f"{len(fit_idx):,} fit / {len(valid_idx):,} validation"
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError(
            "OOF generation failed: some customers have no probability."
        )

    return oof_probability


def build_threshold_table(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    """
    Calculates the full training-OOF cutoff trade-off table —
    one row per threshold implied by the precision-recall curve.
    Every candidate-selection function below reads from this
    same table, so tp/fp/tn/fn are computed once, not per
    candidate method.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probability,
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    rows = []
    for threshold, p, r, f1_value in zip(
        thresholds,
        precision,
        recall,
        f1,
    ):
        predicted = probability >= threshold

        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        tn = int(((predicted == 0) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())

        approved = ~predicted
        approved_bad_rate = (
            float(y_true[approved].mean())
            if approved.sum()
            else np.nan
        )

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1_value),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": sensitivity + specificity - 1,
                "flagged_n": int(predicted.sum()),
                "flagged_rate": float(predicted.mean()),
                "approved_n": int(approved.sum()),
                "approved_bad_rate": approved_bad_rate,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return pd.DataFrame(rows)


def row_to_candidate(threshold_table: pd.DataFrame, row: pd.Series) -> dict:
    int_cols = {"flagged_n", "approved_n", "tp", "fp", "tn", "fn"}
    return {
        key: (int(row[key]) if key in int_cols else float(row[key]))
        for key in threshold_table.columns
    }


def select_max_f1_cutoff(threshold_table: pd.DataFrame) -> dict:
    """
    Maximum-F1 threshold. If several rows tie, the highest
    threshold is selected so the policy flags fewer customers
    for the same F1 score.
    """
    max_f1 = threshold_table["f1"].max()
    tied = threshold_table[np.isclose(threshold_table["f1"], max_f1)]
    selected = tied.sort_values("threshold", ascending=False).iloc[0]
    return row_to_candidate(threshold_table, selected)


def select_max_youden_cutoff(threshold_table: pd.DataFrame) -> dict:
    """Maximum Youden's J = sensitivity + specificity - 1."""
    max_j = threshold_table["youden_j"].max()
    tied = threshold_table[np.isclose(threshold_table["youden_j"], max_j)]
    selected = tied.sort_values("threshold", ascending=False).iloc[0]
    return row_to_candidate(threshold_table, selected)


def select_capacity_constrained_cutoff(
    threshold_table: pd.DataFrame,
    max_flag_rate: float,
) -> dict:
    """
    Threshold flagging as close to max_flag_rate as possible
    without exceeding it — "the review team can handle at most
    X% of applications" as a literal constraint. Falls back to
    the lowest achievable flagged_rate if even the strictest
    threshold in the table exceeds the cap.
    """
    within_capacity = threshold_table[
        threshold_table["flagged_rate"] <= max_flag_rate
    ]
    if within_capacity.empty:
        selected = threshold_table.sort_values("flagged_rate").iloc[0]
    else:
        selected = within_capacity.sort_values(
            "flagged_rate", ascending=False
        ).iloc[0]
    return row_to_candidate(threshold_table, selected)


def select_cost_minimizing_cutoff(
    threshold_table: pd.DataFrame,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict:
    """Threshold minimizing false_negative_cost*FN + false_positive_cost*FP."""
    cost = (
        false_negative_cost * threshold_table["fn"]
        + false_positive_cost * threshold_table["fp"]
    )
    selected = threshold_table.loc[cost.idxmin()]
    candidate = row_to_candidate(threshold_table, selected)
    candidate["total_cost"] = float(cost.min())
    return candidate


def main() -> None:
    print("=" * 62)
    print("SENTINEL CREDIT — TRAINING-ONLY OOF CUTOFF EVIDENCE")
    print("=" * 62)

    if CHAMPION_MODEL != "xgboost":
        raise ValueError(
            "This script currently supports the governed XGBoost "
            f"champion, but config.yaml has champion_model="
            f"{CHAMPION_MODEL!r}."
        )

    frame = load_modelable_data()
    train_ids, metadata = load_training_package()

    verify_dataset_identity(metadata, n_modelable=len(frame))

    X_train, y_train = reconstruct_canonical_training_data(
        frame,
        train_ids,
    )

    best_params = metadata["best_params"]

    print("Dataset identity verified")
    print(f"  dataset_version: {DATASET_VERSION}")
    print(f"  feature_set:     {FEATURE_SET_LABEL}")
    print(f"  model:           {CHAMPION_MODEL}")
    print(f"  training rows:   {len(X_train):,}")
    print(f"  training bads:   {int(y_train.sum()):,}")
    print(f"  training rate:   {y_train.mean():.2%}")
    print(f"  CV folds:        {N_SPLITS}")
    print(f"  best_params:     {best_params}")

    print("\nGenerating calibrated OOF probabilities...")
    oof_probability = generate_oof_probabilities(
        X_train,
        y_train,
        best_params,
    )

    threshold_table = build_threshold_table(
        y_train.to_numpy(),
        oof_probability,
    )

    candidates = {
        "f1_optimal": select_max_f1_cutoff(threshold_table),
        "youden_j": select_max_youden_cutoff(threshold_table),
    }

    if MAX_FLAG_RATE is not None:
        candidates["review_capacity_constrained"] = (
            select_capacity_constrained_cutoff(threshold_table, MAX_FLAG_RATE)
        )
    else:
        print(
            "\n  Skipped review_capacity_constrained — "
            "threshold_recalibration.review_capacity.max_flag_rate "
            "not set in config.yaml."
        )

    if FALSE_NEGATIVE_COST is not None and FALSE_POSITIVE_COST is not None:
        candidates["expected_cost_minimizing"] = select_cost_minimizing_cutoff(
            threshold_table, FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST
        )
    else:
        print(
            "\n  Skipped expected_cost_minimizing — "
            "threshold_recalibration.cost_assumptions not fully set "
            "in config.yaml. This is expected until real cost figures "
            "exist; review_capacity is the more defensible constraint "
            "for v1 in the meantime."
        )

    os.makedirs(REPORT_DIR, exist_ok=True)

    oof_frame = pd.DataFrame(
        {
            "customer_id": train_ids.values,
            TARGET_COLUMN: y_train.values,
            "oof_probability_xgboost": oof_probability,
        }
    )
    oof_path = os.path.join(
        REPORT_DIR,
        "xgboost_training_oof_predictions.csv",
    )
    oof_frame.to_csv(oof_path, index=False)

    tradeoff_path = os.path.join(
        REPORT_DIR,
        "xgboost_oof_threshold_tradeoff.csv",
    )
    threshold_table.to_csv(tradeoff_path, index=False)

    candidates_path = os.path.join(
        REPORT_DIR,
        "threshold_candidates.csv",
    )
    pd.DataFrame(
        [{"method": name, **values} for name, values in candidates.items()]
    ).to_csv(candidates_path, index=False)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "AWAITING HUMAN REVIEW — config.yaml NOT modified by this script",
        "selection_source": "canonical_stratified_training_partition_oof",
        "model": CHAMPION_MODEL,
        "model_version": metadata["model_version"],
        "dataset_version": DATASET_VERSION,
        "feature_set_label": FEATURE_SET_LABEL,
        "n_training_customers": int(len(X_train)),
        "n_training_bads": int(y_train.sum()),
        "training_bad_rate": float(y_train.mean()),
        "n_oof_folds": N_SPLITS,
        "best_params": best_params,
        "candidates": candidates,
        "held_out_sets_used_for_selection": False,
        "note": (
            "All candidates were computed from calibrated training-only "
            "out-of-fold probabilities. Stratified and temporal test "
            "predictions were not used. No candidate is applied "
            "automatically. After choosing one, set model.business_cutoff "
            "in config.yaml to that value, then rerun "
            "'python models\\evaluate_models.py --split stratified' and "
            "'--split temporal' to confirm behaviour on the untouched "
            "test/OOT sets at the frozen cutoff."
        ),
    }

    result_path = os.path.join(
        REPORT_DIR,
        "threshold_recommendation.json",
    )
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\n" + "=" * 62)
    print("CANDIDATE THRESHOLDS (evidence only — human review required)")
    print("=" * 62)
    for name, c in candidates.items():
        extra = f"  total_cost={c['total_cost']:.2f}" if "total_cost" in c else ""
        print(f"\n{name}")
        print(f"  threshold:            {c['threshold']:.4f}")
        print(f"  flagged:              {c['flagged_n']:,} ({c['flagged_rate']:.2%})")
        print(f"  precision / recall:   {c['precision']:.2%} / {c['recall']:.2%}")
        print(f"  bad rate if approved: {c['approved_bad_rate']:.2%}{extra}")

    print("\nNothing has been written to config.yaml.")
    print("Choose a threshold, then manually set model.business_cutoff")
    print("and rerun evaluate_models.py on both splits to confirm.")
    print("\nSaved:")
    print(f"  {oof_path}")
    print(f"  {tradeoff_path}")
    print(f"  {candidates_path}")
    print(f"  {result_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()

"""
Sentinel Credit — Drift and Data Quality Monitoring

Purpose
-------
Compares the canonical training population (the model's reference
distribution) against a scoring population to detect population
drift and basic data-quality problems, before either would show up
as silent performance decay.

Reference population: the exact canonical training partition
(models/artifacts/split_ids_train.csv) — the population the
frozen champion was actually fitted on, matching Decision 18/19's
training-only discipline throughout this project.

Comparison population: defaults to the temporal test partition —
the later-originating customers already used for out-of-time
evaluation. This is a deliberate, honest choice given the project's
scope: Sentinel Credit has no live incoming scoring stream to
monitor, so the closest available proxy for "newly arrived
population" is the most temporally recent slice of data on hand.
In production, this argument would instead point at a batch of
freshly scored customers; --population lets that be substituted
without changing anything else in the script.

Checks
------
- Population Stability Index (PSI) per feature, same calculation
  and thresholds already governed in govern_model_features.py's
  temporal_psi check (config.yaml feature_governance.thresholds),
  so a feature crossing "material_shift" here means the same thing
  it meant during feature governance.
- Data quality: null-rate delta, out-of-training-range rate for
  numeric features, and mean/std drift as a check PSI's binning
  can occasionally miss (a shift entirely within one PSI bin).

This script does not retrain, does not change MODEL_FEATURES, and
does not gate deployment by itself — it produces evidence and
alerts. Feeding that evidence into an automated Approval Gate is
future work (Intelligence Core / Governance Layer, not yet built).

Each run is logged to its own MLflow experiment
(sentinel_credit_monitoring), separate from the training/governance
experiment, so repeated runs accumulate a queryable drift history
over time rather than cluttering baseline model-comparison runs.

See docs/engineering_decision_log.md — Decision 5 (drift monitoring,
queued), Decision 18 (training-only discipline this reuses).
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "monitoring")

# model_features.py lives in models/, not necessarily next to this
# script — this script belongs in pipeline/ (Decision 5's scheduled
# monitoring job, per evaluate_models.py's existing docstring), so
# the import path is set explicitly rather than assumed from
# __file__'s own directory the way single-folder scripts do it.
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
from model_features import MODEL_FEATURES, TARGET_COLUMN, FILTER_COLUMN, FILTER_VALUE  # noqa: E402

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB = CONFIG["database"]
GOV_CFG = CONFIG["model_governance"]
MLFLOW_CFG = CONFIG["mlflow"]
FG_THRESHOLDS = CONFIG["feature_governance"]["thresholds"]

CHAMPION_MODEL = GOV_CFG["champion_model"]
MODEL_VERSION = GOV_CFG["model_version"]
FEATURE_SET_LABEL = GOV_CFG["feature_set_label"]
DATASET_VERSION = GOV_CFG["dataset_version"]

PSI_BINS = CONFIG["feature_governance"]["psi_bins"]
PSI_MODERATE = FG_THRESHOLDS["psi_moderate"]
PSI_HIGH = FG_THRESHOLDS["psi_high"]

MLFLOW_TRACKING_URI = MLFLOW_CFG["tracking_uri"]
MONITORING_EXPERIMENT = "sentinel_credit_monitoring"

# Data-quality thresholds. Not in config.yaml — these are
# monitoring-specific and conservative defaults, not governed
# feature-selection thresholds. Flagged here, not hidden, so
# they're easy to find and adjust.
NULL_RATE_DELTA_ALERT = 0.02        # +2pp null rate vs. training triggers a flag
OUT_OF_RANGE_RATE_ALERT = 0.01      # >1% of values outside training min/max
MEAN_SHIFT_SIGMA_ALERT = 0.5        # mean moved more than 0.5 training std devs


def load_full_population() -> pd.DataFrame:
    # SENTINEL_DB_HOST overrides config.yaml's database.host when
    # set — needed for container execution (Airflow inside Docker
    # can't reach the host's PostgreSQL via "localhost"; it needs
    # host.docker.internal instead). Falls back to config.yaml's
    # value when unset, so direct PowerShell runs are unaffected.
    # Symmetric with the existing SENTINEL_DB_PASSWORD pattern.
    db_host = os.environ.get("SENTINEL_DB_HOST", DB["host"])
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    conn = psycopg2.connect(
        host=db_host, port=DB["port"], dbname=DB["name"],
        user="postgres", password=password,
    )
    try:
        df = pd.read_sql(
            f"SELECT * FROM {DB['schema']}.customer_features", conn
        )
    finally:
        conn.close()
    df["customer_id"] = df["customer_id"].astype(str)
    return df[df[FILTER_COLUMN] == FILTER_VALUE].copy()


def load_reference_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs the exact canonical training population."""
    ids_path = os.path.join(ARTIFACTS_DIR, "split_ids_train.csv")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(
            f"{ids_path} not found. Run train_baseline_models.py first."
        )
    train_ids = pd.read_csv(ids_path)["customer_id"].astype(str)
    indexed = frame.set_index("customer_id")
    missing = train_ids[~train_ids.isin(indexed.index)]
    if len(missing):
        raise ValueError(
            f"{len(missing):,} canonical training IDs are missing from "
            "the current feature store."
        )
    return indexed.loc[train_ids.values].reset_index()


def load_comparison_population(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    source: 'temporal_test' (default, uses split_ids_test.csv is
    the STRATIFIED test set — temporal comparison instead uses the
    temporal split date directly, since no separate temporal test
    ID file is persisted) or a path to a CSV of customer_ids to
    treat as the scoring population.
    """
    if source == "temporal_test":
        model_cfg = CONFIG.get("model", {})
        split_date = model_cfg.get("temporal_split_date")
        if not split_date:
            raise ValueError(
                "model.temporal_split_date is not set in config.yaml — "
                "required to reconstruct the temporal comparison population."
            )
        split_date = pd.Timestamp(split_date)
        first_tx = pd.to_datetime(frame["first_transaction_date"])
        return frame[first_tx >= split_date].copy()

    if not os.path.exists(source):
        raise FileNotFoundError(f"Comparison population file not found: {source}")
    ids = pd.read_csv(source)["customer_id"].astype(str)
    indexed = frame.set_index("customer_id")
    missing = ids[~ids.isin(indexed.index)]
    if len(missing):
        raise ValueError(
            f"{len(missing):,} IDs from {source} are missing from the "
            "current feature store."
        )
    return indexed.loc[ids.values].reset_index()


def population_stability_index(
    expected: pd.Series, actual: pd.Series, n_bins: int
) -> float:
    """
    Same PSI calculation as govern_model_features.py's
    population_stability_index — duplicated rather than imported
    to keep this script runnable standalone (it may run on a
    schedule independent of a governance pass), but identical in
    logic so a "material_shift" here means the same thing there.
    """
    expected = pd.Series(expected).copy()
    actual = pd.Series(actual).copy()
    combined_unique = pd.concat([expected, actual]).nunique(dropna=True)

    if combined_unique <= n_bins:
        expected_bucket = expected.astype("object").where(expected.notna(), "__MISSING__")
        actual_bucket = actual.astype("object").where(actual.notna(), "__MISSING__")
        categories = sorted(set(expected_bucket.astype(str)) | set(actual_bucket.astype(str)))
        expected_dist = expected_bucket.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
        actual_dist = actual_bucket.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    else:
        non_missing_expected = expected.dropna()
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(non_missing_expected.quantile(quantiles).values)
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        expected_bucket = pd.cut(expected, bins=edges, include_lowest=True).astype(str)
        actual_bucket = pd.cut(actual, bins=edges, include_lowest=True).astype(str)
        expected_bucket = expected_bucket.where(expected.notna(), "__MISSING__")
        actual_bucket = actual_bucket.where(actual.notna(), "__MISSING__")
        categories = sorted(set(expected_bucket) | set(actual_bucket))
        expected_dist = expected_bucket.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        actual_dist = actual_bucket.value_counts(normalize=True).reindex(categories, fill_value=0.0)

    epsilon = 1e-6
    expected_dist = expected_dist.clip(lower=epsilon)
    actual_dist = actual_dist.clip(lower=epsilon)
    return float(((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum())


def psi_band(psi: float) -> str:
    if psi < PSI_MODERATE:
        return "stable"
    if psi < PSI_HIGH:
        return "moderate_shift"
    return "material_shift"


def check_feature_drift(reference: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in MODEL_FEATURES:
        ref_series = reference[feature]
        cmp_series = comparison[feature]

        psi = population_stability_index(ref_series, cmp_series, PSI_BINS)

        ref_null_rate = float(ref_series.isna().mean())
        cmp_null_rate = float(cmp_series.isna().mean())
        null_rate_delta = cmp_null_rate - ref_null_rate

        ref_mean = float(ref_series.mean())
        ref_std = float(ref_series.std()) or np.nan
        cmp_mean = float(cmp_series.mean())
        mean_shift_sigma = (
            abs(cmp_mean - ref_mean) / ref_std if ref_std and not np.isnan(ref_std) else np.nan
        )

        ref_min, ref_max = float(ref_series.min()), float(ref_series.max())
        out_of_range = (cmp_series < ref_min) | (cmp_series > ref_max)
        out_of_range_rate = float(out_of_range.mean())

        alerts = []
        band = psi_band(psi)
        if band != "stable":
            alerts.append(f"psi_{band}")
        if null_rate_delta > NULL_RATE_DELTA_ALERT:
            alerts.append("null_rate_increase")
        if out_of_range_rate > OUT_OF_RANGE_RATE_ALERT:
            alerts.append("values_outside_training_range")
        if not np.isnan(mean_shift_sigma) and mean_shift_sigma > MEAN_SHIFT_SIGMA_ALERT:
            alerts.append("mean_shift")

        rows.append({
            "feature": feature,
            "psi": round(psi, 4),
            "psi_band": band,
            "reference_null_rate": round(ref_null_rate, 4),
            "comparison_null_rate": round(cmp_null_rate, 4),
            "null_rate_delta": round(null_rate_delta, 4),
            "reference_mean": round(ref_mean, 4),
            "comparison_mean": round(cmp_mean, 4),
            "mean_shift_sigma": round(mean_shift_sigma, 4) if not np.isnan(mean_shift_sigma) else None,
            "out_of_range_rate": round(out_of_range_rate, 4),
            "alerts": "; ".join(alerts) if alerts else "none",
            "any_alert": bool(alerts),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def check_target_rate_shift(reference: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    """
    Bad-rate drift is not a feature-drift question but is the
    single most decision-relevant number to monitor: if the
    portfolio's actual risk composition is moving, that matters
    even if every individual feature looks stable.
    """
    ref_rate = float(reference[TARGET_COLUMN].mean())
    cmp_rate = float(comparison[TARGET_COLUMN].mean()) if TARGET_COLUMN in comparison.columns else None
    return {
        "reference_bad_rate": round(ref_rate, 4),
        "comparison_bad_rate": round(cmp_rate, 4) if cmp_rate is not None else None,
        "absolute_delta": round(cmp_rate - ref_rate, 4) if cmp_rate is not None else None,
    }


def find_champion_run_id() -> str | None:
    """
    Best-effort link back to the champion training run, so this
    monitoring run can be cross-referenced from MLflow — but
    monitoring should never hard-fail just because that link
    can't be established (e.g. multiple matching runs is fine
    here, unlike governance scripts that must attach evidence to
    exactly one run).
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(MLFLOW_CFG["experiment_name"])
        if experiment is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"params.model_type = '{CHAMPION_MODEL}' and "
                f"tags.feature_set = '{FEATURE_SET_LABEL}'"
            ),
            order_by=["start_time DESC"],
            max_results=1,
        )
        return str(runs.iloc[0]["run_id"]) if not runs.empty else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population", default="temporal_test",
        help=(
            "'temporal_test' (default) uses later-originating customers "
            "as the comparison population. Alternatively, a path to a "
            "CSV with a customer_id column to use as the scoring "
            "population being monitored."
        ),
    )
    parser.add_argument(
        "--skip-mlflow", action="store_true",
        help="Generate local evidence without logging to MLflow.",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("SENTINEL CREDIT — DRIFT AND DATA QUALITY MONITORING")
    print("=" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame = load_full_population()
    reference = load_reference_population(frame)
    comparison = load_comparison_population(frame, args.population)

    print(f"Reference population (canonical training): {len(reference):,} customers")
    print(f"Comparison population ({args.population}): {len(comparison):,} customers")

    if comparison.empty:
        raise RuntimeError(
            "Comparison population is empty — nothing to monitor. Check "
            "--population and, if using 'temporal_test', "
            "model.temporal_split_date in config.yaml."
        )

    drift = check_feature_drift(reference, comparison)
    target_shift = check_target_rate_shift(reference, comparison)

    print("\n" + "-" * 62)
    print("FEATURE DRIFT")
    print("-" * 62)
    print(drift[["feature", "psi", "psi_band", "null_rate_delta",
                  "mean_shift_sigma", "out_of_range_rate", "alerts"]].to_string(index=False))

    print("\n" + "-" * 62)
    print("PORTFOLIO BAD-RATE SHIFT")
    print("-" * 62)
    print(f"  Reference bad rate:   {target_shift['reference_bad_rate']:.2%}")
    if target_shift["comparison_bad_rate"] is not None:
        print(f"  Comparison bad rate:  {target_shift['comparison_bad_rate']:.2%}")
        print(f"  Absolute delta:       {target_shift['absolute_delta']:+.2%}")
    else:
        print("  Comparison population has no target column (unlabelled "
              "scoring batch) — bad-rate comparison skipped.")

    n_alerts = int(drift["any_alert"].sum())
    print("\n" + "-" * 62)
    if n_alerts:
        print(f"ALERT: {n_alerts} of {len(drift)} features triggered a drift "
              f"or data-quality flag.")
        print(drift.loc[drift["any_alert"], ["feature", "alerts"]].to_string(index=False))
    else:
        print("No features triggered a drift or data-quality alert.")
    print("-" * 62)

    drift_path = os.path.join(OUTPUT_DIR, "feature_drift.csv")
    drift.to_csv(drift_path, index=False)

    champion_run_id = find_champion_run_id()

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_population": "canonical_stratified_training_partition",
        "comparison_population_source": args.population,
        "n_reference": int(len(reference)),
        "n_comparison": int(len(comparison)),
        "model_version": MODEL_VERSION,
        "feature_set_label": FEATURE_SET_LABEL,
        "dataset_version": DATASET_VERSION,
        "champion_training_run_id": champion_run_id,
        "psi_thresholds": {"moderate": PSI_MODERATE, "high": PSI_HIGH},
        "data_quality_thresholds": {
            "null_rate_delta_alert": NULL_RATE_DELTA_ALERT,
            "out_of_range_rate_alert": OUT_OF_RANGE_RATE_ALERT,
            "mean_shift_sigma_alert": MEAN_SHIFT_SIGMA_ALERT,
        },
        "target_rate_shift": target_shift,
        "n_features_with_alerts": n_alerts,
        "n_features_total": len(drift),
        "note": (
            "This run does not gate deployment or trigger retraining by "
            "itself. It is evidence for the not-yet-built Approval Gate "
            "(Governance Layer) and Champion/Challenger promotion "
            "workflow. Comparison population is a proxy (temporal test "
            "split) in the absence of a live scoring stream — see module "
            "docstring."
        ),
    }
    metadata_path = os.path.join(OUTPUT_DIR, "monitoring_run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not args.skip_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MONITORING_EXPERIMENT)
        with mlflow.start_run(
            run_name=f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            mlflow.set_tags({
                "model_version": MODEL_VERSION,
                "feature_set": FEATURE_SET_LABEL,
                "dataset_version": DATASET_VERSION,
                "comparison_population_source": args.population,
                "champion_training_run_id": champion_run_id or "unknown",
            })
            mlflow.log_metric("n_features_with_alerts", n_alerts)
            mlflow.log_metric("max_psi", float(drift["psi"].max()))
            mlflow.log_metric("mean_psi", float(drift["psi"].mean()))
            if target_shift["absolute_delta"] is not None:
                mlflow.log_metric("bad_rate_absolute_delta", target_shift["absolute_delta"])
            mlflow.log_artifact(drift_path)
            mlflow.log_artifact(metadata_path)
        print(f"\nLogged to MLflow experiment: {MONITORING_EXPERIMENT}")

    print(f"\nSaved:")
    print(f"  {drift_path}")
    print(f"  {metadata_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()

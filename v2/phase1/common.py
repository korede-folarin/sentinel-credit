"""
Sentinel Credit v2, Phase 1 — shared data, splits and metrics (V2-02).

Plain English: everything the challenger needs to be compared FAIRLY with
the v1.1 champion. It reads the same customers, rebuilds exactly the same
train/test splits, uses the same 13 features, the same missing-value fill,
the same business cutoff, and measures every model with v1's own metric
code (models/evaluate_models.py), imported read-only.

Nothing here writes to v1. Reads: feature_store.customer_features, the
frozen split files in models/artifacts_v1_1/, config.yaml. Writes: only the
v2_results schema (write_frame) and v2 folders.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

import evaluate_models as v1_eval  # noqa: E402  (v1 metric code, read-only)
from model_features import FILTER_COLUMN, FILTER_VALUE, MODEL_FEATURES, TARGET_COLUMN  # noqa: E402

with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
    V1 = yaml.safe_load(f)
with open(os.path.join(ROOT, "v2", "config_v2.yaml"), encoding="utf-8") as f:
    V2 = yaml.safe_load(f)["phase1"]

CUTOFF = float(V1["model"]["business_cutoff"])                  # 0.0657, v1 Decision 27
TEMPORAL_SPLIT_DATE = pd.Timestamp(V1["model"]["temporal_split_date"])
V1_PACKAGE = os.path.join(ROOT, V1["model_governance"]["artifacts_dir"])
SEED = int(V2["random_seed"])
N_SPLITS = 5                                                     # params.yaml cv n_splits (v1)


def db_connect():
    db = V1["database"]
    return psycopg2.connect(host=os.environ.get("SENTINEL_DB_HOST", db["host"]), port=db["port"],
                            dbname=db["name"], user="postgres",
                            password=os.environ.get("SENTINEL_DB_PASSWORD", ""))


def load_population() -> pd.DataFrame:
    """The modelable population, read-only, exactly as v1's training loaded it."""
    conn = db_connect()
    try:
        df = pd.read_sql(f"SELECT * FROM {V1['database']['schema']}.customer_features", conn)
    finally:
        conn.close()
    return df[df[FILTER_COLUMN] == FILTER_VALUE].reset_index(drop=True)


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """v1's missing-value rule: fill with the population median (train_baseline_models.split_data)."""
    return df[MODEL_FEATURES].fillna(df[MODEL_FEATURES].median(numeric_only=True))


def splits(df: pd.DataFrame) -> dict:
    """The champion's splits, rebuilt and checked against the frozen v1 files.

    stratified: the customer IDs saved in the v1.1 package (33,283 / 8,321).
    temporal:   first_transaction_date before / on-or-after the configured date,
                as v1 did; its test IDs are checked against the champion's
                saved temporal test predictions.
    """
    X, y, ids = model_matrix(df), df[TARGET_COLUMN].astype(int), df["customer_id"].astype(str)
    train_ids = set(pd.read_csv(os.path.join(V1_PACKAGE, "split_ids_train.csv"))["customer_id"].astype(str))
    test_ids = set(pd.read_csv(os.path.join(V1_PACKAGE, "split_ids_test.csv"))["customer_id"].astype(str))
    s_train, s_test = ids.isin(train_ids).values, ids.isin(test_ids).values
    first_tx = pd.to_datetime(df["first_transaction_date"])
    t_train, t_test = (first_tx < TEMPORAL_SPLIT_DATE).values, (first_tx >= TEMPORAL_SPLIT_DATE).values

    champ_temporal = pd.read_csv(os.path.join(V1_PACKAGE, "test_predictions_temporal.csv"),
                                 usecols=["customer_id"])["customer_id"].astype(str)
    checks = {
        "stratified_train_matches_v1": int(s_train.sum()) == len(train_ids),
        "stratified_test_matches_v1": int(s_test.sum()) == len(test_ids),
        "stratified_disjoint": not (s_train & s_test).any(),
        "temporal_test_matches_v1": set(ids[t_test]) == set(champ_temporal),
    }
    if not all(checks.values()):
        raise SystemExit(f"Split does not reproduce the champion's: {checks}")
    out = {}
    for name, (tr, te) in {"stratified": (s_train, s_test), "temporal": (t_train, t_test)}.items():
        out[name] = {"X_train": X[tr].reset_index(drop=True), "y_train": y[tr].reset_index(drop=True),
                     "X_test": X[te].reset_index(drop=True), "y_test": y[te].reset_index(drop=True),
                     "ids_test": ids[te].reset_index(drop=True)}
    out["_checks"] = checks
    return out


def champion_predictions(split: str) -> pd.DataFrame:
    """The champion's saved held-out predictions (customer_id, proba)."""
    df = pd.read_csv(os.path.join(V1_PACKAGE, f"test_predictions_{split}.csv"),
                     usecols=["customer_id", "proba_xgboost"])
    return df.rename(columns={"proba_xgboost": "proba"}).assign(customer_id=lambda d: d["customer_id"].astype(str))


def evaluate(y, proba) -> dict:
    """Every number with v1's definitions (models/evaluate_models.py)."""
    y, proba = np.asarray(y).astype(int), np.asarray(proba, dtype=float)
    auc = roc_auc_score(y, proba)
    cm = v1_eval.confusion_matrix_at_cutoff(y, proba, CUTOFF)
    br = v1_eval.bad_rate_at_cutoff(y, proba, CUTOFF)
    return {
        "n_test": int(len(y)), "n_defaults": int(y.sum()),
        "gini": round(v1_eval.gini_from_auc(auc), 4), "auc": round(auc, 4),
        "ks": round(v1_eval.ks_statistic(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "ece": v1_eval.expected_calibration_error(y, proba),
        "business_cutoff": CUTOFF,
        "share_sent_to_review": round(float((proba >= CUTOFF).mean()), 4),
        "recall_at_cutoff": cm["recall"], "precision_at_cutoff": cm["precision"],
        "bad_rate_not_reviewed": br["bad_rate_among_approved"],
    }


def segment_metrics(frame: pd.DataFrame, proba_col: str) -> pd.DataFrame:
    """Error rates by segment at the cutoff, v1's fairness_parity definitions."""
    rows = []
    for seg in ("thin_file_flag", "is_vulnerable"):
        for value, sub in frame.groupby(seg):
            y, p = sub[TARGET_COLUMN].values.astype(int), sub[proba_col].values
            flag = p >= CUTOFF
            rows.append({"segment": seg, "segment_value": str(int(value)), "n_customers": len(sub),
                         "n_defaults": int(y.sum()), "actual_default_rate": round(float(y.mean()), 4),
                         "avg_predicted_pd": round(float(p.mean()), 4),
                         "share_sent_to_review": round(float(flag.mean()), 4),
                         "false_negative_rate": round(float((~flag & (y == 1)).sum() / max((y == 1).sum(), 1)), 4),
                         "false_positive_rate": round(float((flag & (y == 0)).sum() / max((y == 0).sum(), 1)), 4)})
    return pd.DataFrame(rows)


def ensure_schema():
    with open(os.path.join(ROOT, "v2", "sql", "phase1_results.sql"), encoding="utf-8") as f:
        ddl = f.read()
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(ddl)
    finally:
        conn.close()


def write_frame(table: str, frame: pd.DataFrame):
    """Append rows to a v2_results table. Refuses any other schema."""
    schema = V2["results_schema"]
    if not table.startswith(schema + "."):
        raise ValueError(f"v2 writes only to {schema}.*, not {table}")
    cols = list(frame.columns)
    values = [tuple(None if (isinstance(v, float) and np.isnan(v)) else
                    (json.dumps(v) if isinstance(v, (dict, list)) else
                     (v.item() if hasattr(v, "item") else v)) for v in row)
              for row in frame.itertuples(index=False)]
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s", values)
    finally:
        conn.close()

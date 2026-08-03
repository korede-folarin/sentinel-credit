"""
=============================================================
Sentinel Credit — Model Evaluation
=============================================================
File:    evaluate_models.py
Purpose: Full metric suite for the three baseline models:
         discrimination (Gini, KS, PR-AUC), calibration (ECE,
         A/E by decile), threshold performance (precision,
         recall, lift, capture rate, confusion matrix), and
         fairness parity across thin_file_flag and
         is_vulnerable segments. Reads test_predictions.csv
         produced by train_baseline_models.py.

Deliberately out of scope, not omissions:
  - Expected Profit/Loss, Cost Matrix — Expected Loss =
    PD x LGD x EAD. This platform is PD-only (Decision 4
    addendum); LGD and EAD are explicitly out of scope, since
    the generator does not model post-default recovery
    behaviour. A cost-matrix version using assumed unit costs
    (not real LGD/EAD) was considered and declined.
  - PSI / drift monitoring — belongs to Decision 5's scheduled
    Airflow pipeline, which evaluates change over time against
    live data. This script evaluates a trained model once, at
    a point in time. Different job, different file, tracked
    separately in README as in-progress.
Author:  Folarin Korede
Version: 1.0.0

Fairness note: is_vulnerable and thin_file_flag are used here
ONLY as comparison-group labels, exactly as
docs/engineering_decision_log.md Decision 12 (vulnerability
addendum) specifies. Neither is a model input.

See docs/engineering_decision_log.md — Decision 4, Decision 12.
=============================================================
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import psycopg2
import yaml
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    precision_recall_curve, roc_curve,
)

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from model_features import TARGET_COLUMN, FILTER_COLUMN, FILTER_VALUE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB = CONFIG["database"]

# Business cutoff — the score threshold above which a
# customer is treated as high risk. Design assumption,
# revisit once portfolio-level loss appetite is defined.
BUSINESS_CUTOFF = CONFIG.get("model", {}).get("business_cutoff", 0.10)

MODEL_NAMES = ["logistic_regression", "decision_tree", "xgboost"]


def load_predictions(split: str) -> pd.DataFrame:
    path = os.path.join(ARTIFACTS_DIR, f"test_predictions_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run train_baseline_models.py first "
            f"(it now saves predictions for both split strategies)."
        )
    return pd.read_csv(path)


def load_segment_labels() -> pd.DataFrame:
    """
    Pulls is_vulnerable and ab_test_group from the feature
    store for fairness segmentation. thin_file_flag is NOT
    re-fetched here — it's back in MODEL_FEATURES (restored
    30th July 2026 after the exclusion experiment produced an
    identical fairness gap either way — see model_features.py
    NON_MODEL_EXCLUDED comment), so it's already a column in
    the test predictions. Fetching it again would collide on
    merge, silently renaming both to thin_file_flag_x/_y.
    These are NEVER used as model inputs — comparison-group
    labels only.
    """
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    conn = psycopg2.connect(
        host=DB["host"], port=DB["port"], dbname=DB["name"],
        user="postgres", password=password,
    )
    try:
        df = pd.read_sql(
            f"SELECT customer_id, is_vulnerable, "
            f"ab_test_group FROM {DB['schema']}.customer_features "
            f"WHERE {FILTER_COLUMN} = {FILTER_VALUE}",
            conn,
        )
    finally:
        conn.close()
    return df


# ── DISCRIMINATION ─────────────────────────────────────────
def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: max separation between the
    cumulative distributions of scores for good vs bad cases.
    """
    order = np.argsort(proba)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


# ── CALIBRATION ─────────────────────────────────────────────
def expected_calibration_error(y_true, proba, n_bins=10) -> float:
    """
    ECE: weighted average gap between predicted probability
    and observed default rate, across probability bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = proba[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 4)


def actual_vs_expected_by_decile(y_true, proba) -> pd.DataFrame:
    """
    A/E ratio per risk decile — the core backtesting table
    real credit risk teams live by (PRA SS1/23, EBA PD
    backtesting guidance). Ratio near 1.0 means predicted and
    observed default rates agree in that decile.
    """
    df = pd.DataFrame({"y": y_true, "proba": proba})
    df["decile"] = pd.qcut(df["proba"], 10, labels=False, duplicates="drop")

    grouped = df.groupby("decile").agg(
        n=("y", "count"),
        predicted_pd=("proba", "mean"),
        actual_default_rate=("y", "mean"),
    )
    grouped["ae_ratio"] = round(
        grouped["actual_default_rate"] / grouped["predicted_pd"].replace(0, np.nan), 4
    )
    return grouped.round(4)


# ── THRESHOLD OPTIMIZATION ────────────────────────────────────
def find_f1_optimal_threshold(y_true, proba) -> dict:
    """
    Scans every threshold implied by the precision-recall
    curve and returns the one maximizing F1 (harmonic mean of
    precision and recall — equal weight to both). Found, not
    guessed: no assumed cost ratio, no LGD/EAD needed, stays
    inside the PD-only scope (Decision 4 addendum).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one more point than
    # thresholds (the (1, 0) endpoint with no threshold) — drop it.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) > 0,
    )
    best_idx = int(np.argmax(f1))
    return {
        "threshold": round(float(thresholds[best_idx]), 4),
        "f1": round(float(f1[best_idx]), 4),
        "precision": round(float(precision[best_idx]), 4),
        "recall": round(float(recall[best_idx]), 4),
    }


def find_youden_optimal_threshold(y_true, proba) -> dict:
    """
    Scans the ROC curve and returns the threshold maximizing
    Youden's J = Sensitivity + Specificity - 1. Standard
    textbook method for picking an ROC operating point,
    equal weight to catching bads (sensitivity) and correctly
    passing goods (specificity).
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return {
        "threshold": round(float(thresholds[best_idx]), 4),
        "youden_j": round(float(j[best_idx]), 4),
        "sensitivity": round(float(tpr[best_idx]), 4),
        "specificity": round(float(1 - fpr[best_idx]), 4),
    }


# ── THRESHOLD PERFORMANCE ────────────────────────────────────
def lift_and_capture_table(y_true, proba, n_bins=10) -> pd.DataFrame:
    """
    Lift and cumulative capture rate by predicted-risk decile
    (highest risk first). Lift at a decile: how many times
    more concentrated bad cases are in that decile versus the
    overall bad rate. Capture rate: cumulative share of all
    bad cases caught by including deciles down to this one —
    answers "if I review the top N% highest-risk customers,
    what fraction of actual defaulters have I found."
    """
    df = pd.DataFrame({"y": y_true, "proba": proba})
    df["decile"] = pd.qcut(
        df["proba"].rank(method="first", ascending=False), n_bins, labels=False
    )
    overall_bad_rate = df["y"].mean()

    grouped = df.groupby("decile").agg(n=("y", "count"), bad=("y", "sum"))
    grouped["bad_rate"] = grouped["bad"] / grouped["n"]
    grouped["lift"] = round(grouped["bad_rate"] / overall_bad_rate, 2)
    grouped["cum_bad"] = grouped["bad"].cumsum()
    grouped["capture_rate"] = round(grouped["cum_bad"] / grouped["bad"].sum(), 4)
    grouped["pct_population"] = round((grouped["n"].cumsum() / len(df)), 4)

    return grouped[["n", "bad", "bad_rate", "lift", "capture_rate", "pct_population"]].round(4)


def confusion_matrix_at_cutoff(y_true, proba, cutoff: float) -> dict:
    """
    Full TP/FP/TN/FN breakdown at the business cutoff, plus
    precision (of those flagged high-risk, how many actually
    default) alongside the recall already computed in
    bad_rate_at_cutoff.
    """
    pred = (proba >= cutoff).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = round(tp / max(tp + fp, 1), 4)
    recall = round(tp / max(tp + fn, 1), 4)

    return {
        "cutoff": cutoff, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
    }


# ── BUSINESS RELEVANCE ───────────────────────────────────────
def bad_rate_at_cutoff(y_true, proba, cutoff: float) -> dict:
    """
    At the actual approval/decline threshold, what fraction of
    approved (below-cutoff) customers go bad, and what fraction
    of bads would have been caught (recall) at that cutoff.
    More operationally relevant than AUC, since the business
    only ever operates at one threshold.
    """
    flagged = proba >= cutoff
    n_flagged = int(flagged.sum())
    n_caught = int((flagged & (y_true == 1)).sum())
    n_total_bad = int((y_true == 1).sum())

    approved = ~flagged
    bad_rate_approved = (
        float(y_true[approved].mean()) if approved.sum() else None
    )

    return {
        "cutoff": cutoff,
        "n_flagged_high_risk": n_flagged,
        "recall_at_cutoff": round(n_caught / max(n_total_bad, 1), 4),
        "bad_rate_among_approved": (
            round(bad_rate_approved, 4) if bad_rate_approved is not None else None
        ),
    }


# ── FAIRNESS ──────────────────────────────────────────────────
def fairness_parity(df: pd.DataFrame, proba_col: str, segment_col: str) -> pd.DataFrame:
    """
    Default-rate and error-rate parity across a segment.
    segment_col (thin_file_flag, is_vulnerable, ab_test_group)
    is a comparison-group LABEL only — never a model input.
    Checks whether the model is equally accurate across
    segments, not just accurate on average (Decision 12
    vulnerability addendum; FCA Consumer Duty outcomes
    monitoring).
    """
    rows = []
    for seg_value, sub in df.groupby(segment_col):
        y = sub[TARGET_COLUMN].values
        proba = sub[proba_col].values
        pred = (proba >= BUSINESS_CUTOFF).astype(int)

        rows.append({
            "segment": f"{segment_col}={seg_value}",
            "n": len(sub),
            "actual_default_rate": round(float(y.mean()), 4),
            "avg_predicted_pd": round(float(proba.mean()), 4),
            "false_negative_rate": round(
                float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)), 4
            ),
            "false_positive_rate": round(
                float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)), 4
            ),
        })
    return pd.DataFrame(rows)


def print_model_report(name: str, df: pd.DataFrame) -> dict:
    y_true = df[TARGET_COLUMN].values
    proba = df[f"proba_{name}"].values

    auc = roc_auc_score(y_true, proba)
    gini = gini_from_auc(auc)
    ks = ks_statistic(y_true, proba)
    ece = expected_calibration_error(y_true, proba)
    brier = round(float(brier_score_loss(y_true, proba)), 4)
    pr_auc = round(float(average_precision_score(y_true, proba)), 4)

    print("\n" + "=" * 58)
    print(f"{name.upper()}")
    print("=" * 58)
    print(f"Gini:   {gini:.4f}   AUC: {auc:.4f}   KS: {ks:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}   ECE: {ece:.4f}   Brier: {brier:.4f}")

    print("\nActual vs Expected by risk decile:")
    ae = actual_vs_expected_by_decile(y_true, proba)
    print(ae.to_string())

    print("\nLift and capture rate by risk decile (highest risk first):")
    lift_table = lift_and_capture_table(y_true, proba)
    print(lift_table.to_string())

    cm = confusion_matrix_at_cutoff(y_true, proba, BUSINESS_CUTOFF)
    print(f"\nConfusion matrix at CONFIGURED cutoff {BUSINESS_CUTOFF}:")
    print(f"  TP: {cm['tp']:>6}   FP: {cm['fp']:>6}")
    print(f"  FN: {cm['fn']:>6}   TN: {cm['tn']:>6}")
    print(f"  Precision: {cm['precision']:.2%}   Recall: {cm['recall']:.2%}")

    f1_opt = find_f1_optimal_threshold(y_true, proba)
    youden_opt = find_youden_optimal_threshold(y_true, proba)
    print(f"\nOptimal thresholds (found, not guessed):")
    print(f"  F1-optimal:     {f1_opt['threshold']:.4f}  "
          f"(F1={f1_opt['f1']:.4f}, precision={f1_opt['precision']:.2%}, "
          f"recall={f1_opt['recall']:.2%})")
    print(f"  Youden-optimal: {youden_opt['threshold']:.4f}  "
          f"(J={youden_opt['youden_j']:.4f}, sensitivity={youden_opt['sensitivity']:.2%}, "
          f"specificity={youden_opt['specificity']:.2%})")
    if abs(f1_opt['threshold'] - BUSINESS_CUTOFF) > 0.02:
        print(f"  NOTE: configured cutoff ({BUSINESS_CUTOFF}) differs from "
              f"F1-optimal ({f1_opt['threshold']:.4f}) by more than 0.02 — "
              f"worth reviewing config.yaml business_cutoff.")

    cutoff_result = bad_rate_at_cutoff(y_true, proba, BUSINESS_CUTOFF)
    print(f"\nAt configured cutoff {BUSINESS_CUTOFF}:")
    print(f"  Flagged high-risk: {cutoff_result['n_flagged_high_risk']:,}")
    print(f"  Bad rate among approved: {cutoff_result['bad_rate_among_approved']:.2%}")

    print("\nFairness — thin_file_flag:")
    print(fairness_parity(df, f"proba_{name}", "thin_file_flag").to_string(index=False))

    print("\nFairness — is_vulnerable (label only, never a model input):")
    print(fairness_parity(df, f"proba_{name}", "is_vulnerable").to_string(index=False))

    summary_row = {
        "model": name, "gini": round(gini, 4), "auc": round(auc, 4),
        "ks": round(ks, 4), "pr_auc": pr_auc, "ece": ece, "brier": brier,
    }
    lift_table_out = lift_table.reset_index().rename(columns={"index": "decile"})
    lift_table_out.insert(0, "model", name)

    return {"summary": summary_row, "lift_table": lift_table_out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["stratified", "temporal"], default="stratified",
        help="Which split's predictions to evaluate (default: stratified)"
    )
    args = parser.parse_args()

    print("=" * 58)
    print("SENTINEL CREDIT — MODEL EVALUATION")
    print(f"Split: {args.split}")
    print("=" * 58)

    df = load_predictions(args.split)
    segments = load_segment_labels()

    df = df.merge(segments, on="customer_id", how="left")

    summaries = []
    lift_tables = []
    for name in MODEL_NAMES:
        result = print_model_report(name, df)
        summaries.append(result["summary"])
        lift_tables.append(result["lift_table"])

    out_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(summaries).to_csv(
        os.path.join(out_dir, f"model_comparison_{args.split}.csv"), index=False
    )
    pd.concat(lift_tables, ignore_index=True).to_csv(
        os.path.join(out_dir, f"lift_tables_{args.split}.csv"), index=False
    )
    print(f"\nSaved: {out_dir}/model_comparison_{args.split}.csv")
    print(f"Saved: {out_dir}/lift_tables_{args.split}.csv")


if __name__ == "__main__":
    main()

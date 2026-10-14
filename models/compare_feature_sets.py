"""
=============================================================
Sentinel Credit — Feature Set A/B Comparison
=============================================================
File:    compare_feature_sets.py
Purpose: Measures how model behaviour changes when the
         EXPOSURE_TIME_SUSPECTED features are removed.
Author:  Folarin Korede
Version: 1.0.0

This answers a different engineering question from
explain_models.py, which is why it is a separate module:

    explain_models.py  -> why did the model make this
                          prediction?
    this module        -> how does the model's behaviour change
                          when potentially contaminated features
                          are removed?

The second question requires training a SECOND model, which is
an experiment, not an explanation.

BACKGROUND
The corrected has_outcome logic has two asymmetric qualifying
paths: a defaulted customer qualifies regardless of timing,
but a non-defaulted customer only qualifies if all their
repayments have fully matured. Verified: target_default=1
customers average 313.5 days of transaction_span_days against
204.1 for target_default=0. Cumulative exposure features
therefore inflate with available calendar time, independent of
risk. See docs/engineering_decision_log.md Decision 12
addendum, "Unequal Observation Opportunity".

HOW TO READ THE RESULT
If Feature Set B (full) performs dramatically better than
Feature Set A (conservative), that is NOT evidence the
suspected features are genuinely predictive. It is evidence of
how much apparent performance depends on the observation-window
asymmetry. The observation-window analysis determines
defensibility; this comparison only quantifies dependence.

Both models here are uncalibrated, trained with the same
split, seed and hyperparameters, so the comparison isolates
the feature set as the single changed variable.
=============================================================
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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import (
    MODEL_FEATURES, TARGET_COLUMN, FILTER_COLUMN, FILTER_VALUE,
    EXPOSURE_TIME_SUSPECTED,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "feature_set_ab")

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    CONFIG["model_governance"].get("artifacts_dir", "models/artifacts"),
)  # the governed champion package; see config.yaml model_governance

DB = CONFIG["database"]
MODEL_CFG = CONFIG.get("model", {})
RANDOM_SEED = MODEL_CFG.get("random_seed", 42)
TEST_SIZE = MODEL_CFG.get("test_size", 0.2)
BUSINESS_CUTOFF = MODEL_CFG.get("business_cutoff", 0.145)



def load_champion_config() -> dict:
    """
    Reads the champion's selected hyperparameters and split
    from champion_metadata.json rather than hardcoding them.

    Unlike explain_models.py, this module genuinely must
    retrain: Feature Set A and Feature Set B use different
    inputs, so no single saved model can serve both arms. What
    it does NOT do is invent its own hyperparameters — both
    comparison models use the champion's selected settings, so
    the feature set remains the only changed variable.

    The models trained here are EXPERIMENTAL COMPARISON models,
    not champions. They are not persisted to the governed model
    package.
    """
    path = os.path.join(ARTIFACTS_DIR, "champion_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run train_baseline_models.py first — "
            f"this comparison uses the champion's selected "
            f"hyperparameters, not its own."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

FEATURE_SET_A = [f for f in MODEL_FEATURES if f not in EXPOSURE_TIME_SUSPECTED]
FEATURE_SET_B = list(MODEL_FEATURES)


def load_data() -> tuple:
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    conn = psycopg2.connect(
        host=DB["host"], port=DB["port"], dbname=DB["name"],
        user="postgres", password=password,
    )
    try:
        df = pd.read_sql(
            f"SELECT * FROM {DB['schema']}.customer_features", conn
        )
    finally:
        conn.close()

    df = df[df[FILTER_COLUMN] == FILTER_VALUE].copy()
    X = df[MODEL_FEATURES].fillna(df[MODEL_FEATURES].median(numeric_only=True))
    y = df[TARGET_COLUMN]
    ids = df["customer_id"]

    return train_test_split(
        X, y, ids, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def ks_statistic(y_true, proba) -> float:
    order = np.argsort(proba)
    y_sorted = np.asarray(y_true)[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def train_arm(name: str, features: list, X_train, y_train, X_test, y_test,
              xgb_params: dict) -> dict:
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = XGBClassifier(scale_pos_weight=n_neg / max(n_pos, 1), **xgb_params)
    model.fit(X_train[features], y_train)

    proba = model.predict_proba(X_test[features])[:, 1]
    auc = roc_auc_score(y_test, proba)

    # Train-set performance too, so the train/test gap can be
    # compared between arms — a wider gap in one arm suggests
    # it is relying on something that generalises less well.
    proba_train = model.predict_proba(X_train[features])[:, 1]
    auc_train = roc_auc_score(y_train, proba_train)

    return {
        "arm": name,
        "n_features": len(features),
        "gini": round(gini_from_auc(auc), 4),
        "auc": round(auc, 4),
        "ks": round(ks_statistic(y_test, proba), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "gini_train": round(gini_from_auc(auc_train), 4),
        "train_test_gini_gap": round(gini_from_auc(auc_train) - gini_from_auc(auc), 4),
        "model": model,
        "proba": proba,
        "importances": dict(zip(features, model.feature_importances_)),
    }


# Repeated CV settings for the stability check. A single split
# cannot establish that one feature set is no worse than
# another — it shows one draw from a distribution. Repeating
# across folds and seeds answers whether the difference holds
# on average and how variable it is.
CV_N_SPLITS = 5
CV_N_REPEATS = 3


def repeated_cv_comparison(X, y, xgb_params: dict) -> pd.DataFrame:
    """
    Compares both feature sets across RepeatedStratifiedKFold,
    refitting each arm on every fold. Reports per-fold results
    so the variability of the difference is visible, not just
    its mean.

    Stratified because the ~4% positive rate means unstratified
    folds could land very few positives in one fold, making its
    score unreliable — the same reason the training pipeline
    uses StratifiedKFold for hyperparameter search.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS,
        random_state=RANDOM_SEED,
    )
    rows = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        spw = n_neg / max(n_pos, 1)

        for arm_name, features in [
            ("A_conservative", FEATURE_SET_A), ("B_full", FEATURE_SET_B)
        ]:
            m = XGBClassifier(scale_pos_weight=spw, **xgb_params)
            m.fit(X_tr[features], y_tr)

            proba_te = m.predict_proba(X_te[features])[:, 1]
            proba_tr = m.predict_proba(X_tr[features])[:, 1]
            auc_te = roc_auc_score(y_te, proba_te)
            auc_tr = roc_auc_score(y_tr, proba_tr)

            rows.append({
                "fold": fold_i,
                "arm": arm_name,
                "gini": gini_from_auc(auc_te),
                "auc": auc_te,
                "ks": ks_statistic(y_te, proba_te),
                "pr_auc": float(average_precision_score(y_te, proba_te)),
                "train_test_gini_gap": gini_from_auc(auc_tr) - gini_from_auc(auc_te),
            })

    return pd.DataFrame(rows)


def summarise_repeated_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-metric mean, standard deviation, and the paired
    per-fold difference (A minus B). The paired difference
    matters more than comparing two independent means, since
    both arms saw the identical folds.
    """
    metrics = ["gini", "auc", "ks", "pr_auc", "train_test_gini_gap"]
    wide = cv_df.pivot(index="fold", columns="arm", values=metrics)

    rows = []
    for m in metrics:
        a_vals = wide[(m, "A_conservative")]
        b_vals = wide[(m, "B_full")]
        diff = a_vals - b_vals
        rows.append({
            "metric": m,
            "A_mean": round(a_vals.mean(), 4),
            "A_std": round(a_vals.std(), 4),
            "B_mean": round(b_vals.mean(), 4),
            "B_std": round(b_vals.std(), 4),
            "mean_diff_A_minus_B": round(diff.mean(), 4),
            "std_diff": round(diff.std(), 4),
            "folds_A_better": int((diff > 0).sum()),
            "n_folds": int(len(diff)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 62)
    print("SENTINEL CREDIT — FEATURE SET A/B COMPARISON")
    print("=" * 62)
    print(f"Set A (conservative): {len(FEATURE_SET_A)} features")
    print(f"Set B (full):         {len(FEATURE_SET_B)} features")
    print(f"Removed in Set A:     {', '.join(EXPOSURE_TIME_SUSPECTED)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    champion_meta = load_champion_config()
    xgb_params = dict(champion_meta["best_params"])
    xgb_params.pop("scale_pos_weight", None)  # recomputed per arm from data
    xgb_params.update({"random_state": RANDOM_SEED, "eval_metric": "auc"})
    print(f"\nUsing champion hyperparameters ({champion_meta['model_version']}): "
          f"{champion_meta['best_params']}")

    X_train, X_test, y_train, y_test, ids_train, ids_test = load_data()
    print(f"\nTrain {len(X_train):,} / Test {len(X_test):,}  "
          f"bad rate {y_test.mean():.2%}")

    a = train_arm("A_conservative", FEATURE_SET_A, X_train, y_train,
                  X_test, y_test, xgb_params)
    b = train_arm("B_full", FEATURE_SET_B, X_train, y_train,
                  X_test, y_test, xgb_params)

    # ---- Performance comparison ----
    print("\n" + "-" * 62)
    print("PERFORMANCE")
    print("-" * 62)
    metrics = ["n_features", "gini", "auc", "ks", "pr_auc", "brier",
               "gini_train", "train_test_gini_gap"]
    print(f"{'Metric':<24} {'Set A':>12} {'Set B':>12} {'B - A':>10}")
    print("-" * 62)
    rows = []
    for m in metrics:
        diff = b[m] - a[m]
        print(f"{m:<24} {a[m]:>12} {b[m]:>12} {diff:>+10.4f}")
        rows.append({"metric": m, "set_a": a[m], "set_b": b[m], "difference": diff})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "performance_comparison.csv"), index=False
    )

    # ---- Decision-level comparison ----
    flag_a = (a["proba"] >= BUSINESS_CUTOFF).astype(int)
    flag_b = (b["proba"] >= BUSINESS_CUTOFF).astype(int)
    flipped = flag_a != flag_b

    print("\n" + "-" * 62)
    print(f"DECISIONS AT CUTOFF {BUSINESS_CUTOFF}")
    print("-" * 62)
    print(f"Set A flags: {flag_a.sum():,}   Set B flags: {flag_b.sum():,}")
    print(f"Decisions that flip: {flipped.sum():,} of {len(flipped):,} "
          f"({100 * flipped.mean():.2f}%)")

    decisions = pd.DataFrame({
        "customer_id": ids_test.values,
        "actual": y_test.values,
        "proba_a": a["proba"],
        "proba_b": b["proba"],
        "flag_a": flag_a,
        "flag_b": flag_b,
        "flipped": flipped,
        "swing": np.abs(b["proba"] - a["proba"]),
    })
    decisions.to_csv(
        os.path.join(OUT_DIR, "decision_comparison.csv"), index=False
    )

    if flipped.sum():
        print("\nLargest swings among flipped decisions:")
        top = decisions[decisions["flipped"]].nlargest(5, "swing")
        print(top[["customer_id", "actual", "proba_a", "proba_b",
                   "flag_a", "flag_b"]].to_string(index=False))

    # ---- Feature importance shift ----
    print("\n" + "-" * 62)
    print("TOP FEATURES BY ARM")
    print("-" * 62)
    imp_a = pd.Series(a["importances"]).sort_values(ascending=False)
    imp_b = pd.Series(b["importances"]).sort_values(ascending=False)
    print(f"{'Set A (conservative)':<38} {'Set B (full)':<38}")
    for i in range(8):
        left = f"{imp_a.index[i]} ({imp_a.iloc[i]:.3f})" if i < len(imp_a) else ""
        right = f"{imp_b.index[i]} ({imp_b.iloc[i]:.3f})" if i < len(imp_b) else ""
        print(f"{left:<38} {right:<38}")

    pd.DataFrame({
        "feature": list(set(FEATURE_SET_A) | set(FEATURE_SET_B)),
    }).assign(
        importance_a=lambda d: d["feature"].map(a["importances"]),
        importance_b=lambda d: d["feature"].map(b["importances"]),
        removed_in_a=lambda d: d["feature"].isin(EXPOSURE_TIME_SUSPECTED),
    ).sort_values("importance_b", ascending=False).to_csv(
        os.path.join(OUT_DIR, "importance_comparison.csv"), index=False
    )

    # ---- Repeated CV stability check ----
    print("\n" + "-" * 62)
    print(f"REPEATED STRATIFIED CV ({CV_N_SPLITS} folds x {CV_N_REPEATS} repeats "
          f"= {CV_N_SPLITS * CV_N_REPEATS} fits per arm)")
    print("-" * 62)
    print("A single split shows one draw. This shows whether the")
    print("difference holds on average, and how variable it is.\n")

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    cv_df = repeated_cv_comparison(X_all, y_all, xgb_params)
    cv_summary = summarise_repeated_cv(cv_df)
    print(cv_summary.to_string(index=False))

    cv_df.to_csv(os.path.join(OUT_DIR, "repeated_cv_folds.csv"), index=False)
    cv_summary.to_csv(os.path.join(OUT_DIR, "repeated_cv_summary.csv"), index=False)

    gini_row = cv_summary[cv_summary["metric"] == "gini"].iloc[0]
    print(f"\n  Gini: Set A wins {gini_row['folds_A_better']}/"
          f"{gini_row['n_folds']} folds, "
          f"mean difference {gini_row['mean_diff_A_minus_B']:+.4f} "
          f"(sd {gini_row['std_diff']:.4f})")
    print("  Reading: if the mean difference is near zero with Set A winning")
    print("  roughly half the folds, the conservative set is NO WORSE — which")
    print("  is the finding that matters, given the suspected features are")
    print("  hard to defend on observation-window grounds regardless.")

    # ---- Metadata ----
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "business_cutoff": BUSINESS_CUTOFF,
        "champion_model_version": champion_meta["model_version"],
        "xgb_params": xgb_params,
        "model_role": "experimental comparison models, not champions",
        "feature_set_a": FEATURE_SET_A,
        "feature_set_b": FEATURE_SET_B,
        "removed_in_a": EXPOSURE_TIME_SUSPECTED,
        "performance": {m: {"set_a": a[m], "set_b": b[m]} for m in metrics},
        "repeated_cv": {
            "n_splits": CV_N_SPLITS,
            "n_repeats": CV_N_REPEATS,
            "summary": cv_summary.to_dict(orient="records"),
        },
        "decisions_flipped": int(flipped.sum()),
        "decisions_total": int(len(flipped)),
        "pct_flipped": round(100 * float(flipped.mean()), 2),
        "interpretation_note": (
            "A large advantage for Set B is NOT evidence that the removed "
            "features are genuinely predictive. It quantifies how much "
            "apparent performance depends on features confounded with "
            "observation time. Defensibility is determined by the "
            "observation-window analysis (Decision 12 addendum), not by "
            "this comparison."
        ),
    }
    with open(os.path.join(OUT_DIR, "ab_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 62)
    print(f"Saved to {OUT_DIR}")
    print("  performance_comparison.csv")
    print("  decision_comparison.csv")
    print("  importance_comparison.csv")
    print("  repeated_cv_folds.csv")
    print("  repeated_cv_summary.csv")
    print("  ab_metadata.json")
    print("\nReading: a Set B advantage quantifies DEPENDENCE on the")
    print("suspected features, not their validity. See Decision 12.")
    print("=" * 62)


if __name__ == "__main__":
    main()

"""
=============================================================
Sentinel Credit — Baseline Model Training
=============================================================
File:    train_baseline_models.py
Purpose: Trains and compares three PD models — Logistic
         Regression, Decision Tree, XGBoost — per Decision 4.
         Logs each to MLflow for champion/challenger tracking
         (Decision 8).
Author:  Folarin Korede
Version: 1.0.0

See docs/engineering_decision_log.md — Decision 4, Decision 12.
=============================================================
"""

import os
import subprocess
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from model_features import MODEL_FEATURES, TARGET_COLUMN, FILTER_COLUMN, FILTER_VALUE


# ── CONFIGURATION ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

with open(os.path.join(PROJECT_ROOT, "params.yaml"), "r", encoding="utf-8") as f:
    PARAMS = yaml.safe_load(f)

DB = CONFIG["database"]
MODEL_CFG = CONFIG.get("model", {})
RANDOM_SEED = MODEL_CFG.get("random_seed", 42)
TEST_SIZE = MODEL_CFG.get("test_size", 0.2)

CV_CFG = PARAMS.get("cv", {})
N_SPLITS = CV_CFG.get("n_splits", 5)
CV_SCORING = CV_CFG.get("scoring", "average_precision")


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)


def load_data() -> pd.DataFrame:
    """Loads the modelable population from the feature store.

    Raw psycopg2 connection, not a SQLAlchemy engine — pandas
    2.2's engine-detection is broken against SQLAlchemy 1.4
    (pandas-dev/pandas#57049), and this project is pinned to
    SQLAlchemy<2.0 for apache-airflow==2.9.1 compatibility.
    """
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
    return df[df[FILTER_COLUMN] == FILTER_VALUE].copy()


def split_data(df: pd.DataFrame, strategy: str = None) -> tuple:
    """
    Splits into train/test per the given strategy (falls back
    to config['model']['split_strategy'] if not passed).

    'stratified': random split, stratified on target,
    preserving the ~95:5 imbalance in both sets. Tests
    generalization to unseen customers drawn from the same
    time-mixed pool as training — does NOT test temporal
    generalization.

    'temporal': out-of-time (OOT) split on first_transaction_date.
    Train on earlier originations, test on later ones — the
    real deployment condition. Named explicitly in the EBA
    Guidelines on PD/LGD estimation as a required validation
    element ("backtesting, out of time analysis, etc.").
    Requires config['model']['temporal_split_date'].

    customer_id is carried through separately (not as a model
    feature) so evaluate_models.py can join test predictions
    back to segment labels for fairness checks.
    """
    strategy = strategy or MODEL_CFG.get("split_strategy", "stratified")
    X_full = df[MODEL_FEATURES].fillna(df[MODEL_FEATURES].median(numeric_only=True))
    y_full = df[TARGET_COLUMN]
    ids_full = df["customer_id"]

    if strategy == "temporal":
        split_date = MODEL_CFG.get("temporal_split_date")
        if not split_date:
            raise ValueError(
                "split_strategy: temporal requires "
                "model.temporal_split_date in config.yaml"
            )
        split_date = pd.Timestamp(split_date)
        first_tx = pd.to_datetime(df["first_transaction_date"])

        train_mask = first_tx < split_date
        test_mask = first_tx >= split_date

        X_train, X_test = X_full[train_mask], X_full[test_mask]
        y_train, y_test = y_full[train_mask], y_full[test_mask]
        ids_test = ids_full[test_mask]

        print(f"  Temporal split at {split_date.date()}: "
              f"{train_mask.sum():,} train / {test_mask.sum():,} test")

    else:
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X_full, y_full, ids_full,
            test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_full,
        )
        print(f"  Stratified split: {len(X_train):,} train / {len(X_test):,} test")

    return X_train, X_test, y_train, y_test, ids_test


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def train_logistic_regression(X_train, y_train, X_test, y_test) -> dict:
    """
    Regulatory baseline. Coefficients are directly
    interpretable. class_weight='balanced' compensates for
    the ~95:5 imbalance without resampling.

    Hyperparameters (C specifically — regularization strength)
    selected via GridSearchCV with StratifiedKFold on the
    TRAINING set only. Test set is never touched during search.
    """
    # RobustScaler, not StandardScaler: EDA confirmed genuine
    # long right tails in several features (income, order
    # value) — real customers, not generator artefacts. A
    # mean/std-based scaler lets a handful of extreme values
    # distort the scale for everyone; RobustScaler uses
    # median/IQR, which is resistant to that without capping
    # or discarding any real data.
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    grid = PARAMS["logistic_regression"]
    search = GridSearchCV(
        LogisticRegression(random_state=RANDOM_SEED),
        param_grid=grid,
        scoring=CV_SCORING,
        cv=cv_splitter(),
        n_jobs=-1,
    )
    search.fit(X_train_s, y_train)
    best_params = search.best_params_

    # Calibration: class_weight='balanced' fixes the model's
    # ability to LEARN from a rare positive class, but distorts
    # the resulting probability SCALE — predicted PDs end up
    # systematically inflated above the true ~1.9% base rate.
    # Platt scaling (sigmoid) recalibrates predict_proba output
    # against true observed frequencies, using its own internal
    # CV on the training set only. Ranking (Gini/AUC/KS) is
    # unaffected — only the probability scale changes.
    base_model = LogisticRegression(random_state=RANDOM_SEED, **best_params)
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=cv_splitter())
    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # CalibratedClassifierCV fits one base estimator per CV
    # fold internally (calibrated_classifiers_), rather than
    # exposing a single .coef_ directly — average across folds
    # for a representative coefficient set. Interpretability is
    # preserved; the averaging reflects the same CV structure
    # already used for calibration.
    fold_coefs = np.array([
        clf.estimator.coef_[0] for clf in model.calibrated_classifiers_
    ])
    coefs = dict(zip(MODEL_FEATURES, fold_coefs.mean(axis=0)))

    return {
        "name": "logistic_regression",
        "model": model,
        "scaler": scaler,
        "auc": round(auc, 4),
        "gini": round(gini_from_auc(auc), 4),
        "cv_best_score": round(search.best_score_, 4),
        "cv_best_params": search.best_params_,
        "test_proba": proba,
        "coefficients": coefs,
    }


def train_decision_tree(X_train, y_train, X_test, y_test) -> dict:
    """
    Interpretable rules. max_depth/min_samples_leaf selected
    via GridSearchCV — an unconstrained tree overfits on this
    feature set and produces a rule list too deep for a risk
    officer to actually read; too shallow underfits. Search
    finds the balance rather than guessing one value.
    """
    grid = PARAMS["decision_tree"]
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        param_grid=grid,
        scoring=CV_SCORING,
        cv=cv_splitter(),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_

    # Same calibration rationale as Logistic Regression:
    # class_weight='balanced' distorts the probability scale
    # even though it improves the tree's ability to split on
    # the rare positive class.
    base_model = DecisionTreeClassifier(random_state=RANDOM_SEED, **best_params)
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=cv_splitter())
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    fold_importances = np.array([
        clf.estimator.feature_importances_ for clf in model.calibrated_classifiers_
    ])
    importances = dict(zip(MODEL_FEATURES, fold_importances.mean(axis=0)))
    return {
        "name": "decision_tree",
        "model": model,
        "scaler": None,
        "auc": round(auc, 4),
        "gini": round(gini_from_auc(auc), 4),
        "cv_best_score": round(search.best_score_, 4),
        "cv_best_params": search.best_params_,
        "test_proba": proba,
        "feature_importances": importances,
    }


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    """
    Performance champion. scale_pos_weight's base value comes
    from the training fold's actual class ratio (a data
    property), but the MULTIPLIER on it is now genuinely
    searched (params.yaml scale_pos_weight_multiplier) rather
    than fixed at 1.0 — same fix applied to class_weight for
    the other two models, since a fixed single-value "search"
    never actually explores the discrimination/calibration
    tradeoff. Requires SHAP for explanation (Decision 7).
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    base_scale_pos_weight = n_neg / max(n_pos, 1)

    grid = dict(PARAMS["xgboost"])
    multipliers = grid.pop("scale_pos_weight_multiplier")
    grid["scale_pos_weight"] = [round(base_scale_pos_weight * m, 4) for m in multipliers]

    search = GridSearchCV(
        XGBClassifier(random_state=RANDOM_SEED, eval_metric="auc"),
        param_grid=grid,
        scoring=CV_SCORING,
        cv=cv_splitter(),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_

    # Same calibration rationale as the other two models.
    base_model = XGBClassifier(
        random_state=RANDOM_SEED, eval_metric="auc", **best_params,
    )
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=cv_splitter())
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    fold_importances = np.array([
        clf.estimator.feature_importances_ for clf in model.calibrated_classifiers_
    ])
    importances = dict(zip(MODEL_FEATURES, fold_importances.mean(axis=0)))
    return {
        "name": "xgboost",
        "model": model,
        "scaler": None,
        "auc": round(auc, 4),
        "gini": round(gini_from_auc(auc), 4),
        "cv_best_score": round(search.best_score_, 4),
        "cv_best_params": search.best_params_,
        "test_proba": proba,
        "feature_importances": importances,
    }


def log_to_mlflow(result: dict, X_train, y_train, split_strategy: str) -> None:
    """
    Logs the run per Decision 8: parameters, metrics, model
    artefact. This is the champion/challenger evidence base —
    every training run is captured, not just the one deployed.
    """
    with mlflow.start_run(run_name=f"{result['name']}_{split_strategy}"):
        mlflow.log_param("model_type", result["name"])
        mlflow.log_param("n_features", len(MODEL_FEATURES))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("train_bad_rate", round(float(y_train.mean()), 4))
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("split_strategy", split_strategy)
        mlflow.log_param("cv_n_splits", N_SPLITS)
        mlflow.log_param("cv_scoring", CV_SCORING)

        # Log the ACTUAL best hyperparameters GridSearchCV
        # selected, not the static search grid — this is the
        # real audit trail: what was tried (params.yaml, in
        # git) vs what was chosen (here, per run).
        for k, v in result["cv_best_params"].items():
            mlflow.log_param(f"best_{k}", v)

        mlflow.log_metric("cv_score", result["cv_best_score"])
        mlflow.log_metric("gini", result["gini"])
        mlflow.log_metric("auc", result["auc"])
        # NOTE: cv_score is average_precision (PR-AUC), auc is
        # ROC-AUC — different scales, not directly subtractable
        # into a "gap" the way same-metric comparisons work
        # elsewhere (e.g. the stratified/temporal Gini gap).
        # Logged separately rather than as a derived difference.

        mlflow.sklearn.log_model(result["model"], "model")


def print_summary(results: list) -> None:
    print("\n" + "=" * 58)
    print("BASELINE MODEL COMPARISON")
    print("=" * 58)
    print(f"{'Model':<22} {'CV ' + CV_SCORING:>16} {'Test Gini':>10} {'Test AUC':>9}")
    print("-" * 58)
    for r in sorted(results, key=lambda x: x["gini"], reverse=True):
        print(f"{r['name']:<22} {r['cv_best_score']:>16.4f} "
              f"{r['gini']:>10.4f} {r['auc']:>9.4f}")
    print("-" * 58)
    for r in results:
        print(f"{r['name']}: best params {r['cv_best_params']}")
    print("-" * 58)
    print("Full metric suite (KS, calibration, bad-rate-at-cutoff,")
    print("fairness parity) computed in evaluate_models.py")
    print("=" * 58)


def print_split_comparison(all_results: dict) -> None:
    """
    Side-by-side stratified vs temporal Gini per model. The
    gap itself is the finding — a large drop from stratified
    to temporal means the model is weaker at genuine temporal
    generalization than a random split alone would suggest,
    the concept-drift risk Decision 5's monitoring exists for.
    """
    print("\n" + "=" * 58)
    print("STRATIFIED vs TEMPORAL — Gini comparison")
    print("=" * 58)
    print(f"{'Model':<22} {'Stratified':>12} {'Temporal':>10} {'Gap':>8}")
    print("-" * 58)

    strat = {r["name"]: r["gini"] for r in all_results.get("stratified", [])}
    temp = {r["name"]: r["gini"] for r in all_results.get("temporal", [])}

    for name in strat.keys() | temp.keys():
        s = strat.get(name)
        t = temp.get(name)
        gap = round(s - t, 4) if (s is not None and t is not None) else None
        s_str = f"{s:.4f}" if s is not None else "n/a"
        t_str = f"{t:.4f}" if t is not None else "n/a"
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        flag = "  <- check drift" if gap is not None and gap > 0.05 else ""
        print(f"{name:<22} {s_str:>12} {t_str:>10} {gap_str:>8}{flag}")

    print("-" * 58)
    print("Large positive gap (stratified > temporal) suggests the")
    print("model relies on patterns that don't hold up over time —")
    print("worth investigating before deployment, not after.")
    print("=" * 58)


def main() -> None:
    print("=" * 58)
    print("SENTINEL CREDIT — BASELINE MODEL TRAINING")
    print("=" * 58)

    df = load_data()
    print(f"Loaded {len(df):,} modelable customers, "
          f"{len(MODEL_FEATURES)} features")

    mlflow.set_experiment("sentinel_credit_baseline")

    out_dir = os.path.join(PROJECT_ROOT, "models", "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for strategy in ["stratified", "temporal"]:
        print(f"\n{'=' * 58}")
        print(f"SPLIT STRATEGY: {strategy}")
        print("=" * 58)

        if strategy == "temporal" and not MODEL_CFG.get("temporal_split_date"):
            print(f"  Skipped — split_strategy: temporal requires "
                  f"model.temporal_split_date in config.yaml")
            continue

        X_train, X_test, y_train, y_test, ids_test = split_data(df, strategy)

        print(f"Train bad rate {y_train.mean():.2%}  "
              f"Test bad rate {y_test.mean():.2%}")

        # Guard against a degenerate single-class test set — a
        # known, expected consequence of the temporal split
        # under the corrected has_outcome logic (Decision 12
        # addendum, "Unequal Observation Opportunity"): near
        # data_end_date, non-defaulters can no longer qualify
        # for has_outcome=1 (insufficient maturity runway),
        # while defaulters can qualify regardless of timing.
        # Test sets drawn from that period can end up
        # containing only defaulters, making AUC/Gini undefined.
        # Skipped explicitly rather than crashing, since this
        # is a documented, expected data property, not a bug.
        if y_test.nunique() < 2:
            print(f"  SKIPPED — test set contains only one class "
                  f"({int(y_test.sum())}/{len(y_test)} positive). "
                  f"Expected under the current temporal_split_date "
                  f"given the has_outcome maturity requirement — "
                  f"see docs/engineering_decision_log.md Decision 10 "
                  f"addendum (tenor-maturity bias) and Decision 12 "
                  f"addendum (unequal observation opportunity).")
            continue

        results = []
        for train_fn in [train_logistic_regression, train_decision_tree, train_xgboost]:
            print(f"\nTraining {train_fn.__name__}...")
            result = train_fn(X_train, y_train, X_test, y_test)
            log_to_mlflow(result, X_train, y_train, strategy)
            results.append(result)
            print(f"  Gini: {result['gini']:.4f}  AUC: {result['auc']:.4f}")

        print_summary(results)
        all_results[strategy] = results

        # Save predictions per strategy, so evaluate_models.py
        # can be pointed at either.
        eval_frame = X_test.copy()
        eval_frame["customer_id"] = ids_test.values
        eval_frame[TARGET_COLUMN] = y_test.values
        for r in results:
            eval_frame[f"proba_{r['name']}"] = r["test_proba"]
        out_path = os.path.join(out_dir, f"test_predictions_{strategy}.csv")
        eval_frame.to_csv(out_path, index=False)
        print(f"\nTest predictions saved to {out_path}")

    if len(all_results) == 2:
        print_split_comparison(all_results)

    completed_splits = list(all_results.keys())
    if not completed_splits:
        print("\nNo split completed successfully — nothing to evaluate.")
        return

    print(f"\nCompleted split(s): {', '.join(completed_splits)}")
    answer = input(
        "Run evaluate_models.py now for these split(s)? [y/n]: "
    ).strip().lower()

    if answer == "y":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for split in completed_splits:
            print(f"\nRunning evaluate_models.py --split {split} ...\n")
            subprocess.run(
                [sys.executable, os.path.join(script_dir, "evaluate_models.py"),
                 "--split", split],
                check=False,
            )
    else:
        print("\nRun manually when ready, e.g.:")
        for split in completed_splits:
            print(f"  python models\\evaluate_models.py --split {split}")


if __name__ == "__main__":
    main()

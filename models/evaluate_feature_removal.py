"""
Sentinel Credit — Feature Removal Experiment

Purpose
-------
Tests removing one feature from the frozen v1 set (Decision 18) and
produces the evidence a governed decision needs. Built for the
pct_late_payments question (audit finding A6), but the feature to
remove is an argument, not hardcoded.

How the result is judged
------------------------
On feature VALIDITY first: would this information exist at decision
time, and is it independent of the outcome it predicts? A leaky
feature usually looks good in cross-validation, so a Gini drop on
removal is NOT by itself a reason to keep it. The performance
numbers below quantify what removal costs; they do not decide it.

What it runs
------------
1. Paired repeated stratified CV (5 folds x 3 repeats) on the full
   modelable population, both arms on identical folds, champion
   hyperparameters held fixed so the feature set is the only change.
2. Canonical split (the champion's own split_ids_*.csv): both arms
   trained and Platt-calibrated exactly as train_baseline_models.py
   does, scored on the held-out test set, with a paired stratified
   bootstrap 95% CI on test Gini, the difference between arms, and
   the train/test Gini gap (only ~226 test defaults, so a point
   estimate alone cannot say whether a gap is noise).
3. A small hyperparameter check per arm: max_depth {2, 3} x
   learning_rate {0.005, 0.01}, n_estimators picked by early
   stopping on an inner validation slice of each training fold
   (never the scoring fold), plus the champion's fixed configuration
   as the baseline. Records train vs validation PR-AUC and Gini gaps.
4. Calibration check of the reduced arm on held-out data: decile
   reliability table, ECE with both uniform and equal-frequency bins
   (audit finding A8), Brier.
5. Optional (--log-challenger): logs the reduced arm as a challenger
   to MLflow for BOTH splits, with the eval_* metrics
   pipeline/promote_champion.py reads, and saves its artifacts under
   models/challengers/<name>/. The challenger keeps the champion's
   hyperparameters, so a promotion decision is attributable to the
   feature change alone; the hyperparameter check is reported as
   separate evidence.

What it does NOT do
-------------------
- Never writes to models/artifacts/ (the governed champion package)
  or config.yaml. The August champion stays reproducible.
- Never recalibrates business_cutoff. A different model has a
  different score distribution (Decision 13); flag rates at the
  current 0.0671 are reported for context only.
- Never promotes. Promotion is pipeline/promote_champion.py's job,
  with typed human confirmation.

Usage
-----
    python models/evaluate_feature_removal.py --remove pct_late_payments
    python models/evaluate_feature_removal.py --remove pct_late_payments --log-challenger

See docs/engineering_decision_log.md — Decision 14 (the ablation
method this follows), Decision 18 (the frozen set), Decision 21
(challenger workflow), and docs/audit_findings_2026-10-12.md (A6, A7).
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import (  # noqa: E402
    FILTER_COLUMN, FILTER_VALUE, MODEL_FEATURES, TARGET_COLUMN,
)
# Same metric code evaluate_models.py uses, so challenger and
# champion numbers are computed identically.
from evaluate_models import (  # noqa: E402
    expected_calibration_error, gini_from_auc, ks_statistic,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    CONFIG["model_governance"].get("artifacts_dir", "models/artifacts"),
)  # the governed champion package; see config.yaml model_governance

DB = CONFIG["database"]
MODEL_CFG = CONFIG["model"]
GOV_CFG = CONFIG["model_governance"]
MLFLOW_CFG = CONFIG["mlflow"]

RANDOM_SEED = MODEL_CFG.get("random_seed", 42)
BUSINESS_CUTOFF = MODEL_CFG["business_cutoff"]
TEMPORAL_SPLIT_DATE = MODEL_CFG.get("temporal_split_date")

CV_N_SPLITS, CV_N_REPEATS = 5, 3
N_BOOTSTRAP = 2000
HP_GRID = [
    {"max_depth": d, "learning_rate": lr}
    for d in (2, 3) for lr in (0.005, 0.01)
]
HP_MAX_ROUNDS = 3000
HP_EARLY_STOPPING = 100


# ── DATA ──────────────────────────────────────────────────────
def load_population() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=os.environ.get("SENTINEL_DB_HOST", DB["host"]),
        port=DB["port"], dbname=DB["name"], user="postgres",
        password=os.environ.get("SENTINEL_DB_PASSWORD", ""),
    )
    try:
        df = pd.read_sql(f"SELECT * FROM {DB['schema']}.customer_features", conn)
    finally:
        conn.close()
    df = df[df[FILTER_COLUMN] == FILTER_VALUE].copy()
    df["customer_id"] = df["customer_id"].astype(str)
    missing = df[MODEL_FEATURES].isna().sum().sum()
    if missing:
        # train_baseline_models.py median-fills; this experiment
        # refuses instead of silently diverging from it.
        raise ValueError(f"{missing} null model-feature values — investigate first.")
    return df.set_index("customer_id")


def load_champion_metadata() -> dict:
    with open(os.path.join(ARTIFACTS_DIR, "champion_metadata.json"), "r",
              encoding="utf-8") as f:
        return json.load(f)


def canonical_split(df: pd.DataFrame, meta: dict) -> tuple:
    """The champion's own stratified partition, from its saved IDs."""
    if len(df) != meta["dataset"]["n_modelable"]:
        raise ValueError(
            f"Modelable population is {len(df):,}, champion was trained on "
            f"{meta['dataset']['n_modelable']:,} — not the same data."
        )
    tr = pd.read_csv(os.path.join(ARTIFACTS_DIR, "split_ids_train.csv"))["customer_id"].astype(str)
    te = pd.read_csv(os.path.join(ARTIFACTS_DIR, "split_ids_test.csv"))["customer_id"].astype(str)
    return df.loc[tr.values], df.loc[te.values]


def temporal_split(df: pd.DataFrame) -> tuple:
    """Same rule as train_baseline_models.split_data('temporal')."""
    first_tx = pd.to_datetime(df["first_transaction_date"])
    cut = pd.Timestamp(TEMPORAL_SPLIT_DATE)
    return df[first_tx < cut], df[first_tx >= cut]


# ── MODELS ────────────────────────────────────────────────────
def champion_xgb_params(meta: dict) -> dict:
    params = dict(meta["best_params"])
    params.pop("scale_pos_weight", None)  # recomputed from each training set
    params.update({"random_state": RANDOM_SEED, "eval_metric": "auc"})
    return params


def spw(y: pd.Series) -> float:
    return round(int((y == 0).sum()) / max(int((y == 1).sum()), 1), 4)


def fit_calibrated(X: pd.DataFrame, y: pd.Series, params: dict):
    """Exactly train_baseline_models.train_xgboost's calibrated model."""
    base = XGBClassifier(scale_pos_weight=spw(y), **params)
    model = CalibratedClassifierCV(
        base, method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    )
    model.fit(X, y)
    return model


def summary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "gini": round(gini_from_auc(auc), 4),
        "auc": round(auc, 4),
        "ks": round(ks_statistic(y, p), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "ece_uniform_bins": expected_calibration_error(y, p),
        "ece_decile_bins": decile_ece(y, p),
    }


def decile_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False)
    g = d.groupby("decile").agg(
        n=("y", "size"), defaults=("y", "sum"),
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"),
        min_score=("p", "min"), max_score=("p", "max"),
    )
    return g.reset_index()


def decile_ece(y: np.ndarray, p: np.ndarray) -> float:
    g = decile_table(y, p)
    return round(float((g["n"] * (g["mean_predicted"] - g["observed_rate"]).abs()).sum()
                       / g["n"].sum()), 4)


# ── 1. PAIRED REPEATED CV ─────────────────────────────────────
def repeated_cv(df: pd.DataFrame, arms: dict, params: dict) -> pd.DataFrame:
    y = df[TARGET_COLUMN]
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS,
                                 random_state=RANDOM_SEED)
    rows = []
    for fold, (tr_i, te_i) in enumerate(cv.split(df, y)):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        for arm, feats in arms.items():
            m = XGBClassifier(scale_pos_weight=spw(tr[TARGET_COLUMN]), **params)
            m.fit(tr[feats], tr[TARGET_COLUMN])
            p_te = m.predict_proba(te[feats])[:, 1]
            p_tr = m.predict_proba(tr[feats])[:, 1]
            g_te = gini_from_auc(roc_auc_score(te[TARGET_COLUMN], p_te))
            g_tr = gini_from_auc(roc_auc_score(tr[TARGET_COLUMN], p_tr))
            rows.append({
                "fold": fold, "arm": arm, "gini": g_te,
                "ks": ks_statistic(te[TARGET_COLUMN].values, p_te),
                "pr_auc": float(average_precision_score(te[TARGET_COLUMN], p_te)),
                "train_test_gini_gap": g_tr - g_te,
            })
    return pd.DataFrame(rows)


def summarise_paired(cv_df: pd.DataFrame, full: str, reduced: str) -> pd.DataFrame:
    rows = []
    for m in ["gini", "ks", "pr_auc", "train_test_gini_gap"]:
        w = cv_df.pivot(index="fold", columns="arm", values=m)
        diff = w[reduced] - w[full]
        rows.append({
            "metric": m,
            f"{full}_mean": round(w[full].mean(), 4),
            f"{reduced}_mean": round(w[reduced].mean(), 4),
            "mean_diff_reduced_minus_full": round(diff.mean(), 4),
            "sd_diff": round(diff.std(), 4),
            "folds_reduced_higher": int((diff > 0).sum()),
            "n_folds": int(len(diff)),
        })
    return pd.DataFrame(rows)


# ── 2. BOOTSTRAP ──────────────────────────────────────────────
def stratified_bootstrap_idx(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ginis(y_te, preds_te: dict, y_tr, preds_tr: dict) -> pd.DataFrame:
    """Paired: each resample draws one set of test (and train) rows,
    applied to every arm, so the arm difference has its own CI."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for _ in range(N_BOOTSTRAP):
        ite = stratified_bootstrap_idx(y_te, rng)
        itr = stratified_bootstrap_idx(y_tr, rng)
        row = {}
        for arm in preds_te:
            g_te = gini_from_auc(roc_auc_score(y_te[ite], preds_te[arm][ite]))
            g_tr = gini_from_auc(roc_auc_score(y_tr[itr], preds_tr[arm][itr]))
            row[f"{arm}_test_gini"] = g_te
            row[f"{arm}_gap"] = g_tr - g_te
        rows.append(row)
    return pd.DataFrame(rows)


def ci(series: pd.Series) -> tuple:
    return round(float(series.quantile(0.025)), 4), round(float(series.quantile(0.975)), 4)


# ── 3. HYPERPARAMETER CHECK ───────────────────────────────────
def hp_check(train: pd.DataFrame, arms: dict, params: dict,
             champion_best: dict) -> tuple:
    """5-fold CV on the canonical training partition only. Early
    stopping uses a 15% inner slice of each training fold, never the
    validation fold that is scored."""
    y = train[TARGET_COLUMN]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    base = {k: v for k, v in params.items()
            if k not in ("max_depth", "learning_rate", "n_estimators")}
    configs = [{"name": "champion_fixed",
                "max_depth": champion_best["max_depth"],
                "learning_rate": champion_best["learning_rate"],
                "n_estimators": champion_best["n_estimators"],
                "early_stopping": False}]
    configs += [{"name": f"d{c['max_depth']}_lr{c['learning_rate']}_es",
                 **c, "early_stopping": True} for c in HP_GRID]

    rows = []
    for arm, feats in arms.items():
        for cfg in configs:
            for fold, (tr_i, va_i) in enumerate(outer.split(train, y)):
                tr, va = train.iloc[tr_i], train.iloc[va_i]
                params = {**base, "max_depth": cfg["max_depth"],
                          "learning_rate": cfg["learning_rate"]}
                if cfg["early_stopping"]:
                    fit_tr, es = train_test_split(
                        tr, test_size=0.15, stratify=tr[TARGET_COLUMN],
                        random_state=RANDOM_SEED + fold)
                    m = XGBClassifier(
                        scale_pos_weight=spw(fit_tr[TARGET_COLUMN]),
                        n_estimators=HP_MAX_ROUNDS,
                        early_stopping_rounds=HP_EARLY_STOPPING, **params)
                    m.fit(fit_tr[feats], fit_tr[TARGET_COLUMN],
                          eval_set=[(es[feats], es[TARGET_COLUMN])], verbose=False)
                    n_used = int(m.best_iteration) + 1
                    p_tr = m.predict_proba(fit_tr[feats])[:, 1]
                    y_fit = fit_tr[TARGET_COLUMN]
                else:
                    m = XGBClassifier(scale_pos_weight=spw(tr[TARGET_COLUMN]),
                                      n_estimators=cfg["n_estimators"], **params)
                    m.fit(tr[feats], tr[TARGET_COLUMN])
                    n_used = cfg["n_estimators"]
                    p_tr = m.predict_proba(tr[feats])[:, 1]
                    y_fit = tr[TARGET_COLUMN]
                p_va = m.predict_proba(va[feats])[:, 1]
                g_tr = gini_from_auc(roc_auc_score(y_fit, p_tr))
                g_va = gini_from_auc(roc_auc_score(va[TARGET_COLUMN], p_va))
                rows.append({
                    "arm": arm, "config": cfg["name"],
                    "max_depth": cfg["max_depth"],
                    "learning_rate": cfg["learning_rate"],
                    "fold": fold, "n_estimators_used": n_used,
                    "val_pr_auc": float(average_precision_score(va[TARGET_COLUMN], p_va)),
                    "train_pr_auc": float(average_precision_score(y_fit, p_tr)),
                    "val_gini": g_va, "train_gini": g_tr,
                    "train_val_gini_gap": g_tr - g_va,
                })
    folds = pd.DataFrame(rows)
    summary = folds.groupby(["arm", "config"]).agg(
        max_depth=("max_depth", "first"), learning_rate=("learning_rate", "first"),
        n_estimators_median=("n_estimators_used", "median"),
        val_pr_auc_mean=("val_pr_auc", "mean"), val_pr_auc_sd=("val_pr_auc", "std"),
        val_gini_mean=("val_gini", "mean"),
        train_val_gini_gap_mean=("train_val_gini_gap", "mean"),
    ).round(4).reset_index()
    return folds, summary


# ── 5. CHALLENGER LOGGING ─────────────────────────────────────
def log_challenger(name: str, feats: list, params: dict, removed: str,
                   splits: dict, out_dir: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])
    run_ids = {}
    for split, (train, test, model) in splits.items():
        p = model.predict_proba(test[feats])[:, 1]
        m = summary_metrics(test[TARGET_COLUMN].values, p)
        with mlflow.start_run(run_name=f"xgboost_{split}_{name}") as run:
            mlflow.set_tags({
                "feature_set": name,
                "feature_count": str(len(feats)),
                "model_version": f"{GOV_CFG['model_version']}-challenger",
                "dataset_version": GOV_CFG["dataset_version"],
                "split_strategy": split,
                "is_champion": "False",
                "role": "challenger",
                "removed_feature": removed,
                "derived_from": "evaluate_feature_removal.py",
                "pipeline_stage": "challenger_evidence_pending_decision",
            })
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("n_features", len(feats))
            mlflow.log_param("n_train", len(train))
            for k, v in params.items():
                mlflow.log_param(f"best_{k}", v)
            mlflow.log_metrics({
                "eval_gini": m["gini"], "eval_auc": m["auc"], "eval_ks": m["ks"],
                "eval_pr_auc": m["pr_auc"], "eval_brier": m["brier"],
                "eval_ece": m["ece_uniform_bins"],
                "eval_ece_decile_bins": m["ece_decile_bins"],
                "gini": m["gini"], "auc": m["auc"],
            })
            mlflow.sklearn.log_model(model, "model")
            run_ids[split] = run.info.run_id
        joblib.dump(model, os.path.join(out_dir, f"challenger_calibrated_{split}.joblib"))
    return run_ids


# ── MAIN ──────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", required=True, help="Feature to remove from MODEL_FEATURES.")
    ap.add_argument("--log-challenger", action="store_true",
                    help="Log the reduced arm to MLflow as a challenger (both splits).")
    args = ap.parse_args()

    if args.remove not in MODEL_FEATURES:
        raise SystemExit(f"{args.remove} is not in MODEL_FEATURES.")

    full = f"f{len(MODEL_FEATURES)}_full"
    reduced = f"f{len(MODEL_FEATURES) - 1}_no_{args.remove}"
    arms = {full: list(MODEL_FEATURES),
            reduced: [f for f in MODEL_FEATURES if f != args.remove]}

    out_dir = os.path.join(PROJECT_ROOT, "reports", "feature_removal", args.remove)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 66)
    print(f"SENTINEL CREDIT — FEATURE REMOVAL EXPERIMENT: {args.remove}")
    print("=" * 66)

    meta = load_champion_metadata()
    params = champion_xgb_params(meta)
    df = load_population()
    train, test = canonical_split(df, meta)
    print(f"Population {len(df):,}  |  canonical train {len(train):,} / "
          f"test {len(test):,} ({int(test[TARGET_COLUMN].sum())} test defaults)")
    print(f"Champion hyperparameters held fixed: {meta['best_params']}")

    # 1. Paired repeated CV
    print(f"\n[1/4] Paired repeated CV ({CV_N_SPLITS}x{CV_N_REPEATS})...")
    cv_df = repeated_cv(df, arms, params)
    cv_sum = summarise_paired(cv_df, full, reduced)
    cv_df.to_csv(os.path.join(out_dir, "repeated_cv_folds.csv"), index=False)
    cv_sum.to_csv(os.path.join(out_dir, "repeated_cv_summary.csv"), index=False)
    print(cv_sum.to_string(index=False))

    # 2. Canonical split, calibrated, bootstrap CIs
    print(f"\n[2/4] Canonical split, calibrated models, {N_BOOTSTRAP} paired bootstraps...")
    models, preds_te, preds_tr, held_out = {}, {}, {}, {}
    for arm, feats in arms.items():
        models[arm] = fit_calibrated(train[feats], train[TARGET_COLUMN], params)
        preds_te[arm] = models[arm].predict_proba(test[feats])[:, 1]
        preds_tr[arm] = models[arm].predict_proba(train[feats])[:, 1]
        held_out[arm] = summary_metrics(test[TARGET_COLUMN].values, preds_te[arm])
        held_out[arm]["train_gini"] = round(gini_from_auc(
            roc_auc_score(train[TARGET_COLUMN], preds_tr[arm])), 4)
        held_out[arm]["flag_rate_at_current_cutoff"] = round(
            float((preds_te[arm] >= BUSINESS_CUTOFF).mean()), 4)
        flagged = preds_te[arm] >= BUSINESS_CUTOFF
        bad = test[TARGET_COLUMN].values == 1
        held_out[arm]["recall_at_current_cutoff"] = round(
            float((flagged & bad).sum() / max(bad.sum(), 1)), 4)

    boot = bootstrap_ginis(test[TARGET_COLUMN].values, preds_te,
                           train[TARGET_COLUMN].values, preds_tr)
    boot.to_csv(os.path.join(out_dir, "bootstrap_samples.csv"), index=False)
    boot_diff = boot[f"{reduced}_test_gini"] - boot[f"{full}_test_gini"]
    ci_rows = []
    for arm in arms:
        lo, hi = ci(boot[f"{arm}_test_gini"])
        glo, ghi = ci(boot[f"{arm}_gap"])
        ci_rows.append({
            "arm": arm, **held_out[arm],
            "test_gini_ci_low": lo, "test_gini_ci_high": hi,
            "train_test_gap": round(held_out[arm]["train_gini"] - held_out[arm]["gini"], 4),
            "gap_ci_low": glo, "gap_ci_high": ghi,
            "gap_distinguishable_from_zero": bool(glo > 0),
        })
    canon = pd.DataFrame(ci_rows)
    canon.to_csv(os.path.join(out_dir, "canonical_split_comparison.csv"), index=False)
    dlo, dhi = ci(boot_diff)
    print(canon[["arm", "gini", "test_gini_ci_low", "test_gini_ci_high", "train_gini",
                 "train_test_gap", "gap_ci_low", "gap_ci_high", "pr_auc", "ks",
                 "ece_decile_bins", "brier"]].to_string(index=False))
    print(f"  Gini difference (reduced - full): {held_out[reduced]['gini'] - held_out[full]['gini']:+.4f}"
          f"  95% CI [{dlo:+.4f}, {dhi:+.4f}]")

    # Decision flips at the current cutoff (context only)
    flips = (preds_te[full] >= BUSINESS_CUTOFF) != (preds_te[reduced] >= BUSINESS_CUTOFF)

    # 2b. Temporal split, same paired comparison. Both arms are refitted
    # on the temporal training set with the same (champion) parameters,
    # so the difference isolates the feature, as on the canonical split.
    print(f"\n[2b] Temporal split, calibrated models, {N_BOOTSTRAP} paired bootstraps...")
    t_train, t_test = temporal_split(df)
    t_models, t_preds_te, t_preds_tr, t_held_out = {}, {}, {}, {}
    for arm, feats in arms.items():
        t_models[arm] = fit_calibrated(t_train[feats], t_train[TARGET_COLUMN], params)
        t_preds_te[arm] = t_models[arm].predict_proba(t_test[feats])[:, 1]
        t_preds_tr[arm] = t_models[arm].predict_proba(t_train[feats])[:, 1]
        t_held_out[arm] = summary_metrics(t_test[TARGET_COLUMN].values, t_preds_te[arm])
    t_boot = bootstrap_ginis(t_test[TARGET_COLUMN].values, t_preds_te,
                             t_train[TARGET_COLUMN].values, t_preds_tr)
    t_boot.to_csv(os.path.join(out_dir, "bootstrap_samples_temporal.csv"), index=False)
    tdlo, tdhi = ci(t_boot[f"{reduced}_test_gini"] - t_boot[f"{full}_test_gini"])
    t_diff = round(t_held_out[reduced]["gini"] - t_held_out[full]["gini"], 4)
    pd.DataFrame([{"arm": a, **t_held_out[a]} for a in arms]).to_csv(
        os.path.join(out_dir, "temporal_split_comparison.csv"), index=False)
    print(f"  temporal test {len(t_test):,} ({int(t_test[TARGET_COLUMN].sum())} defaults): "
          f"Gini {t_held_out[full]['gini']:.4f} -> {t_held_out[reduced]['gini']:.4f}, "
          f"difference {t_diff:+.4f}  95% CI [{tdlo:+.4f}, {tdhi:+.4f}]")

    # 3. Hyperparameter check
    print("\n[3/4] Hyperparameter check (canonical training partition only)...")
    hp_folds, hp_sum = hp_check(train, arms, params, meta["best_params"])
    hp_folds.to_csv(os.path.join(out_dir, "hyperparameter_check_folds.csv"), index=False)
    hp_sum.to_csv(os.path.join(out_dir, "hyperparameter_check_summary.csv"), index=False)
    print(hp_sum.to_string(index=False))

    # 4. Calibration of the reduced arm
    print(f"\n[4/4] Calibration on held-out test ({reduced})...")
    dec = decile_table(test[TARGET_COLUMN].values, preds_te[reduced])
    dec.to_csv(os.path.join(out_dir, f"calibration_deciles_{reduced}.csv"), index=False)
    decile_table(test[TARGET_COLUMN].values, preds_te[full]).to_csv(
        os.path.join(out_dir, f"calibration_deciles_{full}.csv"), index=False)
    print(dec.round(4).to_string(index=False))

    # 5. Challenger (optional)
    run_ids = None
    if args.log_challenger:
        print("\n[5] Logging challenger to MLflow (stratified + temporal)...")
        ch_dir = os.path.join(PROJECT_ROOT, "models", "challengers", reduced)
        os.makedirs(ch_dir, exist_ok=True)
        feats = arms[reduced]
        run_ids = log_challenger(
            reduced, feats, params, args.remove,
            {"stratified": (train, test, models[reduced]),
             "temporal": (t_train, t_test, t_models[reduced])},
            ch_dir,
        )
        with open(os.path.join(ch_dir, "challenger_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "feature_list": feats, "removed_feature": args.remove,
                "hyperparameters": meta["best_params"],
                "hyperparameter_note": "Champion's selected values, held fixed so "
                                       "the feature change is the only difference.",
                "calibration_method": "sigmoid", "split_ids": "champion's canonical files",
                "mlflow_run_ids": run_ids,
                "business_cutoff_note": "Not recalibrated. 0.0671 belongs to the "
                                        "14-feature champion (Decision 13/19).",
            }, f, indent=2)
        print(f"  Challenger run IDs: {run_ids}")

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "removed_feature": args.remove,
        "arms": arms,
        "champion_model_version": meta["model_version"],
        "hyperparameters_held_fixed": meta["best_params"],
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "held_out": held_out,
        "gini_difference_reduced_minus_full": {
            "point": round(held_out[reduced]["gini"] - held_out[full]["gini"], 4),
            "ci95": [dlo, dhi],
        },
        "decisions_flipped_at_current_cutoff": int(flips.sum()),
        "decisions_total": int(len(flips)),
        "challenger_mlflow_run_ids": run_ids,
        "judgement_rule": (
            "Judged on feature validity (would the information exist at "
            "decision time, independent of the outcome?). A Gini drop on "
            "removal is not by itself a reason to keep a feature: leaky "
            "features usually look good in cross-validation."
        ),
    }
    with open(os.path.join(out_dir, "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Evidence file read by promote_champion.py --validity-override.
    # Ties the paired-bootstrap Gini-difference CIs to the exact
    # challenger runs they describe.
    evidence = {
        "generated_at": metadata["generated_at"],
        "removed_feature": args.remove,
        "challenger_run_ids": run_ids,
        "n_bootstrap": N_BOOTSTRAP,
        "hyperparameters_held_fixed": meta["best_params"],
        "gini_difference_challenger_minus_champion_arm": {
            "stratified": {"point": round(held_out[reduced]["gini"] - held_out[full]["gini"], 4),
                           "ci95": [dlo, dhi]},
            "temporal": {"point": t_diff, "ci95": [tdlo, tdhi]},
        },
        "method": "paired stratified bootstrap of held-out Gini, both arms "
                  "refitted on identical data with identical parameters",
    }
    with open(os.path.join(out_dir, "promotion_evidence.json"), "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("\n" + "=" * 66)
    print(f"Decisions flipping at current cutoff {BUSINESS_CUTOFF}: "
          f"{int(flips.sum()):,} of {len(flips):,} ({100 * flips.mean():.2f}%)")
    print(f"Saved to {out_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()

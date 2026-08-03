"""
=============================================================
Sentinel Credit — Pre-Modelling Exploratory Data Analysis
=============================================================
File:    eda_baseline.py
Purpose: Standard bank-practice EDA on feature_store.
         customer_features before any model is trained.
         Six checks: target/bad-rate, missingness,
         univariate WoE/IV screening, correlation/VIF,
         outlier flagging, train/test split stability.
Author:  Folarin Korede
Version: 1.0.0

Why this exists: PD models in production credit risk teams
are never trained straight off a feature table. This script
is that step. IV screening in particular doubles as a
leakage detector — an IV above 0.5 on any feature is a signal
to investigate before training, not after.

See docs/engineering_decision_log.md — Decision 4, Decision 12.
=============================================================
"""

# ── IMPORTS ───────────────────────────────────────────────
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import psycopg2
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
from model_features import MODEL_FEATURES, TARGET_COLUMN, FILTER_COLUMN, FILTER_VALUE


# ── CONFIGURATION ─────────────────────────────────────────
def load_config() -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
DB = CONFIG["database"]
MODEL_CFG = CONFIG.get("model", {})
RANDOM_SEED = MODEL_CFG.get("random_seed", 42)


def load_feature_store() -> pd.DataFrame:
    """
    Loads feature_store.customer_features from PostgreSQL.
    Requires the DB password via environment variable
    SENTINEL_DB_PASSWORD (never hardcoded, never committed).

    Uses a raw psycopg2 connection rather than a SQLAlchemy
    engine — pandas 2.2's engine-detection is broken against
    SQLAlchemy 1.4 (pandas-dev/pandas#57049), and this project
    is pinned to SQLAlchemy<2.0 to satisfy apache-airflow==2.9.1.
    A raw DBAPI2 connection sidesteps that broken code path
    entirely and is fully supported by pd.read_sql().
    """
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    conn = psycopg2.connect(
        host=DB["host"],
        port=DB["port"],
        dbname=DB["name"],
        user="postgres",
        password=password,
    )
    try:
        df = pd.read_sql(
            f"SELECT * FROM {DB['schema']}.customer_features", conn
        )
    finally:
        conn.close()
    return df


# ── SECTION 1: TARGET / BAD-RATE CHECK ────────────────────
def check_target_imbalance(df: pd.DataFrame) -> dict:
    """
    Overall bad rate, plus split by key segments. Confirms
    the class imbalance problem and where it concentrates,
    before any modelling decision (class weights, threshold
    choice) is made.
    """
    modelable = df[df[FILTER_COLUMN] == FILTER_VALUE]
    overall_rate = float(modelable[TARGET_COLUMN].mean())

    segments = {}
    for seg_col in ["thin_file_flag", "is_vulnerable", "ab_test_group"]:
        if seg_col not in modelable.columns:
            continue
        seg_rates = (
            modelable.groupby(seg_col)[TARGET_COLUMN]
            .agg(["mean", "count"])
            .rename(columns={"mean": "bad_rate", "count": "n"})
        )
        segments[seg_col] = seg_rates.round(4).to_dict("index")

    return {
        "n_total": int(len(df)),
        "n_modelable": int(len(modelable)),
        "overall_bad_rate": round(overall_rate, 4),
        "n_bad": int(modelable[TARGET_COLUMN].sum()),
        "segment_bad_rates": segments,
    }


def print_target_imbalance(result: dict) -> None:
    print("\n" + "=" * 58)
    print("1. TARGET / BAD-RATE CHECK")
    print("=" * 58)
    print(f"Total customers:       {result['n_total']:,}")
    print(f"Modelable (has_outcome=1): {result['n_modelable']:,}")
    print(f"Overall bad rate:      {result['overall_bad_rate']:.2%}")
    print(f"Bad count:             {result['n_bad']:,}")
    if result["n_bad"] < 300:
        print("  WARNING: fewer than 300 positive cases — "
              "model evaluation will be unstable.")
    for seg_col, rates in result["segment_bad_rates"].items():
        print(f"\nBy {seg_col}:")
        for val, stats in rates.items():
            print(f"  {val}: bad rate {stats['bad_rate']:.2%}  "
                  f"n={stats['n']:,}")


# ── SECTION 2: MISSINGNESS ────────────────────────────────
def check_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing rate per candidate feature. NULLs here are
    informative by design (e.g. avg_order_value_gbp is NULL
    for customers with zero approved transactions), not
    random — this check confirms NULLs land where expected,
    not somewhere they shouldn't.
    """
    cols = [c for c in MODEL_FEATURES if c in df.columns]
    miss = df[cols].isna().mean().round(4).sort_values(ascending=False)
    return miss.reset_index().rename(
        columns={"index": "feature", 0: "missing_rate"}
    )


def print_missingness(miss_df: pd.DataFrame) -> None:
    print("\n" + "=" * 58)
    print("2. MISSINGNESS")
    print("=" * 58)
    nonzero = miss_df[miss_df["missing_rate"] > 0]
    if nonzero.empty:
        print("No missing values in MODEL_FEATURES.")
    else:
        for _, row in nonzero.iterrows():
            print(f"  {row['feature']:<40} {row['missing_rate']:.2%}")


# ── SECTION 3: UNIVARIATE WoE / IV SCREENING ──────────────
# Weight of Evidence and Information Value — the standard
# first-pass feature screen in credit scoring. Thresholds
# read from config.yaml — see eda_thresholds section.
# These are an industry convention (Siddiqi, Credit Risk
# Scorecards), not derived from this dataset's own numbers
# the way hellinger_threshold was. IV < 0.02: no signal,
# drop. 0.02-0.3: usable. IV > 0.5: investigate for leakage
# before including — an unusually strong single feature is
# more often a leak than a discovery.
EDA_CFG = CONFIG.get("eda_thresholds", {})
IV_DROP_THRESHOLD = EDA_CFG.get("iv_drop_threshold", 0.02)
IV_LEAKAGE_THRESHOLD = EDA_CFG.get("iv_leakage_threshold", 0.50)
VIF_THRESHOLD = EDA_CFG.get("vif_threshold", 10.0)
CORRELATION_THRESHOLD = EDA_CFG.get("correlation_threshold", 0.70)
OUTLIER_IQR_MULTIPLIER = EDA_CFG.get("outlier_iqr_multiplier", 1.5)


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int = 10
) -> dict:
    """
    Bins a numeric feature into n_bins equal-frequency
    buckets (or uses categories directly if non-numeric),
    then computes WoE and IV per bin.
    """
    data = df[[feature, target]].dropna()
    if data.empty:
        return {"feature": feature, "iv": None, "note": "all NaN"}

    if pd.api.types.is_numeric_dtype(data[feature]) and data[feature].nunique() > n_bins:
        try:
            data["bucket"] = pd.qcut(
                data[feature], n_bins, duplicates="drop"
            )
        except ValueError:
            data["bucket"] = data[feature]
    else:
        # Binary/low-cardinality numeric or categorical: bucket by
        # exact value rather than quantile, which collapses to a
        # single bin on skewed binary splits and silently reports
        # IV=0 for genuinely predictive features (e.g. thin_file_flag).
        data["bucket"] = data[feature]

    grouped = data.groupby("bucket")[target].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    if total_bad == 0 or total_good == 0:
        return {"feature": feature, "iv": None, "note": "no variation in target"}

    grouped["dist_bad"] = (grouped["bad"] + 0.5) / (total_bad + 0.5 * len(grouped))
    grouped["dist_good"] = (grouped["good"] + 0.5) / (total_good + 0.5 * len(grouped))
    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv_bin"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

    iv = float(grouped["iv_bin"].sum())
    return {"feature": feature, "iv": round(iv, 4), "n_bins": len(grouped)}


def run_iv_screening(df: pd.DataFrame) -> pd.DataFrame:
    modelable = df[df[FILTER_COLUMN] == FILTER_VALUE]
    results = []
    for feat in MODEL_FEATURES:
        if feat not in modelable.columns:
            continue
        results.append(compute_woe_iv(modelable, feat, TARGET_COLUMN))
    iv_df = pd.DataFrame(results).sort_values(
        "iv", ascending=False, na_position="last"
    )
    return iv_df


def print_iv_screening(iv_df: pd.DataFrame) -> None:
    print("\n" + "=" * 58)
    print("3. UNIVARIATE SCREENING — Weight of Evidence / IV")
    print("=" * 58)
    for _, row in iv_df.iterrows():
        if row["iv"] is None:
            print(f"  {row['feature']:<40} IV: n/a ({row.get('note','')})")
            continue
        flag = ""
        if row["iv"] < IV_DROP_THRESHOLD:
            flag = "  <- below drop threshold, weak/no signal"
        elif row["iv"] > IV_LEAKAGE_THRESHOLD:
            flag = "  <- ABOVE 0.5: investigate for leakage before using"
        print(f"  {row['feature']:<40} IV: {row['iv']:.4f}{flag}")
    print(f"\nThresholds: drop < {IV_DROP_THRESHOLD}, "
          f"investigate for leakage > {IV_LEAKAGE_THRESHOLD}")


# ── SECTION 4: CORRELATION / MULTICOLLINEARITY ────────────
def check_correlation_and_vif(df: pd.DataFrame) -> tuple:
    """
    Correlation matrix and VIF on numeric MODEL_FEATURES.
    VIF > 10 is the conventional red flag for multicollinearity
    — matters most for Logistic Regression, since correlated
    inputs distort its coefficients and undermine the
    interpretability claim it's chosen for (Decision 4).
    """
    modelable = df[df[FILTER_COLUMN] == FILTER_VALUE]
    numeric_feats = [
        f for f in MODEL_FEATURES
        if f in modelable.columns
        and pd.api.types.is_numeric_dtype(modelable[f])
    ]
    X = modelable[numeric_feats].fillna(modelable[numeric_feats].median())

    corr = X.corr().round(3)

    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = None
        vif_data.append({"feature": col, "vif": round(vif, 2) if vif else None})
    vif_df = pd.DataFrame(vif_data).sort_values(
        "vif", ascending=False, na_position="last"
    )

    return corr, vif_df


def print_correlation_and_vif(corr: pd.DataFrame, vif_df: pd.DataFrame) -> None:
    print("\n" + "=" * 58)
    print("4. CORRELATION / MULTICOLLINEARITY (VIF)")
    print("=" * 58)
    high_corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) > CORRELATION_THRESHOLD:
                high_corr_pairs.append((cols[i], cols[j], val))
    if high_corr_pairs:
        print("Highly correlated pairs (|r| > 0.7):")
        for a, b, v in high_corr_pairs:
            print(f"  {a} <-> {b}: {v:.3f}")
    else:
        print("No feature pairs with |r| > 0.7.")

    print("\nVIF (standard multicollinearity flag, see config "
          f"eda_thresholds.vif_threshold = {VIF_THRESHOLD}):")
    for _, row in vif_df.iterrows():
        flag = "  <- HIGH" if row["vif"] and row["vif"] > VIF_THRESHOLD else ""
        vif_str = f"{row['vif']:.2f}" if row["vif"] is not None else "n/a"
        print(f"  {row['feature']:<40} VIF: {vif_str}{flag}")


# ── SECTION 5: OUTLIER FLAGGING ───────────────────────────
def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier flagging on numeric MODEL_FEATURES.
    Reports the rate, does not automatically cap — capping
    is a modelling decision to make deliberately, not
    silently inside an EDA script.
    """
    modelable = df[df[FILTER_COLUMN] == FILTER_VALUE]
    numeric_feats = [
        f for f in MODEL_FEATURES
        if f in modelable.columns
        and pd.api.types.is_numeric_dtype(modelable[f])
    ]
    rows = []
    for feat in numeric_feats:
        col = modelable[feat].dropna()
        if col.empty:
            continue
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        lower, upper = (
            q1 - OUTLIER_IQR_MULTIPLIER * iqr,
            q3 + OUTLIER_IQR_MULTIPLIER * iqr,
        )
        outlier_rate = float(((col < lower) | (col > upper)).mean())
        rows.append({"feature": feat, "outlier_rate": round(outlier_rate, 4)})
    return pd.DataFrame(rows).sort_values("outlier_rate", ascending=False)


def print_outliers(outlier_df: pd.DataFrame) -> None:
    print("\n" + "=" * 58)
    print("5. OUTLIER FLAGGING (IQR method)")
    print("=" * 58)
    for _, row in outlier_df.iterrows():
        if row["outlier_rate"] > 0:
            print(f"  {row['feature']:<40} {row['outlier_rate']:.2%}")


# ── SECTION 6: SPLIT STABILITY SANITY CHECK ───────────────
def check_split_stability(df: pd.DataFrame, test_size: float = None) -> dict:
    """
    Confirms a random train/test split doesn't accidentally
    skew bad rate or key segment composition before any
    model is trained — a sanity check, not a substitute for
    the real PSI drift monitoring built in Decision 5.
    """
    from sklearn.model_selection import train_test_split

    test_size = test_size or MODEL_CFG.get("test_size", 0.2)
    modelable = df[df[FILTER_COLUMN] == FILTER_VALUE]

    train, test = train_test_split(
        modelable, test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=modelable[TARGET_COLUMN]
    )

    return {
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "train_bad_rate": round(float(train[TARGET_COLUMN].mean()), 4),
        "test_bad_rate": round(float(test[TARGET_COLUMN].mean()), 4),
    }


def print_split_stability(result: dict) -> None:
    print("\n" + "=" * 58)
    print("6. TRAIN/TEST SPLIT STABILITY")
    print("=" * 58)
    print(f"Train: {result['train_n']:,}  bad rate {result['train_bad_rate']:.2%}")
    print(f"Test:  {result['test_n']:,}  bad rate {result['test_bad_rate']:.2%}")
    diff = abs(result["train_bad_rate"] - result["test_bad_rate"])
    print(f"Difference: {diff:.2%}"
          f"{'  <- check stratification' if diff > 0.005 else ''}")


# ── ENTRY POINT ───────────────────────────────────────────
def main() -> None:
    print("=" * 58)
    print("SENTINEL CREDIT — PRE-MODELLING EDA")
    print("=" * 58)

    df = load_feature_store()
    print(f"Loaded {len(df):,} customers from "
          f"{DB['schema']}.customer_features")

    target_result = check_target_imbalance(df)
    print_target_imbalance(target_result)

    # Save so downstream reporting/charts read a real file,
    # not numbers copied out of the console.
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports"
    )
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{
        "n_total": target_result["n_total"],
        "n_modelable": target_result["n_modelable"],
        "n_not_modelable": target_result["n_total"] - target_result["n_modelable"],
        "n_bad": target_result["n_bad"],
        "n_good": target_result["n_modelable"] - target_result["n_bad"],
        "overall_bad_rate": target_result["overall_bad_rate"],
    }]).to_csv(os.path.join(out_dir, "eda_target_summary.csv"), index=False)
    print(f"\nSaved: {out_dir}/eda_target_summary.csv")

    miss_df = check_missingness(df)
    print_missingness(miss_df)

    iv_df = run_iv_screening(df)
    print_iv_screening(iv_df)

    corr, vif_df = check_correlation_and_vif(df)
    print_correlation_and_vif(corr, vif_df)

    outlier_df = flag_outliers(df)
    print_outliers(outlier_df)

    split_result = check_split_stability(df)
    print_split_stability(split_result)

    print("\n" + "=" * 58)
    print("EDA COMPLETE — review flags above before training")
    print("=" * 58)


if __name__ == "__main__":
    main()

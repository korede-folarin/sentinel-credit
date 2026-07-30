"""
=============================================================
Sentinel Credit — Synthetic Data Validation
=============================================================
File:    validate_synthetic.py
Purpose: Validates generator output on the three dimensions
         defined by the FCA/ICO/Alan Turing Institute
         framework: fidelity, utility, privacy.
Author:  Folarin Korede
Version: 1.1.0

Change from 1.0.0: privacy thresholds (singling-out,
near-duplicate) and the utility Gini-retention band are now
read from config.yaml rather than hardcoded, matching the
fidelity thresholds. See docs/engineering_decision_log.md —
Decision 2, "Known Gaps" note.

See docs/engineering_decision_log.md — Decision 2.
=============================================================
"""

# ── IMPORTS ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import os
import json
import yaml
from datetime import datetime


# ── CONFIGURATION ─────────────────────────────────────────
def load_config() -> dict:
    """Load project configuration from config.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
GEN    = CONFIG["data_generator"]
VAL    = CONFIG["validation"]
RANDOM_SEED = GEN["random_seed"]

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR = os.path.join(PROJECT_ROOT, GEN["output_dir"])


def load_outputs() -> tuple:
    """Load the three generated tables from the output dir."""
    customers = pd.read_csv(
        os.path.join(DATA_DIR, "customers.csv")
    )
    transactions = pd.read_csv(
        os.path.join(DATA_DIR, "transactions.csv")
    )
    repayments = pd.read_csv(
        os.path.join(DATA_DIR, "repayments.csv")
    )
    return customers, transactions, repayments


# ── SECTION 2: FIDELITY ───────────────────────────────────
# Fidelity tests whether the generator produced the
# distributions it was calibrated to produce. Since no real
# UK BNPL dataset exists, the comparison target is the set
# of published statistics in config.yaml.
# See docs/engineering_decision_log.md — Decision 2.

# Thresholds — design assumptions, read from config.yaml.
# At n=50,000 sampling error is tiny, so a well-calibrated
# generator should sit far below these.
HELLINGER_THRESHOLD  = VAL["hellinger_threshold"]
PREVALENCE_TOLERANCE = VAL["prevalence_tolerance"]


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger distance between two discrete distributions.
    Bounded 0 (identical) to 1 (no overlap).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(
        np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence, base 2. Bounded 0 to 1.
    Symmetric and defined even when a bin has zero mass.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def check_categorical(
    observed: pd.Series,
    target: dict,
    name: str
) -> dict:
    """
    Compares an observed categorical column against its
    config.yaml target distribution.
    """
    obs_props = observed.value_counts(normalize=True)

    categories = list(target.keys())
    p_target   = np.array([target[c] for c in categories])
    p_observed = np.array(
        [obs_props.get(c, 0.0) for c in categories]
    )

    h  = hellinger(p_observed, p_target)
    js = js_divergence(p_observed, p_target)
    max_diff = float(np.max(np.abs(p_observed - p_target)))

    return {
        "check":        name,
        "type":         "categorical",
        "hellinger":    round(h, 5),
        "js_divergence": round(js, 5),
        "max_abs_diff": round(max_diff, 5),
        "passed":       bool(h < HELLINGER_THRESHOLD),
        "detail": {
            c: {
                "target":   round(float(target[c]), 4),
                "observed": round(float(obs_props.get(c, 0.0)), 4),
            }
            for c in categories
        },
    }


def check_prevalence(
    observed: pd.Series,
    target: float,
    name: str
) -> dict:
    """
    Compares an observed binary flag rate against its
    config.yaml target prevalence.
    """
    rate = float(observed.mean())
    diff = abs(rate - target)

    return {
        "check":    name,
        "type":     "prevalence",
        "target":   round(target, 4),
        "observed": round(rate, 4),
        "abs_diff": round(diff, 4),
        "passed":   bool(diff < PREVALENCE_TOLERANCE),
    }


def run_fidelity(
    customers: pd.DataFrame,
    transactions: pd.DataFrame
) -> list:
    """
    Runs every fidelity check and returns a list of results.
    Each check compares generator output against the
    published statistic it was calibrated to.
    """
    results = []

    # Categorical distributions vs config targets
    results.append(check_categorical(
        customers["age_bracket"],
        GEN["age_distribution"],
        "age_distribution — Klarna Wikipink UK 2024"
    ))
    results.append(check_categorical(
        customers["income_band"],
        GEN["income_bands"],
        "income_bands — ONS FYE 2023 shifted"
    ))
    results.append(check_categorical(
        transactions["merchant_category"],
        GEN["merchant_categories"],
        "merchant_categories — Klarna UK directory"
    ))

    # Binary prevalences vs config targets
    results.append(check_prevalence(
        customers["thin_file_flag"],
        GEN["thin_file_prevalence"],
        "thin_file — PwC/TotallyMoney 2022"
    ))
    results.append(check_prevalence(
        customers["vulnerability_resilience"],
        GEN["vulnerability_prevalence"],
        "resilience driver — FCA Financial Lives 2022"
    ))
    results.append(check_prevalence(
        customers["mental_health_flag"],
        GEN["mental_health_prevalence"],
        "health driver — NHS Digital"
    ))

    # A/B split — engineered 50/50
    results.append(check_prevalence(
        (customers["ab_test_group"] == "B_relaxed").astype(int),
        GEN["ab_test"]["group_B_size"],
        "ab_test group split — engineered 50/50"
    ))

    # Merchant category structural completeness
    results.append(check_merchant_ordering(transactions))

    return results


def check_merchant_ordering(transactions: pd.DataFrame) -> dict:
    """
    Structural completeness check on merchant categories.
    Confirms every category configured in config.yaml
    appears in the generated data, and no unexpected
    category is present. This guards against typos or
    dropped categories in the generator.
    """
    configured = set(GEN["merchant_categories"].keys())
    observed   = set(transactions["merchant_category"].unique())
    missing    = configured - observed
    unexpected = observed - configured

    return {
        "check":  "merchant categories — completeness",
        "type":   "structural",
        "missing":    sorted(missing),
        "unexpected": sorted(unexpected),
        "passed": bool(not missing and not unexpected),
    }


def print_fidelity(results: list) -> None:
    """Prints fidelity results as a readable console table."""
    print("\n" + "=" * 58)
    print("FIDELITY — output vs calibration targets")
    print("=" * 58)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r["type"] == "categorical":
            print(
                f"[{status}] {r['check']}\n"
                f"       hellinger {r['hellinger']:.4f}  "
                f"js {r['js_divergence']:.4f}  "
                f"max diff {r['max_abs_diff']:.4f}"
            )
        elif r["type"] == "prevalence":
            print(
                f"[{status}] {r['check']}\n"
                f"       target {r['target']:.2%}  "
                f"observed {r['observed']:.2%}  "
                f"diff {r['abs_diff']:.2%}"
            )
        else:
            print(f"[{status}] {r['check']}")
    n_pass = sum(r["passed"] for r in results)
    print("-" * 58)
    print(f"{n_pass}/{len(results)} fidelity checks passed")


# ── SECTION 3: UTILITY (TSTR) ─────────────────────────────
# Utility tests whether a model trained on SYNTHETIC data
# learns real predictive signal. The strongest test is
# Train-on-Synthetic, Test-on-Real (TSTR): train a credit
# model on our synthetic data, then score it on the real
# Home Credit Default Risk dataset.
#
# Honest limitation: Home Credit is a different market with
# different features, partly PCA-transformed. TSTR here
# tests DIRECTIONAL signal preservation — whether the model
# ranks risk in the right direction on real data — NOT exact
# performance parity. Documented in Decision 2.
#
# Reference file expected at:
#   validation/reference/home_credit.csv
# If absent, utility falls back to an internal check (train
# on synthetic, test on a held-out synthetic slice) and the
# report records that real TSTR was not run.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

REFERENCE_PATH = os.path.join(
    PROJECT_ROOT, "validation", "reference", "home_credit.csv"
)

# Utility acceptance thresholds — read from config.yaml.
# GINI_MIN: minimum acceptable Gini for either the TSTR real
# score or the internal held-out score to count as "learned
# real signal" at all.
# GINI_RETENTION_TOLERANCE (gini_tolerance): how far the
# synthetic-vs-real Gini gap is allowed to be, expressed as
# an absolute Gini-point band, before the TSTR result is
# flagged as diverging further than expected. This is
# reported alongside the pass/fail, not the sole gate on it,
# because a small n or a genuinely harder real dataset can
# widen the gap without indicating a broken generator.
GINI_MIN = 0.10
GINI_RETENTION_TOLERANCE = VAL["gini_tolerance"]

# Features shared between our synthetic schema and a typical
# real credit dataset. Only numeric, non-identifying,
# non-protected fields are used for modelling.
SYNTHETIC_MODEL_FEATURES = [
    "age",
    "annual_income_gbp",
    "existing_monthly_obligations_gbp",
    "credit_score",
    "months_credit_history",
    "num_credit_accounts",
    "num_defaults_24m",
    "thin_file_flag",
]

# Cross-market TSTR uses only features with a GENUINE
# equivalent in Home Credit. Chosen by mapping validity
# alone, before seeing any result — not by importance.
# Excluded (no true equivalent in Home Credit):
#   months_credit_history  -> employment length (different)
#   num_credit_accounts    -> loan prolongations (different)
#   num_defaults_24m       -> social-circle defaults (different)
#   thin_file_flag         -> derived from ext score (invented cutoff)
CROSS_MARKET_FEATURES = [
    "age",
    "annual_income_gbp",
    "existing_monthly_obligations_gbp",
    "credit_score",
]


def gini_from_auc(auc: float) -> float:
    """Gini = 2 * AUC - 1. Standard credit scoring metric."""
    return 2.0 * auc - 1.0


def build_customer_default_label(
    customers: pd.DataFrame,
    repayments: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds a customer-level default label for modelling.
    A customer is labelled default=1 if any of their
    transactions defaulted.
    """
    cust_default = (
        repayments.groupby("customer_id")["default_flag"]
        .max()
        .rename("default")
    )
    df = customers.merge(
        cust_default, on="customer_id", how="inner"
    )
    return df


def train_credit_model(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple:
    """
    Trains a logistic regression credit model.
    Returns the fitted model and the fitted scaler so the
    same transform can be applied to the test set.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler


def evaluate_model(
    model,
    scaler,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """Scores a fitted model on a test set. Returns AUC/Gini."""
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, proba)
    return {
        "auc":  round(float(auc), 4),
        "gini": round(float(gini_from_auc(auc)), 4),
        "n":    int(len(y)),
    }


def load_reference() -> pd.DataFrame:
    """
    Loads the real Home Credit reference dataset if present.
    Returns None if the file is not available.
    """
    if not os.path.exists(REFERENCE_PATH):
        return None
    return pd.read_csv(REFERENCE_PATH)


def map_reference_features(ref: pd.DataFrame) -> pd.DataFrame:
    """
    Maps Home Credit columns onto our synthetic feature
    schema so a model trained on synthetic data can score
    the real data. Home Credit uses different column names,
    so we map the closest equivalent and document each.

    Home Credit -> synthetic equivalent:
      DAYS_BIRTH / -365        -> age
      AMT_INCOME_TOTAL         -> annual_income_gbp
      AMT_ANNUITY              -> existing_monthly_obligations_gbp
      EXT_SOURCE_2 * 999       -> credit_score (proxy, scaled)
      DAYS_EMPLOYED / -30      -> months_credit_history (proxy)
      CNT_CREDIT_PROLONG       -> num_credit_accounts (proxy)
      DEF_30_CNT_SOCIAL_CIRCLE -> num_defaults_24m (proxy)
      (derived)                -> thin_file_flag
      TARGET                   -> default
    """
    out = pd.DataFrame()
    out["age"] = (-ref["DAYS_BIRTH"] / 365).round().astype(int)
    out["annual_income_gbp"] = ref["AMT_INCOME_TOTAL"]
    out["existing_monthly_obligations_gbp"] = (
        ref.get("AMT_ANNUITY", 0).fillna(0) / 12
    )
    # EXT_SOURCE_2 is 0-1; scale to a 300-999 style score
    ext = ref.get("EXT_SOURCE_2", pd.Series(0.5, index=ref.index))
    out["credit_score"] = (300 + ext.fillna(0.5) * 699).round()
    emp = ref.get("DAYS_EMPLOYED", pd.Series(0, index=ref.index))
    out["months_credit_history"] = (
        (-emp / 30).clip(lower=0, upper=600).fillna(0).round()
    )
    out["num_credit_accounts"] = (
        ref.get("CNT_CREDIT_PROLONG", pd.Series(0, index=ref.index))
        .fillna(0).astype(int)
    )
    out["num_defaults_24m"] = (
        ref.get("DEF_30_CNT_SOCIAL_CIRCLE",
                pd.Series(0, index=ref.index))
        .fillna(0).astype(int)
    )
    # Thin file proxy: low external score and short employment
    out["thin_file_flag"] = (
        (ext.fillna(0.5) < 0.3)
    ).astype(int)
    out["default"] = ref["TARGET"]
    return out.dropna()


def feature_direction_agreement(
    model_syn,
    model_real,
    feature_names: list
) -> dict:
    """
    Compares how each feature drives default risk in a model
    trained on synthetic vs a model trained on real data.

    Two genuine measures:
      Sign agreement — fraction of features whose logistic
        coefficient has the SAME sign in both models. Sign is
        the direction the feature pushes default odds.
      Spearman rho — rank correlation of feature importance
        (|coefficient|) between the two models. Do features
        rank in the same order of importance in both worlds.

    Both features are standardised before fitting, so
    coefficient magnitude is a fair importance measure.
    """
    coef_syn  = model_syn.coef_[0]
    coef_real = model_real.coef_[0]

    same_sign = np.sign(coef_syn) == np.sign(coef_real)
    sign_agreement = float(np.mean(same_sign))

    rho, _ = spearmanr(np.abs(coef_syn), np.abs(coef_real))

    per_feature = {}
    for i, f in enumerate(feature_names):
        per_feature[f] = {
            "coef_synthetic": round(float(coef_syn[i]), 4),
            "coef_real":      round(float(coef_real[i]), 4),
            "same_direction": bool(same_sign[i]),
        }

    return {
        "sign_agreement":  round(sign_agreement, 4),
        "spearman_rho":    round(float(rho), 4),
        "per_feature":     per_feature,
    }


def run_utility(
    customers: pd.DataFrame,
    repayments: pd.DataFrame
) -> dict:
    """
    Runs the utility check.
    If the real Home Credit reference is present, runs full
    TSTR. Otherwise runs an internal held-out check and
    records that real TSTR was not performed.
    """
    df = build_customer_default_label(customers, repayments)
    X_syn = df[SYNTHETIC_MODEL_FEATURES]
    y_syn = df["default"]

    model, scaler = train_credit_model(X_syn, y_syn)

    ref = load_reference()

    if ref is not None:
        real = map_reference_features(ref)
        X_real = real[CROSS_MARKET_FEATURES]
        y_real = real["default"]

        X_syn_cm = df[CROSS_MARKET_FEATURES]
        model_cm, scaler_cm = train_credit_model(X_syn_cm, y_syn)

        real_eval = evaluate_model(model_cm, scaler_cm, X_real, y_real)
        syn_eval  = evaluate_model(model_cm, scaler_cm, X_syn_cm, y_syn)

        retention = (
            round(real_eval["gini"] / syn_eval["gini"], 4)
            if syn_eval["gini"] > 0 else None
        )

        # Retention band check — how far apart are the two
        # Gini scores in absolute terms, against the
        # config-driven tolerance. Reported, not the sole gate.
        gini_gap = round(
            abs(real_eval["gini"] - syn_eval["gini"]), 4
        )
        within_tolerance = bool(gini_gap <= GINI_RETENTION_TOLERANCE)

        scaler_r = StandardScaler()
        Xr = scaler_r.fit_transform(X_real)
        model_real = LogisticRegression(max_iter=1000)
        model_real.fit(Xr, y_real)
        direction = feature_direction_agreement(
            model_cm, model_real, CROSS_MARKET_FEATURES
        )

        passed = bool(
            real_eval["gini"] > GINI_MIN
            and direction["sign_agreement"] >= 0.75
        )
        result = {
            "mode":               "TSTR + feature direction (real Home Credit)",
            "trained_on":         "synthetic",
            "tested_on":          "real Home Credit",
            "synthetic_gini":     syn_eval["gini"],
            "real_gini":          real_eval["gini"],
            "gini_retention":     retention,
            "gini_gap":           gini_gap,
            "gini_gap_tolerance": GINI_RETENTION_TOLERANCE,
            "gini_gap_within_tolerance": within_tolerance,
            "real_auc":           real_eval["auc"],
            "real_n":             real_eval["n"],
            "sign_agreement":     direction["sign_agreement"],
            "spearman_rho":       direction["spearman_rho"],
            "per_feature":        direction["per_feature"],
            "passed":             passed,
            "note": (
                "Test A: synthetic-trained model scored on real "
                "data (Gini retention). Test B: feature "
                "coefficient sign agreement and importance rank "
                "correlation. gini_gap is reported against "
                "gini_tolerance as supporting context, not the "
                "pass/fail gate, since a genuinely harder real "
                "dataset can widen the gap without indicating a "
                "broken generator. Proxy mappings between BNPL "
                "and Home Credit mean a disagreeing feature may "
                "reflect market difference, not generator error."
            ),
        }
    else:
        split = int(len(df) * 0.8)
        model_h, scaler_h = train_credit_model(
            X_syn.iloc[:split], y_syn.iloc[:split]
        )
        held = evaluate_model(
            model_h, scaler_h,
            X_syn.iloc[split:], y_syn.iloc[split:]
        )
        mode = "INTERNAL (held-out synthetic — real file absent)"
        result = {
            "mode":         mode,
            "trained_on":   "synthetic (80%)",
            "tested_on":    "synthetic held-out (20%)",
            "holdout_gini": held["gini"],
            "holdout_auc":  held["auc"],
            "holdout_n":    held["n"],
            "passed":       bool(held["gini"] > GINI_MIN),
            "note": (
                "Real Home Credit reference not found at "
                "validation/reference/home_credit.csv. "
                "Ran internal held-out check instead. Drop "
                "the real file in to enable full TSTR."
            ),
        }

    return result


def print_utility(result: dict) -> None:
    """Prints the utility result."""
    print("\n" + "=" * 58)
    print("UTILITY — can a model learn from the synthetic data")
    print("=" * 58)
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] mode: {result['mode']}")
    if "real_gini" in result:
        print(f"       Test A — TSTR")
        print(f"         synthetic Gini {result['synthetic_gini']:.4f}  "
              f"real Gini {result['real_gini']:.4f}  "
              f"retention {result['gini_retention']}")
        tol_status = "within" if result["gini_gap_within_tolerance"] else "outside"
        print(f"         gini gap {result['gini_gap']:.4f}  "
              f"({tol_status} tolerance {result['gini_gap_tolerance']:.4f})")
        print(f"       Test B — feature direction")
        print(f"         sign agreement {result['sign_agreement']:.2%}  "
              f"rank rho {result['spearman_rho']:.4f}")
    else:
        print(f"       held-out Gini {result['holdout_gini']:.4f}  "
              f"AUC {result['holdout_auc']:.4f}  "
              f"n {result['holdout_n']:,}")
    print(f"       {result['note']}")


# ── SECTION 4: PRIVACY ────────────────────────────────────
# Privacy tests whether individuals could be re-identified
# from the synthetic data. Because our generator uses only
# published AGGREGATE statistics as inputs (never real
# individual records), there is no real person to re-identify
# by construction. These checks guard against accidental
# memorisation-style patterns.
# See docs/engineering_decision_log.md — Decision 2.

from sklearn.neighbors import NearestNeighbors

PRIVACY_QUASI_IDENTIFIERS = [
    "age", "gender", "income_band", "employment_status",
    "age_bracket", "device_type",
]

# Privacy thresholds — design assumptions, read from
# config.yaml rather than hardcoded. These are two distinct
# checks with two distinct natural ceilings; a single shared
# threshold cannot correctly govern both, so each gets its
# own key. See docs/engineering_decision_log.md — Decision 2.
SINGLING_OUT_THRESHOLD   = VAL["singling_out_threshold"]
NEAR_DUPLICATE_THRESHOLD = VAL["near_duplicate_threshold"]


def singling_out_check(customers: pd.DataFrame) -> dict:
    """
    Singling-out risk: can a combination of quasi-identifiers
    isolate exactly one record? We count how many customers
    are UNIQUE on the quasi-identifier set. A high unique rate
    on low-cardinality attributes would be a concern.

    Because our records are synthetic (not real people), a
    unique combination points to no real individual. We still
    report the rate as a standard privacy diagnostic.
    """
    qi = [c for c in PRIVACY_QUASI_IDENTIFIERS
          if c in customers.columns]
    group_sizes = customers.groupby(qi).size()
    n_unique = int((group_sizes == 1).sum())
    unique_rate = float(n_unique / len(customers))

    return {
        "check":           "singling-out",
        "quasi_identifiers": qi,
        "n_unique_records": n_unique,
        "unique_rate":     round(unique_rate, 4),
        "threshold":       SINGLING_OUT_THRESHOLD,
        # Threshold is a design assumption: on 6 low-to-mid
        # cardinality attributes a modest unique rate is
        # expected and acceptable for synthetic data.
        "passed":          bool(unique_rate < SINGLING_OUT_THRESHOLD),
        "note": (
            "Records are synthetic; a unique quasi-identifier "
            "combination corresponds to no real individual. "
            "Reported as a standard diagnostic."
        ),
    }


def nearest_neighbour_check(
    customers: pd.DataFrame,
    sample_size: int = 5000
) -> dict:
    """
    Nearest-neighbour distance within the synthetic set.
    If many records sit extremely close to another record on
    numeric features, the generator may be producing near
    duplicates. Healthy synthetic data shows well-separated
    records, not clusters of near-identical rows.

    We sample for speed and measure the distance from each
    sampled record to its nearest OTHER record.
    """
    numeric = [
        "age", "annual_income_gbp",
        "existing_monthly_obligations_gbp", "credit_score",
        "months_credit_history",
    ]
    numeric = [c for c in numeric if c in customers.columns]

    data = customers[numeric].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    n = min(sample_size, len(scaled))
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(scaled), size=n, replace=False)
    sample = scaled[idx]

    nn = NearestNeighbors(n_neighbors=2).fit(scaled)
    distances, _ = nn.kneighbors(sample)
    nearest = distances[:, 1]

    mean_dist = float(np.mean(nearest))
    near_dup_rate = float(np.mean(nearest < 0.01))

    return {
        "check":          "nearest-neighbour distance",
        "features":       numeric,
        "sample_size":    n,
        "mean_nn_distance": round(mean_dist, 4),
        "near_duplicate_rate": round(near_dup_rate, 4),
        "threshold":      NEAR_DUPLICATE_THRESHOLD,
        # Design assumption: under 1% near-duplicates is
        # healthy separation for a generator of this size.
        "passed":         bool(near_dup_rate < NEAR_DUPLICATE_THRESHOLD),
        "note": (
            "Standardised numeric distance. Low near-duplicate "
            "rate indicates well-separated synthetic records."
        ),
    }


def run_privacy(customers: pd.DataFrame) -> list:
    """Runs all privacy checks and returns a list of results."""
    return [
        singling_out_check(customers),
        nearest_neighbour_check(customers),
    ]


def print_privacy(results: list) -> None:
    """Prints privacy results."""
    print("\n" + "=" * 58)
    print("PRIVACY — re-identification risk")
    print("=" * 58)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r["check"] == "singling-out":
            print(f"[{status}] singling-out")
            print(f"       unique records {r['n_unique_records']:,}  "
                  f"rate {r['unique_rate']:.2%}  "
                  f"threshold {r['threshold']:.2%}")
        else:
            print(f"[{status}] nearest-neighbour")
            print(f"       mean distance {r['mean_nn_distance']:.4f}  "
                  f"near-duplicate rate {r['near_duplicate_rate']:.2%}  "
                  f"threshold {r['threshold']:.2%}")
    n_pass = sum(r["passed"] for r in results)
    print("-" * 58)
    print(f"{n_pass}/{len(results)} privacy checks passed")


# ── ENTRY POINT ───────────────────────────────────────────
def main() -> None:
    print("=" * 58)
    print("SENTINEL CREDIT — SYNTHETIC DATA VALIDATION")
    print("=" * 58)

    customers, transactions, repayments = load_outputs()
    print(f"Loaded: {len(customers):,} customers  "
          f"{len(transactions):,} transactions  "
          f"{len(repayments):,} repayments")

    fidelity_results = run_fidelity(customers, transactions)
    print_fidelity(fidelity_results)

    utility_result = run_utility(customers, repayments)
    print_utility(utility_result)

    privacy_results = run_privacy(customers)
    print_privacy(privacy_results)


if __name__ == "__main__":
    main()

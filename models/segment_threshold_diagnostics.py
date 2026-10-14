"""
Sentinel Credit — Segment-Level Threshold Diagnostics

Purpose
-------
Answers Decision 19's operating-policy review questions that
evaluate_models.py's aggregate output cannot.

Checks 2-3 (score distribution, threshold sweep by segment)
established that the thin_file_flag=0 defaulted population sits
entirely below the current operating range on both splits, and
that no single global threshold closes the gap without an
extreme false-positive cost in thin_file_flag=1. That rules out
the threshold itself as the cause.

Check 4 answers the resulting question: can the model
discriminate risk WITHIN each segment at all? Within-segment
AUC/Gini/KS, with bootstrap confidence intervals given the small
defaulted-customer counts in thin_file_flag=0 (61 stratified, 34
temporal) — a single point estimate at that sample size can look
more definitive than it is.

Interpretation is NOT automated here. AUC materially above 0.5
and stable across splits means the model still ranks within the
segment — the issue is then the absolute score scale interacting
with a single global policy, not a ranking failure. AUC near 0.5
means little useful within-segment discrimination — a model/
feature limitation threshold engineering cannot fix. Stratified
looking reasonable while temporal collapses would point to
segment-specific instability over time rather than either of the
above. None of these labels are assigned automatically by this
script; they require reading the numbers.

This script does NOT select or apply a segment-specific
threshold — that is a governance decision under Decision 12.
Even strong within-segment AUC does not by itself mean
"just calibration" — a single globally calibrated model can
still separate scores this way because the segment itself
carries strong risk information. Ranking survives; what policy
to build on top of that (percentiles, segment-adjusted cutoffs,
or something else) is a separate decision.

Reads test_predictions_<split>.csv, already produced by
train_baseline_models.py.

See docs/engineering_decision_log.md — Decision 12, 19.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(
    PROJECT_ROOT, "reports", "threshold_selection", "segment_diagnostics"
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import TARGET_COLUMN  # noqa: E402

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    CONFIG["model_governance"].get("artifacts_dir", "models/artifacts"),
)  # the governed champion package; see config.yaml model_governance

BUSINESS_CUTOFF = CONFIG["model"]["business_cutoff"]
CHAMPION_MODEL = CONFIG["model_governance"]["champion_model"]

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


def load_predictions(split: str) -> pd.DataFrame:
    path = os.path.join(ARTIFACTS_DIR, f"test_predictions_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run train_baseline_models.py first.")
    df = pd.read_csv(path)
    if "thin_file_flag" not in df.columns:
        raise ValueError(f"{path} has no thin_file_flag column.")
    return df


def score_distribution_by_segment(df: pd.DataFrame, proba_col: str) -> pd.DataFrame:
    """Check 2: where DEFAULTED customers' scores actually sit, by segment."""
    defaulted = df[df[TARGET_COLUMN] == 1]
    rows = []
    for seg, sub in defaulted.groupby("thin_file_flag"):
        proba = sub[proba_col]
        rows.append({
            "thin_file_flag": seg,
            "n_defaulted": len(sub),
            "min": round(float(proba.min()), 4),
            "p10": round(float(proba.quantile(0.10)), 4),
            "p25": round(float(proba.quantile(0.25)), 4),
            "median": round(float(proba.median()), 4),
            "p75": round(float(proba.quantile(0.75)), 4),
            "p90": round(float(proba.quantile(0.90)), 4),
            "max": round(float(proba.max()), 4),
            "mean": round(float(proba.mean()), 4),
            "n_above_current_cutoff": int((proba >= BUSINESS_CUTOFF).sum()),
            "pct_above_current_cutoff": round(
                100 * float((proba >= BUSINESS_CUTOFF).mean()), 2
            ),
        })
    return pd.DataFrame(rows)


def threshold_sweep_by_segment(
    df: pd.DataFrame, proba_col: str, thresholds: np.ndarray
) -> pd.DataFrame:
    """Check 3: recall/precision/flagged-rate/FPR by segment, one global threshold at a time."""
    rows = []
    for t in thresholds:
        pred = (df[proba_col] >= t).astype(int)
        for seg in (0, 1):
            mask = df["thin_file_flag"] == seg
            y = df.loc[mask, TARGET_COLUMN].to_numpy()
            p = pred[mask].to_numpy()
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            tn = int(((p == 0) & (y == 0)).sum())
            rows.append({
                "threshold": round(float(t), 4),
                "thin_file_flag": seg,
                "n": int(mask.sum()),
                "n_bad": int(y.sum()),
                "recall": round(tp / max(tp + fn, 1), 4),
                "precision": round(tp / max(tp + fp, 1), 4),
                "flagged_rate": round(float(p.mean()), 4),
                "fpr": round(fp / max(fp + tn, 1), 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
    return pd.DataFrame(rows)


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    order = np.argsort(proba)
    y_sorted = y_true[order]
    total_bad = max(int(y_sorted.sum()), 1)
    total_good = max(int((1 - y_sorted).sum()), 1)
    cum_bad = np.cumsum(y_sorted) / total_bad
    cum_good = np.cumsum(1 - y_sorted) / total_good
    return float(np.max(np.abs(cum_bad - cum_good)))


def bootstrap_auc_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
) -> tuple:
    """
    Percentile bootstrap CI on AUC. Resamples with replacement
    and skips any resample that happens to contain only one
    class (unavoidable at n_defaults=34-61, especially since the
    positive class is already rare within these segments).
    Returns (ci_low, ci_high, n_usable_resamples) — the third
    value matters: if it's far below n_boot, the interval itself
    is on thin ice and should be read that way.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if y_b.min() == y_b.max():
            continue
        aucs.append(roc_auc_score(y_b, proba[idx]))
    if len(aucs) < 50:
        return float("nan"), float("nan"), len(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), len(aucs)


def within_segment_discrimination(df: pd.DataFrame, proba_col: str) -> pd.DataFrame:
    """
    Check 4: does the model rank risk WITHIN each segment, not
    just across the whole population. This is the diagnostic
    that separates a policy/scale problem from a model-signal
    problem — see the module docstring for the interpretation
    rule, which is deliberately not automated into this function.
    """
    rows = []
    for seg in (0, 1):
        sub = df[df["thin_file_flag"] == seg]
        y = sub[TARGET_COLUMN].to_numpy()
        p = sub[proba_col].to_numpy()
        n = len(sub)
        n_bad = int(y.sum())

        if n_bad == 0 or n_bad == n:
            auc = ci_lo = ci_hi = ks = float("nan")
            n_boot_used = 0
        else:
            auc = float(roc_auc_score(y, p))
            ci_lo, ci_hi, n_boot_used = bootstrap_auc_ci(y, p)
            ks = ks_statistic(y, p)

        gini = 2 * auc - 1 if not np.isnan(auc) else float("nan")

        rows.append({
            "thin_file_flag": seg,
            "n": n,
            "n_defaults": n_bad,
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "auc_ci95_low": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
            "auc_ci95_high": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
            "n_bootstrap_usable": n_boot_used,
            "gini": round(gini, 4) if not np.isnan(gini) else None,
            "ks": round(ks, 4) if not np.isnan(ks) else None,
        })
    return pd.DataFrame(rows)


def print_check4(discrimination: pd.DataFrame, split: str) -> None:
    print("\n" + "=" * 62)
    print("WITHIN-SEGMENT DISCRIMINATION")
    print("=" * 62)
    print(f"\nSplit: {split}\n")
    for _, row in discrimination.iterrows():
        print(f"thin_file_flag={int(row['thin_file_flag'])}")
        print(f"  n:         {row['n']:,}")
        print(f"  defaults:  {row['n_defaults']:,}")
        if row["auc"] is None:
            print("  AUC:       n/a (single-class segment)")
        else:
            print(
                f"  AUC:       {row['auc']:.4f}  "
                f"(95% CI: {row['auc_ci95_low']:.4f}-{row['auc_ci95_high']:.4f}, "
                f"n_bootstrap_usable={row['n_bootstrap_usable']})"
            )
            print(f"  Gini:      {row['gini']:.4f}")
            print(f"  KS:        {row['ks']:.4f}")
        print()
    if discrimination["n_defaults"].min() < 100:
        print(
            "  CAUTION: at least one segment has under 100 defaulted "
            "customers in this split. Treat the AUC point estimate with "
            "the confidence interval, not on its own, and do not average "
            "or compare across splits as if each were equally precise."
        )
    print(
        "  Ranking surviving within a segment does not by itself mean "
        "'just calibration' — a single globally calibrated model can "
        "separate scores this way because the segment carries real risk "
        "information. What policy to build on that (percentiles, "
        "segment-adjusted cutoffs, or otherwise) remains a governance "
        "decision under Decision 12, not something this script decides."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="stratified", choices=["stratified", "temporal"]
    )
    parser.add_argument(
        "--model", default=None, help="Defaults to config's champion_model."
    )
    args = parser.parse_args()

    model_name = args.model or CHAMPION_MODEL
    proba_col = f"proba_{model_name}"

    print("=" * 62)
    print("SEGMENT-LEVEL THRESHOLD DIAGNOSTICS")
    print(f"Split: {args.split}  Model: {model_name}  "
          f"Current cutoff: {BUSINESS_CUTOFF}")
    print("=" * 62)
    print("Does NOT select or apply a segment-specific threshold — see")
    print("Decision 12. Establishes whether the thin_file_flag=0 FNR")
    print("pattern is a genuine score-ranking gap, and whether any single")
    print("global threshold could close it. See Decision 19.\n")

    df = load_predictions(args.split)
    if proba_col not in df.columns:
        raise ValueError(
            f"{proba_col} not found in test_predictions_{args.split}.csv"
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    dist = score_distribution_by_segment(df, proba_col)
    print("Check 2 — calibrated score distribution among DEFAULTED customers:")
    print(dist.to_string(index=False))
    dist_path = os.path.join(
        OUT_DIR, f"score_distribution_by_segment_{args.split}.csv"
    )
    dist.to_csv(dist_path, index=False)

    lo, hi = 0.005, min(0.30, float(df[proba_col].max()))
    thresholds = np.linspace(lo, hi, 60)
    sweep = threshold_sweep_by_segment(df, proba_col, thresholds)
    sweep_path = os.path.join(
        OUT_DIR, f"threshold_sweep_by_segment_{args.split}.csv"
    )
    sweep.to_csv(sweep_path, index=False)

    print("\nCheck 3 — recall/FPR by segment across a threshold sweep "
          "(sample points):")
    sample_points = sorted(set(
        [round(BUSINESS_CUTOFF, 4)] + list(np.round(np.linspace(lo, hi, 8), 4))
    ))
    for t in sample_points:
        row = sweep[np.isclose(sweep["threshold"], t, atol=1e-3)]
        r0 = row[row["thin_file_flag"] == 0]
        r1 = row[row["thin_file_flag"] == 1]
        if r0.empty or r1.empty:
            continue
        flag = "  <- current cutoff" if abs(t - BUSINESS_CUTOFF) < 1e-3 else ""
        print(
            f"  t={t:.4f}  "
            f"thin=0: recall={r0['recall'].iloc[0]:.2%} "
            f"flagged={r0['flagged_rate'].iloc[0]:.2%}   "
            f"thin=1: recall={r1['recall'].iloc[0]:.2%} "
            f"fpr={r1['fpr'].iloc[0]:.2%}{flag}"
        )

    discrimination = within_segment_discrimination(df, proba_col)
    print_check4(discrimination, args.split)
    discrimination_path = os.path.join(
        OUT_DIR, f"within_segment_discrimination_{args.split}.csv"
    )
    discrimination.to_csv(discrimination_path, index=False)

    print("\nSaved:")
    print(f"  {dist_path}")
    print(f"  {sweep_path}")
    print(f"  {discrimination_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()

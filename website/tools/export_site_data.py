"""
Sentinel Credit — Aggregated evidence for the public demo site

Writes website/public/data/cutoff_simulator.json: counts of the champion's
held-out (stratified test) customers per score band x segment x outcome.
Counts only, never customer rows (the repository never publishes data
rows). The browser recomputes flagged share, recall, precision, and each
segment's false-negative and false-positive rates at any cutoff by
summing bands.

Exactness: band edges include the configured cutoff and the Decision 19
comparator thresholds, so metrics at those cutoffs are exact; between
edges they are exact at band boundaries (width BAND_WIDTH).

Self-check before writing: the metrics rebuilt from the counts at the
configured cutoff must equal the saved evaluation evidence
(reports/evaluation/stratified/xgboost/), or the script refuses.

Usage:  python website/tools/export_site_data.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJECT_ROOT, "website", "public", "data", "cutoff_simulator.json")
EVAL_DIR = os.path.join(PROJECT_ROOT, "reports", "evaluation", "stratified", "xgboost")
BAND_WIDTH = 0.0005

with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)
GOV = CONFIG["model_governance"]
CUTOFF = float(CONFIG["model"]["business_cutoff"])


def load():
    pkg = os.path.join(PROJECT_ROOT, GOV["artifacts_dir"])
    preds = pd.read_csv(os.path.join(pkg, "test_predictions_stratified.csv"),
                        usecols=["customer_id", "target_default", "thin_file_flag",
                                 "proba_xgboost"])
    conn = psycopg2.connect(
        host=os.environ.get("SENTINEL_DB_HOST", CONFIG["database"]["host"]),
        port=CONFIG["database"]["port"], dbname=CONFIG["database"]["name"],
        user="postgres", password=os.environ.get("SENTINEL_DB_PASSWORD", ""))
    try:
        vul = pd.read_sql("SELECT customer_id, is_vulnerable FROM feature_store.customer_features",
                          conn)
    finally:
        conn.close()
    frame = preds.merge(vul, on="customer_id", how="left", validate="one_to_one")
    if frame["is_vulnerable"].isna().any():
        raise ValueError("Some held-out customers have no vulnerability label.")
    return frame


def edges_for(scores):
    cand = pd.read_csv(os.path.join(PROJECT_ROOT, "reports", "threshold_selection",
                                    "threshold_candidates.csv"))
    special = [CUTOFF] + cand["threshold"].astype(float).tolist()
    top = float(np.ceil(scores.max() / BAND_WIDTH) * BAND_WIDTH) + BAND_WIDTH
    grid = np.round(np.arange(0.0, top + 1e-12, BAND_WIDTH), 6)
    return np.unique(np.concatenate([grid, special])), cand


def counts(frame, edges, mask):
    """Per band [edges[i], edges[i+1]): number of defaulters and non-defaulters."""
    band = np.searchsorted(edges, frame.loc[mask, "proba_xgboost"].to_numpy(), side="right") - 1
    bad = np.bincount(band[frame.loc[mask, "target_default"].to_numpy() == 1],
                      minlength=len(edges) - 1)
    good = np.bincount(band[frame.loc[mask, "target_default"].to_numpy() == 0],
                       minlength=len(edges) - 1)
    return bad.tolist(), good.tolist()


def metrics_at(group, edges, cutoff):
    """Same arithmetic the website runs: flagged = bands with lower edge >= cutoff."""
    k = int(np.searchsorted(edges, cutoff - 1e-12))
    bad, good = np.array(group["bad"]), np.array(group["good"])
    tp, fp = bad[k:].sum(), good[k:].sum()
    fn, tn = bad[:k].sum(), good[:k].sum()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "fnr": fn / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)}


def main():
    frame = load()
    edges, cand = edges_for(frame["proba_xgboost"])
    groups = {
        "all": np.ones(len(frame), bool),
        "thin_file_flag=0": frame["thin_file_flag"].to_numpy() == 0,
        "thin_file_flag=1": frame["thin_file_flag"].to_numpy() == 1,
        "is_vulnerable=0": frame["is_vulnerable"].to_numpy() == 0,
        "is_vulnerable=1": frame["is_vulnerable"].to_numpy() == 1,
    }
    data = {}
    for name, mask in groups.items():
        bad, good = counts(frame, edges, mask)
        data[name] = {"bad": bad, "good": good}

    # Self-check against the saved evaluation evidence at the configured cutoff.
    with open(os.path.join(EVAL_DIR, "evaluation_metadata.json"), encoding="utf-8") as f:
        ev = json.load(f)["configured_cutoff"]
    got = metrics_at(data["all"], edges, CUTOFF)
    problems = [k for k in ("tp", "fp", "fn", "tn") if got[k] != ev[k]]
    for seg_file, seg in [("fairness_thin_file_flag.csv", "thin_file_flag"),
                          ("fairness_is_vulnerable.csv", "is_vulnerable")]:
        saved = pd.read_csv(os.path.join(EVAL_DIR, seg_file)).set_index("segment")
        for g in (0, 1):
            m = metrics_at(data[f"{seg}={g}"], edges, CUTOFF)
            for rate in ("fnr", "fpr"):
                col = "false_negative_rate" if rate == "fnr" else "false_positive_rate"
                if abs(m[rate] - float(saved.loc[f"{seg}={g}", col])) > 5e-5:
                    problems.append(f"{seg}={g} {rate}")
    if problems:
        sys.exit(f"Counts do not reproduce the saved evaluation at the cutoff: {problems}")
    print(f"Self-check PASS at cutoff {CUTOFF}: confusion matrix {got['tp']}/{got['fp']}/"
          f"{got['fn']}/{got['tn']} matches evaluate_models.py; segment FNR/FPR match.")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "champion held-out stratified test set; counts per score band, no rows",
        "model_version": GOV["model_version"], "feature_set": GOV["feature_set_label"],
        "n_customers": int(len(frame)), "n_defaulters": int(frame["target_default"].sum()),
        "configured_cutoff": CUTOFF,
        "comparators": [{"method": r["method"], "threshold": float(r["threshold"])}
                        for _, r in cand.iterrows()],
        "band_width": BAND_WIDTH,
        "edges": [round(float(e), 8) for e in edges],
        "groups": data,
        "note": "Synthetic data. Flagged = score >= cutoff. Metrics are exact at band edges.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {os.path.relpath(OUT, PROJECT_ROOT)} ({os.path.getsize(OUT) / 1024:.0f} KB, "
          f"{len(edges) - 1} bands, {len(groups)} groups)")


if __name__ == "__main__":
    main()

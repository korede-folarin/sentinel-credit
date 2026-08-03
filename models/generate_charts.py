"""
=============================================================
Sentinel Credit — Report Charts
=============================================================
File:    generate_charts.py
Purpose: Builds the three summary charts from files the
         pipeline actually produces — never from numbers
         copied out of console output. Requires
         eda_baseline.py and evaluate_models.py to have been
         run first (they save the source CSVs this script
         reads).
Author:  Folarin Korede
Version: 1.0.0
=============================================================
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CHARTS_DIR = os.path.join(REPORTS_DIR, "charts")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def chart_eda_summary() -> None:
    """
    Reads reports/eda_target_summary.csv (written by
    eda_baseline.py). Two panels: modelable vs not-modelable,
    and default vs non-default within the modelable population.
    """
    path = os.path.join(REPORTS_DIR, "eda_target_summary.csv")
    if not os.path.exists(path):
        print(f"SKIP eda summary chart — {path} not found. "
              f"Run eda/eda_baseline.py first.")
        return

    row = pd.read_csv(path).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(
        ["Modelable\n(has_outcome=1)", "Not modelable\n(censored)"],
        [row["n_modelable"], row["n_not_modelable"]],
        color=["#4C72B0", "#B0B0B0"],
    )
    axes[0].set_title(f"Population — {int(row['n_total']):,} total customers")
    for i, v in enumerate([row["n_modelable"], row["n_not_modelable"]]):
        axes[0].text(i, v, f"{int(v):,}", ha="center", va="bottom")

    axes[1].bar(
        ["No default", "Default"],
        [row["n_good"], row["n_bad"]],
        color=["#4C72B0", "#C44E52"],
    )
    axes[1].set_title(
        f"Modelable population — {row['overall_bad_rate']:.2%} bad rate"
    )
    for i, v in enumerate([row["n_good"], row["n_bad"]]):
        axes[1].text(i, v, f"{int(v):,}", ha="center", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "eda_target_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def chart_model_comparison(split: str) -> None:
    """
    Reads reports/model_comparison_{split}.csv (written by
    evaluate_models.py). Grouped bar chart across the metrics
    that share a 0-1 scale: Gini, AUC-ROC, KS.
    """
    path = os.path.join(REPORTS_DIR, f"model_comparison_{split}.csv")
    if not os.path.exists(path):
        print(f"SKIP model comparison chart — {path} not found. "
              f"Run models/evaluate_models.py --split {split} first.")
        return

    df = pd.read_csv(path).set_index("model")
    metrics = ["gini", "auc", "ks"]
    labels = ["Gini", "AUC-ROC", "KS"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[metric], width, label=label, color=color)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f"{val:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.40, color="grey", linestyle="--", linewidth=1)
    ax.text(len(df) - 0.5, 0.41, "Gini 0.40 (acceptable)", fontsize=7, color="grey")
    ax.set_title(f"Model comparison — {split} split")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, f"model_comparison_{split}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def chart_lift_by_model(split: str) -> None:
    """
    Reads reports/lift_tables_{split}.csv (written by
    evaluate_models.py). Lift by decile for every model, plus
    a random-baseline reference line at lift=1.0 — flagging
    customers at random gives a lift of exactly 1 by
    definition, so this is the honest zero-skill comparison
    line, not an arbitrary one.
    """
    path = os.path.join(REPORTS_DIR, f"lift_tables_{split}.csv")
    if not os.path.exists(path):
        print(f"SKIP lift chart — {path} not found. "
              f"Run models/evaluate_models.py --split {split} first.")
        return

    df = pd.read_csv(path)
    colors = {"logistic_regression": "#4C72B0", "decision_tree": "#55A868",
              "xgboost": "#C44E52"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].sort_values("decile")
        ax.plot(
            sub["decile"], sub["lift"], marker="o",
            label=model_name.replace("_", " ").title(),
            color=colors.get(model_name, "grey"), linewidth=2,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Random (lift = 1.0)")
    ax.set_xlabel("Risk decile (0 = highest predicted risk)")
    ax.set_ylabel("Lift (vs random selection)")
    ax.set_title(f"Lift by risk decile, all models — {split} split")
    ax.set_xticks(range(10))
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, f"lift_comparison_{split}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["stratified", "temporal"], default="stratified",
        help="Which evaluated split's results to chart (default: stratified)"
    )
    args = parser.parse_args()

    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("=" * 58)
    print("SENTINEL CREDIT — REPORT CHARTS")
    print("=" * 58)

    chart_eda_summary()
    chart_model_comparison(args.split)
    chart_lift_by_model(args.split)


if __name__ == "__main__":
    main()

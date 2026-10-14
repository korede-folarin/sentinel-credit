"""
Sentinel Credit, Model Performance

Displays what models/evaluate_models.py and models/select_business_cutoff.py
already established, for the stratified and the out-of-time (temporal)
split. Nothing is recomputed here: discrimination, calibration, lift,
the operating point at the configured cutoff, and error rates by segment
all come from the saved evaluation evidence.
"""

import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theme  # noqa: E402
from utils import load_config, load_csv, load_json, missing_evidence  # noqa: E402

theme.apply("Model Performance")
cfg = load_config()
theme.header(
    "Model validation",
    "Model Performance",
    "How well the champion separates defaulters from non-defaulters, whether its "
    "probabilities are calibrated, and what the configured cutoff does in practice.",
)

split = st.radio("Evaluation split", ["stratified", "temporal"], horizontal=True,
                 format_func=lambda s: {"stratified": "Stratified hold-out",
                                        "temporal": "Out-of-time (temporal)"}[s])
ev = f"evaluation/{split}/xgboost"
meta = load_json(f"{ev}/evaluation_metadata.json")
comparison = load_csv(f"model_comparison_{split}.csv")

if meta is None or comparison is None:
    missing_evidence(f"models/evaluate_models.py --split {split}")
    st.stop()

theme.chips({"Model": f"xgboost {meta['model_version']}", "Features": meta["feature_set"],
             "Cutoff evaluated": meta["business_cutoff"], "Split": split})

# ── OPERATING POINT ───────────────────────────────────────────
theme.section("At the configured cutoff")
cc, s = meta["configured_cutoff"], meta["summary"]
n = cc["tp"] + cc["fp"] + cc["tn"] + cc["fn"]
c = st.columns(5)
c[0].metric("Gini", f"{s['gini']:.3f}")
c[1].metric("Flagged for review", f"{cc['n_flagged_high_risk'] / n:.1%}",
            help=f"{cc['n_flagged_high_risk']:,} of {n:,} test customers.")
c[2].metric("Recall", f"{cc['recall']:.1%}", help="Share of defaulters flagged.")
c[3].metric("Precision", f"{cc['precision']:.1%}", help="Share of flagged who defaulted.")
c[4].metric("Bad rate if not flagged", f"{cc['bad_rate_among_approved']:.2%}")

# ── MODEL COMPARISON ──────────────────────────────────────────
theme.section("Three-model benchmark · Decision 4")
left, right = st.columns([2, 3], gap="large")
labels = {"xgboost": "XGBoost (champion)", "logistic_regression": "Logistic regression",
          "decision_tree": "Decision tree"}
cmp_ = comparison.assign(Model=comparison["model"].map(labels)).sort_values("gini")
with left:
    fig = go.Figure(go.Bar(
        x=cmp_["gini"], y=cmp_["Model"], orientation="h", text=cmp_["gini"].map("{:.3f}".format),
        textposition="outside", cliponaxis=False, textfont=dict(color=theme.STONE),
        marker_color=[theme.GOLD if m == "xgboost" else theme.STONE for m in cmp_["model"]],
        hovertemplate="%{y}<br>Gini %{x:.4f}<extra></extra>"))
    fig.update_layout(title="Gini by model", xaxis=dict(range=[0, max(cmp_["gini"]) * 1.3]), margin=dict(r=40),
                      xaxis_title="Gini (2·AUC − 1)", yaxis_title=None)
    theme.chart(fig, 260)
with right:
    st.dataframe(
        cmp_.sort_values("gini", ascending=False)[["Model", "gini", "auc", "ks", "pr_auc",
                                                   "brier", "ece"]],
        hide_index=True, use_container_width=True,
        column_config={k: st.column_config.NumberColumn(v, format="%.4f") for k, v in
                       {"gini": "Gini", "auc": "AUC", "ks": "KS", "pr_auc": "PR-AUC",
                        "brier": "Brier", "ece": "ECE"}.items()})
    theme.note("ECE here uses ten fixed bins on 0–1; scores rarely exceed 0.15, so it is "
               "flattering. The decile calibration chart below is the more honest view "
               "(audit finding A8).")

# ── CALIBRATION AND LIFT ──────────────────────────────────────
theme.section("Calibration and ranking")
left, right = st.columns(2, gap="large")
ave = load_csv(f"{ev}/actual_vs_expected.csv")
lift = load_csv(f"{ev}/lift_and_capture.csv")
with left:
    if ave is None:
        missing_evidence(f"models/evaluate_models.py --split {split}")
    else:
        top = float(max(ave["predicted_pd"].max(), ave["actual_default_rate"].max())) * 1.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect calibration",
                                 line=dict(color=theme.GRAPHITE_LINE, dash="dot")))
        fig.add_trace(go.Scatter(
            x=ave["predicted_pd"], y=ave["actual_default_rate"], mode="lines+markers",
            name="Champion, by score decile", line=dict(color=theme.GOLD, width=2),
            customdata=ave[["decile", "n"]],
            hovertemplate="Decile %{customdata[0]}<br>predicted %{x:.2%}<br>actual %{y:.2%}"
                          "<br>%{customdata[1]:,} customers<extra></extra>"))
        fig.update_layout(title="Predicted vs observed default rate",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="Mean predicted PD", yaxis_title="Observed default rate")
        theme.chart(fig, 340)
with right:
    if lift is None:
        missing_evidence(f"models/evaluate_models.py --split {split}")
    else:
        lift = lift.sort_values("decile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lift["pct_population"], y=lift["pct_population"],
                                 mode="lines", name="Random", line=dict(color=theme.GRAPHITE_LINE,
                                                                       dash="dot")))
        fig.add_trace(go.Scatter(
            x=lift["pct_population"], y=lift["capture_rate"], mode="lines+markers",
            name="Champion", line=dict(color=theme.GOLD, width=2), customdata=lift["lift"],
            hovertemplate="Riskiest %{x:.0%} of customers<br>captures %{y:.0%} of defaults"
                          "<br>decile lift %{customdata:.2f}×<extra></extra>"))
        fig.update_layout(title="Cumulative defaults captured", xaxis_tickformat=".0%",
                          yaxis_tickformat=".0%", xaxis_title="Customers reviewed, riskiest first",
                          yaxis_title="Defaults captured")
        theme.chart(fig, 340)

# ── FAIRNESS ──────────────────────────────────────────────────
theme.section("Error rates by segment, at the cutoff")
cols = st.columns(2, gap="large")
for col, (seg, title) in zip(cols, [("thin_file_flag", "Credit-file depth"),
                                    ("is_vulnerable", "Vulnerability (monitoring label)")]):
    table = load_csv(f"{ev}/fairness_{seg}.csv")
    with col:
        st.markdown(f"**{title}**")
        if table is None:
            missing_evidence(f"models/evaluate_models.py --split {split}")
            continue
        names = {"thin_file_flag=0": "Established file", "thin_file_flag=1": "Thin file",
                 "is_vulnerable=0": "Not vulnerable", "is_vulnerable=1": "Vulnerable"}
        show = table.assign(Segment=table["segment"].map(names).fillna(table["segment"]))
        for c_ in ["actual_default_rate", "avg_predicted_pd", "false_negative_rate",
                   "false_positive_rate"]:
            show[c_] = show[c_] * 100
        st.dataframe(
            show[["Segment", "n", "actual_default_rate", "avg_predicted_pd",
                  "false_negative_rate", "false_positive_rate"]],
            hide_index=True, use_container_width=True,
            column_config={
                "n": st.column_config.NumberColumn("Customers", format="%d"),
                "actual_default_rate": st.column_config.NumberColumn("Actual DR", format="%.2f%%"),
                "avg_predicted_pd": st.column_config.NumberColumn("Mean PD", format="%.2f%%"),
                "false_negative_rate": st.column_config.NumberColumn("FNR", format="%.1f%%",
                                                                     help="Defaulters missed."),
                "false_positive_rate": st.column_config.NumberColumn("FPR", format="%.1f%%",
                                                                     help="Non-defaulters flagged."),
            })
thin = load_csv(f"{ev}/fairness_thin_file_flag.csv")
if thin is not None:
    established = thin[thin["segment"] == "thin_file_flag=0"]
    if len(established):
        fnr = float(established.iloc[0]["false_negative_rate"])
        theme.note(
            f"Established-file defaulters: {fnr:.0%} missed at this cutoff, while thin-file "
            "customers carry most false positives. Ranking still works within the "
            "established-file segment; the shared probability scale keeps its scores below "
            "the cutoff. Segment-specific treatment is an open governance decision "
            "(Decisions 12, 19).")

# ── CUTOFF EVIDENCE ───────────────────────────────────────────
theme.section("Cutoff selection evidence · training-only out-of-fold · Decision 19")
cand = load_csv("threshold_selection/threshold_candidates.csv")
if cand is None:
    missing_evidence("models/select_business_cutoff.py")
else:
    labels_m = {"f1_optimal": "F1-optimal (comparator)", "youden_j": "Youden's J (comparator)",
                "review_capacity_constrained": "Review capacity 15% (selected)"}
    view = cand.assign(Method=cand["method"].map(labels_m).fillna(cand["method"]))
    for c_ in ["flagged_rate", "precision", "recall", "approved_bad_rate"]:
        view[c_] = view[c_] * 100
    st.dataframe(
        view[["Method", "threshold", "flagged_rate", "precision", "recall", "approved_bad_rate"]],
        hide_index=True, use_container_width=True,
        column_config={
            "threshold": st.column_config.NumberColumn("PD threshold", format="%.4f"),
            "flagged_rate": st.column_config.NumberColumn("Flagged", format="%.1f%%"),
            "precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
            "recall": st.column_config.NumberColumn("Recall", format="%.1f%%"),
            "approved_bad_rate": st.column_config.NumberColumn("Bad rate not flagged",
                                                               format="%.2f%%"),
        })
    theme.note(f"Configured cutoff: {cfg['model']['business_cutoff']}. The 15% review "
               "capacity is still a placeholder, not a confirmed operational figure.")

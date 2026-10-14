"""
Sentinel Credit, Executive Overview

Entry point. Answers, in order: what this platform is, the current
portfolio and model risk position, what needs attention, and where to
go next. Every figure is read from a live query, config.yaml, or
evidence the pipeline produced; nothing is invented for display. All
data is synthetic (see docs/project_rationale.md).
"""

import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import theme  # noqa: E402
from utils import (  # noqa: E402
    db_available, governance_status, load_config, load_csv, load_json, query_db,
)

theme.apply("Executive Overview")
cfg = load_config()
gov = cfg["model_governance"]

theme.header(
    "Executive overview",
    "Sentinel Credit",
    "BNPL credit-risk intelligence: a governed champion model, live portfolio "
    "data, drift monitoring and an evidence-based approval gate. All customer "
    "data is synthetic.",
)
theme.chips({
    "Champion": f"{gov['champion_model']} {gov['model_version']}",
    "Feature set": gov["feature_set_label"],
    "Decision cutoff": cfg["model"]["business_cutoff"],
    "Package": gov.get("artifacts_dir", "models/artifacts"),
})

ok, reason = db_available()

# ── PORTFOLIO POSITION ────────────────────────────────────────
theme.section("Portfolio position · live from PostgreSQL")
if not ok:
    theme.empty_state("Live portfolio data unavailable.",
                      f"Set SENTINEL_DB_PASSWORD and check PostgreSQL is running. ({reason})")
else:
    apps = query_db(
        "SELECT count(*) AS applications, "
        "sum(application_approved) AS approved, "
        "sum(CASE WHEN application_approved = 1 THEN order_value_gbp END) AS approved_value "
        "FROM raw.transactions"
    ).iloc[0]
    book = query_db(
        "SELECT count(*) AS customers, avg(target_default) AS bad_rate "
        "FROM feature_store.customer_features WHERE has_outcome = 1"
    ).iloc[0]
    approval = float(apps["approved"]) / float(apps["applications"])
    c = st.columns(5)
    c[0].metric("Applications", f"{int(apps['applications']):,}")
    c[1].metric("Approval rate", f"{approval:.1%}")
    c[2].metric("Approved value", f"£{float(apps['approved_value']) / 1e6:,.1f}m")
    c[3].metric("Customers with outcome", f"{int(book['customers']):,}")
    c[4].metric("Customer default rate", f"{float(book['bad_rate']):.2%}",
                help="Share of customers reaching 90 days past due (CRR/EBA definition).")

# ── MODEL RISK POSITION ───────────────────────────────────────
theme.section("Model risk position · held-out test customers")
comparison = load_csv("model_comparison_stratified.csv")
preds_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          gov.get("artifacts_dir", "models/artifacts"),
                          "test_predictions_stratified.csv")
c = st.columns(4)
if comparison is not None and (row := comparison[comparison["model"] == "xgboost"]).shape[0]:
    r = row.iloc[0]
    c[0].metric("Gini", f"{r['gini']:.3f}", help="2·AUC − 1 on the held-out stratified test set.")
    c[1].metric("PR-AUC", f"{r['pr_auc']:.3f}")
else:
    c[0].metric("Gini", "—")
    c[1].metric("PR-AUC", "—")
if os.path.exists(preds_path):
    import pandas as pd
    p = pd.read_csv(preds_path, usecols=["proba_xgboost", "target_default"])
    flagged = p["proba_xgboost"] >= cfg["model"]["business_cutoff"]
    caught = (flagged & (p["target_default"] == 1)).sum() / max(p["target_default"].sum(), 1)
    c[2].metric("Flagged for review", f"{flagged.mean():.1%}",
                help="Share of held-out customers at or above the decision cutoff.")
    c[3].metric("Defaults caught at cutoff", f"{caught:.1%}")
else:
    c[2].metric("Flagged for review", "—")
    c[3].metric("Defaults caught at cutoff", "—")

# ── ATTENTION ─────────────────────────────────────────────────
theme.section("Requires attention")
items = []
status = governance_status() if ok else None
if status is None:
    items.append(("Approval gate status unavailable", "neutral",
                  "The governance view could not be read."))
else:
    s = status["gate_status"]
    level = {"PASS": "positive", "STALE": "warning"}.get(s, "critical")
    detail = (f"Evidence {float(status['evidence_age_hours_now']):.1f}h old "
              f"(limit {int(status['max_monitoring_age_hours'])}h).")
    items.append((f"Approval gate: {s}", level, detail))

monitoring = load_json("monitoring/monitoring_run_metadata.json")
if monitoring:
    n_alerts = monitoring["n_features_with_alerts"]
    items.append((f"Drift alerts: {n_alerts} of {monitoring['n_features_total']}",
                  "positive" if n_alerts == 0 else "critical",
                  f"Last monitoring run {monitoring['generated_at'][:16].replace('T', ' ')}."))

thin = load_csv("evaluation/stratified/xgboost/fairness_thin_file_flag.csv")
if thin is not None:
    t0 = thin[thin["segment"] == "thin_file_flag=0"]
    if len(t0) and float(t0.iloc[0]["false_negative_rate"]) >= 0.999:
        items.append(("Open governance item: established-file defaults missed", "warning",
                      "Customers with a full credit file (thin_file_flag=0) who default "
                      "are all scored below the cutoff (FNR 100%). Ranking within the "
                      "segment survives; the shared probability scale does not reach the "
                      "cutoff. Decision 19."))

for title, level, detail in items:
    a, b = st.columns([1.1, 4])
    a.markdown(theme.pill(level.replace("positive", "ok").replace("critical", "action"),
                          level), unsafe_allow_html=True)
    b.markdown(f"**{title}**  \n<span class='sc-note'>{detail}</span>", unsafe_allow_html=True)

# ── TRENDS ────────────────────────────────────────────────────
theme.section("Portfolio trends · by origination month")
if ok:
    orig = query_db("SELECT month, n_originations, total_value_gbp FROM reporting.originations_monthly")
    dr = query_db("SELECT month, n_resolved, default_rate FROM reporting.default_rate_monthly")
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=orig["month"], y=orig["n_originations"],
                               marker_color=theme.STONE,
                               hovertemplate="%{x|%b %Y}<br>%{y:,} approved orders<extra></extra>"))
        fig.update_layout(title="Approved originations per month",
                          yaxis_title="Orders", xaxis_title=None)
        theme.chart(fig, 300)
    with right:
        fig = go.Figure(go.Scatter(
            x=dr["month"], y=dr["default_rate"], mode="lines+markers",
            line=dict(color=theme.GOLD, width=2), marker=dict(size=4),
            customdata=dr["n_resolved"],
            hovertemplate="%{x|%b %Y} cohort<br>%{y:.2%} defaulted<br>"
                          "%{customdata:,} resolved orders<extra></extra>"))
        fig.update_layout(title="Default rate by origination cohort",
                          yaxis_tickformat=".1%", yaxis_title="Orders defaulting", xaxis_title=None)
        theme.chart(fig, 300)
    theme.note("Origination growth reflects the synthetic population's onboarding schedule, "
               "not market demand (Decision 22). Default rates are per approved order, "
               "attributed to the month it originated.")
else:
    theme.empty_state("Trends unavailable.", "Requires the live database.")

# ── NAVIGATION ────────────────────────────────────────────────
theme.section("Workspaces")
nav = [
    ("pages/1_Credit_Risk_Assessment.py", "Credit Risk Assessment",
     "Score an applicant with the champion model."),
    ("pages/2_Portfolio_Intelligence.py", "Portfolio Intelligence",
     "Originations, cohort default rates, segments and forecasts."),
    ("pages/3_Model_Performance.py", "Model Performance",
     "Discrimination, calibration, cutoff evidence and segment error rates."),
    ("pages/4_Explainability.py", "Explainability", "Global and case-level SHAP drivers."),
    ("pages/5_Model_Monitoring.py", "Model Monitoring", "Feature drift and cohort stability."),
    ("pages/6_Governance_and_Audit.py", "Governance & Audit",
     "Approval gate history, promotion record and model identity."),
]
cols = st.columns(3)
for i, (path, label, desc) in enumerate(nav):
    with cols[i % 3]:
        st.page_link(path, label=label)
        theme.note(desc)

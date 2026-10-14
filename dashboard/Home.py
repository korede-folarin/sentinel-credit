"""
Sentinel Credit, Dashboard Home

Portfolio overview and entry point for the risk officer facing
Streamlit app. Every metric shown here is read from evidence
already produced by the pipeline, config.yaml, or a live query,
nothing on this page is invented for display purposes.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    governance_header,
    load_json,
    load_csv,
    missing_evidence,
    query_db,
)

st.set_page_config(page_title="Sentinel Credit, Overview", layout="wide")

st.title("Sentinel Credit")
st.caption(
    "Closed loop ML platform for BNPL credit risk. Synthetic data, "
    "PD only scope. See docs/engineering_decision_log.md for the full "
    "governance record behind every number on this page."
)

st.divider()
st.subheader("Governance identity")
governance_header()

st.divider()
st.subheader("Champion performance")

comparison = load_csv("model_comparison_stratified.csv")
if comparison is None:
    missing_evidence("models/evaluate_models.py --split stratified")
else:
    champion_row = comparison[comparison["model"] == "xgboost"]
    if not champion_row.empty:
        row = champion_row.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Gini", f"{row['gini']:.4f}")
        cols[1].metric("AUC", f"{row['auc']:.4f}")
        cols[2].metric("KS", f"{row['ks']:.4f}")
        cols[3].metric("PR AUC", f"{row['pr_auc']:.4f}")

st.divider()
st.subheader("Portfolio, live from the database")

try:
    live = query_db(
        "SELECT count(*) AS n_customers, "
        "round(avg(target_default)::numeric, 4) AS bad_rate "
        "FROM feature_store.customer_features WHERE has_outcome = 1"
    )
    cols = st.columns(2)
    cols[0].metric("Modelable customers", f"{int(live.iloc[0]['n_customers']):,}")
    cols[1].metric("Bad rate", f"{float(live.iloc[0]['bad_rate']):.2%}")
except Exception as error:
    st.info(
        "Live database query unavailable in this session. Set "
        "SENTINEL_DB_PASSWORD and confirm the database is reachable. "
        f"Detail: {error}"
    )

st.divider()
st.subheader("Governance status")

monitoring = load_json("monitoring/monitoring_run_metadata.json")
gate = load_json("governance/approval_gate_result.json")

cols = st.columns(2)
with cols[0]:
    st.markdown("**Drift monitoring**")
    if monitoring is None:
        missing_evidence("pipeline/monitor_drift.py")
    else:
        n_alerts = monitoring["n_features_with_alerts"]
        n_total = monitoring["n_features_total"]
        if n_alerts == 0:
            st.success(f"No alerts, {n_total} of {n_total} features stable.")
        else:
            st.error(f"{n_alerts} of {n_total} features alerting.")
        st.caption(f"Generated: {monitoring['generated_at']}")

with cols[1]:
    st.markdown("**Approval gate**")
    if gate is None:
        missing_evidence("pipeline/approval_gate.py")
    else:
        status = gate["gate_status"]
        if status == "PASS":
            st.success("PASS")
        else:
            st.error("BLOCK")
        st.caption(f"Evaluated: {gate.get('evaluated_at', 'unknown')}")

st.divider()
st.caption(
    "Use the pages in the left sidebar for model performance, "
    "explainability, fairness, drift and forecasting detail."
)

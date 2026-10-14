"""
Sentinel Credit, Model Performance

Reads model_comparison and lift_tables CSVs already produced by
models/evaluate_models.py, for both the stratified and temporal
splits. Nothing here recomputes a metric, it displays what the
governed evaluation already established.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_csv, missing_evidence

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance")

split = st.radio("Split", ["stratified", "temporal"], horizontal=True)

comparison = load_csv(f"model_comparison_{split}.csv")
lift = load_csv(f"lift_tables_{split}.csv")

if comparison is None:
    missing_evidence(f"models/evaluate_models.py --split {split}")
else:
    st.subheader("Model comparison")
    st.dataframe(
        comparison.sort_values("gini", ascending=False),
        width='stretch',
    )

    fig = px.bar(
        comparison.sort_values("gini", ascending=False),
        x="model", y="gini", title="Gini by model",
        color="model",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

if lift is None:
    missing_evidence(f"models/evaluate_models.py --split {split}")
else:
    st.subheader("Lift and capture rate, champion model")
    champion_lift = lift[lift["model"] == "xgboost"]
    if not champion_lift.empty:
        fig = px.line(
            champion_lift, x="decile", y="capture_rate",
            title="Cumulative default capture by risk decile",
            markers=True,
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(champion_lift, width='stretch')

st.divider()
st.subheader("Business cutoff selection evidence")
candidates = load_csv("threshold_selection/threshold_candidates.csv")
if candidates is None:
    missing_evidence("models/select_business_cutoff.py")
else:
    st.dataframe(candidates, width='stretch')
    st.caption(
        "See docs/business_cutoff_selection_report.md for the full "
        "reasoning behind the selected threshold."
    )

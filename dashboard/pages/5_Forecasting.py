"""
Sentinel Credit, Forecasting

Reads forecasting/forecast_portfolio.py's output. Shows both
forecast series with the same caution the underlying report
carries, especially for origination volume, where growth reflects
a fixed synthetic onboarding schedule rather than market demand.
That caveat is shown here directly, not only in the markdown
report.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_json, image_path, missing_evidence

st.set_page_config(page_title="Forecasting", layout="wide")
st.title("Forecasting")

metadata = load_json("forecasting/forecast_metadata.json")
if metadata is None:
    missing_evidence("forecasting/forecast_portfolio.py")
    st.stop()

st.caption(
    f"LSTM status: {metadata['lstm_status']}"
)

for result in metadata["results"]:
    st.divider()
    st.subheader(result["series_name"])
    st.info(result["caveat"])

    metrics = result["metrics"]
    cols = st.columns(3)
    cols[0].metric("Backtest months held out", metrics["n_holdout_months"])
    cols[1].metric("MAE", metrics["mae"])
    cols[2].metric(
        "MAPE",
        f"{metrics['mape_pct']}%" if metrics["mape_pct"] is not None else "n/a",
    )

    slug = result["series_name"].lower().replace(" ", "_")
    plot = image_path(f"forecasting/forecast_{slug}.png")
    if plot:
        st.image(
            plot,
            caption=(
                "Solid line is fitted and backtested history. "
                "Dashed line beyond the vertical marker is "
                "unvalidated forecast."
            ),
        )

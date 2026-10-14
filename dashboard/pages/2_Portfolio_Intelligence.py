"""
Sentinel Credit, Portfolio Intelligence

Portfolio composition and trend from the live reporting views
(sql/reporting/create_reporting_views.sql), plus the Prophet forecasts
produced by forecasting/forecast_portfolio.py (Decision 22), redrawn
from their CSV output. Forecast caveats are shown with the charts, not
only in the report: origination growth reflects the synthetic
onboarding schedule, and anything beyond the data's end is unvalidated.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theme  # noqa: E402
from utils import REPORTS_DIR, db_available, load_csv, load_json, query_db  # noqa: E402

theme.apply("Portfolio Intelligence")
theme.header(
    "Portfolio intelligence",
    "Portfolio Intelligence",
    "Origination volume, cohort default rates, segment composition and backtested "
    "forecasts. Synthetic portfolio, January 2022 to October 2024 originations.",
)

ok, reason = db_available()

# ── SEGMENTS ──────────────────────────────────────────────────
theme.section("Segment composition · customers with an observed outcome")
if not ok:
    theme.empty_state("Segment data unavailable.", f"PostgreSQL is not reachable: {reason}")
else:
    seg = query_db("SELECT thin_file_flag, is_vulnerable, n_customers, actual_default_rate, "
                   "avg_credit_score, avg_savings_balance_gbp FROM reporting.segment_summary")
    seg["Segment"] = (seg["thin_file_flag"].map({0: "Established file", 1: "Thin file"}) + " · "
                      + seg["is_vulnerable"].map({0: "not vulnerable", 1: "vulnerable"}))
    total = seg["n_customers"].sum()
    seg["Share"] = seg["n_customers"] / total
    s = seg.sort_values("actual_default_rate")
    fig = go.Figure(go.Bar(
        x=s["actual_default_rate"], y=s["Segment"], orientation="h",
        marker_color=[theme.GOLD if "Thin" in x else theme.STONE for x in s["Segment"]],
        customdata=s["n_customers"],
        hovertemplate="%{y}<br>%{x:.2%} default rate<br>%{customdata:,} customers<extra></extra>"))
    fig.update_layout(title="Default rate by segment", xaxis_tickformat=".1%",
                      xaxis_title="Customers defaulting", yaxis_title=None)
    theme.chart(fig, 280)
    # NumberColumn's "%" format does not scale, so convert to percent first.
    table = seg.assign(share_pct=seg["Share"] * 100,
                       rate_pct=seg["actual_default_rate"].astype(float) * 100)
    st.dataframe(
        table[["Segment", "n_customers", "share_pct", "rate_pct", "avg_credit_score",
               "avg_savings_balance_gbp"]],
        hide_index=True, use_container_width=True,
        column_config={
            "n_customers": st.column_config.NumberColumn("Customers", format="%d"),
            "share_pct": st.column_config.NumberColumn("Share", format="%.1f%%"),
            "rate_pct": st.column_config.NumberColumn("Default rate", format="%.2f%%"),
            "avg_credit_score": st.column_config.NumberColumn("Avg credit score", format="%.0f"),
            "avg_savings_balance_gbp": st.column_config.NumberColumn("Avg savings (£)",
                                                                     format="£%.0f"),
        })
    theme.note("Vulnerability is a monitoring label, never a model input (Decision 12). "
               "Segments are shown to check outcomes are equitable, not to score.")

# ── VOLUME AND VALUE ──────────────────────────────────────────
theme.section("Approved volume and value · by origination month")
if ok:
    orig = query_db("SELECT month, n_originations, total_value_gbp, avg_order_value_gbp "
                    "FROM reporting.originations_monthly")
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=orig["month"], y=orig["total_value_gbp"] / 1e6,
                               marker_color=theme.STONE,
                               hovertemplate="%{x|%b %Y}<br>£%{y:.2f}m approved<extra></extra>"))
        fig.update_layout(title="Approved order value", yaxis_title="£ million", xaxis_title=None)
        theme.chart(fig, 300)
    with right:
        fig = go.Figure(go.Scatter(x=orig["month"], y=orig["avg_order_value_gbp"],
                                   mode="lines", line=dict(color=theme.GOLD, width=2),
                                   hovertemplate="%{x|%b %Y}<br>£%{y:,.0f} average<extra></extra>"))
        fig.update_layout(title="Average approved order value", yaxis_title="£",
                          xaxis_title=None)
        theme.chart(fig, 300)

# ── FORECASTS ─────────────────────────────────────────────────
theme.section("Forecasts · Prophet, backtested on the last 6 months")
meta = load_json("forecasting/forecast_metadata.json")
if meta is None:
    theme.empty_state("No forecast evidence yet.", "Run forecasting/forecast_portfolio.py.")
else:
    files = {"Default Rate": "forecast_default_rate.csv",
             "Origination Volume": "forecast_origination_volume.csv"}
    cols = st.columns(2, gap="large")
    for col, result in zip(cols, meta["results"]):
        name = result["series_name"]
        frame = load_csv(f"forecasting/{files.get(name, '')}")
        with col:
            if frame is None:
                theme.empty_state(f"{name}: forecast CSV missing.",
                                  "Run forecasting/forecast_portfolio.py.")
                continue
            frame["ds"] = pd.to_datetime(frame["ds"])
            is_rate = "Rate" in name
            fmt = ".2%" if is_rate else ",.0f"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([frame["ds"], frame["ds"][::-1]]),
                y=pd.concat([frame["yhat_upper"], frame["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(184,154,97,0.12)", line=dict(width=0),
                hoverinfo="skip", name="80% interval"))
            fit = frame[~frame["is_beyond_data"]]
            beyond = frame[frame["is_beyond_data"]]
            fig.add_trace(go.Scatter(x=fit["ds"], y=fit["yhat"], mode="lines",
                                     line=dict(color=theme.GOLD, width=2), name="Model fit",
                                     hovertemplate=f"%{{x|%b %Y}}<br>%{{y:{fmt}}}<extra>fit</extra>"))
            fig.add_trace(go.Scatter(x=beyond["ds"], y=beyond["yhat"], mode="lines",
                                     line=dict(color=theme.GOLD, width=2, dash="dash"),
                                     name="Forecast beyond data (unvalidated)",
                                     hovertemplate=f"%{{x|%b %Y}}<br>%{{y:{fmt}}}<extra>forecast</extra>"))
            actual = frame.dropna(subset=["y"])
            fig.add_trace(go.Scatter(
                x=actual["ds"], y=actual["y"], mode="markers", name="Actual",
                marker=dict(size=6, color=[theme.CRIMSON if h else theme.IVORY_TEXT
                                           for h in actual["is_holdout"]]),
                hovertemplate=f"%{{x|%b %Y}}<br>%{{y:{fmt}}}<extra>actual</extra>"))
            if len(beyond):
                fig.add_vline(x=beyond["ds"].min(), line=dict(color=theme.STONE, dash="dot",
                                                              width=1))
            fig.update_layout(title=name, yaxis_tickformat=".1%" if is_rate else ",",
                              xaxis_title=None, yaxis_title="Default rate" if is_rate else "Orders")
            theme.chart(fig, 340)
            m = result["metrics"]
            a, b, c = st.columns(3)
            a.metric("Held-out months", m["n_holdout_months"])
            b.metric("MAE", f"{m['mae']:.4f}" if is_rate else f"{m['mae']:,.0f}")
            c.metric("MAPE", f"{m['mape_pct']:.1f}%" if m["mape_pct"] is not None else "n/a")
            theme.note(result["caveat"])
    theme.note("Red markers are the held-out backtest months. The dashed line after the dotted "
               "marker extends beyond the data and is not validated against anything. "
               f"LSTM: {meta['lstm_status']}")

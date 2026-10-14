"""
Sentinel Credit, Model Monitoring

Drift and data-quality evidence from pipeline/monitor_drift.py, read
live from reporting.feature_drift_log and reporting.feature_drift_by_cohort
(Decisions 20, 24, 25).

Two honest limits are stated on the page: the comparison population is a
proxy (later-originating customers not in the training reference; there
is no live scoring stream), and repeated runs on the unchanged snapshot
reproduce the same values, so the run history is not a trend. The
origination-cohort view is the time axis that carries real variation.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theme  # noqa: E402
from utils import db_available, load_config, load_json, query_db  # noqa: E402

theme.apply("Model Monitoring")
cfg = load_config()
th = cfg["feature_governance"]["thresholds"]
MODERATE, HIGH = th["psi_moderate"], th["psi_high"]

theme.header(
    "Model monitoring",
    "Population Stability",
    "Has the population the model scores moved away from the population it was trained "
    "on? Population Stability Index (PSI) per feature, plus data-quality checks.",
)
ok, reason = db_available()
if not ok:
    theme.empty_state("Monitoring tables unavailable.", f"PostgreSQL is not reachable: {reason}")
    st.stop()

meta = load_json("monitoring/monitoring_run_metadata.json")
latest = query_db(
    "SELECT * FROM reporting.feature_drift_log "
    "WHERE logged_at = (SELECT max(logged_at) FROM reporting.feature_drift_log)")
if latest.empty:
    theme.empty_state("No drift runs logged yet.", "Run pipeline/monitor_drift.py.")
    st.stop()

latest["logged_at"] = pd.to_datetime(latest["logged_at"], utc=True)

# ── LATEST RUN ────────────────────────────────────────────────
theme.section("Latest run")
n_alerts = int(latest["any_alert"].sum())
c = st.columns(4)
c[0].metric("Features alerting", f"{n_alerts} of {len(latest)}")
c[1].metric("Highest PSI", f"{latest['psi'].astype(float).max():.3f}")
c[2].metric("Run logged", latest["logged_at"].max().strftime("%d %b %Y %H:%M UTC"))
if meta and meta.get("target_rate_shift", {}).get("absolute_delta") is not None:
    shift = meta["target_rate_shift"]
    c[3].metric("Bad-rate shift", f"{shift['absolute_delta']:+.2%}",
                help=f"Reference {shift['reference_bad_rate']:.2%}, "
                     f"comparison {shift['comparison_bad_rate']:.2%}. Reported, not gated.")
if meta and "n_comparison_overlap_excluded" in meta:
    theme.note(f"Reference: {meta['n_reference']:,} training customers. Comparison: "
               f"{meta['n_comparison']:,} later-originating customers the model never saw "
               f"({meta['n_comparison_overlap_excluded']:,} also in training were excluded).")

data = latest.assign(psi=latest["psi"].astype(float)).sort_values("psi")
band_colour = {"stable": theme.STONE, "moderate_shift": theme.AMBER, "material_shift": theme.CRIMSON}
fig = go.Figure(go.Bar(
    x=data["psi"], y=data["feature"], orientation="h",
    marker_color=[band_colour.get(b, theme.STONE) for b in data["psi_band"]],
    customdata=data[["psi_band", "alerts"]],
    hovertemplate="%{y}<br>PSI %{x:.4f} (%{customdata[0]})<br>alerts: %{customdata[1]}"
                  "<extra></extra>"))
for x, label in [(MODERATE, "monitor"), (HIGH, "action")]:
    fig.add_vline(x=x, line=dict(color=theme.GRAPHITE_LINE, dash="dot"),
                  annotation_text=f"{x} {label}", annotation_font_color=theme.STONE,
                  annotation_position="top")
fig.update_layout(title="PSI by feature",
                  xaxis_title="PSI", yaxis_title=None,
                  xaxis=dict(range=[0, max(HIGH * 1.1, data["psi"].max() * 1.15)]))
theme.chart(fig, 28 * len(data) + 120)

with st.expander("Data-quality checks for this run"):
    dq = latest[["feature", "psi", "null_rate_delta", "mean_shift_sigma", "out_of_range_rate",
                 "alerts"]].copy()
    st.dataframe(dq, hide_index=True, use_container_width=True, column_config={
        "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
        "null_rate_delta": st.column_config.NumberColumn("Null-rate change", format="%.4f"),
        "mean_shift_sigma": st.column_config.NumberColumn("Mean shift (σ)", format="%.3f"),
        "out_of_range_rate": st.column_config.NumberColumn("Outside training range",
                                                           format="%.4f")})

# ── COHORTS ───────────────────────────────────────────────────
theme.section("By origination quarter · held-out customers")
cohorts = query_db("SELECT * FROM reporting.feature_drift_by_cohort")
if cohorts.empty:
    theme.empty_state("No cohort view yet.", "Run pipeline/monitor_drift.py.")
else:
    scored = cohorts[cohorts["scored"]]
    pivot = scored.pivot(index="feature", columns="cohort_period", values="psi").astype(float)
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    floors = scored.groupby("cohort_period")["psi_noise_floor"].first().astype(float)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        zmin=0, zmax=HIGH,
        colorscale=[[0, theme.GRAPHITE], [MODERATE / HIGH, theme.GOLD], [1, theme.CRIMSON]],
        colorbar=dict(title=dict(text="PSI", font=dict(color=theme.STONE)),
                      tickfont=dict(color=theme.STONE)),
        customdata=[[floors[c_] for c_ in pivot.columns]] * len(pivot),
        hovertemplate="%{y}<br>%{x}<br>PSI %{z:.4f}<br>noise floor %{customdata:.4f}"
                      "<extra></extra>"))
    fig.update_layout(title="PSI by origination quarter",
                      xaxis_title="Origination quarter", yaxis_title=None)
    theme.chart(fig, 28 * len(pivot) + 140)
    skipped = sorted(cohorts.loc[~cohorts["scored"], "cohort_period"].unique())
    theme.note("Each quarter is compared with the training reference. A PSI near that "
               "quarter's noise floor (shown on hover) is indistinguishable from sampling "
               "noise. The final quarter covers October 2024 only, so its customers have the "
               "shortest histories."
               + (f" Too few customers to score: {', '.join(skipped)}." if skipped else ""))

# ── HISTORY ───────────────────────────────────────────────────
theme.section("Run history")
history = query_db(
    "SELECT coalesce(run_id, '(before run ids)') AS run_id, min(logged_at) AS logged_at, "
    "max(psi) AS max_psi, sum(any_alert::int) AS alerts "
    "FROM reporting.feature_drift_log GROUP BY run_id ORDER BY 2 DESC")
history["logged_at"] = pd.to_datetime(history["logged_at"], utc=True).dt.tz_localize(None)
st.dataframe(history, hide_index=True, use_container_width=True, column_config={
    "run_id": "Run", "logged_at": st.column_config.DatetimeColumn("Logged (UTC)"),
    "max_psi": st.column_config.NumberColumn("Max PSI", format="%.4f"),
    "alerts": st.column_config.NumberColumn("Alerts", format="%d")})
theme.note("Runs on the unchanged data snapshot reproduce the same values, so this history "
           "is a record of runs, not a drift trend (Decision 25). The scheduler only runs while "
           "the local Airflow stack is up (Decision 24).")

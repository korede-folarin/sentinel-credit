"""
Sentinel Credit, Explainability

SHAP evidence produced by models/explain_models.py for the champion
(Decision 14): global importance and the governed local case studies,
redrawn from the saved CSV/JSON output.

SHAP explains the champion's uncalibrated explanation model, fitted on
the same data with the same settings. Global rankings carry over to the
calibrated model because calibration is monotonic; local values are
contributions to the raw risk margin, not to the calibrated probability.
The page states this wherever values are shown.
"""

import glob
import json
import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theme  # noqa: E402
from utils import REPORTS_DIR, load_csv, load_json, missing_evidence  # noqa: E402

theme.apply("Explainability")
theme.header(
    "Explainability",
    "What Drives the Model",
    "Which inputs the champion relies on overall, and how they combine for "
    "individual customers. SHAP values on the champion's explanation model.",
)

meta = load_json("shap/shap_metadata.json")
importance = load_csv("shap/global_feature_importance.csv")
if meta is None or importance is None:
    missing_evidence("models/explain_models.py")
    st.stop()

theme.chips({"Model": f"{meta.get('model', 'xgboost')} {meta.get('model_version', '')}",
             "Generated": str(meta.get("generated_at", ""))[:16].replace("T", " "),
             "Explains": "uncalibrated explanation model (raw margin)"})

# ── GLOBAL ────────────────────────────────────────────────────
theme.section("Global importance · mean absolute SHAP")
imp = importance.sort_values("mean_abs_shap")
fig = go.Figure(go.Bar(
    x=imp["pct_of_total"], y=imp["feature"], orientation="h",
    text=imp["pct_of_total"].map("{:.1f}%".format), textposition="outside", cliponaxis=False,
    textfont=dict(color=theme.STONE),
    marker_color=[theme.GOLD if r <= 3 else theme.STONE for r in imp["rank"]],
    customdata=imp["mean_abs_shap"],
    hovertemplate="%{y}<br>%{x:.1f}% of total importance<br>mean |SHAP| %{customdata:.4f}"
                  "<extra></extra>"))
fig.update_layout(title="Share of total importance", xaxis_title="% of total mean |SHAP|",
                  yaxis_title=None, xaxis=dict(range=[0, imp["pct_of_total"].max() * 1.32]), margin=dict(r=40))
theme.chart(fig, 30 * len(imp) + 110)
theme.note("Day-zero customer attributes lead; no exposure-time feature remains in the model "
           "(Decisions 14, 18). pct_late_payments was removed in v1.1 for leaking outcome "
           "information (Decision 27).")

# ── WHEN THE MODEL ASKS FOR A PERSON (Decision 31) ────────────
theme.section("When the model asks for a person · held-out customers")
_ev_path = os.path.join(os.path.dirname(REPORTS_DIR), "website", "public", "data",
                        "review_evidence.json")
if not os.path.exists(_ev_path):
    missing_evidence("models/build_explanation_rules.py")
else:
    with open(_ev_path, encoding="utf-8") as f:
        ev = json.load(f)
    a, b, c = st.columns(3)
    a.metric("Held-out customers", f"{ev['n_customers']:,}")
    b.metric("Risk officer review required", f"{ev['review_required']:,}")
    c.metric("Review rate", f"{ev['review_share']:.1%}")

    def _pct(x):
        return "<0.1%" if 0 < x < 0.001 else f"{x:.1%}"

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**How often each check fires**")
        st.dataframe(
            {"Outcome": ["No issue: no review" if r["error_type"] == "none" else r["label"]
                         for r in ev["by_error_type"]],
             "Customers": [f"{r['n']:,}" for r in ev["by_error_type"]],
             "Share": [_pct(r["share"]) for r in ev["by_error_type"]],
             "Later defaulted": [r["defaults"] for r in ev["by_error_type"]]},
            use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Which detail the decision rests on**")
        rows = ev["single_detail_by_detail"]
        st.dataframe(
            {"Detail": [r["name"] for r in rows],
             "Recommendation": ["Refer" if r["recommendation"] == "refer" else "Pass" for r in rows],
             "Customers": [r["n"] for r in rows],
             "Median PD": [_pct(r["median_probability"]) if "median_probability" in r else "—"
                           for r in rows],
             "If detail typical": [_pct(r["median_probability_if_typical"])
                                   if "median_probability_if_typical" in r else "—" for r in rows],
             "Share of effect": [f"{r['median_share_of_effect']:.0%}"
                                 if "median_share_of_effect" in r else "—" for r in rows]},
            use_container_width=True, hide_index=True)
    theme.note(
        "The three Decision 31 checks (likely data-input error, the known blind spot for full "
        "credit files, a recommendation resting on one detail) run on every held-out customer. "
        "'Rests on one detail' means setting that detail alone to typical would change the "
        "recommendation and it carries at least 60% of the total effect: only average order value "
        "(customers who pass only because their orders are small) and a thin credit file "
        "(customers referred only because of it) ever do. PD columns are group medians. "
        f"Model {ev['model_version']}, rules {ev['rules_version']}; same file as the public site.")

# ── LOCAL CASES ───────────────────────────────────────────────
theme.section("Case studies · individual customers")
cases = []
for path in sorted(glob.glob(os.path.join(REPORTS_DIR, "shap", "local", "*_values.json"))):
    with open(path, encoding="utf-8") as f:
        cases.append(json.load(f))
if not cases:
    missing_evidence("models/explain_models.py")
    st.stop()

titles = {"high_risk": "Highest-risk customer", "low_risk": "Lowest-risk customer",
          "near_threshold_provisional": "Near the cutoff",
          "false_negative_provisional": "Missed defaulter (false negative)",
          "false_positive_provisional": "Flagged non-defaulter (false positive)",
          "single_feature_dominant": "One feature dominates"}
choice = st.selectbox("Case", range(len(cases)),
                      format_func=lambda i: titles.get(cases[i]["case_type"],
                                                       cases[i]["case_type"]))
case = cases[choice]
refer = case["decision"] == "flag_high_risk"
a, b, c, d = st.columns(4)
a.metric("Calibrated PD", f"{case['calibrated_probability']:.2%}")
b.markdown("**Recommendation**<br>" + theme.pill(
    "Refer for review" if refer else "Below threshold", "critical" if refer else "positive"),
    unsafe_allow_html=True)
c.metric("Actual outcome", "Defaulted" if case["actual_outcome"] == 1 else "Did not default")
d.metric("Customer", case["customer_id"])

contrib = sorted(case["contributions"], key=lambda r: abs(r["shap_value"]), reverse=True)[:10]
contrib = contrib[::-1]
fig = go.Figure(go.Bar(
    x=[r["shap_value"] for r in contrib], y=[r["feature"] for r in contrib], orientation="h",
    marker_color=[theme.CRIMSON if r["shap_value"] > 0 else theme.EMERALD for r in contrib],
    customdata=[r["value"] for r in contrib],
    hovertemplate="%{y} = %{customdata}<br>contribution %{x:+.3f}<extra></extra>"))
fig.update_layout(title="Contributions for this customer",
                  xaxis_title="← lowers risk · raises risk →  (SHAP, raw margin)",
                  yaxis_title=None)
theme.chart(fig, 30 * len(contrib) + 110)
theme.note(case.get("note", ""))
if "provisional" in case.get("case_type", ""):
    theme.note("'Provisional' cases depend on the cutoff: they are classified at the current "
               f"cutoff ({case['business_cutoff']}) and change if the cutoff changes.")

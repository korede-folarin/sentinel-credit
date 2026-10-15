"""
Sentinel Credit, Credit Risk Assessment

Scores one application with the governed champion: a new applicant
(profile + this order) or an existing customer from the feature store.
Uses dashboard/scoring.py, which shares feature construction with the
FastAPI service, so both score an applicant identically.

What the result is, and is not:
- a calibrated probability of default and a review decision at the
  configured cutoff — real model outputs;
- no credit score, credit limit or approval is invented here;
- the champion is a customer-level portfolio-risk model (Decision 18),
  so a new-applicant score illustrates the mechanism rather than a
  validated application-time underwriting decision. Shown on the page.

Nothing typed into this page is stored or logged.
"""

import html
import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theme  # noqa: E402
import scoring  # noqa: E402
from utils import db_available, load_config  # noqa: E402

theme.apply("Credit Risk Assessment")
cfg = load_config()
gov = cfg["model_governance"]

theme.header(
    "Decision support",
    "Credit Risk Assessment",
    "Score a BNPL application with the governed champion model. The result is a "
    "probability of default and a review recommendation at the configured cutoff.",
)
theme.chips({"Model": f"{gov['champion_model']} {gov['model_version']}",
             "Features": gov["feature_set_label"],
             "Review cutoff": f"PD ≥ {cfg['model']['business_cutoff']:.2%}"})
st.markdown(
    "<div class='sc-note' style='margin:0.2rem 0 0.6rem'>Demonstration on synthetic data. "
    "Do not enter real personal or financial details. Nothing entered here is stored "
    "or logged.</div>", unsafe_allow_html=True)

ok, reason = db_available()
ranges = scoring.training_ranges() if ok else None

LABELS = {
    "monthly_income_gbp": "Monthly income (£)",
    "existing_monthly_obligations_gbp": "Existing monthly obligations (£)",
    "savings_balance_gbp": "Savings balance (£)",
    "credit_score": "Credit score",
    "months_credit_history": "Months of credit history",
    "num_credit_accounts": "Credit accounts",
    "num_defaults_24m": "Defaults in last 24 months",
    "thin_file_flag": "Thin credit file",
    "max_affordability_ratio": "Affordability ratio",
    "avg_order_value_gbp": "Order value (£)",
    "avg_n_instalments": "Instalments",
    "pct_high_risk_category": "High-risk merchant category",
    "pct_medium_risk_category": "Medium-risk merchant category",
}
# The form starts at the typical applicant (training medians), the same
# reference the plain-English reasons compare against (Decision 30).
TYPICAL = scoring.typical_applicant()


def start(feature: str):
    return TYPICAL[feature]


mode = st.radio("Applicant", ["New applicant", "Existing customer"], horizontal=True,
                label_visibility="collapsed")

features, context = None, {}
if mode == "New applicant":
    with st.form("assessment"):
        left, right = st.columns([3, 2], gap="large")
        with left:
            theme.section("Income and commitments")
            a, b, c = st.columns(3)
            income = a.number_input(LABELS["monthly_income_gbp"], 1.0, 50000.0,
                                    float(start("monthly_income_gbp")), step=50.0)
            obligations = b.number_input(LABELS["existing_monthly_obligations_gbp"], 0.0,
                                         50000.0, float(start("existing_monthly_obligations_gbp")),
                                         step=25.0)
            savings = c.number_input(LABELS["savings_balance_gbp"], 0.0, 1_000_000.0,
                                     float(start("savings_balance_gbp")), step=100.0)
            theme.section("Credit file")
            a, b = st.columns(2)
            score_in = a.number_input(LABELS["credit_score"], 0, 1000, int(start("credit_score")))
            history = b.number_input(LABELS["months_credit_history"], 0, 720,
                                     int(start("months_credit_history")))
            a, b = st.columns(2)
            accounts = a.number_input(LABELS["num_credit_accounts"], 0, 50,
                                      int(start("num_credit_accounts")))
            defaults = b.number_input(LABELS["num_defaults_24m"], 0, 20,
                                      int(start("num_defaults_24m")))
            thin = st.checkbox("Thin credit file (limited bureau history)",
                               value=bool(TYPICAL["thin_file_flag"]))
        with right:
            theme.section("This order")
            order_value = st.number_input(LABELS["avg_order_value_gbp"], 1.0, 20000.0,
                                          float(start("order_value_gbp")), step=10.0)
            plans = [3, 6, 12]
            instalments = st.selectbox(LABELS["avg_n_instalments"], plans,
                                       index=plans.index(int(start("n_instalments"))),
                                       help="BNPL plans offered: Pay in 3, 6 or 12.")
            category = st.selectbox(
                "Merchant category", scoring.MERCHANT_CATEGORIES,
                index=list(scoring.MERCHANT_CATEGORIES).index(start("merchant_category")))
            st.write("")
            submitted = st.form_submit_button("Assess application", use_container_width=True)
    if submitted:
        profile = {"monthly_income_gbp": income, "existing_monthly_obligations_gbp": obligations,
                   "savings_balance_gbp": savings, "credit_score": score_in,
                   "months_credit_history": history, "num_credit_accounts": accounts,
                   "num_defaults_24m": defaults, "thin_file_flag": int(thin)}
        try:
            features = scoring.new_customer(profile, order_value, int(instalments), category)
            context = {"type": "New applicant", "category": category,
                       "inputs": {**profile, "order_value_gbp": order_value,
                                  "n_instalments": int(instalments),
                                  "merchant_category": category}}
        except (KeyError, ValueError) as error:
            st.error(f"Could not build features: {error}")
else:
    if not ok:
        theme.empty_state("Existing-customer lookup needs the database.",
                          f"PostgreSQL is not reachable: {reason}")
    else:
        with st.form("lookup"):
            cid = st.text_input("Customer ID", placeholder="e.g. C6E79F37DD1A2B3C4",
                                help="Scores the customer's governed feature-store record.")
            submitted = st.form_submit_button("Assess customer")
        if submitted and cid.strip():
            row = scoring.lookup_customer(cid)
            if row is None:
                st.warning("No customer with that ID in the feature store.")
            else:
                features = row
                context = {"type": "Existing customer", "id": cid.strip().upper()}

if features is not None:
    try:
        with st.spinner("Scoring…"):
            result = scoring.score(features)
            result["outputs"] = (scoring.explain_new(context["inputs"]) if "inputs" in context
                                 else scoring.explain_existing(features))
            st.session_state["assessment_result"] = {"result": result, "context": context}
    except Exception as error:  # noqa: BLE001 - shown, never replaced with a made-up score
        st.session_state.pop("assessment_result", None)
        st.error(f"Scoring failed: {error}")

# ── RESULT ────────────────────────────────────────────────────
saved = st.session_state.get("assessment_result")
if saved:
    res, ctx = saved["result"], saved["context"]
    refer = res["decision"] == "flag_high_risk"
    verdict = "Refer for manual review" if refer else "Below review threshold"
    level = "critical" if refer else "positive"
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="sc-surface">
          <div class="sc-kicker">Assessment · {ctx['type']}{' · ' + ctx['id'] if 'id' in ctx else ''}</div>
          <div style="display:flex;flex-wrap:wrap;gap:2.4rem;align-items:flex-end;margin-top:0.4rem">
            <div><div class="sc-result-label">Probability of default</div>
                 <div class="sc-result-value">{res['probability']:.1%}</div></div>
            <div><div class="sc-result-label">Recommendation</div>
                 <div style="margin-top:0.5rem">{theme.pill(verdict, level)}</div></div>
            <div><div class="sc-result-label">Relative risk</div>
                 <div style="font-size:1.05rem;margin-top:0.45rem;color:{theme.INK}">
                 Higher than {res['percentile']:.0%} of {res['heldout_n']:,} held-out customers</div></div>
            <div><div class="sc-result-label">Affordability ratio</div>
                 <div style="font-size:1.05rem;margin-top:0.45rem;color:{theme.INK}">
                 {res['features']['max_affordability_ratio']:.2f}</div></div>
          </div>
          <div class="sc-note" style="margin-top:0.9rem">Review cutoff PD ≥ {res['cutoff']:.2%}.
          Affordability ratio = (existing obligations + this instalment) ÷ monthly income.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    outputs = res.get("outputs")
    if outputs:
        esc = html.escape
        ex, off = outputs["explanation"], outputs["risk_officer"]
        flag = ""
        if ex["review"]["required"]:
            flag = (f"<div style='border:1px solid {theme.AMBER};border-left:3px solid {theme.AMBER};"
                    f"padding:0.6rem 0.9rem;margin:0.3rem 0 0.8rem'><b style='color:{theme.AMBER}'>"
                    f"{esc(ex['review']['heading'])}</b><div>{esc(ex['review']['text'])}</div></div>")
        items = "".join(f"<li>{esc(r['text'])}</li>" for r in ex["reasons"])
        data = "".join(f"<li>{esc(d)}</li>" for d in ex["data"])
        steps = "".join(f"<li>{esc(s)}</li>" for s in ex["next_steps"])
        st.markdown(
            f"""<div style="background:{theme.GRAPHITE};border:1px solid {theme.GRAPHITE_LINE};border-left:3px solid {theme.GOLD};padding:1.2rem 1.4rem;margin-top:1rem">
              <div class="sc-kicker">For the applicant</div>
              {flag}
              <div style="font-family:'Cormorant Garamond', Georgia, serif;font-size:1.5rem;font-weight:600">
                {esc(ex['title'])}</div>
              <p style="margin:0.4rem 0 0.6rem">{esc(ex['intro'])}</p>
              {'<ol style="margin:0 0 0.4rem 1.2rem">' + items + '</ol>' if items else ''}
              {'<div class="sc-note">' + esc(ex['link']) + '</div>' if ex['link'] else ''}
              {'<p style="margin:0.5rem 0">' + esc(ex['other']) + '</p>' if ex['other'] else ''}
              {'<div class="sc-note" style="margin-top:0.5rem">' + esc(ex['data_intro']) + '</div>' if ex['data_intro'] else ''}
              {'<ul style="margin:0.2rem 0 0.4rem 1.2rem">' + data + '</ul>' if data else ''}
              <div class="sc-kicker" style="margin-top:0.8rem">Next steps</div>
              <ul style="margin:0.2rem 0 0.5rem 1.2rem">{steps}</ul>
              <div class="sc-note">{esc(ex['method'])}</div>
            </div>""",
            unsafe_allow_html=True)

        theme.section("For a risk officer")
        st.markdown(
            theme.pill("Risk officer review required", "warning") if off["review_required"]
            else theme.pill("No review flag", "positive"), unsafe_allow_html=True)
        st.markdown(f"**{off['error_label']}**  \n{off['genre_reason']}"
                    + (f"  \n{off['divergence_note']}" if off["divergence_note"] else ""))
        st.dataframe({"Check": [c["name"] for c in off["checks"]],
                      "Result": ["Flag" if c["triggered"] else "Clear" for c in off["checks"]],
                      "Detail": [c["detail"] for c in off["checks"]]},
                     use_container_width=True, hide_index=True)
        theme.note("Limits come from the training data; the blind-spot percentile and the "
                   "single-detail share are declared choices (Decision 31). Review required is a "
                   "statement, not a routing.")
        with st.expander("Decision record"):
            st.json({**outputs["decision_record"],
                     "calibrated_probability": round(res["probability"], 6)})

    if res["drivers"] is not None:
        theme.section("What drove this score")
        d = res["drivers"].head(8).iloc[::-1]
        colours = [theme.CRIMSON if v > 0 else theme.EMERALD for v in d["shap"]]
        fig = go.Figure(go.Bar(
            x=d["shap"], y=[LABELS.get(f, f) for f in d["feature"]], orientation="h",
            marker_color=colours, customdata=d["value"],
            hovertemplate="%{y}<br>input value %{customdata}<br>"
                          "contribution %{x:+.3f}<extra></extra>"))
        fig.update_layout(title="Largest contributions",
                          xaxis_title="← lowers risk · raises risk →  (SHAP, raw margin)",
                          yaxis_title=None)
        theme.chart(fig, 330)
        theme.note("Contributions come from the champion's uncalibrated explanation model. "
                   "They show direction and relative strength, not changes in the calibrated "
                   "probability (Decision 14).")

    if ranges is not None:
        outside = [LABELS.get(f, f) for f, v in res["features"].items()
                   if f in ranges.index and not (ranges.loc[f, "min"] <= v <= ranges.loc[f, "max"])]
        if outside:
            st.warning("Outside the range seen in training, so the score is an extrapolation: "
                       + ", ".join(outside) + ".")

    with st.expander("Model inputs used"):
        st.dataframe(
            {"Feature": [LABELS.get(f, f) for f in res["features"]],
             "Value": [f"{v:,.4g}" if isinstance(v, (int, float)) else str(v)
                       for v in res["features"].values()]},
            use_container_width=True, hide_index=True)

    st.info(
        "Scope: the champion is a customer-level portfolio-risk model trained on lifetime "
        "behaviour (Decision 18). For a new applicant, the score demonstrates the scoring "
        "mechanism; it is not a validated application-time underwriting decision.",
        icon=None,
    )

"""
Sentinel Credit, Governance and Audit

The governance record: approval-gate status (STALE-aware, read from
reporting.governance_status_latest, Decision 24), the gate's history,
the most recent promotion record (Decisions 21, 27), the champion's
identity, and the decision record parsed from the Engineering Decision
Log itself. Read-only: nothing on this page can promote, approve or
change a model.
"""

import json
import os
import re
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theme  # noqa: E402
from utils import (  # noqa: E402
    PROJECT_ROOT, db_available, governance_status, load_config, load_json, query_db,
)

theme.apply("Governance & Audit")
cfg = load_config()
gov = cfg["model_governance"]
theme.header(
    "Governance and audit",
    "Governance & Audit",
    "Who decided what, on which evidence. Approval-gate status, promotion history, "
    "the champion's identity, and the engineering decision record.",
)

ok, reason = db_available()

# ── GATE ──────────────────────────────────────────────────────
theme.section("Approval gate · current status")
status = governance_status() if ok else None
if status is None:
    theme.empty_state("Gate status unavailable.",
                      "The governance view could not be read: "
                      + (reason or "no gate evaluation recorded yet."))
else:
    s = status["gate_status"]
    level = {"PASS": "positive", "STALE": "warning"}.get(s, "critical")
    a, b, c, d = st.columns([1.2, 1, 1, 1])
    a.markdown("**Status**<br>" + theme.pill(s, level), unsafe_allow_html=True)
    b.metric("Evidence age", f"{float(status['evidence_age_hours_now']):.1f} h",
             help="Age of the monitoring evidence now: its age at evaluation plus the time since.")
    c.metric("Limit", f"{int(status['max_monitoring_age_hours'])} h")
    d.metric("Drift alerts at evaluation", f"{int(status['n_alerts'])} of "
             f"{int(status['n_features_total'])}")
    theme.note("PASS needs fresh monitoring evidence and zero drift alerts. A PASS older than "
               "the limit reads STALE automatically; a stopped pipeline cannot keep showing PASS.")

if ok:
    log = query_db("SELECT id, gate_status, evaluated_at, n_alerts, n_features_total, "
                   "monitoring_age_hours FROM reporting.governance_status_log "
                   "ORDER BY evaluated_at DESC")
    if not log.empty:
        log["evaluated_at"] = pd.to_datetime(log["evaluated_at"], utc=True).dt.tz_localize(None)
        with st.expander(f"Gate history · {len(log)} evaluations"):
            st.dataframe(log, hide_index=True, use_container_width=True, column_config={
                "id": "#", "gate_status": "Recorded status",
                "evaluated_at": st.column_config.DatetimeColumn("Evaluated (UTC)"),
                "n_alerts": "Alerts", "n_features_total": "Features",
                "monitoring_age_hours": st.column_config.NumberColumn("Evidence age at eval (h)",
                                                                      format="%.2f")})

# ── CHAMPION ──────────────────────────────────────────────────
theme.section("Champion identity")
pkg = os.path.join(PROJECT_ROOT, gov.get("artifacts_dir", "models/artifacts"))
meta_path = os.path.join(pkg, "champion_metadata.json")
champ = None
if os.path.exists(meta_path):
    with open(meta_path, encoding="utf-8") as f:
        champ = json.load(f)
theme.chips({
    "Model": f"{gov['champion_model']} {gov['model_version']}",
    "Feature set": gov["feature_set_label"],
    "Features": champ["n_features"] if champ else "—",
    "Trained": champ["trained_at"][:16].replace("T", " ") if champ else "—",
    "Hyperparameters": champ.get("xgboost_params_source", "grid_search") if champ else "—",
    "Cutoff": cfg["model"]["business_cutoff"],
    "Package": gov.get("artifacts_dir", "models/artifacts"),
})
theme.note("Each champion has its own package folder; earlier champions are never overwritten. "
           "Rollback steps are recorded in config.yaml beside artifacts_dir (Decision 27).")

# ── PROMOTION ─────────────────────────────────────────────────
theme.section("Most recent promotion record")
promo = load_json("promotion/promotion_recommendation.json")
if promo is None:
    theme.empty_state("No promotion record yet.", "Run pipeline/promote_champion.py.")
else:
    applied = promo.get("applied")
    path = promo.get("promotion_path", "gini_win")
    a, b, c = st.columns(3)
    a.markdown("**Outcome**<br>" + theme.pill("Applied" if applied else "Not applied",
                                              "positive" if applied else "neutral"),
               unsafe_allow_html=True)
    b.markdown("**Promotion path**<br>" + theme.pill(path.replace("_", " "), "neutral"),
               unsafe_allow_html=True)
    c.markdown(f"**Recorded**<br><span class='sc-note'>{promo.get('generated_at', '')}</span>",
               unsafe_allow_html=True)
    rows = []
    for split in ("stratified", "temporal"):
        r = promo.get(split, {})
        rows.append({"Split": split, "Champion Gini": r.get("champion_gini"),
                     "Challenger Gini": r.get("challenger_gini"), "Δ Gini": r.get("delta_gini"),
                     "Material win": r.get("material_win")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(k, format="%.4f")
                                for k in ("Champion Gini", "Challenger Gini", "Δ Gini")})
    v = promo.get("validity_override")
    if v:
        lines = [f"**Validity override · {v['decision_ref']}** · removed "
                 f"`{v['removed_feature']}` · rule: {v['rule']}"]
        for split, sv in v["splits"].items():
            lines.append(f"- {split}: Gini difference {sv['point']:+.4f}, 95% CI "
                         f"[{sv['ci95'][0]:+.4f}, {sv['ci95'][1]:+.4f}]"
                         f" → {'loss could be zero' if sv['non_inferior'] else 'loss is real'}")
        st.markdown("\n".join(lines))
    theme.note("Promotion needs a material Gini win on both splits, or an evidence-bound "
               "validity override for removing an invalid feature, plus a PASS from the gate "
               "and a typed human confirmation (Decisions 21, 27).")

# ── DECISION RECORD ───────────────────────────────────────────
theme.section("Engineering decision record")
log_path = os.path.join(PROJECT_ROOT, "docs", "engineering_decision_log.md")
if os.path.exists(log_path):
    with open(log_path, encoding="utf-8") as f:
        text = f.read()
    heads = re.findall(r"^## Decision (\d+) — (.+)$", text, flags=re.M)
    version = re.search(r"^\*\*Version:\*\* (.+)$", text, flags=re.M)
    table = pd.DataFrame([(int(n), t) for n, t in heads], columns=["#", "Decision"])
    table = table.sort_values("#", ascending=False)
    st.dataframe(table, hide_index=True, use_container_width=True, height=320)
    theme.note(f"Parsed from docs/engineering_decision_log.md"
               + (f" (version {version.group(1)})" if version else "")
               + ". Open findings are tracked in docs/audit_findings_2026-10-12.md.")
else:
    theme.empty_state("Decision log not found.", "Expected docs/engineering_decision_log.md.")

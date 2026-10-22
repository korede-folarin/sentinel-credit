"""
Sentinel Credit, Applicant Explanations and Decision Checks (Decisions 30, 31, 33)

Turns a scored application into three outputs (Decision 7):

- explanation (applicant): the outcome first, then why, in the applicant's
  own details. Approved, or needs a manual review; never declined (the
  model has no decline rule). Model issues are never shown to the
  applicant (Decision 33). Then: details to check, what could change the
  outcome (model-tested single changes), and what happens next.
- risk_officer: error type, the three checks, divergence, context.
- decision_record (regulatory): model identity, cutoff, methods, rules
  version, lineage, approver (not recorded, Decision 23).

Shared by the FastAPI service and the Streamlit dashboard. The public site
runs a line-for-line JavaScript port (website/public/assets/reasons.js),
checked word for word against this module by the site's parity page.

Reasons (Decision 30): each input's effect is measured with the real model
by setting that one input to the typical value and re-scoring:
    effect = P(applicant) - P(applicant with this input typical)
It describes how the model responds, not what causes default.

Checks (Decision 31), limits from serving/explanation_rules.json (built
from the training partition by models/build_explanation_rules.py):
1. data input: a model feature outside its training range, or a
   combination never seen in training;
2. missing information: a full credit file (the model's known blind spot)
   scoring at or above the group's configured percentile;
3. over-reliance: one detail decides the recommendation and carries most
   of the total effect.
Any of them means risk officer review required. Outcome (Decision 33):
approved only if the model passes the application AND no check fires;
otherwise the applicant is told it needs a manual review.

What could change the outcome (Decision 33): for a manual review that is
not a data-input case, single changes to what an applicant can act on
(order value, plan length, existing commitments, savings), each scored by
the model with everything else unchanged. A change qualifies if the model
would then approve automatically (below the cutoff and outside the blind
spot). The smallest qualifying change per detail is shown, up to three.
"""

import json
import math
import os

from serving.features import PROFILE_FIELDS, build_new_customer_features

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = PROFILE_FIELDS + ["order_value_gbp", "n_instalments", "merchant_category"]
FEATURES = PROFILE_FIELDS + ["max_affordability_ratio", "avg_order_value_gbp", "avg_n_instalments",
                             "pct_high_risk_category", "pct_medium_risk_category"]
MONEY = {"monthly_income_gbp", "existing_monthly_obligations_gbp", "savings_balance_gbp",
         "order_value_gbp", "avg_order_value_gbp"}
PERCENT = {"max_affordability_ratio", "pct_high_risk_category", "pct_medium_risk_category"}
# For a new applicant, the input a model feature is shown as.
FEATURE_TO_INPUT = {"avg_order_value_gbp": "order_value_gbp", "avg_n_instalments": "n_instalments"}
# Details an applicant can act on, in the order guidance is listed on ties.
ACTIONABLE = ["order_value_gbp", "n_instalments", "existing_monthly_obligations_gbp",
              "savings_balance_gbp"]
SPAN_FEATURE = {"order_value_gbp": "avg_order_value_gbp", "n_instalments": "avg_n_instalments"}


def _load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def load_templates() -> dict:
    return _load("reason_templates.json")


def load_typical(kind: str = "new") -> dict:
    doc = _load("typical_applicant.json")
    return doc["values"] if kind == "new" else doc["existing_customer_values"]


def load_rules() -> dict:
    return _load("explanation_rules.json")


def _round_half_up(v: float) -> int:
    """Matches JavaScript Math.round for the values shown (whole pounds etc.)."""
    return int(math.floor(v + 0.5))


def _round10(v: float) -> int:
    return 10 * _round_half_up(v / 10)


def _capitalise(s: str) -> str:
    return s[:1].upper() + s[1:]


def format_value(field, v, templates: dict) -> str:
    if field == "merchant_category":
        return str(v)
    if field in MONEY:
        return f"£{_round_half_up(v):,}"
    if field in PERCENT:
        return f"{_round_half_up(v * 100)}%"
    if field == "avg_n_instalments":
        return f"{_round_half_up(v * 10) / 10:.1f} instalments"
    n = _round_half_up(v)
    units = templates["units"].get(field)
    if units:
        return f"{n} {units[0] if n == 1 else units[1]}"
    return str(n)


def _features(inputs: dict) -> dict:
    profile = {f: inputs[f] for f in PROFILE_FIELDS}
    return build_new_customer_features(profile, inputs["order_value_gbp"],
                                       inputs["n_instalments"], inputs["merchant_category"])


def effects(inputs: dict, typical: dict, score_many, kind: str = "new") -> tuple:
    """(probability, {input: effect}) for every input that differs from typical.
    score_many: list of feature dicts -> list of probabilities. For an
    existing customer the inputs are the model features themselves."""
    to_features = _features if kind == "new" else (lambda d: {f: d[f] for f in FEATURES})
    order = INPUTS if kind == "new" else FEATURES
    variants, names = [to_features(inputs)], []
    for f in order:
        if inputs[f] != typical[f]:
            variants.append(to_features({**inputs, f: typical[f]}))
            names.append(f)
    probs = score_many(variants)
    return probs[0], {f: probs[0] - p for f, p in zip(names, probs[1:])}


def _sentence(field, inputs, typical, effect, t):
    if field == "merchant_category" or field == "thin_file_flag":
        direction = "higher" if effect > 0 else "lower"
    else:
        direction = "higher" if inputs[field] > typical[field] else "lower"
    return t["fields"][field][direction].format(
        value=format_value(field, inputs[field], t),
        typical=format_value(field, typical[field], t))


def build_reasons(inputs: dict, typical: dict, effect_by_input: dict, templates: dict = None,
                  kind: str = "new") -> dict:
    """The details that counted most against and in favour (Decision 30), as
    sentences in the applicant's own values. Deterministic ordering: effect
    size, then input order, so the JavaScript port produces identical text."""
    t = templates or load_templates()
    tol = t["numerical_tolerance"]
    order = {f: i for i, f in enumerate(INPUTS if kind == "new" else FEATURES)}
    against = sorted([(f, e) for f, e in effect_by_input.items() if e > tol],
                     key=lambda x: (-x[1], order[x[0]]))
    favour = sorted([(f, e) for f, e in effect_by_input.items() if e < -tol],
                    key=lambda x: (x[1], order[x[0]]))
    n = t["max_reasons"]
    return {
        "against": [{"input": f, "text": _sentence(f, inputs, typical, e, t)} for f, e in against[:n]],
        "favour": [{"input": f, "text": _sentence(f, inputs, typical, e, t)} for f, e in favour[:n]],
    }


# ── Decision 31: checks ───────────────────────────────────────────

def _feature_of(name: str, features: dict) -> str:
    """The model feature whose 'past customers like you' band describes an input."""
    if name == "order_value_gbp":
        return "avg_order_value_gbp"
    if name == "n_instalments":
        return "avg_n_instalments"
    if name == "merchant_category":
        return ("pct_medium_risk_category" if features["pct_medium_risk_category"] >= 1
                and features["pct_high_risk_category"] < 1 else "pct_high_risk_category")
    return name


def band_text(feature: str, value, rules: dict) -> str:
    for b in rules["bands"][feature]:
        if (b["lo"] is None or value >= b["lo"]) and (b["hi"] is None or value < b["hi"]):
            return b["text"]
    return rules["overall"]["text"]


def data_input_issues(features: dict, inputs: dict, kind: str, rules: dict, t: dict) -> list:
    issues, words = [], t["issues"]
    for f in FEATURES:
        r, v = rules["ranges"][f], features[f]
        if (r["check_min"] and v < r["min"]) or v > r["max"]:
            key = FEATURE_TO_INPUT.get(f, f) if kind == "new" else f
            shown = inputs[key] if key in inputs else v
            issues.append(words["out_of_range"].format(label=t["fields"][key]["label"],
                                                       value=format_value(key, shown, t)))
    c = rules["consistency"]
    months = format_value("months_credit_history", features["months_credit_history"], t)
    accounts = format_value("num_credit_accounts", features["num_credit_accounts"], t)
    if features["thin_file_flag"] == 1:
        if features["months_credit_history"] > c["thin_file_max_months_credit_history"]:
            issues.append(words["thin_long_history"].format(months=months))
        if features["num_credit_accounts"] > c["thin_file_max_num_credit_accounts"]:
            issues.append(words["thin_many_accounts"].format(accounts=accounts))
    else:
        if features["months_credit_history"] < c["full_file_min_months_credit_history"]:
            issues.append(words["full_short_history"].format(months=months))
        if features["num_credit_accounts"] < c["full_file_min_num_credit_accounts"]:
            issues.append(words["full_no_accounts"].format(accounts=accounts))
    if (features["existing_monthly_obligations_gbp"] / features["monthly_income_gbp"]
            > c["max_commitments_to_income"]):
        issues.append(words["commitments_vs_income"].format(
            obligations=format_value("existing_monthly_obligations_gbp",
                                     features["existing_monthly_obligations_gbp"], t),
            income=format_value("monthly_income_gbp", features["monthly_income_gbp"], t)))
    return issues


def single_detail(probability: float, effect_by_input: dict, cutoff: float, order: list,
                  share: float):
    """The input that decides the recommendation on its own, or None."""
    total = sum(abs(e) for e in effect_by_input.values())
    if total <= 0:
        return None
    rank = {f: i for i, f in enumerate(order)}
    name, e = sorted(effect_by_input.items(), key=lambda x: (-abs(x[1]), rank[x[0]]))[0]
    flips = (probability >= cutoff) != ((probability - e) >= cutoff)
    return name if (flips and abs(e) / total >= share) else None


# ── Decision 33: what could change the outcome ────────────────────

def change_candidates(inputs: dict, rules: dict) -> list:
    """(input, value) single changes an applicant could make, nearest first."""
    out = []
    cur = inputs["order_value_gbp"]
    seen = []
    for k in range(1, 20):                       # smaller orders, down to £20
        v = max(20, _round10(cur * (1 - k / 20)))
        if v < cur and v not in seen:
            seen.append(v)
    out += [("order_value_gbp", v) for v in seen]
    out += [("n_instalments", n) for n in (3, 6, 12) if n != inputs["n_instalments"]]
    cur = inputs["existing_monthly_obligations_gbp"]
    seen = []
    for k in range(1, 11):                       # lower commitments, down to £0
        v = _round10(cur * (1 - k / 10))
        if v < cur and v not in seen:
            seen.append(v)
    out += [("existing_monthly_obligations_gbp", v) for v in seen]
    cur, top = inputs["savings_balance_gbp"], rules["ranges"]["savings_balance_gbp"]["max"]
    seen = []
    for k in range(1, 21):                       # more savings, up to the training maximum
        v = _round10(cur + (top - cur) * k / 20)
        if v > cur and v not in seen:
            seen.append(v)
    out += [("savings_balance_gbp", v) for v in seen]
    return out


def outcome_changes(inputs: dict, cutoff: float, rules: dict, t: dict, score_many) -> list:
    """Up to three sentences: the smallest single change per actionable detail
    that the model says would make this an automatic approval."""
    cands = change_candidates(inputs, rules)
    if not cands:
        return []
    probs = score_many([_features({**inputs, f: v}) for f, v in cands])
    line = rules["blind_spot"]["line"]
    full_file = inputs["thin_file_flag"] == 0
    best = {}
    for (f, v), p in zip(cands, probs):
        if p >= cutoff or (full_file and p >= line):
            continue
        r = rules["ranges"][SPAN_FEATURE.get(f, f)]
        d = abs(v - inputs[f]) / (r["max"] - r["min"])
        if f not in best or d < best[f][0]:
            best[f] = (d, v)
    ranked = sorted(best.items(), key=lambda x: (x[1][0], ACTIONABLE.index(x[0])))[:3]
    return [t["changes"][f].format(value=format_value(f, v, t),
                                   current=format_value(f, inputs[f], t))
            for f, (_, v) in ranked]


# ── the three outputs ─────────────────────────────────────────────

def review(inputs: dict, features: dict, probability: float, effect_by_input: dict,
           reasons: dict, cutoff: float, kind: str = "new", templates: dict = None,
           rules: dict = None, score_many=None) -> dict:
    """Apply the checks (Decision 31) and build the three outputs (Decision 33)."""
    t = templates or load_templates()
    rules = rules or load_rules()
    order = INPUTS if kind == "new" else FEATURES
    refer = probability >= cutoff
    bs = rules["blind_spot"]

    issues = data_input_issues(features, inputs, kind, rules, t)
    full_file = features["thin_file_flag"] == 0
    blind = full_file and probability >= bs["line"]
    decisive = single_detail(probability, effect_by_input, cutoff, order,
                             rules["choices"]["single_detail_share"])
    if issues:
        error_type = "data_input"
    elif blind:
        error_type = "missing_information"
    elif decisive:
        error_type = "overreliance"
    else:
        error_type = "none"
    required = error_type != "none"
    outcome = "manual_review" if (refer or required) else "approved"

    # applicant output: outcome, why, details to check, what could change, next
    o = t["outcome"][outcome]
    why = reasons["against"] if outcome == "manual_review" else reasons["favour"]
    changes = []
    if outcome == "manual_review" and kind == "new" and error_type != "data_input" and score_many:
        changes = outcome_changes(inputs, cutoff, rules, t, score_many)
    show_changes = outcome == "manual_review" and kind == "new" and error_type != "data_input"
    explanation = {
        "outcome": outcome,
        "headline": o["headline"],
        "summary": o["summary"],
        "why_intro": o["why_intro"] if why else o["why_none"],
        "why": why,
        "check_intro": t["check"]["intro"] if issues else "",
        "check": [_capitalise(s) + "." for s in issues],
        "changes_intro": (t["changes_text"]["intro"] if changes
                          else t["changes_text"]["none"]) if show_changes else "",
        "changes": changes,
        "changes_note": t["changes_text"]["note"] if changes else "",
        "next_steps": list(o["next_steps"]),
        "method": t["method"][kind],
    }

    # risk officer output
    w = t["officer"]
    cw = w["checks"]
    pct = round(bs["percentile"] * 100)
    if not full_file:
        blind_detail = cw["missing_information"]["not_applicable"]
    else:
        blind_detail = cw["missing_information"]["triggered" if blind else "ok"].format(
            pct=pct, line=bs["line_text"])
    context = []
    if reasons["against"]:
        f = _feature_of(reasons["against"][0]["input"], features)
        context.append(band_text(f, features[f], rules))
    if blind:
        context += [bs["text_segment"], bs["text_top"]]
    officer = {
        "error_type": error_type,
        "error_label": w["error_labels"][error_type],
        "review_required": required,
        "model_recommendation": "refer" if refer else "pass",
        "customer_outcome": outcome,
        "divergence": required and not refer,
        "divergence_note": w["divergence_note"] if (required and not refer) else "",
        "checks": [
            {"id": "data_input", "name": cw["data_input"]["name"], "triggered": bool(issues),
             "detail": "; ".join(issues) if issues else cw["data_input"]["ok"]},
            {"id": "missing_information", "name": cw["missing_information"]["name"],
             "triggered": blind, "detail": blind_detail},
            {"id": "overreliance", "name": cw["overreliance"]["name"],
             "triggered": decisive is not None,
             "detail": (cw["overreliance"]["triggered"].format(name=t["fields"][decisive]["name"])
                        if decisive else cw["overreliance"]["ok"])},
        ],
        "context": context,
    }

    # regulatory record (deterministic part; callers add probability and time)
    m = rules["model"]
    record = {
        "record_type": "sentinel_credit_decision_record",
        "record_version": "2",
        "applicant_type": kind,
        "model_version": m["model_version"],
        "feature_set_label": m["feature_set_label"],
        "model_trained_at": m["trained_at"],
        "business_cutoff": m["cutoff"],
        "model_recommendation": "refer" if refer else "pass",
        "risk_officer_review_required": required,
        "customer_outcome": outcome,
        "divergence": required and not refer,
        "error_type": error_type,
        "rules_version": rules["rules_version"],
        "rules_generated_at": rules["generated_at"],
        "approver_id": None,
        "approver_note": t["record"]["approver_note"],
        "lineage": [s.format(model_version=m["model_version"], cutoff=m["cutoff"],
                             rules_version=rules["rules_version"]) for s in t["record"]["lineage"]],
    }
    return {"explanation": explanation, "risk_officer": officer, "decision_record": record}


def explain(inputs: dict, score_many, cutoff: float, typical: dict = None,
            templates: dict = None, rules: dict = None) -> dict:
    """Score, explain and check one new applicant: inputs are the raw application.
    Returns probability, effects and the three outputs."""
    t = templates or load_templates()
    typical = typical or load_typical("new")
    probability, eff = effects(inputs, typical, score_many, "new")
    reasons = build_reasons(inputs, typical, eff, t, "new")
    out = review(inputs, _features(inputs), probability, eff, reasons, cutoff, "new", t, rules,
                 score_many)
    return {"probability": probability, "effects": eff, **out}


def explain_existing(features: dict, score_many, cutoff: float, typical: dict = None,
                     templates: dict = None, rules: dict = None) -> dict:
    """The same three outputs for an existing customer, from their 13 model features."""
    t = templates or load_templates()
    typical = typical or load_typical("existing")
    inputs = {f: features[f] for f in FEATURES}
    probability, eff = effects(inputs, typical, score_many, "existing")
    reasons = build_reasons(inputs, typical, eff, t, "existing")
    out = review(inputs, inputs, probability, eff, reasons, cutoff, "existing", t, rules,
                 score_many)
    return {"probability": probability, "effects": eff, **out}

"""
Sentinel Credit, Plain-English Reasons and Decision Checks (Decisions 30, 31)

Turns a scored application into three outputs (Decision 7):

- explanation (customer): plain English, in a genre matched to the error
  type detected; says "Risk officer review required" when a check needs a
  person. No routing: the flag is the output.
- risk_officer: error type, genre and why, the three checks, divergence.
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
   combination never seen in training (e.g. a limited credit file with
   more history than any thin-file customer had);
2. missing information: a full credit file (the model's known blind spot:
   it never scores this group at the cutoff) scoring at or above the
   group's configured percentile;
3. over-reliance: one detail decides the recommendation (setting it alone
   to typical flips it) and carries most of the total effect.
Priority 1 > 2 > 3. Genre (Belton et al. 2025, Decision 7): data input ->
features-based; missing information and over-reliance -> data-centric,
no per-detail attributions; none -> combination. Any detected issue means
risk officer review required.
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


def _join(labels):
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def build_reasons(inputs: dict, typical: dict, probability: float, effect_by_input: dict,
                  refer: bool, templates: dict = None, kind: str = "new") -> dict:
    """Customer-facing reasons (Decision 30). Deterministic ordering: effect
    size, then input order, so the JavaScript port produces identical text."""
    t = templates or load_templates()
    tol = t["numerical_tolerance"]
    order = {f: i for i, f in enumerate(INPUTS if kind == "new" else FEATURES)}
    against = sorted([(f, e) for f, e in effect_by_input.items() if e > tol],
                     key=lambda x: (-x[1], order[x[0]]))
    favour = sorted([(f, e) for f, e in effect_by_input.items() if e < -tol],
                    key=lambda x: (x[1], order[x[0]]))
    if refer:
        main = against[:t["max_reasons"]]
        side = favour[:t["max_favourable"]]
        block = t["refer"]
        side_text = (f"{block['favour_intro']} {_join([t['fields'][f]['label'] for f, _ in side])}."
                     if side else "")
    else:
        main = favour[:t["max_reasons"]]
        side = against[:t["max_favourable"]]
        block = t["pass"]
        side_text = (f"{block['against_intro']} {_join([t['fields'][f]['label'] for f, _ in side])}."
                     if side else "")
    return {
        "decision": "refer" if refer else "pass",
        "title": block["title"],
        "intro": block["intro"] if main else block["none"],
        "reasons": [{"input": f, "text": _sentence(f, inputs, typical, e, t)} for f, e in main],
        "link": block.get("link", "") if (refer and main) else "",
        "other": side_text,
        "next_steps": list(t["next_steps"]["refer" if refer else "pass"]),
        "method": t["method"][kind],
    }


# ── Decision 31: checks, genre, three outputs ─────────────────────

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
    issues, words = [], t["review"]["issues"]
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


def review(inputs: dict, features: dict, probability: float, effect_by_input: dict,
           reasons: dict, cutoff: float, kind: str = "new", templates: dict = None,
           rules: dict = None) -> dict:
    """Apply Decision 31 to a Decision 30 explanation: returns the three outputs."""
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
        error_type, genre = "data_input", "features"
    elif blind:
        error_type, genre = "missing_information", "data_centric"
    elif decisive:
        error_type, genre = "overreliance", "data_centric"
    else:
        error_type, genre = "none", "combination"
    required = error_type != "none"

    # customer output
    ex = dict(reasons)
    words = t["review"]
    data = []
    if genre == "combination":
        data = [band_text(_feature_of(ex["reasons"][0]["input"], features),
                          features[_feature_of(ex["reasons"][0]["input"], features)], rules)
                if ex["reasons"] else rules["overall"]["text"]]
    elif error_type == "overreliance":
        f = _feature_of(decisive, features)
        data = [band_text(f, features[f], rules), rules["overall"]["text"]]
    elif error_type == "missing_information":
        data = [bs["text_segment"], bs["text_top"], rules["overall"]["text"]]
    if genre == "data_centric":
        ex["intro"] = words["data_only_intro"]
        ex["reasons"], ex["link"], ex["other"] = [], "", ""
    if error_type == "data_input":
        review_text = words["data_input"].format(issues="; ".join(issues))
    elif error_type == "missing_information":
        review_text = words["missing_information"].format(top=bs["top_percent"])
    elif error_type == "overreliance":
        review_text = words["overreliance"]
    else:
        review_text = ""
    next_steps = list(ex["next_steps"])
    if required and not refer:
        next_steps = [words["next_step"]] + next_steps
    explanation = {
        "decision": ex["decision"],
        "title": ex["title"],
        "review": {"required": required, "heading": words["heading"] if required else "",
                   "text": review_text},
        "intro": ex["intro"],
        "reasons": ex["reasons"],
        "link": ex["link"],
        "other": ex["other"],
        "data_intro": words["data_intro"] if (data and genre == "combination") else "",
        "data": data,
        "next_steps": next_steps,
        "method": rules["data_method"] if genre == "data_centric" else ex["method"],
    }

    # risk officer output
    o = t["officer"]
    checks_words = o["checks"]
    pct = round(bs["percentile"] * 100)
    if not full_file:
        blind_detail = checks_words["missing_information"]["not_applicable"]
    else:
        blind_detail = checks_words["missing_information"]["triggered" if blind else "ok"].format(
            pct=pct, line=bs["line_text"])
    officer = {
        "error_type": error_type,
        "error_label": o["error_labels"][error_type],
        "genre": genre,
        "genre_reason": o["genre_reasons"][genre],
        "review_required": required,
        "model_recommendation": "refer" if refer else "pass",
        "divergence": required and not refer,
        "divergence_note": o["divergence_note"] if (required and not refer) else "",
        "checks": [
            {"id": "data_input", "name": checks_words["data_input"]["name"],
             "triggered": bool(issues),
             "detail": "; ".join(issues) if issues else checks_words["data_input"]["ok"]},
            {"id": "missing_information", "name": checks_words["missing_information"]["name"],
             "triggered": blind, "detail": blind_detail},
            {"id": "overreliance", "name": checks_words["overreliance"]["name"],
             "triggered": decisive is not None,
             "detail": (checks_words["overreliance"]["triggered"].format(
                 name=t["fields"][decisive]["name"]) if decisive
                 else checks_words["overreliance"]["ok"])},
        ],
    }

    # regulatory record (deterministic part; callers add probability and time)
    m = rules["model"]
    record = {
        "record_type": "sentinel_credit_decision_record",
        "record_version": "1",
        "applicant_type": kind,
        "model_version": m["model_version"],
        "feature_set_label": m["feature_set_label"],
        "model_trained_at": m["trained_at"],
        "business_cutoff": m["cutoff"],
        "model_recommendation": "refer" if refer else "pass",
        "risk_officer_review_required": required,
        "divergence": required and not refer,
        "error_type": error_type,
        "explanation_genre": genre,
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
    reasons = build_reasons(inputs, typical, probability, eff, probability >= cutoff, t, "new")
    out = review(inputs, _features(inputs), probability, eff, reasons, cutoff, "new", t, rules)
    return {"probability": probability, "effects": eff, **out}


def explain_existing(features: dict, score_many, cutoff: float, typical: dict = None,
                     templates: dict = None, rules: dict = None) -> dict:
    """The same three outputs for an existing customer, from their 13 model features."""
    t = templates or load_templates()
    typical = typical or load_typical("existing")
    inputs = {f: features[f] for f in FEATURES}
    probability, eff = effects(inputs, typical, score_many, "existing")
    reasons = build_reasons(inputs, typical, probability, eff, probability >= cutoff, t, "existing")
    out = review(inputs, inputs, probability, eff, reasons, cutoff, "existing", t, rules)
    return {"probability": probability, "effects": eff, **out}

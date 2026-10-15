"""
Sentinel Credit, Plain-English Reasons for a New Applicant (Decision 30)

Turns a scored application into customer-facing reasons: the details that
counted most against it (or in its favour), in plain English, plus next
steps. Shared by the FastAPI service and the Streamlit dashboard; the
public site runs a line-for-line JavaScript port
(website/public/assets/reasons.js) checked word for word against this
module by the site's parity page.

Method: each input's effect is measured with the real model by setting
that one input to the typical applicant's value and re-scoring, keeping
everything else as entered:
    effect = P(applicant) - P(applicant with this input typical)
A positive effect means the applicant's value raised the probability of
default relative to typical. This is a one-input-at-a-time comparison:
it describes how the model responds, not what causes default.

Wording rules (serving/reason_templates.json): plain English, the
customer's own details, "the model links", never causal, never the
review cutoff (showing a decision rule harms error-spotting, Belton et
al. 2025; Decision 7), and a prompt to check the details are correct.
"""

import json
import math
import os

from serving.features import PROFILE_FIELDS, build_new_customer_features

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = PROFILE_FIELDS + ["order_value_gbp", "n_instalments", "merchant_category"]
MONEY = {"monthly_income_gbp", "existing_monthly_obligations_gbp", "savings_balance_gbp",
         "order_value_gbp"}


def load_templates() -> dict:
    with open(os.path.join(HERE, "reason_templates.json"), encoding="utf-8") as f:
        return json.load(f)


def load_typical() -> dict:
    with open(os.path.join(HERE, "typical_applicant.json"), encoding="utf-8") as f:
        return json.load(f)["values"]


def _round_half_up(v: float) -> int:
    """Matches JavaScript Math.round for the values shown (whole pounds etc.)."""
    return int(math.floor(v + 0.5))


def format_value(field: str, v, templates: dict) -> str:
    if field == "merchant_category":
        return str(v)
    if field in MONEY:
        return f"£{_round_half_up(v):,}"
    n = _round_half_up(v)
    units = templates["units"].get(field)
    if units:
        return f"{n} {units[0] if n == 1 else units[1]}"
    return str(n)


def _features(inputs: dict) -> dict:
    profile = {f: inputs[f] for f in PROFILE_FIELDS}
    return build_new_customer_features(profile, inputs["order_value_gbp"],
                                       inputs["n_instalments"], inputs["merchant_category"])


def effects(inputs: dict, typical: dict, score_many) -> tuple:
    """(probability, {input: effect}) for every input that differs from typical.
    score_many: list of feature dicts -> list of probabilities."""
    variants, names = [_features(inputs)], []
    for f in INPUTS:
        if inputs[f] != typical[f]:
            variants.append(_features({**inputs, f: typical[f]}))
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
                  refer: bool, templates: dict = None) -> dict:
    """Customer-facing explanation. Deterministic ordering: effect size, then
    input order, so the JavaScript port produces identical text."""
    t = templates or load_templates()
    tol = t["numerical_tolerance"]
    order = {f: i for i, f in enumerate(INPUTS)}
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
        "method": ("Each detail was compared with a typical applicant by changing that one detail "
                   "and re-scoring with the model."),
    }


def explain(inputs: dict, score_many, cutoff: float, typical: dict = None,
            templates: dict = None) -> dict:
    """Score and explain one new applicant: inputs are the raw application."""
    typical = typical or load_typical()
    probability, eff = effects(inputs, typical, score_many)
    out = build_reasons(inputs, typical, probability, eff, probability >= cutoff, templates)
    out["probability"] = probability
    return out

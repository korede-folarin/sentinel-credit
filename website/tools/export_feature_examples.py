"""
Sentinel Credit — Feature-construction and explanation examples for the browser checks

Writes website/public/model/feature_examples.json: generated example
applications (no customer records) with the features, probability and the
full three-output explanation (Decisions 30, 31) the Python path produces.
parity.html checks the JavaScript ports (model.js feature construction,
reasons.js) against these, word for word.

The reference files (typical applicant, explanation rules, wording) are
built by models/build_explanation_rules.py, which also copies them to
website/public/model/. Run that first after any promotion.

The applications are mixed so every error type is exercised: consistent
profiles, deliberately inconsistent ones (data input), weak full-file
profiles (the blind spot) and risky thin-file profiles (referrals).

Usage:  python website/tools/export_feature_examples.py
"""

import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from serving.features import MERCHANT_CATEGORIES, PROFILE_FIELDS, build_new_customer_features  # noqa: E402
from serving.reasons import explain, load_rules, load_templates, load_typical  # noqa: E402

MODEL_DIR = os.path.join(PROJECT_ROOT, "website", "public", "model")
TYPICAL = load_typical("new")
TEST_SHARE = 0.3   # test-only single-detail share for code-path coverage; live value in config


def application(i, rng):
    thin = int(i % 2)
    app = {
        "monthly_income_gbp": float(rng.choice([800, 1250.5, 2200, 3333.33, 6000, 15000])
                                    + rng.integers(0, 400)),
        "existing_monthly_obligations_gbp": float(rng.choice([0, 75.25, 300, 812.5])),
        "savings_balance_gbp": float(rng.choice([0, 150, 1500, 9000])),
        "credit_score": int(rng.integers(300, 999)),
        "months_credit_history": int(rng.integers(0, 24) if thin else rng.integers(12, 180)),
        "num_credit_accounts": int(rng.integers(0, 2) if thin else rng.integers(1, 8)),
        "num_defaults_24m": int(rng.integers(0, 3)),
        "thin_file_flag": thin,
        "order_value_gbp": float(rng.choice([19.99, 250, 333.33, 1299, 1900])),
        "n_instalments": int([3, 6, 12][i % 3]),
        "merchant_category": MERCHANT_CATEGORIES[i % len(MERCHANT_CATEGORIES)],
    }
    kind = i % 6
    if kind in (0, 3):   # risky thin file: referral wording
        app.update({"thin_file_flag": 1, "savings_balance_gbp": float(rng.choice([0, 20, 40])),
                    "credit_score": int(rng.integers(300, 450)), "num_credit_accounts": int(rng.integers(0, 2)),
                    "num_defaults_24m": 2, "months_credit_history": int(rng.integers(0, 12)),
                    "monthly_income_gbp": float(rng.choice([900, 1300.5, 1800])),
                    "existing_monthly_obligations_gbp": float(rng.choice([300, 500])),
                    "n_instalments": 12, "merchant_category": str(rng.choice(["Electronics", "Travel"]))})
        if kind == 3:    # one strong favourable detail: near the cutoff, may decide alone
            app.update(rng.choice([{"savings_balance_gbp": 9000.0}, {"credit_score": 900},
                                   {"months_credit_history": 23}, {"n_instalments": 3},
                                   {"merchant_category": "Food & Grocery"}]))
    elif kind == 1:   # weak full file: the blind spot
        app.update({"thin_file_flag": 0, "savings_balance_gbp": float(rng.choice([0, 40, 150])),
                    "credit_score": int(rng.integers(300, 480)), "num_defaults_24m": 2,
                    "months_credit_history": int(rng.integers(12, 30)),
                    "num_credit_accounts": int(rng.integers(1, 3)),
                    "monthly_income_gbp": float(rng.choice([900, 1300.5, 1800]))})
    elif kind in (4, 5):   # the typical applicant with a few details changed: near the cutoff
        app = dict(TYPICAL)
        if rng.random() < 0.6:
            app.update({"thin_file_flag": 1, "months_credit_history": int(rng.integers(0, 24)),
                        "num_credit_accounts": int(rng.integers(0, 2))})
        palette = [{"savings_balance_gbp": float(rng.choice([0, 40, 150]))},
                   {"credit_score": int(rng.integers(380, 560))},
                   {"num_defaults_24m": int(rng.integers(1, 3))},
                   {"n_instalments": 12},
                   {"merchant_category": str(rng.choice(["Electronics", "Travel"]))},
                   {"monthly_income_gbp": float(rng.choice([900, 1300.5]))}]
        for k in rng.choice(len(palette), size=int(rng.integers(1, 4)), replace=False):
            app.update(palette[k])
    elif kind == 2:   # inconsistent or out-of-range details: data input
        app.update(rng.choice([
            {"thin_file_flag": 1, "months_credit_history": int(rng.integers(30, 300))},
            {"thin_file_flag": 0, "num_credit_accounts": 0},
            {"existing_monthly_obligations_gbp": 1900.0, "monthly_income_gbp": 1200.0},
            {"monthly_income_gbp": 30000.0},
            {"order_value_gbp": 4999.5},
        ]))
    return app


def main():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pkg = os.path.join(PROJECT_ROOT, cfg["model_governance"]["artifacts_dir"])
    cutoff = float(cfg["model"]["business_cutoff"])
    model = joblib.load(os.path.join(pkg, "champion_calibrated.joblib"))
    with open(os.path.join(pkg, "feature_list.json"), encoding="utf-8") as f:
        features = json.load(f)["features"]
    rules = load_rules()
    if rules["model"]["model_version"] != cfg["model_governance"]["model_version"]:
        raise SystemExit("explanation_rules.json is for another champion: "
                         "run models/build_explanation_rules.py first.")

    def score_many(feature_dicts):
        return model.predict_proba(pd.DataFrame(feature_dicts)[features])[:, 1].tolist()

    templates, typical = load_templates(), load_typical("new")
    rng = np.random.default_rng(7)
    examples = []
    for i in range(300):
        app = application(i, rng)
        profile = {f: app[f] for f in PROFILE_FIELDS}
        feats = build_new_customer_features(profile, app["order_value_gbp"],
                                            app["n_instalments"], app["merchant_category"])
        # Over-reliance cannot fire for a new applicant at the live share (Decision 31:
        # reaching the cutoff takes several risky details, none dominant). To test that
        # code path word for word, every 5th example runs with a lowered, test-only share.
        override = {"single_detail_share": TEST_SHARE} if i % 5 == 4 else None
        r = ({**rules, "choices": {**rules["choices"], **override}} if override else rules)
        out = explain(app, score_many, cutoff, typical, templates, r)
        examples.append({"rules_override": override, "profile": profile, "order_value_gbp": app["order_value_gbp"],
                         "n_instalments": app["n_instalments"],
                         "merchant_category": app["merchant_category"],
                         "expected_features": feats,
                         "expected_probability": float(out["probability"]),
                         "expected_outputs": {k: out[k] for k in
                                              ("explanation", "risk_officer", "decision_record")}})

    payload = {"source": "generated applications (no customer records)",
               "built_with": "serving/features.py + serving/reasons.py + champion_calibrated.joblib",
               "features": features, "examples": examples}
    with open(os.path.join(MODEL_DIR, "feature_examples.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    probs = np.array([e["expected_probability"] for e in examples])
    kinds = Counter(e["expected_outputs"]["risk_officer"]["error_type"] for e in examples)
    live = Counter(e["expected_outputs"]["risk_officer"]["error_type"] for e in examples
                   if not e["rules_override"])
    print(f"At the live rules: {dict(live)}")
    print(f"Wrote feature_examples.json: {len(examples)} applications, "
          f"{int((probs >= cutoff).sum())} referred; error types {dict(kinds)}")
    missing = {"none", "data_input", "missing_information", "overreliance"} - set(kinds)
    if missing:
        raise SystemExit(f"Examples do not exercise every error type: missing {missing}")


if __name__ == "__main__":
    main()

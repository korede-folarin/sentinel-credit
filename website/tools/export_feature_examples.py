"""
Sentinel Credit — Feature-construction and reason examples for the browser checks

1. Writes serving/typical_applicant.json: the typical applicant the
   plain-English reasons compare against (Decision 30) — medians of the
   champion's canonical TRAINING partition, rounded to the precision they
   are shown at (so the value displayed is exactly the value used), with
   3 instalments and the most common merchant category.
2. Copies it and serving/reason_templates.json into website/public/model/
   so the site uses the same files as the API and the dashboard.
3. Writes website/public/model/feature_examples.json: generated example
   applications (no customer records) with the features, probability and
   full reason text the Python path produces. parity.html checks the
   JavaScript ports (features.js / reasons.js) against these, word for word.

Usage:  python website/tools/export_feature_examples.py
"""

import json
import os
import shutil
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psycopg2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from serving.features import MERCHANT_CATEGORIES, PROFILE_FIELDS, build_new_customer_features  # noqa: E402
from serving.reasons import explain, load_templates  # noqa: E402

MODEL_DIR = os.path.join(PROJECT_ROOT, "website", "public", "model")
TYPICAL_PATH = os.path.join(PROJECT_ROOT, "serving", "typical_applicant.json")
MOST_COMMON_CATEGORY = "Fashion & Clothing"   # config.yaml data_generator.merchant_categories (0.30)


def typical_applicant(cfg, pkg):
    ids = pd.read_csv(os.path.join(pkg, "split_ids_train.csv"))["customer_id"].astype(str)
    conn = psycopg2.connect(
        host=os.environ.get("SENTINEL_DB_HOST", cfg["database"]["host"]),
        port=cfg["database"]["port"], dbname=cfg["database"]["name"], user="postgres",
        password=os.environ.get("SENTINEL_DB_PASSWORD", ""))
    try:
        cols = PROFILE_FIELDS + ["avg_order_value_gbp"]
        frame = pd.read_sql(f"SELECT customer_id, {', '.join(cols)} "
                            "FROM feature_store.customer_features WHERE has_outcome = 1", conn)
    finally:
        conn.close()
    frame = frame[frame["customer_id"].astype(str).isin(set(ids))]
    med = frame[cols].median()
    values = {f: int(round(float(med[f]))) for f in PROFILE_FIELDS}
    values["order_value_gbp"] = int(round(float(med["avg_order_value_gbp"])))
    values["n_instalments"] = 3
    values["merchant_category"] = MOST_COMMON_CATEGORY
    return {
        "_about": ("Typical applicant for plain-English reasons (Decision 30): medians of the "
                   "champion's canonical training partition, rounded to whole units; 3 instalments "
                   "(Pay in 3, the dominant plan); the most common merchant category."),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_version": cfg["model_governance"]["model_version"],
        "n_training_customers": int(len(frame)),
        "values": values,
    }


def main():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pkg = os.path.join(PROJECT_ROOT, cfg["model_governance"]["artifacts_dir"])
    cutoff = float(cfg["model"]["business_cutoff"])
    model = joblib.load(os.path.join(pkg, "champion_calibrated.joblib"))
    with open(os.path.join(pkg, "feature_list.json"), encoding="utf-8") as f:
        features = json.load(f)["features"]

    typical_doc = typical_applicant(cfg, pkg)
    with open(TYPICAL_PATH, "w", encoding="utf-8") as f:
        json.dump(typical_doc, f, indent=2)
    typical = typical_doc["values"]
    print(f"Typical applicant ({typical_doc['n_training_customers']:,} training customers): {typical}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    shutil.copyfile(TYPICAL_PATH, os.path.join(MODEL_DIR, "typical_applicant.json"))
    shutil.copyfile(os.path.join(PROJECT_ROOT, "serving", "reason_templates.json"),
                    os.path.join(MODEL_DIR, "reason_templates.json"))

    def score_many(feature_dicts):
        return model.predict_proba(pd.DataFrame(feature_dicts)[features])[:, 1].tolist()

    templates = load_templates()
    rng = np.random.default_rng(7)
    examples = []
    for i in range(240):
        app = {
            "monthly_income_gbp": float(rng.choice([800, 1250.5, 2200, 3333.33, 6000, 15000])
                                        + rng.integers(0, 400)),
            "existing_monthly_obligations_gbp": float(rng.choice([0, 75.25, 300, 812.5, 1900])),
            "savings_balance_gbp": float(rng.choice([0, 150, 1500, 9000, 60000])),
            "credit_score": int(rng.integers(300, 999)),
            "months_credit_history": int(rng.integers(0, 300)),
            "num_credit_accounts": int(rng.integers(0, 12)),
            "num_defaults_24m": int(rng.integers(0, 4)),
            "thin_file_flag": int(i % 2),
            "order_value_gbp": float(rng.choice([1, 19.99, 250, 333.33, 1299, 4999.5])),
            "n_instalments": int([3, 6, 12][i % 3]),
            "merchant_category": MERCHANT_CATEGORIES[i % len(MERCHANT_CATEGORIES)],
        }
        if i % 4 == 0:   # deliberately risky profiles, so the referral wording is well covered
            app.update({"thin_file_flag": 1, "savings_balance_gbp": float(rng.choice([0, 40, 150])),
                        "credit_score": int(rng.integers(300, 520)),
                        "num_defaults_24m": int(rng.integers(1, 4)),
                        "months_credit_history": int(rng.integers(0, 24))})
        profile = {f: app[f] for f in PROFILE_FIELDS}
        feats = build_new_customer_features(profile, app["order_value_gbp"],
                                            app["n_instalments"], app["merchant_category"])
        exp = explain(app, score_many, cutoff, typical, templates)
        examples.append({"profile": profile, "order_value_gbp": app["order_value_gbp"],
                         "n_instalments": app["n_instalments"],
                         "merchant_category": app["merchant_category"],
                         "expected_features": feats,
                         "expected_probability": float(exp["probability"]),
                         "expected_reasons": {k: v for k, v in exp.items() if k != "probability"}})

    payload = {"source": "generated applications (no customer records)",
               "built_with": "serving/features.py + serving/reasons.py + champion_calibrated.joblib",
               "features": features, "examples": examples}
    with open(os.path.join(MODEL_DIR, "feature_examples.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    probs = np.array([e["expected_probability"] for e in examples])
    refer = int((probs >= cutoff).sum())
    print(f"Wrote feature_examples.json: {len(examples)} applications, {refer} referred; "
          f"reasons per referral up to {templates['max_reasons']}")
    sample = next(e for e in examples if e["expected_probability"] >= cutoff)["expected_reasons"]
    print("\nSample referral explanation:\n  " + sample["title"] + "\n  " + sample["intro"])
    for r in sample["reasons"]:
        print("   - " + r["text"])
    print("  " + sample["link"] + "\n  " + sample["other"])


if __name__ == "__main__":
    main()

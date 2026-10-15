"""
Sentinel Credit — Feature-construction examples for the browser port check

The website builds model features in JavaScript (website/public/assets/
model.js buildFeatures), a port of serving/features.py. This writes
generated example applications with the features and probability the
Python path produces, so parity.html can check the port end to end:
raw inputs -> JS features -> ONNX in the browser == Python features ->
Python champion.

Generated inputs only (no customer records). Includes edge cases: every
merchant category, every plan, thin file on/off, tiny and large orders,
and affordability ratios that need rounding.

Usage:  python website/tools/export_feature_examples.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from serving.features import MERCHANT_CATEGORIES, build_new_customer_features  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "website", "public", "model", "feature_examples.json")


def main():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pkg = os.path.join(PROJECT_ROOT, cfg["model_governance"]["artifacts_dir"])
    model = joblib.load(os.path.join(pkg, "champion_calibrated.joblib"))
    with open(os.path.join(pkg, "feature_list.json"), encoding="utf-8") as f:
        features = json.load(f)["features"]

    rng = np.random.default_rng(7)
    apps = []
    for i in range(240):
        apps.append({
            "profile": {
                "monthly_income_gbp": float(rng.choice([800, 1250.5, 2200, 3333.33, 6000, 15000])
                                            + rng.integers(0, 400)),
                "existing_monthly_obligations_gbp": float(rng.choice([0, 75.25, 300, 812.5, 1900])),
                "savings_balance_gbp": float(rng.choice([0, 150, 1500, 9000, 60000])),
                "credit_score": int(rng.integers(300, 999)),
                "months_credit_history": int(rng.integers(0, 300)),
                "num_credit_accounts": int(rng.integers(0, 12)),
                "num_defaults_24m": int(rng.integers(0, 4)),
                "thin_file_flag": int(i % 2),
            },
            "order_value_gbp": float(rng.choice([1, 19.99, 250, 333.33, 1299, 4999.5])),
            "n_instalments": int([3, 6, 12][i % 3]),
            "merchant_category": MERCHANT_CATEGORIES[i % len(MERCHANT_CATEGORIES)],
        })

    rows = []
    for a in apps:
        feats = build_new_customer_features(a["profile"], a["order_value_gbp"],
                                            a["n_instalments"], a["merchant_category"])
        rows.append(feats)
    X = pd.DataFrame(rows)[features]
    probs = model.predict_proba(X)[:, 1]
    payload = {"source": "generated applications (no customer records)",
               "built_with": "serving/features.py + champion_calibrated.joblib",
               "features": features,
               "examples": [{**a, "expected_features": r, "expected_probability": float(p)}
                            for a, r, p in zip(apps, rows, probs)]}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {os.path.relpath(OUT, PROJECT_ROOT)}: {len(apps)} applications, "
          f"PD range {probs.min():.4f}-{probs.max():.4f}, "
          f"{int((probs >= cfg['model']['business_cutoff']).sum())} above the cutoff")


if __name__ == "__main__":
    main()

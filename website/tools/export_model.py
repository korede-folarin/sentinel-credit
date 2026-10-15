"""
Sentinel Credit — Export the champion to ONNX for in-browser scoring

Converts the governed champion (config.yaml model_governance.artifacts_dir)
— a CalibratedClassifierCV of five XGBoost models, each with its own
sigmoid (Platt) calibrator, probabilities averaged — into one ONNX graph,
then proves the ONNX output matches the Python model before anything is
published.

Parity check (pass rule fixed in advance, PARITY_* below):
- every modelable customer in the feature store, plus edge cases at the
  extremes of the website's input ranges;
- max |P_python − P_onnx| must be below PARITY_MAX_ABS_DIFF, and
- no customer's decision may flip at the configured cutoff.
A failure writes the report and exits non-zero; no model is published.

Outputs (website/public/model/):
- sentinel_champion.onnx         the model
- model_contract.json            feature order, cutoff, version, ranges
- parity_vectors.json            inputs + expected Python probabilities,
                                 replayed by the browser parity page
- python_parity_report.json      this script's evidence

Usage:  python website/tools/export_model.py
"""

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import psycopg2
import yaml
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "website", "public", "model")

PARITY_MAX_ABS_DIFF = 1e-5      # float32 tree thresholds vs float64 Python
N_BROWSER_VECTORS = 2000        # replayed in the browser parity page
RANDOM_SEED = 42

with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)
GOV = CONFIG["model_governance"]
PKG = os.path.join(PROJECT_ROOT, GOV["artifacts_dir"])
CUTOFF = float(CONFIG["model"]["business_cutoff"])


def load_package():
    with open(os.path.join(PKG, "feature_list.json"), encoding="utf-8") as f:
        features = json.load(f)["features"]
    with open(os.path.join(PKG, "champion_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(os.path.join(PKG, "champion_calibrated.joblib"))
    return features, meta, model


def load_population(features):
    conn = psycopg2.connect(
        host=os.environ.get("SENTINEL_DB_HOST", CONFIG["database"]["host"]),
        port=CONFIG["database"]["port"], dbname=CONFIG["database"]["name"],
        user="postgres", password=os.environ.get("SENTINEL_DB_PASSWORD", ""))
    try:
        frame = pd.read_sql(f"SELECT customer_id, {', '.join(features)} "
                            "FROM feature_store.customer_features WHERE has_outcome = 1", conn)
    finally:
        conn.close()
    if frame[features].isna().any().any():
        raise ValueError("Null model-feature values in the feature store; investigate first.")
    return frame


def convert(model, n_features):
    """
    onnxmltools reads XGBoost tree dumps and requires split features named
    'f0'..'fN'. The champion's boosters carry real names, so names are
    stripped on a COPY (column order unchanged; the parity check compares
    the ONNX graph against the original, unmodified model).
    """
    import copy
    model = copy.deepcopy(model)
    for calibrated in model.calibrated_classifiers_:
        calibrated.estimator.get_booster().feature_names = None
    update_registered_converter(
        XGBClassifier, "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
    onx = convert_sklearn(
        model, initial_types=[("input", FloatTensorType([None, n_features]))],
        options={id(model): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3})
    onnx.checker.check_model(onx)
    return onx


def edge_cases(features, ranges):
    """Inputs at and beyond the extremes the website will allow."""
    rows = []
    lo, hi, med = ranges["min"], ranges["max"], ranges["median"]
    rows.append(med.to_dict())
    rows.append(lo.to_dict())
    rows.append(hi.to_dict())
    for f in features:                      # one feature at a time at each extreme
        for v in (lo[f], hi[f]):
            r = med.to_dict()
            r[f] = v
            rows.append(r)
    rng = np.random.default_rng(RANDOM_SEED)  # random points inside the box
    for _ in range(500):
        rows.append({f: float(rng.uniform(lo[f], hi[f])) for f in features})
    return pd.DataFrame(rows)[features]


def generated_vectors(model, features, ranges, n_random, n_boundary):
    """Synthetic test inputs that are not customer records."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    lo, hi = ranges["min"].to_numpy(float), ranges["max"].to_numpy(float)

    def score(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=features))[:, 1]

    # Uniform random applicants almost never reach the cutoff (measured: 0 of
    # 20,000; risk depends on feature combinations, not single extremes). So
    # high-risk anchors are FOUND: greedy coordinate ascent from random starts —
    # repeatedly set the one feature (over an 11-point grid of its range) that
    # raises the score most — until the score passes the cutoff. Then jittered
    # copies of the anchors that stay above it add diversity. Bounded throughout.
    grid = np.linspace(0, 1, 11)
    anchors = []
    for _ in range(40):                                      # bounded number of starts
        x = rng.uniform(lo, hi)
        for _step in range(len(features) * 2):
            cands = np.repeat(x[None, :], len(features) * len(grid), axis=0)
            for j in range(len(features)):
                cands[j * len(grid):(j + 1) * len(grid), j] = lo[j] + grid * (hi[j] - lo[j])
            p = score(cands)
            best = int(np.argmax(p))
            if p[best] <= score(x[None, :])[0] + 1e-12:
                break
            x = cands[best]
            if p[best] >= CUTOFF:
                anchors.append(x.copy())
                break
        if len(anchors) >= 20:
            break
    if not anchors:
        raise RuntimeError("No high-risk anchors found; cannot build boundary vectors.")
    anchors = np.array(anchors)
    high = anchors.copy()
    for _ in range(30):                                      # bounded jitter rounds
        base = anchors[rng.integers(0, len(anchors), size=4 * n_boundary)]
        cand = np.clip(base + rng.normal(0, 0.03, base.shape) * (hi - lo), lo, hi)
        high = np.vstack([high, cand[score(cand) >= CUTOFF]])
        if len(high) >= n_boundary:
            break
    if len(high) < n_boundary:
        raise RuntimeError(f"Only {len(high)} high-risk anchors found; cannot build boundary vectors.")
    high = high[rng.permutation(len(high))[:n_boundary]]
    low = rng.uniform(lo, hi, size=(n_boundary, len(features)))
    low_p = score(low)
    low = low[low_p < CUTOFF][:n_boundary]
    high = high[:len(low)]

    # Vectorised bisection on every pair at once: score(low) < cutoff <= score(high).
    t_lo, t_hi = np.zeros(len(low)), np.ones(len(low))
    for _ in range(50):
        t = (t_lo + t_hi) / 2
        p_t = score(low + t[:, None] * (high - low))
        below = p_t < CUTOFF
        t_lo, t_hi = np.where(below, t, t_lo), np.where(below, t_hi, t)
    # Keep the endpoint whose score is nearer the cutoff (trees are step functions,
    # so the exact cutoff may sit on a jump between the two).
    a_pts, b_pts = low + t_lo[:, None] * (high - low), low + t_hi[:, None] * (high - low)
    pa, pb = score(a_pts), score(b_pts)
    boundary = np.where((np.abs(pa - CUTOFF) <= np.abs(pb - CUTOFF))[:, None], a_pts, b_pts)

    frame = pd.concat([pd.DataFrame(boundary, columns=features),
                       pd.DataFrame(rng.uniform(lo, hi, size=(n_random, len(features))),
                                    columns=features)], ignore_index=True)
    return frame.astype(np.float32).astype(float)   # exactly representable in float32


def main():
    features, meta, model = load_package()
    print(f"Champion {meta['model_name']} {meta['model_version']} · {len(features)} features "
          f"· package {GOV['artifacts_dir']} · cutoff {CUTOFF}")

    pop = load_population(features)
    X = pop[features]
    ranges = X.agg(["min", "median", "max"]).T
    X_edge = edge_cases(features, ranges)

    onx = convert(model, len(features))
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "sentinel_champion.onnx")
    with open(model_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"ONNX written: {os.path.relpath(model_path, PROJECT_ROOT)} "
          f"({os.path.getsize(model_path) / 1024:.0f} KB)")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    prob_name = next(n for n in out_names if "prob" in n.lower())

    def onnx_proba(frame):
        return sess.run([prob_name], {"input": frame.to_numpy(np.float32)})[0][:, 1]

    report = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "model_version": meta["model_version"], "feature_set": GOV["feature_set_label"],
              "cutoff": CUTOFF, "rule": {"max_abs_diff_below": PARITY_MAX_ABS_DIFF,
                                         "decision_flips_allowed": 0},
              "onnx_outputs": out_names, "sets": {}}
    passed = True
    for name, frame in [("modelable_population", X), ("edge_cases", X_edge)]:
        p_py = model.predict_proba(frame)[:, 1]
        p_ox = onnx_proba(frame)
        diff = np.abs(p_py - p_ox)
        flips = int(((p_py >= CUTOFF) != (p_ox >= CUTOFF)).sum())
        nearest = float(np.min(np.abs(p_py - CUTOFF)))
        ok = bool(diff.max() < PARITY_MAX_ABS_DIFF and flips == 0)
        passed &= ok
        report["sets"][name] = {"n": int(len(frame)), "max_abs_diff": float(diff.max()),
                                "mean_abs_diff": float(diff.mean()),
                                "decision_flips": flips,
                                "closest_python_score_to_cutoff": nearest, "pass": ok}
        print(f"  {name:<22} n={len(frame):>6,}  max|Δ|={diff.max():.2e}  "
              f"mean|Δ|={diff.mean():.2e}  flips={flips}  -> {'PASS' if ok else 'FAIL'}")

    # Vectors for the browser parity page. GENERATED inputs only, never
    # customer rows (the repository never publishes data rows):
    # - random applicants inside the training ranges;
    # - applicants constructed to sit at the cutoff, by bisecting the line
    #   between a generated low-risk and high-risk profile (the score is a
    #   step function, so they land as close to it as the trees allow) —
    #   where a decision flip would show first.
    gen = generated_vectors(model, features, ranges, n_random=N_BROWSER_VECTORS - 400,
                            n_boundary=400)
    gen_p = model.predict_proba(gen)[:, 1]
    vectors = {"features": features, "cutoff": CUTOFF,
               "source": "generated inputs only (no customer rows)",
               "n_at_cutoff": int(len(gen) - (N_BROWSER_VECTORS - 400)),
               "rows": gen.to_numpy(float).tolist(),
               "expected": gen_p.round(12).tolist()}
    with open(os.path.join(OUT_DIR, "parity_vectors.json"), "w", encoding="utf-8") as f:
        json.dump(vectors, f)
    near = np.abs(gen_p - CUTOFF)
    n_b = vectors["n_at_cutoff"]
    print(f"  browser vectors: {len(gen):,} generated ({n_b} at the cutoff); closest |p − c| = "
          f"{near.min():.1e}, median of boundary set {np.median(np.sort(near)[:n_b]):.1e}")
    idx = gen.index

    contract = {
        "model": meta["model_name"], "model_version": meta["model_version"],
        "feature_set": GOV["feature_set_label"], "trained_at": meta["trained_at"],
        "calibration": "sigmoid (Platt), 5-fold CalibratedClassifierCV, averaged",
        "features": features, "input_name": "input", "probability_output": prob_name,
        "cutoff": CUTOFF, "cutoff_rule": "PD >= cutoff -> refer for manual review",
        "training_ranges": {f: {k: float(ranges.loc[f, k]) for k in ("min", "median", "max")}
                            for f in features},
        "scope": ("Customer-level portfolio-risk model (Decision 18). For a new applicant the "
                  "score illustrates the mechanism; it is not a validated application-time "
                  "underwriting decision."),
        "data": "Synthetic. No real customer data.",
        "parity": {k: {"max_abs_diff": v["max_abs_diff"], "decision_flips": v["decision_flips"]}
                   for k, v in report["sets"].items()},
    }
    with open(os.path.join(OUT_DIR, "model_contract.json"), "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    report["passed"] = passed
    with open(os.path.join(OUT_DIR, "python_parity_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nPython parity: {'PASS' if passed else 'FAIL'} "
          f"(browser vectors: {len(idx):,}, incl. {vectors['n_at_cutoff']} on the decision boundary)")
    if not passed:
        os.remove(model_path)
        sys.exit("Parity failed: ONNX model removed, nothing to publish.")


if __name__ == "__main__":
    main()

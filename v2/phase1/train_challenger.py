"""
Sentinel Credit v2, Phase 1 — train and evaluate the PyTorch challenger (V2-02).

Plain English: build a neural-network challenger to the v1.1 XGBoost champion
and measure it on exactly the same footing:
- the same 13 features, the same customers, the same stratified and
  out-of-time splits (rebuilt and checked against the v1 files);
- the same Platt calibration (5-fold, sigmoid) and the same 0.0657 cutoff;
- v1's own metric code.

Steps, per split:
1. Pick the network size by 5-fold cross-validated AUC on the TRAINING data
   only (small fixed search).
2. Fit the calibrated model on the training data; score the held-out test set.
3. Log to MLflow experiment sentinel_v2_phase1_pytorch with the champion's tag
   set (dataset_version = the v1 dataset's real value).
4. Write results to the v2_results tables; save the model under v2/artifacts.

Then explain the stratified model with Captum Integrated Gradients and set
it beside the champion's SHAP importance. Nothing is promoted here; see
evaluate_challenger.py for the Approval Gate.

Usage (project root, database running, SENTINEL_DB_PASSWORD set):
    python v2/phase1/train_challenger.py
"""

# PyTorch must load before MLflow, XGBoost and pandas on Windows: imported
# later, its DLLs fail to initialise (WinError 1114). Found on the first run.
import torch  # noqa: I001
from captum.attr import IntegratedGradients

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
from torch_models import MLPClassifierTorch  # noqa: E402

CFG = c.V2


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def mlp(params):
    m = CFG["mlp"]
    return MLPClassifierTorch(hidden=tuple(params["hidden"]), dropout=params["dropout"],
                              learning_rate=m["learning_rate"], weight_decay=m["weight_decay"],
                              batch_size=m["batch_size"], max_epochs=m["max_epochs"],
                              validation_fraction=m["validation_fraction"], patience=m["patience"],
                              random_state=c.SEED)


def search(X, y):
    """Small fixed search, training data only: mean 5-fold CV AUC per candidate."""
    folds = StratifiedKFold(n_splits=c.N_SPLITS, shuffle=True, random_state=c.SEED)
    results = []
    for cand in CFG["mlp"]["candidates"]:
        aucs = []
        for tr, va in folds.split(X, y):
            model = mlp(cand).fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
        results.append({**cand, "cv_auc": round(float(np.mean(aucs)), 4), "cv_auc_sd": round(float(np.std(aucs)), 4)})
        print(f"    hidden={cand['hidden']} dropout={cand['dropout']}: CV AUC {results[-1]['cv_auc']:.4f} "
              f"(sd {results[-1]['cv_auc_sd']:.4f})")
    best = max(results, key=lambda r: r["cv_auc"])
    return best, results


def explain(base, X_train, X_test, features, run_label):
    """Captum Integrated Gradients on the uncalibrated network's logit."""
    e = CFG["explain"]
    sample = X_test.sample(n=min(e["sample_size"], len(X_test)), random_state=c.SEED)
    x = torch.tensor(base.scaler_.transform(sample.values.astype(np.float32)), dtype=torch.float32)
    baseline = torch.zeros_like(x)          # standardised training mean
    attr = IntegratedGradients(base.model_).attribute(x, baselines=baseline, n_steps=e["n_steps"])
    mean_abs = attr.abs().mean(dim=0).detach().numpy()
    mlp_imp = pd.DataFrame({"feature": features, "mean_abs_attribution": mean_abs})
    mlp_imp["share_of_total"] = (mlp_imp["mean_abs_attribution"] / mlp_imp["mean_abs_attribution"].sum()).round(4)
    mlp_imp["rank"] = mlp_imp["mean_abs_attribution"].rank(ascending=False, method="first").astype(int)
    mlp_imp = mlp_imp.assign(run_label=run_label, model="pytorch_mlp", method="captum_integrated_gradients")

    shap = pd.read_csv(os.path.join(c.ROOT, "reports", "shap", "global_feature_importance.csv"))
    shap = shap[shap["feature"].isin(features)]
    champ = pd.DataFrame({"feature": shap["feature"], "mean_abs_attribution": shap["mean_abs_shap"],
                          "share_of_total": (shap["pct_of_total"] / 100).round(4), "rank": shap["rank"]})
    champ = champ.assign(run_label=run_label, model="xgboost", method="shap_tree_v1_report")
    merged = mlp_imp.merge(champ[["feature", "rank"]], on="feature", suffixes=("_mlp", "_champion"))
    rho = spearmanr(merged["rank_mlp"], merged["rank_champion"]).statistic
    return pd.concat([mlp_imp, champ], ignore_index=True), round(float(rho), 4), merged.sort_values("rank_mlp")


def main():
    t0 = time.time()
    run_label = "phase1_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    art_dir = os.path.join(c.ROOT, CFG["artifacts_dir"], run_label)
    rep_dir = os.path.join(c.ROOT, CFG["reports_dir"])
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    print(f"Run label: {run_label}")

    df = c.load_population()
    S = c.splits(df)
    print(f"Population {len(df):,}; split checks: {S['_checks']}")
    seg_lookup = df.set_index(df["customer_id"].astype(str))[["thin_file_flag", "is_vulnerable"]]

    mlflow.set_tracking_uri("file:" + os.path.join(c.ROOT, "mlruns").replace("\\", "/"))
    mlflow.set_experiment(CFG["mlflow_experiment"])
    c.ensure_schema()

    comparison_rows, segment_frames, summary = [], [], {"run_label": run_label, "splits": {}}
    for split in ("stratified", "temporal"):
        d = S[split]
        print(f"\n== {split}: {len(d['X_train']):,} train / {len(d['X_test']):,} test ==")
        t = time.time()
        best, tried = search(d["X_train"], d["y_train"])
        print(f"  chosen: hidden={best['hidden']} dropout={best['dropout']}")
        model = CalibratedClassifierCV(mlp(best), method="sigmoid",
                                       cv=StratifiedKFold(n_splits=c.N_SPLITS, shuffle=True, random_state=c.SEED))
        model.fit(d["X_train"], d["y_train"])
        proba = model.predict_proba(d["X_test"])[:, 1]
        train_secs = round(time.time() - t, 1)
        m = c.evaluate(d["y_test"], proba)

        champ = c.champion_predictions(split).set_index("customer_id").loc[d["ids_test"]]
        cm = c.evaluate(d["y_test"], champ["proba"].values)
        print(f"  challenger Gini {m['gini']:.4f} AUC {m['auc']:.4f} ECE {m['ece']:.4f} recall@cutoff "
              f"{m['recall_at_cutoff']:.4f} | champion Gini {cm['gini']:.4f} ({train_secs}s)")

        frame = pd.DataFrame({"customer_id": d["ids_test"], c.TARGET_COLUMN: d["y_test"].values,
                              "proba_mlp": proba, "proba_champion": champ["proba"].values})
        frame = frame.join(seg_lookup, on="customer_id")
        seg_m, seg_c = c.segment_metrics(frame, "proba_mlp"), c.segment_metrics(frame, "proba_champion")

        with mlflow.start_run(run_name=f"pytorch_mlp_{split}_f13") as run:
            mlflow.set_tags({
                "dataset_version": CFG["dataset_version"], "feature_set": "13_features_v1_1",
                "feature_count": len(c.MODEL_FEATURES), "model_version": CFG["model_version"],
                "pipeline_stage": "v2_phase1_challenger", "split_strategy": split,
                "evaluation_split": split, "evaluation_logged": "True", "evaluation_cutoff": c.CUTOFF,
                "is_champion": "False", "git_commit": git_commit(), "decision_log": "v2 V2-02",
                "data_end_date": str(c.V1["data_generator"]["data_end_date"]),
                "origination_end_date": str(c.V1["data_generator"]["origination_end_date"]),
                "run_label": run_label})
            mlflow.log_params({"model_type": "pytorch_mlp", "hidden": str(best["hidden"]),
                               "dropout": best["dropout"], "learning_rate": CFG["mlp"]["learning_rate"],
                               "weight_decay": CFG["mlp"]["weight_decay"], "batch_size": CFG["mlp"]["batch_size"],
                               "calibration": "sigmoid_5fold", "cv_n_splits": c.N_SPLITS, "n_features": 13,
                               "n_train": len(d["X_train"]), "business_cutoff": c.CUTOFF, "random_seed": c.SEED,
                               "search": json.dumps(tried)})
            def gap(col, seg, table=seg_m):
                sub = table[table.segment == seg][col]
                return float(sub.max() - sub.min())
            mlflow.log_metrics({"eval_gini": m["gini"], "eval_auc": m["auc"], "eval_ks": m["ks"],
                                "eval_pr_auc": m["pr_auc"], "eval_brier": m["brier"], "eval_ece": m["ece"],
                                "eval_cutoff_recall": m["recall_at_cutoff"],
                                "eval_cutoff_precision": m["precision_at_cutoff"],
                                "eval_bad_rate_among_approved": m["bad_rate_not_reviewed"] or 0.0,
                                "eval_thin_file_fnr_gap": gap("false_negative_rate", "thin_file_flag"),
                                "eval_thin_file_fpr_gap": gap("false_positive_rate", "thin_file_flag"),
                                "eval_vulnerability_fnr_gap": gap("false_negative_rate", "is_vulnerable"),
                                "eval_vulnerability_fpr_gap": gap("false_positive_rate", "is_vulnerable"),
                                "cv_auc": best["cv_auc"], "train_seconds": train_secs})
            mlflow.sklearn.log_model(model, "model")
            run_id = run.info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", CFG["registered_model_name"])

        joblib.dump(model, os.path.join(art_dir, f"challenger_calibrated_{split}.joblib"))
        frame.to_csv(os.path.join(art_dir, f"test_predictions_{split}.csv"), index=False)
        for role, name, metrics, rid in (("challenger", "pytorch_mlp", m, run_id), ("champion", "xgboost", cm, None)):
            comparison_rows.append({"run_label": run_label, "split": split, "model": name, "role": role,
                                    "mlflow_run_id": rid, "dataset_version": CFG["dataset_version"], **metrics})
        for name, seg in (("pytorch_mlp", seg_m), ("xgboost", seg_c)):
            segment_frames.append(seg.assign(run_label=run_label, split=split, model=name))
        summary["splits"][split] = {"challenger": m, "champion": cm, "mlflow_run_id": run_id,
                                    "chosen": best, "search": tried, "train_seconds": train_secs}

    # Explanation model: the chosen network, uncalibrated, on the full stratified training data.
    d = S["stratified"]
    base = mlp(summary["splits"]["stratified"]["chosen"]).fit(d["X_train"], d["y_train"])
    attributions, rho, ranks = explain(base, d["X_train"], d["X_test"], c.MODEL_FEATURES, run_label)
    joblib.dump(base, os.path.join(art_dir, "challenger_explanation_base.joblib"))
    print(f"\nCaptum vs champion SHAP: Spearman rank correlation {rho}")
    print(ranks[["feature", "rank_mlp", "rank_champion"]].to_string(index=False))
    summary["explain"] = {"spearman_rank_vs_champion_shap": rho,
                          "ranking": ranks[["feature", "rank_mlp", "rank_champion"]].to_dict("records")}

    c.write_frame("v2_results.model_comparison", pd.DataFrame(comparison_rows))
    c.write_frame("v2_results.segment_metrics", pd.concat(segment_frames, ignore_index=True)[
        ["run_label", "split", "model", "segment", "segment_value", "n_customers", "n_defaults",
         "actual_default_rate", "avg_predicted_pd", "share_sent_to_review", "false_negative_rate",
         "false_positive_rate"]])
    c.write_frame("v2_results.feature_attributions", attributions[
        ["run_label", "model", "method", "feature", "mean_abs_attribution", "share_of_total", "rank"]])

    summary["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(rep_dir, f"{run_label}_challenger.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\nDone in {summary['total_seconds']}s. Wrote v2_results.* rows and "
          f"{os.path.relpath(rep_dir, c.ROOT)}/{run_label}_challenger.json")
    print(f"Next: python v2/phase1/evaluate_challenger.py --run-label {run_label}")


if __name__ == "__main__":
    main()

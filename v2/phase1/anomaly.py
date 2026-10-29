"""
Sentinel Credit v2, Phase 1 — autoencoder anomaly detection (V2-02).

Plain English: teach a network what NORMAL customers look like (customers in
the training data who did not default), then measure how badly it rebuilds
each held-out customer: the reconstruction error. A high error means "unlike
the normal customers it learned from". The question this answers honestly:
do unusual customers default more, and does that add anything the champion's
risk score does not already know?

Reported per split (each split trains its own autoencoder on its own training
non-defaulters, so the out-of-time test never leaks into training):
- AUC of reconstruction error as a default signal on its own;
- default rate by error decile (1 = most normal, 10 = most unusual);
- Spearman correlation between error and the champion's PD;
- among customers the champion does NOT send to review, the default rate in
  the most unusual decile against the rest (would the error catch misses?).

Usage: python v2/phase1/anomaly.py --run-label <label from train_challenger.py>
"""

# PyTorch must load first on Windows (DLL initialisation order, see train_challenger.py).
import torch  # noqa: F401,I001

import argparse
import json
import os
import sys
import time

import joblib
import mlflow
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
from torch_models import reconstruction_error, train_autoencoder  # noqa: E402

CFG = c.V2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-label", required=True)
    run_label = ap.parse_args().run_label
    art_dir = os.path.join(c.ROOT, CFG["artifacts_dir"], run_label)
    rep_dir = os.path.join(c.ROOT, CFG["reports_dir"])
    os.makedirs(art_dir, exist_ok=True)

    df = c.load_population()
    S = c.splits(df)
    mlflow.set_tracking_uri("file:" + os.path.join(c.ROOT, "mlruns").replace("\\", "/"))
    mlflow.set_experiment(CFG["mlflow_experiment"])
    a = CFG["autoencoder"]
    rows, summary = [], {"run_label": run_label, "splits": {}}

    for split in ("stratified", "temporal"):
        d = S[split]
        t = time.time()
        normal = d["X_train"][d["y_train"].values == 0]
        scaler, model, epochs, val_mse = train_autoencoder(
            normal, a["hidden"], a["learning_rate"], a["batch_size"], a["max_epochs"],
            a["validation_fraction"], a["patience"], c.SEED)
        err = reconstruction_error(scaler, model, d["X_test"])
        y = d["y_test"].values.astype(int)
        champ = c.champion_predictions(split).set_index("customer_id").loc[d["ids_test"], "proba"].values

        auc = roc_auc_score(y, err)
        rho = spearmanr(err, champ).statistic
        decile = pd.qcut(pd.Series(err).rank(method="first"), 10, labels=False) + 1
        frame = pd.DataFrame({"decile": decile, "y": y, "err": err, "pd": champ})
        table = frame.groupby("decile").agg(n_customers=("y", "size"), n_defaults=("y", "sum"),
                                             default_rate=("y", "mean"),
                                             mean_reconstruction_error=("err", "mean"),
                                             mean_champion_pd=("pd", "mean")).reset_index()
        not_reviewed = frame[frame["pd"] < c.CUTOFF]
        top = not_reviewed[not_reviewed["decile"] == 10]["y"]
        rest = not_reviewed[not_reviewed["decile"] < 10]["y"]
        secs = round(time.time() - t, 1)
        result = {"auc_error_alone": round(auc, 4), "gini_error_alone": round(2 * auc - 1, 4),
                  "spearman_error_vs_champion_pd": round(float(rho), 4),
                  "not_reviewed_top_decile_default_rate": round(float(top.mean()), 4) if len(top) else None,
                  "not_reviewed_top_decile_n": int(len(top)),
                  "not_reviewed_other_deciles_default_rate": round(float(rest.mean()), 4),
                  "epochs": epochs, "best_val_mse": round(float(val_mse), 4),
                  "n_normal_train": int(len(normal)), "seconds": secs,
                  "decile_default_rates": table["default_rate"].round(4).tolist()}
        summary["splits"][split] = result
        print(f"{split}: error AUC {result['auc_error_alone']}, Spearman vs champion PD "
              f"{result['spearman_error_vs_champion_pd']}, deciles {result['decile_default_rates']}")

        with mlflow.start_run(run_name=f"autoencoder_{split}_f13"):
            mlflow.set_tags({"dataset_version": CFG["dataset_version"], "feature_set": "13_features_v1_1",
                             "model_version": CFG["model_version"], "pipeline_stage": "v2_phase1_anomaly",
                             "split_strategy": split, "is_champion": "False", "decision_log": "v2 V2-02",
                             "run_label": run_label})
            mlflow.log_params({"model_type": "pytorch_autoencoder", "hidden": str(a["hidden"]),
                               "trained_on": "training non-defaulters", "random_seed": c.SEED})
            mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})
        joblib.dump({"scaler": scaler, "state_dict": model.state_dict(), "hidden": a["hidden"]},
                    os.path.join(art_dir, f"autoencoder_{split}.joblib"))
        rows.append(table.rename(columns={"decile": "error_decile"}).assign(run_label=run_label, split=split))

    out = pd.concat(rows, ignore_index=True)
    out["default_rate"] = out["default_rate"].round(4)
    out["mean_champion_pd"] = out["mean_champion_pd"].round(4)
    c.write_frame("v2_results.anomaly_deciles", out[["run_label", "split", "error_decile", "n_customers",
                                                    "n_defaults", "default_rate", "mean_reconstruction_error",
                                                    "mean_champion_pd"]])
    with open(os.path.join(rep_dir, f"{run_label}_anomaly.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"Wrote v2_results.anomaly_deciles and {CFG['reports_dir']}/{run_label}_anomaly.json")


if __name__ == "__main__":
    main()

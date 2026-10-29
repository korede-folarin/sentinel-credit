"""
Sentinel Credit v2, Phase 1 — compare with the champion and run the Approval
Gate, without promoting anything (V2-02).

Plain English: ask the same question v1's promotion workflow asks ("does the
challenger beat the champion on BOTH splits, and is monitoring evidence clean
enough to allow a change?"), and record the answer whatever it is, including
"challenger did not win". No model is promoted: the v2_results.gate_outcomes
table even refuses a row with promoted = true.

How, read-only against v1:
- pipeline/promote_champion.py: its functions find_champion_run,
  get_run_metrics and evaluate_promotion are imported and called. Its main()
  is NOT run, because that would register the challenger under the v1 model
  name and overwrite v1's promotion record.
- pipeline/approval_gate.py: load_monitoring_evidence and evaluate_gate are
  called. write_governance_status is NOT called (it writes v1's reporting).
- The promotion threshold stays v1's: champion_promotion.min_gini_delta is
  unset on purpose, so no Gini difference can count as a material win.

Writes: one v2_results.gate_outcomes row, one MLflow run in the v2
experiment, and v2/reports/phase1/<label>_gate.json.

Usage: python v2/phase1/evaluate_challenger.py --run-label <label>
"""

import argparse
import json
import os
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402

sys.path.insert(0, os.path.join(c.ROOT, "pipeline"))
import approval_gate as v1_gate  # noqa: E402  (read-only use)
import promote_champion as v1_promo  # noqa: E402  (read-only use)

CFG = c.V2


def challenger_runs(run_label):
    client = MlflowClient()
    exp = client.get_experiment_by_name(CFG["mlflow_experiment"])
    found = {}
    for split in ("stratified", "temporal"):
        runs = client.search_runs([exp.experiment_id],
                                  filter_string=(f"tags.run_label = '{run_label}' and "
                                                 f"tags.split_strategy = '{split}' and "
                                                 "params.model_type = 'pytorch_mlp'"))
        if len(runs) != 1:
            raise SystemExit(f"Expected one challenger run for {split} in {run_label}, found {len(runs)}")
        found[split] = runs[0].info.run_id
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-label", required=True)
    run_label = ap.parse_args().run_label
    mlflow.set_tracking_uri("file:" + os.path.join(c.ROOT, "mlruns").replace("\\", "/"))

    chall = challenger_runs(run_label)
    split_results, champ_runs = {}, {}
    for split in ("stratified", "temporal"):
        champ_runs[split], selection = v1_promo.find_champion_run(split)
        champ = v1_promo.get_run_metrics(champ_runs[split])
        chal = v1_promo.get_run_metrics(chall[split])
        split_results[split] = {**v1_promo.evaluate_promotion(champ, chal), "champion_run_selection": selection,
                                "champion_run_id": champ_runs[split], "challenger_run_id": chall[split]}

    both_win = all(r["material_win"] for r in split_results.values())
    evidence = v1_gate.load_monitoring_evidence()
    gate = v1_gate.evaluate_gate(evidence)
    min_delta = v1_promo.MIN_GINI_DELTA
    recommended = bool(both_win and gate["gate_status"] == "PASS")

    ds, dt = split_results["stratified"]["delta_gini"], split_results["temporal"]["delta_gini"]
    if min_delta is None:
        verdict = ("challenger did not win: no promotion threshold is set (v1 champion_promotion.min_gini_delta "
                   "is unset on purpose), so no Gini difference can count as a material win")
        if ds <= 0 and dt <= 0:
            verdict = ("challenger did not win: lower Gini than the champion on both splits; "
                       + verdict.split(": ", 1)[1])
    elif not both_win:
        verdict = "challenger did not win: no material Gini improvement on both splits"
    elif gate["gate_status"] != "PASS":
        verdict = "challenger won on Gini but the Approval Gate blocked it"
    else:
        verdict = "promotion would be supported; not applied (Phase 1 never promotes)"
    outcome = (f"{verdict}. Gini difference (challenger minus champion): stratified {ds:+.4f}, "
               f"temporal {dt:+.4f}. Approval Gate: {gate['gate_status']}.")
    print(outcome)

    detail = {"splits": split_results, "approval_gate": gate, "min_gini_delta": min_delta,
              "registered_model_name": CFG["registered_model_name"],
              "note": "Read-only use of v1 promotion and gate logic; nothing promoted or registered under v1."}
    row = pd.DataFrame([{
        "run_label": run_label, "challenger_model": "pytorch_mlp",
        "champion_run_stratified": champ_runs["stratified"], "champion_run_temporal": champ_runs["temporal"],
        "challenger_run_stratified": chall["stratified"], "challenger_run_temporal": chall["temporal"],
        "delta_gini_stratified": ds, "delta_gini_temporal": dt, "min_gini_delta": min_delta,
        "material_win_both_splits": both_win, "approval_gate_status": gate["gate_status"],
        "promotion_recommended": recommended, "promoted": False, "outcome": outcome, "detail": detail}])
    c.write_frame("v2_results.gate_outcomes", row)

    mlflow.set_experiment(CFG["mlflow_experiment"])
    with mlflow.start_run(run_name="promotion_evaluation"):
        mlflow.set_tags({"run_label": run_label, "pipeline_stage": "v2_phase1_promotion_evaluation",
                         "dataset_version": CFG["dataset_version"], "decision_log": "v2 V2-02",
                         "approval_gate_status": gate["gate_status"], "promoted": "False",
                         "outcome": outcome[:4900]})
        mlflow.log_metrics({"delta_gini_stratified": ds, "delta_gini_temporal": dt,
                            "material_win_both_splits": int(both_win), "promotion_recommended": int(recommended)})
    with open(os.path.join(c.ROOT, CFG["reports_dir"], f"{run_label}_gate.json"), "w", encoding="utf-8") as f:
        json.dump({"outcome": outcome, **detail}, f, indent=1, default=str)
    print(f"Logged to v2_results.gate_outcomes, MLflow and {CFG['reports_dir']}/{run_label}_gate.json")


if __name__ == "__main__":
    main()

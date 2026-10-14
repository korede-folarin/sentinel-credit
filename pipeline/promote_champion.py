"""
Sentinel Credit — Champion/Challenger Promotion Workflow

Purpose
-------
Compares a challenger model against the currently governed
champion, on both held-out splits, using the same metrics
evaluate_models.py already logs — and produces a promotion
recommendation. It does not promote anything by itself.

This is the same "evidence, not silent action" discipline used
throughout the project (select_business_cutoff.py never writes
config.yaml; recalibrate_threshold.py was retired specifically
because it blurred that line). Promotion changes what model
actually makes lending-relevant decisions — of everything this
project automates, this is the last thing that should happen
without an explicit, confirmed human action.

Registry model
--------------
Uses MLflow's alias-based Model Registry (not the deprecated
stage-based API), since mlflow>=2.13.0 is already pinned project-
wide. Two aliases on one registered model name
(config.yaml champion_promotion.registered_model_name):
  "champion"   — the model currently governing real decisions.
  "challenger" — a candidate being evaluated against it.
Registering a run's model is idempotent here: if a run_id is
already registered as a version, that version is reused rather
than creating a duplicate.

Promotion rule
--------------
Recommends promotion only if the challenger beats the champion's
Gini by more than champion_promotion.min_gini_delta on BOTH
splits — stratified and temporal — not just one. This mirrors the
project's standing concern about temporal stability (Decision 16,
18): a challenger that wins on stratified but not temporal is
exactly the pattern that would have flagged XGBoost's own
generalization if it had gone the other way. PR-AUC and KS are
reported as corroborating evidence, not gating criteria — matching
how the current champion was itself selected on combined evidence,
not a single dominant metric (training_summary.md, Section 7).

min_gini_delta is illustrative (see config.yaml comment) — no real
promotion has happened yet to calibrate a defensible threshold
against, the same honest position taken for the review-capacity
and cost-assumption thresholds elsewhere in this project.

Applying a promotion (--apply) always requires typed confirmation,
and refuses outright if the promotion criteria are not met —
--apply cannot force a promotion the evidence does not support.

See docs/engineering_decision_log.md — Decision 4 (champion
selection basis), Decision 8 (MLflow tracking), Decision 16/18
(temporal stability concern this reuses).
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime

import mlflow
import yaml
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "promotion")

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

GOV_CFG = CONFIG["model_governance"]
MLFLOW_CFG = CONFIG["mlflow"]
PROMO_CFG = CONFIG.get("champion_promotion", {})

CHAMPION_MODEL = GOV_CFG["champion_model"]
FEATURE_SET_LABEL = GOV_CFG["feature_set_label"]
DATASET_VERSION = GOV_CFG["dataset_version"]
MODEL_VERSION = GOV_CFG["model_version"]

MLFLOW_TRACKING_URI = MLFLOW_CFG["tracking_uri"]
MLFLOW_EXPERIMENT = MLFLOW_CFG["experiment_name"]

REGISTERED_MODEL_NAME = PROMO_CFG.get(
    "registered_model_name", "sentinel_credit_champion"
)
MIN_GINI_DELTA = PROMO_CFG.get("min_gini_delta")

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


# Two champion reruns count as "the same Gini" when they agree to
# the 4 decimal places evaluate_models.py logs eval_gini at. This is
# rounding precision, not a business threshold.
RERUN_GINI_TOLERANCE = 0.0001


def find_champion_run(split: str) -> tuple[str, dict]:
    """
    Finds the current champion's run for one split, via the
    is_champion tag train_baseline_models.py sets, combined with the
    governance identity every other script in this project checks
    against.

    Several matching runs are expected: re-running the same champion
    should reproduce it. Only runs evaluate_models.py has scored
    (eval_gini logged) are considered. Selection rule, comparing the
    latest evaluated run with the one before it:
      - same Gini (within rounding): use the latest.
      - latest Gini LOWER: roll back — use the previous run. A worse
        rerun of the same champion means something changed (data,
        code, environment) and is not adopted silently.
      - latest Gini HIGHER: use the latest, flagged, since reruns of
        the same champion are expected to match.
    Returns (run_id, selection) where selection records which rule
    applied, for the recommendation file. See Decision 21 addendum.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {MLFLOW_EXPERIMENT!r} not found.")

    filter_string = (
        f"params.model_type = '{CHAMPION_MODEL}' and "
        f"tags.split_strategy = '{split}' and "
        f"tags.feature_set = '{FEATURE_SET_LABEL}' and "
        f"tags.dataset_version = '{DATASET_VERSION}' and "
        f"tags.model_version = '{MODEL_VERSION}' and "
        f"tags.is_champion = 'True'"
    )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    if not runs:
        raise RuntimeError(
            f"No champion run found for split={split} matching current "
            f"governance identity (feature_set={FEATURE_SET_LABEL}, "
            f"dataset_version={DATASET_VERSION}, model_version={MODEL_VERSION})."
        )

    evaluated = [r for r in runs if "eval_gini" in r.data.metrics]
    skipped = [r.info.run_id for r in runs if "eval_gini" not in r.data.metrics]
    if not evaluated:
        raise RuntimeError(
            f"{len(runs)} champion run(s) found for split={split}, but none "
            "has eval_* metrics. Run evaluate_models.py first."
        )

    latest = evaluated[0]
    selection = {
        "n_matching_runs": len(runs),
        "n_evaluated_runs": len(evaluated),
        "skipped_unevaluated_run_ids": skipped,
        "latest_run_id": latest.info.run_id,
        "latest_gini": latest.data.metrics["eval_gini"],
    }

    if len(evaluated) == 1:
        selection["rule"] = "single_evaluated_run"
        chosen = latest
    else:
        previous = evaluated[1]
        delta = latest.data.metrics["eval_gini"] - previous.data.metrics["eval_gini"]
        selection.update({
            "previous_run_id": previous.info.run_id,
            "previous_gini": previous.data.metrics["eval_gini"],
            "latest_minus_previous_gini": round(delta, 4),
        })
        if abs(delta) <= RERUN_GINI_TOLERANCE:
            selection["rule"] = "reruns_match_latest_used"
            chosen = latest
        elif delta < 0:
            selection["rule"] = "ROLLED_BACK_latest_worse"
            chosen = previous
            print(
                f"  WARNING ({split}): latest champion run "
                f"{latest.info.run_id[:8]} has Gini "
                f"{latest.data.metrics['eval_gini']:.4f}, LOWER than the previous "
                f"run {previous.info.run_id[:8]} "
                f"({previous.data.metrics['eval_gini']:.4f}). Rolled back to the "
                "previous run. Investigate why a rerun of the same champion "
                "got worse before relying on either."
            )
        else:
            selection["rule"] = "latest_better_flagged"
            chosen = latest
            print(
                f"  NOTE ({split}): latest champion run {latest.info.run_id[:8]} "
                f"has Gini {delta:+.4f} above the previous run. Using the latest, "
                "but reruns of the same champion are expected to match."
            )

    print(f"  split={split}: {len(runs)} matching champion run(s), "
          f"{len(evaluated)} evaluated; rule: {selection['rule']}; "
          f"using {chosen.info.run_id[:8]}.")
    selection["chosen_run_id"] = chosen.info.run_id
    return chosen.info.run_id, selection


def get_run_metrics(run_id: str) -> dict:
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    required = ["eval_gini", "eval_auc", "eval_ks", "eval_pr_auc"]
    missing = [m for m in required if m not in metrics]
    if missing:
        raise ValueError(
            f"Run {run_id} is missing metrics {missing} — has "
            "evaluate_models.py been run against it? (training alone "
            "logs 'gini'/'auc' from the training-time test split, not "
            "the eval_* metrics evaluate_models.py adds afterward.)"
        )
    return {
        "run_id": run_id,
        "model_type": run.data.params.get("model_type", "unknown"),
        "gini": metrics["eval_gini"],
        "auc": metrics["eval_auc"],
        "ks": metrics["eval_ks"],
        "pr_auc": metrics["eval_pr_auc"],
    }


def register_run_model(run_id: str, alias: str) -> int:
    """
    Registers a run's model artifact under REGISTERED_MODEL_NAME
    if not already registered, then sets the given alias on it.
    Idempotent on run_id — reuses an existing version rather than
    creating a duplicate if this run has already been registered
    (e.g. re-running this script against the same champion run).
    """
    client = MlflowClient()

    try:
        existing_versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}' and run_id='{run_id}'"
        )
    except Exception:
        existing_versions = []

    if existing_versions:
        version = existing_versions[0].version
    else:
        try:
            client.get_registered_model(REGISTERED_MODEL_NAME)
        except Exception:
            client.create_registered_model(REGISTERED_MODEL_NAME)
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        version = result.version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version)
    return int(version)


def check_approval_gate() -> dict:
    """
    Runs pipeline/approval_gate.py and returns its result. This is
    the one place --apply's material-win criteria are not the only
    gate — active drift or stale monitoring evidence blocks a
    promotion regardless of how strong the Gini improvement looks.
    Run as a subprocess (same pattern as monitor_drift_dag.py
    invoking monitor_drift.py) so this reuses the exact gate logic
    a human running it directly would see, not a duplicated check.
    """
    gate_script = os.path.join(PROJECT_ROOT, "pipeline", "approval_gate.py")
    result = subprocess.run(
        [sys.executable, gate_script, "--quiet"],
        capture_output=True, text=True,
    )
    result_path = os.path.join(PROJECT_ROOT, "reports", "governance", "approval_gate_result.json")
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"gate_status": "BLOCK", "checks": [
        {"check": "approval_gate_ran", "passed": False,
         "detail": f"approval_gate.py exited {result.returncode} with no result file. "
                    f"stderr: {result.stderr[:500]}"}
    ]}


def check_validity_override(decision_ref: str, evidence_path: str,
                            run_id_stratified: str, run_id_temporal: str) -> dict:
    """
    Second promotion path, for a challenger that removes a feature
    found to be INVALID (e.g. leaking the outcome). Removing leakage
    usually costs a little apparent Gini, so the Gini-win rule above
    can never approve it (Decision 26). This path replaces "material
    Gini win" with "the Gini loss is not distinguishable from zero":
    on BOTH splits, the 95% paired-bootstrap CI of (challenger minus
    champion) Gini must reach zero or above. No new threshold is
    introduced; the CI comes from the evidence file.

    Refuses (SystemExit) rather than degrading silently if:
      - decision_ref is not a heading in the decision log,
      - the evidence file is missing, or describes different runs.
    The approval gate and typed PROMOTE still apply on --apply.
    """
    log_path = os.path.join(PROJECT_ROOT, "docs", "engineering_decision_log.md")
    with open(log_path, "r", encoding="utf-8") as f:
        headings = [line for line in f if line.startswith("## ")]
    if not any(line.startswith(f"## {decision_ref} ") for line in headings):
        raise SystemExit(
            f"--validity-override {decision_ref!r}: no heading '## {decision_ref} ...' "
            "in docs/engineering_decision_log.md. Write the decision first."
        )
    if not evidence_path or not os.path.exists(evidence_path):
        raise SystemExit("--validity-override requires --validity-evidence pointing at "
                         "promotion_evidence.json from evaluate_feature_removal.py.")
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    expected = {"stratified": run_id_stratified, "temporal": run_id_temporal}
    if evidence.get("challenger_run_ids") != expected:
        raise SystemExit(
            "Evidence file describes challenger runs "
            f"{evidence.get('challenger_run_ids')}, not {expected}. "
            "Re-run evaluate_feature_removal.py --log-challenger and use its run IDs."
        )

    splits = {}
    for split in ("stratified", "temporal"):
        d = evidence["gini_difference_challenger_minus_champion_arm"][split]
        splits[split] = {**d, "non_inferior": bool(d["ci95"][1] >= 0)}
    return {
        "decision_ref": decision_ref,
        "evidence_path": os.path.relpath(evidence_path, PROJECT_ROOT),
        "removed_feature": evidence.get("removed_feature"),
        "rule": "95% CI of Gini difference reaches >= 0 on both splits",
        "splits": splits,
        "passed": all(s["non_inferior"] for s in splits.values()),
    }


def evaluate_promotion(champion: dict, challenger: dict) -> dict:
    delta_gini = round(challenger["gini"] - champion["gini"], 4)
    delta_auc = round(challenger["auc"] - champion["auc"], 4)
    delta_ks = round(challenger["ks"] - champion["ks"], 4)
    delta_pr_auc = round(challenger["pr_auc"] - champion["pr_auc"], 4)

    material_win = (
        MIN_GINI_DELTA is not None and delta_gini > MIN_GINI_DELTA
    )

    return {
        "champion_gini": champion["gini"],
        "challenger_gini": challenger["gini"],
        "delta_gini": delta_gini,
        "delta_auc": delta_auc,
        "delta_ks": delta_ks,
        "delta_pr_auc": delta_pr_auc,
        "material_win": material_win,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenger-run-id-stratified", required=True,
        help="MLflow run_id for the challenger's stratified evaluation.",
    )
    parser.add_argument(
        "--challenger-run-id-temporal", required=True,
        help="MLflow run_id for the challenger's temporal evaluation.",
    )
    parser.add_argument(
        "--validity-override", metavar="DECISION_REF",
        help=(
            "Promote a challenger that removes an INVALID feature (e.g. "
            "leakage) without a Gini win. Value is the decision-log "
            "heading authorising it, e.g. 'Decision 27'; it must exist in "
            "docs/engineering_decision_log.md. Requires --validity-evidence. "
            "Allowed only if, on BOTH splits, the 95%% CI of the Gini "
            "difference (challenger minus champion) reaches zero or above. "
            "Approval gate and typed PROMOTE still apply."
        ),
    )
    parser.add_argument(
        "--validity-evidence", metavar="PATH",
        help="promotion_evidence.json written by models/evaluate_feature_removal.py.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help=(
            "Set the 'champion' alias to the challenger if BOTH splits "
            "show a material win. Refuses if criteria are not met, "
            "regardless of this flag. Always requires typed confirmation."
        ),
    )
    args = parser.parse_args()

    print("=" * 64)
    print("SENTINEL CREDIT — CHAMPION/CHALLENGER PROMOTION EVIDENCE")
    print("=" * 64)

    if MIN_GINI_DELTA is None:
        print(
            "\n  NOTE: champion_promotion.min_gini_delta is not set in "
            "config.yaml. Deltas will be computed and reported, but "
            "material_win cannot be determined without a threshold — "
            "no promotion will ever be recommended until this is set. "
            "See config.yaml for why an illustrative default was not "
            "invented here."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nRegistered model name: {REGISTERED_MODEL_NAME}")
    print(f"Current champion: {CHAMPION_MODEL} "
          f"(feature_set={FEATURE_SET_LABEL}, dataset={DATASET_VERSION})")

    champion_run_stratified, selection_stratified = find_champion_run("stratified")
    champion_run_temporal, selection_temporal = find_champion_run("temporal")

    champion_metrics_stratified = get_run_metrics(champion_run_stratified)
    champion_metrics_temporal = get_run_metrics(champion_run_temporal)
    challenger_metrics_stratified = get_run_metrics(args.challenger_run_id_stratified)
    challenger_metrics_temporal = get_run_metrics(args.challenger_run_id_temporal)

    result_stratified = evaluate_promotion(
        champion_metrics_stratified, challenger_metrics_stratified
    )
    result_temporal = evaluate_promotion(
        champion_metrics_temporal, challenger_metrics_temporal
    )

    print("\n" + "-" * 64)
    print("COMPARISON")
    print("-" * 64)
    for split, champ, chall, res in [
        ("stratified", champion_metrics_stratified, challenger_metrics_stratified, result_stratified),
        ("temporal", champion_metrics_temporal, challenger_metrics_temporal, result_temporal),
    ]:
        print(f"\n{split}")
        print(f"  Champion  ({champ['model_type']}, run {champ['run_id'][:8]}...): "
              f"Gini={champ['gini']:.4f} AUC={champ['auc']:.4f} "
              f"KS={champ['ks']:.4f} PR-AUC={champ['pr_auc']:.4f}")
        print(f"  Challenger({chall['model_type']}, run {chall['run_id'][:8]}...): "
              f"Gini={chall['gini']:.4f} AUC={chall['auc']:.4f} "
              f"KS={chall['ks']:.4f} PR-AUC={chall['pr_auc']:.4f}")
        print(f"  Delta Gini: {res['delta_gini']:+.4f}  "
              f"(material win: {res['material_win']})")

    both_material = result_stratified["material_win"] and result_temporal["material_win"]

    validity = None
    if args.validity_override:
        validity = check_validity_override(
            args.validity_override, args.validity_evidence,
            args.challenger_run_id_stratified, args.challenger_run_id_temporal,
        )

    print("\n" + "-" * 64)
    if both_material:
        promotion_supported, promotion_path = True, "gini_win"
        print("RECOMMENDATION: promotion supported — material Gini improvement "
              "on BOTH splits.")
    elif validity is not None:
        promotion_supported = validity["passed"]
        promotion_path = "validity_override"
        print(f"VALIDITY OVERRIDE ({validity['decision_ref']}): removes "
              f"'{validity['removed_feature']}' without a Gini win.")
        for split in ("stratified", "temporal"):
            s = validity["splits"][split]
            print(f"  {split}: Gini difference {s['point']:+.4f}, 95% CI "
                  f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] -> "
                  f"{'loss could be zero' if s['non_inferior'] else 'loss is REAL'}")
        if promotion_supported:
            print("RECOMMENDATION: promotion supported under validity override — "
                  "the Gini loss is not distinguishable from zero on either split.")
        else:
            print("RECOMMENDATION: promotion NOT supported — validity override "
                  "refused: the Gini loss is distinguishable from zero on at "
                  "least one split.")
    else:
        promotion_supported, promotion_path = False, "gini_win"
        print("RECOMMENDATION: promotion NOT supported — requires a material "
              "Gini improvement on both stratified and temporal, not just one.")
    print("-" * 64)

    challenger_version = register_run_model(
        args.challenger_run_id_stratified, CHALLENGER_ALIAS
    )
    print(f"\nChallenger registered: {REGISTERED_MODEL_NAME} "
          f"version {challenger_version}, alias '{CHALLENGER_ALIAS}'")

    recommendation = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "registered_model_name": REGISTERED_MODEL_NAME,
        "champion_model_type": CHAMPION_MODEL,
        "feature_set_label": FEATURE_SET_LABEL,
        "dataset_version": DATASET_VERSION,
        "min_gini_delta_threshold": MIN_GINI_DELTA,
        "stratified": {
            "champion_run_id": champion_run_stratified,
            "champion_run_selection": selection_stratified,
            "challenger_run_id": args.challenger_run_id_stratified,
            **result_stratified,
        },
        "temporal": {
            "champion_run_id": champion_run_temporal,
            "champion_run_selection": selection_temporal,
            "challenger_run_id": args.challenger_run_id_temporal,
            **result_temporal,
        },
        "promotion_recommended": promotion_supported,
        "promotion_path": promotion_path,
        "validity_override": validity,
        "applied": False,
        "note": (
            "Promotion changes which model governs real decisions. This "
            "script never applies a promotion without --apply AND typed "
            "confirmation, and refuses --apply outright if the criteria "
            "above are not met."
        ),
    }

    if args.apply:
        if not promotion_supported:
            print("\n--apply was set, but promotion criteria are not met. "
                  "Refusing — no alias change made.")
        else:
            gate_result = check_approval_gate()
            if gate_result["gate_status"] != "PASS":
                print(
                    "\n--apply was set and Gini criteria were met, but the "
                    "APPROVAL GATE returned BLOCK — refusing regardless. "
                    "A material Gini improvement does not override active "
                    "drift or stale monitoring evidence. See "
                    "reports/governance/approval_gate_result.json for detail."
                )
                for check in gate_result.get("checks", []):
                    if check["passed"] is False:
                        print(f"  BLOCKED: {check['check']} — {check.get('detail','')}")
                recommendation["applied"] = False
                recommendation["approval_gate"] = gate_result
            else:
                confirm = input(
                    f"\nApproval gate: PASS. Type 'PROMOTE' to set the "
                    f"'{CHAMPION_ALIAS}' alias to challenger run "
                    f"{args.challenger_run_id_stratified[:8]}... "
                    f"(currently: {champion_run_stratified[:8]}...): "
                ).strip()
                if confirm == "PROMOTE":
                    client = MlflowClient()
                    client.set_registered_model_alias(
                        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, challenger_version
                    )
                    # Record WHY this version became champion on the
                    # registry entry itself, not only in the JSON file.
                    version_tags = {
                        "promotion_path": promotion_path,
                        "promoted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "replaced_champion_run_stratified": champion_run_stratified,
                    }
                    if validity is not None:
                        version_tags.update({
                            "decision_ref": validity["decision_ref"],
                            "removed_feature": validity["removed_feature"],
                        })
                    for key, value in version_tags.items():
                        client.set_model_version_tag(
                            REGISTERED_MODEL_NAME, challenger_version, key, str(value)
                        )
                    recommendation["applied"] = True
                    recommendation["approval_gate"] = gate_result
                    print(f"\nApplied: {REGISTERED_MODEL_NAME} version "
                          f"{challenger_version} is now aliased '{CHAMPION_ALIAS}'.")
                    print("NOTE: config.yaml's model_governance.champion_model "
                          "and champion_metadata.json are NOT updated by this "
                          "script — update those manually to match, the same "
                          "'evidence generated, human applies' boundary used "
                          "for business_cutoff.")
                else:
                    print("\nConfirmation text did not match 'PROMOTE' exactly — "
                          "no alias change made.")

    recommendation_path = os.path.join(OUTPUT_DIR, "promotion_recommendation.json")
    with open(recommendation_path, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)
    print(f"\nSaved: {recommendation_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()

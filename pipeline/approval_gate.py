"""
Sentinel Credit — Automated Approval Gate

Purpose
-------
Reads evidence already produced by pipeline/monitor_drift.py and
decides PASS or BLOCK for any action that changes what governs
real decisions (a champion promotion, most immediately). This is
not a new check — it consumes existing evidence and applies a
governed decision rule to it, the same way promote_champion.py
consumes evaluate_models.py's metrics rather than recomputing
them.

Gate criteria
-------------
1. Monitoring evidence must exist and be recent — stale monitoring
   data approving a promotion would defeat the point of having
   monitoring at all. max_monitoring_age_hours has a real,
   defensible default (48h) tied to monitor_drift_dag.py's actual
   @daily schedule: 48h tolerates one missed run before the gate
   refuses to trust the evidence, not an arbitrary number. Unlike
   min_gini_delta or the cost-assumption thresholds elsewhere in
   this project, this one IS defaulted, because it is derived
   directly from an operational fact (the schedule) already
   decided in Decision 20, not asserting a business judgement that
   hasn't been made yet.
2. Zero active drift/data-quality alerts in that evidence.
3. Portfolio bad-rate shift is reported for visibility but does
   NOT gate — no defensible threshold for "how much bad-rate
   drift is acceptable" has been established anywhere in this
   project, and inventing one here would be the same mistake
   already declined for cost_assumptions and min_gini_delta.

A PASS here is necessary, not sufficient, for promotion —
promote_champion.py's own Gini-improvement criteria still apply
independently. A BLOCK here overrides everything else: an
otherwise-material Gini improvement does not get applied during
active drift.

See docs/engineering_decision_log.md — Decision 20 (monitoring),
Decision 21 (promotion workflow this gates).
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITORING_METADATA_PATH = os.path.join(
    PROJECT_ROOT, "reports", "monitoring", "monitoring_run_metadata.json"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "governance")

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

GATE_CFG = CONFIG.get("approval_gate", {})
MAX_MONITORING_AGE_HOURS = GATE_CFG.get("max_monitoring_age_hours", 48)


def load_monitoring_evidence() -> dict:
    if not os.path.exists(MONITORING_METADATA_PATH):
        return None
    with open(MONITORING_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_gate(evidence: dict) -> dict:
    checks = []

    if evidence is None:
        checks.append({
            "check": "monitoring_evidence_exists",
            "passed": False,
            "detail": (
                f"No monitoring evidence found at {MONITORING_METADATA_PATH}. "
                "Run pipeline/monitor_drift.py first."
            ),
        })
        return {"gate_status": "BLOCK", "checks": checks}

    checks.append({
        "check": "monitoring_evidence_exists",
        "passed": True,
        "detail": f"Found: {MONITORING_METADATA_PATH}",
    })

    generated_at = datetime.fromisoformat(evidence["generated_at"])
    age = datetime.now() - generated_at
    age_hours = age.total_seconds() / 3600
    age_ok = age_hours <= MAX_MONITORING_AGE_HOURS

    checks.append({
        "check": "monitoring_evidence_recent",
        "passed": age_ok,
        "detail": (
            f"Monitoring evidence is {age_hours:.1f}h old "
            f"(max allowed: {MAX_MONITORING_AGE_HOURS}h, generated {generated_at.isoformat()})."
            + ("" if age_ok else " STALE — rerun pipeline/monitor_drift.py before proceeding.")
        ),
    })

    n_alerts = evidence["n_features_with_alerts"]
    n_total = evidence["n_features_total"]
    no_alerts = n_alerts == 0

    checks.append({
        "check": "no_active_drift_alerts",
        "passed": no_alerts,
        "detail": (
            f"{n_alerts} of {n_total} features currently alerting."
            + ("" if no_alerts else " See reports/monitoring/feature_drift.csv for detail.")
        ),
    })

    # Reported, not gated — see module docstring for why.
    bad_rate_delta = evidence.get("target_rate_shift", {}).get("absolute_delta")
    checks.append({
        "check": "portfolio_bad_rate_shift (informational only, not gating)",
        "passed": None,
        "detail": (
            f"Absolute bad-rate delta vs training reference: {bad_rate_delta:+.4f}"
            if bad_rate_delta is not None else "No bad-rate comparison available."
        ),
    })

    gating_checks = [c for c in checks if c["passed"] is not None]
    gate_status = "PASS" if all(c["passed"] for c in gating_checks) else "BLOCK"

    return {"gate_status": gate_status, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress printed output — used when called as a library check "
             "from another script (e.g. promote_champion.py).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    evidence = load_monitoring_evidence()
    result = evaluate_gate(evidence)

    result["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    result["max_monitoring_age_hours"] = MAX_MONITORING_AGE_HOURS

    result_path = os.path.join(OUTPUT_DIR, "approval_gate_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if not args.quiet:
        print("=" * 62)
        print("SENTINEL CREDIT — AUTOMATED APPROVAL GATE")
        print("=" * 62)
        for check in result["checks"]:
            status = (
                "PASS" if check["passed"] is True
                else "BLOCK" if check["passed"] is False
                else "INFO"
            )
            print(f"  [{status:>5}] {check['check']}")
            print(f"           {check['detail']}")
        print("\n" + "-" * 62)
        print(f"GATE STATUS: {result['gate_status']}")
        print("-" * 62)
        print(f"\nSaved: {result_path}")
        print("=" * 62)

    sys.exit(0 if result["gate_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

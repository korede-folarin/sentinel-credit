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
import traceback
import warnings
from datetime import datetime, timedelta

import psycopg2
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
DB_CFG = CONFIG.get("database", {})


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
        return {
            "gate_status": "BLOCK", "checks": checks,
            "n_alerts": None, "n_features_total": None,
            "monitoring_age_hours": None,
        }

    checks.append({
        "check": "monitoring_evidence_exists",
        "passed": True,
        "detail": f"Found: {MONITORING_METADATA_PATH}",
    })

    # Timezone-aware comparison: monitor_drift.py writes generated_at
    # with a UTC offset, so evidence produced in the UTC Airflow
    # container and a gate run on the Windows host agree on its age.
    # A naive value (metadata written before this change) is read as
    # this machine's local time, the old behaviour. See Decision 24.
    generated_at = datetime.fromisoformat(evidence["generated_at"])
    if generated_at.tzinfo is None:
        generated_at = generated_at.astimezone()
    age = datetime.now().astimezone() - generated_at
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

    return {
        "gate_status": gate_status,
        "checks": checks,
        # Real numbers, not buried in the printed detail strings —
        # write_governance_status() reads these directly rather
        # than parsing prose back apart, which would be fragile.
        "n_alerts": n_alerts,
        "n_features_total": n_total,
        "monitoring_age_hours": round(age_hours, 2),
    }


def write_governance_status(result: dict) -> bool:
    """
    Writes the gate result this run already computed into
    reporting.governance_status_log, so Power BI (or any other
    live reporting tool) can read the real, current gate status
    through the same DirectQuery connection already used for
    every other reporting view, instead of a manually typed
    value standing in for it.

    This is purely additive: it runs after evaluate_gate() has
    already decided PASS or BLOCK, and it never changes that
    decision. If the database is unreachable, this prints a
    warning and returns False rather than failing the whole gate
    check — the JSON evidence file main() already wrote is the
    thing promote_champion.py actually depends on, this database
    write is a reporting convenience layered on top of it, not a
    new dependency for the gate itself.
    """
    if not DB_CFG:
        print("  Skipping governance status write — no database "
              "section in config.yaml.")
        return False

    db_host = os.environ.get("SENTINEL_DB_HOST", DB_CFG.get("host", "localhost"))
    db_password = os.environ.get("SENTINEL_DB_PASSWORD", "")

    try:
        conn = psycopg2.connect(
            host=db_host, port=DB_CFG.get("port", 5432),
            dbname=DB_CFG.get("name", "sentinel_credit"),
            user="postgres", password=db_password,
        )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO reporting.governance_status_log
                (gate_status, evaluated_at, n_alerts, n_features_total,
                 monitoring_age_hours, max_monitoring_age_hours)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                result["gate_status"],
                result["evaluated_at"],
                result["n_alerts"],
                result["n_features_total"],
                result["monitoring_age_hours"],
                result["max_monitoring_age_hours"],
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        print("  Governance status written to "
              "reporting.governance_status_log.")
        return True
    except Exception as error:
        print(f"  WARNING: could not write governance status to the "
              f"database ({error}). The evidence file was still saved "
              f"correctly — this only affects the live Power BI feed.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress printed output — used when called as a library check "
             "from another script (e.g. promote_champion.py).",
    )
    parser.add_argument(
        "--require-db-write", action="store_true",
        help="Exit 2 if the governance status cannot be written to the "
             "database. Used by the Airflow DAG, where a silently missing "
             "row would only surface later as a STALE badge.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    evidence = load_monitoring_evidence()
    result = evaluate_gate(evidence)

    # Offset-aware, so a row written from the UTC container and one
    # written on the Windows host land on one timeline (Decision 24).
    result["evaluated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    result["max_monitoring_age_hours"] = MAX_MONITORING_AGE_HOURS

    result_path = os.path.join(OUTPUT_DIR, "approval_gate_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Additive: writes the same result to the database for Power BI's
    # live feed. Runs after the JSON file above is already safely
    # saved, and never affects the gate_status — see
    # write_governance_status()'s own docstring. Only with
    # --require-db-write does a failed write change the exit code.
    written = write_governance_status(result)
    if args.require_db_write and not written:
        sys.exit(2)

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

    # Exit codes: 0 PASS, 1 BLOCK, 2 the gate could not complete or
    # (with --require-db-write) could not record its result.
    sys.exit(0 if result["gate_status"] == "PASS" else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # An unhandled error would otherwise exit 1, indistinguishable
        # from a recorded BLOCK verdict to a caller such as the DAG.
        traceback.print_exc()
        sys.exit(2)

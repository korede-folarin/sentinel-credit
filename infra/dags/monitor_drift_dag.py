"""
Sentinel Credit — Drift Monitoring DAG

Purpose
-------
Runs pipeline/monitor_drift.py on a schedule and separates two
concerns Airflow's UI can distinguish cleanly:

- compute_drift: did the check itself run successfully? A
  failure here means something is broken (DB unreachable, feature
  store schema changed, etc.) — an infrastructure problem.
- evaluate_alerts: did the check find drift? A failure here means
  the pipeline ran fine but the DATA moved — a business signal,
  not a bug. Kept as a separate task so "the monitoring job broke"
  and "the monitoring job found something" show up as visibly
  different failure reasons in the Airflow UI, not one conflated
  red X.
- run_approval_gate: re-evaluates pipeline/approval_gate.py on the
  evidence compute_drift just wrote, and records the result in
  reporting.governance_status_log, so the Power BI gate badge stays
  current while the stack runs (Decision 24). Runs alongside
  evaluate_alerts, not after it: found drift must still produce a
  BLOCK row, not no row. The task fails only if the gate cannot
  record its result; a BLOCK verdict is a governance outcome,
  already surfaced as a failure by evaluate_alerts.

No task retrains, promotes a model, or changes MODEL_FEATURES.

max_active_runs=1: when the stack comes back after downtime,
catch-up runs would otherwise execute concurrently and race on the
same CSV and metadata JSON (observed 12th October 2026, audit A18).

Container setup this DAG assumes (see infra/docker-compose.override.yml):
- The project repo is mounted at /usr/local/airflow/project.
- Airflow Variables sentinel_db_host and sentinel_db_password are
  set (Admin > Variables in the UI, or `airflow variables set`).
  sentinel_db_host should be host.docker.internal for local Astro
  dev, since "localhost" inside the container means the container
  itself, not the Windows host running PostgreSQL.

See docs/engineering_decision_log.md — Decision 5 (drift
monitoring, queued), pipeline/monitor_drift.py's own docstring for
the population-choice and check-design rationale this DAG doesn't
repeat.
"""

import json
import os
import subprocess
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable

PROJECT_MOUNT = "/usr/local/airflow/project"
SCRIPT_PATH = os.path.join(PROJECT_MOUNT, "pipeline", "monitor_drift.py")
GATE_SCRIPT_PATH = os.path.join(PROJECT_MOUNT, "pipeline", "approval_gate.py")
METADATA_PATH = os.path.join(
    PROJECT_MOUNT, "reports", "monitoring", "monitoring_run_metadata.json"
)


def _script_env() -> dict:
    """Subprocess environment: DB settings from Airflow Variables,
    never written to disk or the task log."""
    env = os.environ.copy()
    env["SENTINEL_DB_HOST"] = Variable.get("sentinel_db_host")
    env["SENTINEL_DB_PASSWORD"] = Variable.get("sentinel_db_password")
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    return env

default_args = {
    "owner": "sentinel-credit",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    # With max_active_runs=1, a task that never returns would block
    # every later scheduled run. Healthy runs finish in about a
    # minute; a run blocked on I/O through the Windows bind mount
    # (observed 13th October 2026, audit A21) is failed here instead
    # of hanging indefinitely.
    "execution_timeout": timedelta(minutes=20),
}


@dag(
    dag_id="sentinel_credit_drift_monitoring",
    # @daily fires only while the local Astro/Docker stack is up —
    # not continuous daily monitoring. See Decision 20 addendum.
    description="PSI drift and data-quality check for the v1 champion (scheduled while the local stack is running)",
    schedule="@daily",
    start_date=datetime(2026, 8, 11),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["sentinel-credit", "monitoring", "drift"],
)
def sentinel_credit_drift_monitoring():

    @task
    def compute_drift() -> str:
        """
        Runs monitor_drift.py as a subprocess rather than importing
        it, so this task exercises the exact same script a human
        would run manually from PowerShell — no separate "DAG
        version" of the monitoring logic to drift out of sync with
        the one actually documented and tested.
        """
        env = _script_env()

        result = subprocess.run(
            ["python", SCRIPT_PATH],
            cwd=PROJECT_MOUNT,  # config.yaml's mlflow tracking_uri
                                 # ("file:./mlruns") resolves relative
                                 # to cwd, not to config.yaml's own
                                 # location — this must be the project
                                 # root for it to land in the same
                                 # mlruns/ used by manual runs.
            env=env,
            capture_output=True,
            text=True,
        )

        # Sent to the Airflow task log either way, so a failure is
        # debuggable from the UI without needing container access.
        print(result.stdout)
        if result.stderr:
            print("---- stderr ----")
            print(result.stderr)

        if result.returncode != 0:
            raise AirflowFailException(
                f"monitor_drift.py exited with code {result.returncode} — "
                "see task log above for the script's own output."
            )

        return METADATA_PATH

    @task
    def evaluate_alerts(metadata_path: str) -> None:
        """
        Reads the metadata monitor_drift.py already wrote and
        decides Airflow's response — this task does not recompute
        anything, it only interprets output compute_drift already
        produced and validated.
        """
        if not os.path.exists(metadata_path):
            raise AirflowFailException(
                f"{metadata_path} not found — compute_drift may not have "
                "written its output correctly."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        n_alerts = metadata["n_features_with_alerts"]
        n_total = metadata["n_features_total"]
        bad_rate_delta = metadata["target_rate_shift"].get("absolute_delta")

        print(f"Features checked: {n_total}")
        print(f"Features with alerts: {n_alerts}")
        if bad_rate_delta is not None:
            print(f"Portfolio bad-rate absolute delta: {bad_rate_delta:+.4f}")

        if n_alerts > 0:
            raise AirflowFailException(
                f"{n_alerts} of {n_total} features triggered a drift or "
                f"data-quality alert. See {metadata_path} and "
                f"{os.path.join(PROJECT_MOUNT, 'reports', 'monitoring', 'feature_drift.csv')} "
                "for details. This is a DATA finding, not a pipeline bug — "
                "the check itself ran successfully."
            )

    @task
    def run_approval_gate(metadata_path: str) -> str:
        """
        Re-evaluates the approval gate against the evidence
        compute_drift just wrote. Exit codes from approval_gate.py:
        0 PASS and 1 BLOCK are both a successfully RECORDED verdict
        (this task succeeds and returns it); 2 means the result could
        not be written to reporting.governance_status_log, which
        fails the task, since the badge would otherwise only reveal
        the problem 48h later as STALE.
        """
        result = subprocess.run(
            ["python", GATE_SCRIPT_PATH, "--require-db-write"],
            cwd=PROJECT_MOUNT,
            env=_script_env(),
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("---- stderr ----")
            print(result.stderr)

        if result.returncode not in (0, 1):
            raise AirflowFailException(
                f"approval_gate.py exited with code {result.returncode} — "
                "the gate result was not recorded. See task log above."
            )
        return "PASS" if result.returncode == 0 else "BLOCK"

    metadata_path = compute_drift()
    evaluate_alerts(metadata_path)
    run_approval_gate(metadata_path)


sentinel_credit_drift_monitoring()

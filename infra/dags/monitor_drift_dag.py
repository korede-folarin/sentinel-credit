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

Neither task retrains, promotes a model, or changes MODEL_FEATURES.
This DAG produces evidence and a visible alert state; feeding that
into an automated Approval Gate or Champion/Challenger promotion
workflow is the next Intelligence Core / Governance Layer piece,
not yet built.

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
METADATA_PATH = os.path.join(
    PROJECT_MOUNT, "reports", "monitoring", "monitoring_run_metadata.json"
)

default_args = {
    "owner": "sentinel-credit",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="sentinel_credit_drift_monitoring",
    description="Daily PSI drift and data-quality check for the v1 champion",
    schedule="@daily",
    start_date=datetime(2026, 8, 11),
    catchup=False,
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
        db_host = Variable.get("sentinel_db_host")
        db_password = Variable.get("sentinel_db_password")

        env = os.environ.copy()
        env["SENTINEL_DB_HOST"] = db_host
        env["SENTINEL_DB_PASSWORD"] = db_password
        env["MLFLOW_ALLOW_FILE_STORE"] = "true"

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

    metadata_path = compute_drift()
    evaluate_alerts(metadata_path)


sentinel_credit_drift_monitoring()

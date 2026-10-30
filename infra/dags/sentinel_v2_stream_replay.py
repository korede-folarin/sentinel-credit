"""
Sentinel Credit v2, Phase 4 — Stream Replay DAG (V2-03)

Purpose
-------
Runs one simulated replay through the v2 streaming stack and checks it.
New in v2; the v1 drift DAG (monitor_drift_dag.py) is unchanged.

- check_stream_stack: the Flink job is RUNNING (Flink REST API). If not,
  the stack is down: an infrastructure problem, nothing is replayed.
- request_replay: writes a 'requested' row to v2_stream.replay_runs. The
  replay agent in the streaming stack (v2/streaming/simulator.py agent)
  picks it up and sends the events to Kafka. Airflow never talks to Kafka
  directly, so its image needs no new packages.
- wait_for_replay: waits until the agent marks the run finished.
- check_replay: runs v2/streaming/stream_check.py (completeness, parity
  with the v1 feature store, latency, throughput, lag, drift). Fails if any
  event was lost or double-applied, or any customer's state differs from v1.

Simulated stream of synthetic data, shadow mode: nothing here answers an
applicant, retrains, or promotes a model. Manual trigger only.

Container setup this DAG assumes (as for monitor_drift_dag.py):
- The project repo is mounted at /usr/local/airflow/project.
- Airflow Variables sentinel_db_host and sentinel_db_password are set.
- The streaming stack is running (v2/streaming/stream.ps1 start, submit).
  Its Flink REST port is reached at host.docker.internal:8081.

Scoring parity inside check_replay is skipped here: Airflow's image has
newer scikit-learn/XGBoost than the champion was saved with, so the check
records "skipped" rather than load the model under the wrong versions.
"""

import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

import psycopg2
import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable

PROJECT_MOUNT = "/usr/local/airflow/project"
CHECK_SCRIPT = os.path.join(PROJECT_MOUNT, "v2", "streaming", "stream_check.py")
FLINK_REST = "http://host.docker.internal:8081"
JOB_NAME = "sentinel_v2_stream_scorer"


def _db():
    return psycopg2.connect(host=Variable.get("sentinel_db_host"), port=5432, dbname="sentinel_credit",
                            user="postgres", password=Variable.get("sentinel_db_password"))


@dag(
    dag_id="sentinel_v2_stream_replay",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    params={"rate": 100, "customers": 4000},
    default_args={"retries": 0, "execution_timeout": timedelta(minutes=45)},
    tags=["sentinel", "v2", "streaming", "phase4"],
)
def sentinel_v2_stream_replay():

    @task
    def check_stream_stack():
        jobs = requests.get(f"{FLINK_REST}/jobs/overview", timeout=10).json()["jobs"]
        running = [j for j in jobs if j["name"] == JOB_NAME and j["state"] == "RUNNING"]
        if not running:
            raise AirflowFailException(f"{JOB_NAME} is not RUNNING; start the stack with stream.ps1")
        return running[0]["jid"]

    @task
    def request_replay(params=None):
        rate, customers = int(params["rate"]), int(params["customers"])
        # Named replay_id, not run_id: Airflow reserves run_id for its own context.
        replay_id = f"stream_{rate}eps_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
        conn = _db()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO v2_stream.replay_runs
                    (run_id, status, requested_by, rate_eps, n_customers, seed, duplicate_rate, late_rate,
                     late_max_s, dataset_version, model_version)
                    VALUES (%s, 'requested', 'airflow', %s, %s, 42, 0.01, 0.02, 600,
                            'full_horizon_2025_v1', 'v2-phase4-stream')""", (replay_id, rate, customers))
        finally:
            conn.close()
        return replay_id

    @task(execution_timeout=timedelta(minutes=40))
    def wait_for_replay(replay_id: str):
        while True:
            conn = _db()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT status, note FROM v2_stream.replay_runs WHERE run_id = %s", (replay_id,))
                    status, note = cur.fetchone()
            finally:
                conn.close()
            if status == "finished":
                return replay_id
            if status == "failed":
                raise AirflowFailException(f"replay {replay_id} failed: {note}")
            time.sleep(15)

    @task
    def check_replay(replay_id: str):
        env = os.environ.copy()
        env["SENTINEL_DB_HOST"] = Variable.get("sentinel_db_host")
        env["SENTINEL_DB_PASSWORD"] = Variable.get("sentinel_db_password")
        result = subprocess.run(["python", CHECK_SCRIPT, "--run-id", replay_id], env=env, cwd=PROJECT_MOUNT,
                                capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr[-4000:])
        if result.returncode != 0:
            raise AirflowFailException(f"stream checks failed for {replay_id}")

    stack = check_stream_stack()
    replay_id = request_replay()
    stack >> replay_id
    check_replay(wait_for_replay(replay_id))


sentinel_v2_stream_replay()

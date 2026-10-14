"""
Sentinel Credit, Dashboard Shared Utilities

Purpose
-------
Common config loading, path resolution and safe file loading
shared by every page in this Streamlit app. Loading functions
never fabricate placeholder data when a file is missing. They
return None and the calling page shows an explicit, honest
message pointing at which script to run first, matching the
"evidence, not invention" standard used across every report and
script in this project.
"""

import json
import os

import pandas as pd
import psycopg2
import streamlit as st
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


@st.cache_data
def load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(relative_path: str):
    path = os.path.join(REPORTS_DIR, relative_path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(relative_path: str):
    path = os.path.join(REPORTS_DIR, relative_path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def image_path(relative_path: str):
    path = os.path.join(REPORTS_DIR, relative_path)
    return path if os.path.exists(path) else None


def missing_evidence(script_hint: str) -> None:
    from theme import empty_state
    empty_state("No evidence for this view yet.",
                f"Run {script_hint}, then refresh this page.")


def _connect():
    config = load_config()
    db_cfg = config["database"]
    host = os.environ.get("SENTINEL_DB_HOST", db_cfg["host"])
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    return psycopg2.connect(
        host=host, port=db_cfg["port"], dbname=db_cfg["name"],
        user="postgres", password=password, connect_timeout=5,
    )


@st.cache_resource
def _connection_holder() -> dict:
    return {"conn": None}


def get_db_connection():
    """
    Live connection to the database, for anything this dashboard
    needs to query directly rather than reading a saved report.
    Reads SENTINEL_DB_HOST and SENTINEL_DB_PASSWORD the same way
    every other script in this project does, so the dashboard
    works whether it is run natively or inside the Airflow
    container, with no separate configuration path invented for
    it. Reconnects if the cached connection was closed (e.g. a
    PostgreSQL restart) instead of failing every later query.
    """
    holder = _connection_holder()
    if holder["conn"] is None or holder["conn"].closed:
        holder["conn"] = _connect()
    return holder["conn"]


@st.cache_data(ttl=300, show_spinner=False)
def query_db(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Read-only query. Always pass user-supplied values via params."""
    conn = get_db_connection()
    try:
        return pd.read_sql(sql, conn, params=params or None)
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        conn.close()
        return pd.read_sql(sql, get_db_connection(), params=params or None)
    finally:
        if not conn.closed:
            conn.rollback()  # never leave a read transaction open


def governance_status():
    """
    Latest approval-gate status from reporting.governance_status_latest,
    which derives STALE at read time (Decision 24). Returns a dict, or
    None if the database is unavailable — callers show that honestly.
    """
    try:
        frame = query_db("SELECT * FROM reporting.governance_status_latest")
    except Exception:  # noqa: BLE001 - surfaced by the caller as unavailable
        return None
    return None if frame.empty else frame.iloc[0].to_dict()


def db_available() -> tuple:
    """(True, None) if a trivial query succeeds, else (False, reason)."""
    try:
        query_db("SELECT 1 AS ok")
        return True, None
    except Exception as error:  # noqa: BLE001
        return False, str(error).splitlines()[0][:200]

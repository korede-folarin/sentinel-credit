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
    st.warning(
        f"No evidence found for this view yet. Run {script_hint} "
        "and refresh this page."
    )


@st.cache_resource
def get_db_connection():
    """
    Live connection to the database, for anything this dashboard
    needs to query directly rather than reading a saved report.
    Reads SENTINEL_DB_HOST and SENTINEL_DB_PASSWORD the same way
    every other script in this project does, so the dashboard
    works whether it is run natively or inside the Airflow
    container, with no separate configuration path invented for
    it.
    """
    config = load_config()
    db_cfg = config["database"]
    host = os.environ.get("SENTINEL_DB_HOST", db_cfg["host"])
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    return psycopg2.connect(
        host=host, port=db_cfg["port"], dbname=db_cfg["name"],
        user="postgres", password=password,
    )


@st.cache_data(ttl=300)
def query_db(sql: str) -> pd.DataFrame:
    conn = get_db_connection()
    return pd.read_sql(sql, conn)


def governance_header() -> None:
    config = load_config()
    gov = config["model_governance"]
    cols = st.columns(4)
    cols[0].metric("Champion model", gov["champion_model"])
    cols[1].metric("Feature set", gov["feature_set_label"])
    cols[2].metric("Model version", gov["model_version"])
    cols[3].metric("Business cutoff", config["model"]["business_cutoff"])

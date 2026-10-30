"""
Sentinel Credit v2, Phase 2: shared settings, paths and connections (V2-04).

No credentials in code: the database password comes only from the
SENTINEL_DB_PASSWORD environment variable, passed to psycopg2 as a keyword
argument, never built into a connection string or logged.
"""

import os
import re

import psycopg2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
    V1 = yaml.safe_load(f)
with open(os.path.join(ROOT, "v2", "config_v2.yaml"), encoding="utf-8") as f:
    P2 = yaml.safe_load(f)["phase2"]

V1_FEATURE_SQL = os.path.join(ROOT, "sql", "02_feature_store.sql")
# Data root: keep it OUTSIDE any synced folder (OneDrive would try to upload
# gigabytes). Set SENTINEL_V2_DATA_DIR; the repo-relative default is a fallback.
DATA_ROOT = os.environ.get("SENTINEL_V2_DATA_DIR", os.path.join(ROOT, P2["data_dir"]))
DATA_DIR = os.path.join(DATA_ROOT, "scaled")
PANDAS_OUT = os.path.join(DATA_ROOT, "pandas_out")
SPARK_OUT = os.path.join(DATA_ROOT, "spark_out")
RESULTS_DIR = os.path.join(ROOT, "v2", "reports", "phase2")
RESULTS_SCHEMA = P2["results_schema"]                    # v2_scale
V1_EXPORT = "v1_raw_export"                              # v1 raw tables copied to Parquet, read-only


def dataset_version(n_customers: int) -> str:
    return f"{P2['dataset_prefix']}_seed{P2['seed']}_n{n_customers}"


def dataset_dir(version: str) -> str:
    return os.path.join(DATA_DIR, version)


def raw_schema(n_customers: int) -> str:
    """PostgreSQL schema holding one scaled dataset's raw tables."""
    return f"v2_scale_raw_n{n_customers}"


def db_connect():
    db = V1["database"]
    return psycopg2.connect(host=os.environ.get("SENTINEL_DB_HOST", db["host"]), port=db["port"],
                            dbname=db["name"], user="postgres",
                            password=os.environ.get("SENTINEL_DB_PASSWORD", ""))


def v1_sql_for(n_customers: int) -> str:
    """
    v1's feature-store SQL, read-only, pointed at one scaled dataset: raw.* ->
    v2_scale_raw_n<N>.*, feature_store.customer_features ->
    v2_scale.customer_features_n<N>, index names suffixed. Refuses to return
    anything that still names a v1 schema, because the v1 script begins with
    DROP TABLE feature_store.customer_features.
    """
    with open(V1_FEATURE_SQL, encoding="utf-8") as f:
        sql = f.read()
    target = f"{RESULTS_SCHEMA}.customer_features_n{n_customers}"
    sql = sql.replace("CREATE SCHEMA IF NOT EXISTS feature_store;", f"CREATE SCHEMA IF NOT EXISTS {RESULTS_SCHEMA};")
    sql = sql.replace("feature_store.customer_features", target)
    sql = re.sub(r"\braw\.", f"{raw_schema(n_customers)}.", sql)
    sql = re.sub(r"\b(idx_cf_\w+)", rf"\1_n{n_customers}", sql)
    code = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())     # ignore comments
    if re.search(r"\bfeature_store\b", code) or re.search(r"(?<![\w.])raw\.", code):
        raise RuntimeError("Refusing to run: the rewritten SQL still names a v1 schema")
    return sql

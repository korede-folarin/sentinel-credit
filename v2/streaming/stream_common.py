"""
Sentinel Credit v2, Phase 4: settings and connections shared by the
simulator, the sink and the checks (V2-03).

No credentials in code: the database password comes only from the
SENTINEL_DB_PASSWORD environment variable and is passed to psycopg2 as a
keyword argument, never built into a connection string or logged.
"""

import os

import psycopg2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# Inside the tools container the repo files are mounted at SENTINEL_HOME;
# on the host they sit two folders up from here.
SENTINEL = os.environ.get("SENTINEL_HOME")
if SENTINEL:
    V1_CONFIG = os.path.join(SENTINEL, "config.yaml")
    V2_CONFIG = os.path.join(SENTINEL, "config_v2.yaml")
else:
    ROOT = os.path.dirname(os.path.dirname(HERE))
    V1_CONFIG = os.path.join(ROOT, "config.yaml")
    V2_CONFIG = os.path.join(ROOT, "v2", "config_v2.yaml")

with open(V1_CONFIG, encoding="utf-8") as f:
    V1 = yaml.safe_load(f)
with open(V2_CONFIG, encoding="utf-8") as f:
    P4 = yaml.safe_load(f)["phase4"]

SCHEMA = P4["results_schema"]
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092")


def db_connect():
    db = V1["database"]
    return psycopg2.connect(host=os.environ.get("SENTINEL_DB_HOST", db["host"]),
                            port=int(os.environ.get("SENTINEL_DB_PORT", db["port"])),
                            dbname=db["name"], user="postgres",
                            password=os.environ.get("SENTINEL_DB_PASSWORD", ""))

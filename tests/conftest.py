"""
Shared test setup for Sentinel Credit (v2 V2-01).

Plain English: helpers every test file uses: where the project is, which
copy of the champion model to load, and whether the local database or the
local (git-ignored) model package is available. Tests that need those are
skipped, not failed, when they are absent (for example on GitHub's CI
machines), and the skip reason says why.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The committed copy of the v1.1 champion package (published with the
# Streamlit demo, v1 Decision 32). Available everywhere, including CI.
REPO_MODEL_DIR = os.path.join(ROOT, "dashboard", "demo_data", "model")
# The governed, git-ignored package the API uses by default (local only).
LOCAL_MODEL_DIR = os.path.join(ROOT, "models", "artifacts_v1_1")


def load_module(name, relpath):
    """Import a script by path (pipeline/ and data_generator/ are not packages)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _db_available():
    if not os.environ.get("SENTINEL_DB_PASSWORD"):
        return False
    try:
        import psycopg2
        import yaml
        with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
            db = yaml.safe_load(f)["database"]
        psycopg2.connect(host=os.environ.get("SENTINEL_DB_HOST", db["host"]), port=db["port"],
                         dbname=db["name"], user="postgres",
                         password=os.environ["SENTINEL_DB_PASSWORD"], connect_timeout=3).close()
        return True
    except Exception:  # noqa: BLE001
        return False


DB_AVAILABLE = _db_available()
LOCAL_PACKAGE = os.path.exists(os.path.join(LOCAL_MODEL_DIR, "champion_calibrated.joblib"))

needs_db = pytest.mark.skipif(not DB_AVAILABLE, reason="needs the local PostgreSQL database")
needs_local_package = pytest.mark.skipif(
    not LOCAL_PACKAGE, reason="needs the git-ignored model package models/artifacts_v1_1")

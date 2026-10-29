"""
Sentinel Credit v2 — the v1 manifest: proof that v1 is unchanged (V2-01).

Plain English: before any v2 work, take a fingerprint of everything that
defines the finished v1 platform, so that later we can prove none of it
moved. Technical term: an integrity manifest of content hashes, row counts,
schema definitions and experiment-tracking records.

What it records (v2/manifests/v1_manifest.json):
- repo_artefacts: SHA-256 of every v1 artefact and definition committed to
  Git (the champion model copy, the ONNX model, model contract, explanation
  rules and wording, feature definitions, feature-store SQL, the generator).
- config_v1: SHA-256 of the config.yaml sections v1 depends on, so new v2
  sections can be added without tripping the check.
- local_artefacts: SHA-256 of the git-ignored model packages on this machine.
- database: exact row counts for raw.* and feature_store.*; column names and
  types for every table and view in raw, feature_store and reporting.
  (reporting.* row counts are not frozen: monitoring appends to them.)
- mlflow: every run that exists now, with its params and metrics hashed.
  New runs may appear later (monitoring keeps running); existing ones must
  not change or disappear.
- key_metrics: champion Gini, calibration (Brier, ECE) and business cutoff.

Usage:
    python v2/tools/v1_manifest.py build          # once, at tag v1.0-evaluated
    python v2/tools/v1_manifest.py verify --repo  # CI: committed files only
    python v2/tools/v1_manifest.py verify --full  # local: also DB, MLflow, artefacts

Needs, for build and --full: the local PostgreSQL (SENTINEL_DB_PASSWORD in
the environment, never in code) and the local MLflow store (./mlruns).
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "v2", "manifests", "v1_manifest.json")
TAG = "v1.0-evaluated"

# Committed v1 artefacts and definitions. Any change to these fails CI.
REPO_ARTEFACTS = [
    "dashboard/demo_data/model/*",
    "website/public/model/*",
    "website/public/data/*.json",
    "serving/explanation_rules.json",
    "serving/reason_templates.json",
    "serving/typical_applicant.json",
    "serving/features.py",
    "serving/reasons.py",
    "models/model_features.py",
    "sql/*.sql",
    "data_generator/bnpl_data_generator.py",
]
CONFIG_V1_SECTIONS = ["data_generator", "model", "model_governance", "explainability",
                      "feature_governance", "threshold_recalibration", "champion_promotion",
                      "approval_gate"]
LOCAL_ARTEFACTS = ["models/artifacts/*", "models/artifacts_v1_1/*"]
FROZEN_COUNT_SCHEMAS = ["raw", "feature_store"]
SCHEMA_SCHEMAS = ["raw", "feature_store", "reporting"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(path):
    """Text files hashed with line endings normalised, so a Windows checkout
    (CRLF) and the Linux CI checkout (LF) give the same fingerprint."""
    with open(path, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    binary = path.endswith((".joblib", ".onnx", ".png", ".woff2"))
    return sha256(path) if binary else text_sha256(path)


def expand(patterns):
    out = []
    for p in patterns:
        out += [f for f in glob.glob(os.path.join(ROOT, p)) if os.path.isfile(f)]
    return sorted(set(os.path.relpath(f, ROOT).replace("\\", "/") for f in out))


def config_v1_hash():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    subset = {k: cfg.get(k) for k in CONFIG_V1_SECTIONS}
    return hashlib.sha256(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-list", "-n", "1", TAG], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def db_connect():
    import psycopg2
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        db = yaml.safe_load(f)["database"]
    return psycopg2.connect(host=os.environ.get("SENTINEL_DB_HOST", db["host"]), port=db["port"],
                            dbname=db["name"], user="postgres",
                            password=os.environ.get("SENTINEL_DB_PASSWORD", ""), connect_timeout=10)


def read_database():
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("""SELECT table_schema, table_name, column_name, data_type
                       FROM information_schema.columns WHERE table_schema = ANY(%s)
                       ORDER BY table_schema, table_name, ordinal_position""", (SCHEMA_SCHEMAS,))
        schemas = {}
        for s, t, c, d in cur.fetchall():
            schemas.setdefault(f"{s}.{t}", []).append([c, d])
        cur.execute("""SELECT table_schema, table_name FROM information_schema.tables
                       WHERE table_schema = ANY(%s) AND table_type = 'BASE TABLE'
                       ORDER BY 1, 2""", (FROZEN_COUNT_SCHEMAS,))
        counts = {}
        for s, t in cur.fetchall():
            cur.execute(f'SELECT count(*) FROM "{s}"."{t}"')
            counts[f"{s}.{t}"] = cur.fetchone()[0]
        return {"row_counts": counts, "schemas": schemas}
    finally:
        conn.close()


def read_mlflow():
    """Read the MLflow file store directly, read-only. The monitoring runs
    were written by a newer MLflow inside the Airflow container, in a format
    the project's MLflow 2.13 client cannot parse (V2-01 finding), so the
    files themselves are fingerprinted: each run's params and metrics."""
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        uri = yaml.safe_load(f)["mlflow"]["tracking_uri"]
    store = os.path.join(ROOT, uri[len("file:./"):]) if uri.startswith("file:./") else uri[len("file:"):]
    runs = {}
    for exp_meta in sorted(glob.glob(os.path.join(store, "*", "meta.yaml"))):
        exp_dir = os.path.dirname(exp_meta)
        with open(exp_meta, encoding="utf-8") as f:
            exp = yaml.safe_load(f) or {}
        for run_meta in sorted(glob.glob(os.path.join(exp_dir, "*", "meta.yaml"))):
            run_dir = os.path.dirname(run_meta)
            with open(run_meta, encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            body = {}
            for kind in ("params", "metrics"):
                for fp in sorted(glob.glob(os.path.join(run_dir, kind, "**", "*"), recursive=True)):
                    if os.path.isfile(fp):
                        rel = os.path.relpath(fp, run_dir).replace("\\", "/")
                        body[rel] = text_sha256(fp)
            run_id = meta.get("run_id") or meta.get("run_uuid") or os.path.basename(run_dir)
            runs[run_id] = {"experiment": exp.get("name"), "status": meta.get("status"),
                            "content_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
    return runs


def key_metrics():
    path = os.path.join(ROOT, "reports", "evaluation", "stratified", "xgboost", "evaluation_metadata.json")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    s = m["summary"]
    return {"model_version": m["model_version"], "feature_set": m["feature_set"],
            "training_run_id": m["training_run_id"], "dataset_version": m["dataset_version"],
            "gini": s["gini"], "auc": s["auc"], "brier": s["brier"], "ece": s["ece"],
            "business_cutoff": m["business_cutoff"]}


def build():
    manifest = {
        "_about": ("Fingerprint of Sentinel Credit v1 at tag v1.0-evaluated (v2 log V2-01). "
                   "Built by v2/tools/v1_manifest.py; verified in CI (--repo) and locally (--full)."),
        "tag": TAG, "commit": git_commit(),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_artefacts": {p: file_hash(os.path.join(ROOT, p)) for p in expand(REPO_ARTEFACTS)},
        "config_v1_sha256": config_v1_hash(),
        "local_artefacts": {p: sha256(os.path.join(ROOT, p)) for p in expand(LOCAL_ARTEFACTS)},
        "database": read_database(),
        "mlflow_runs": read_mlflow(),
        "key_metrics": key_metrics(),
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=False)
        f.write("\n")
    db = manifest["database"]
    print(f"Wrote {os.path.relpath(MANIFEST, ROOT)}: {len(manifest['repo_artefacts'])} repo artefacts, "
          f"{len(manifest['local_artefacts'])} local artefacts, {len(db['row_counts'])} tables counted, "
          f"{len(db['schemas'])} schemas, {len(manifest['mlflow_runs'])} MLflow runs.")


def verify(full):
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    problems = []
    for p, h in man["repo_artefacts"].items():
        path = os.path.join(ROOT, p)
        if not os.path.exists(path):
            problems.append(f"missing: {p}")
        elif file_hash(path) != h:
            problems.append(f"changed: {p}")
    added = set(expand(REPO_ARTEFACTS)) - set(man["repo_artefacts"])
    problems += [f"new file in a frozen v1 location: {p}" for p in sorted(added)]
    if config_v1_hash() != man["config_v1_sha256"]:
        problems.append("changed: a v1 section of config.yaml")
    checked = ["repo artefacts", "config v1 sections"]
    if full:
        for p, h in man["local_artefacts"].items():
            path = os.path.join(ROOT, p)
            if not os.path.exists(path):
                problems.append(f"missing local artefact: {p}")
            elif sha256(path) != h:
                problems.append(f"changed local artefact: {p}")
        db = read_database()
        for t, n in man["database"]["row_counts"].items():
            if db["row_counts"].get(t) != n:
                problems.append(f"row count changed: {t} {n} -> {db['row_counts'].get(t)}")
        for t, cols in man["database"]["schemas"].items():
            if db["schemas"].get(t) != cols:
                problems.append(f"schema changed: {t}")
        runs = read_mlflow()
        for rid, r in man["mlflow_runs"].items():
            now = runs.get(rid)
            if now is None:
                problems.append(f"MLflow run missing: {rid} ({r['experiment']})")
            elif now["content_sha256"] != r["content_sha256"] or now["status"] != r["status"]:
                problems.append(f"MLflow run changed: {rid} ({r['experiment']})")
        if key_metrics() != man["key_metrics"]:
            problems.append("champion key metrics changed")
        checked += ["local artefacts", "database counts and schemas", "MLflow runs", "key metrics"]
    print(f"Checked: {', '.join(checked)}.")
    if problems:
        print("v1 CHANGED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("v1 unchanged.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    v = sub.add_parser("verify")
    g = v.add_mutually_exclusive_group(required=True)
    g.add_argument("--repo", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    build() if a.cmd == "build" else verify(a.cmd == "verify" and a.full)

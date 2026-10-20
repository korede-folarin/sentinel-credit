"""
Sentinel Credit — Export the public demo snapshot for the Streamlit dashboard

The hosted demo (Streamlit Community Cloud, SENTINEL_DEMO=1) has no
database, no git-ignored reports and no model package. This script builds
dashboard/demo_data/ from the local platform:

1. Runs every dashboard page headless (Streamlit AppTest) in recording
   mode. Every database query result and every report file a page reads
   is captured.
2. Keeps only aggregate data:
   - a query whose result has a customer_id column is refused (never
     snapshotted), e.g. existing-customer lookup;
   - per-customer SHAP case studies (reports/shap/local/) are excluded;
   - every other report file is checked: a CSV with a customer_id column
     or a JSON with a customer_id key anywhere stops the export.
3. Copies the champion package's model files (no data rows) and replaces
   the per-customer held-out prediction file with 1,001 score quantiles.
4. Writes training_ranges.csv (min / median / max per feature) and
   MANIFEST.json listing every file, its source and what was refused.

Re-run after any promotion, report refresh or dashboard change, then
commit dashboard/demo_data/.

Usage:  python dashboard/tools/export_demo_data.py
Needs the local PostgreSQL feature store (SENTINEL_DB_PASSWORD).
"""

import json
import os
import shutil
import sys
import tempfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

DASHBOARD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD)
OUT = os.path.join(DASHBOARD, "demo_data")
RECORD = tempfile.mkdtemp(prefix="sentinel_demo_record_")
os.environ["SENTINEL_DEMO_RECORD_DIR"] = RECORD   # before utils is imported anywhere
os.environ.pop("SENTINEL_DEMO", None)
sys.path.insert(0, DASHBOARD)
sys.path.insert(0, PROJECT_ROOT)
warnings.filterwarnings("ignore")

MODEL_FILES = ["champion_calibrated.joblib", "champion_explanation_base.joblib",
               "feature_list.json", "champion_metadata.json"]
EXCLUDED_REPORT_PREFIXES = ("shap/local/",)


def has_customer_key(obj) -> bool:
    if isinstance(obj, dict):
        return any(k == "customer_id" or has_customer_key(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(has_customer_key(v) for v in obj)
    return False


def relative_paths(text: str) -> str:
    """Absolute local paths (which name the machine's user) -> project-relative."""
    root = PROJECT_ROOT
    for form in (root.replace("\\", "\\\\"), root, root.replace("\\", "/")):   # JSON-escaped first
        text = text.replace(form, ".")
    return text


def leaked_home_paths(folder: str) -> list:
    home = os.path.expanduser("~")
    needles = {home, home.replace("\\", "\\\\"), home.replace("\\", "/")}
    hits = []
    for d, _, files in os.walk(folder):
        for n in files:
            if n.endswith((".json", ".csv")):
                with open(os.path.join(d, n), encoding="utf-8", errors="ignore") as f:
                    t = f.read()
                if any(x in t for x in needles):
                    hits.append(os.path.relpath(os.path.join(d, n), folder))
    return hits


def read_jsonl(name):
    path = os.path.join(RECORD, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pages():
    from streamlit.testing.v1 import AppTest
    pages = ["Home.py"] + sorted(os.path.join("pages", f)
                                 for f in os.listdir(os.path.join(DASHBOARD, "pages"))
                                 if f.endswith(".py"))
    for page in pages:
        at = AppTest.from_file(os.path.join(DASHBOARD, page), default_timeout=300)
        at.run()
        if page.endswith("3_Model_Performance.py"):
            at.radio[0].set_value("temporal").run()      # the second evaluation split
        if page.endswith("1_Credit_Risk_Assessment.py"):
            at.button[0].click().run()                    # one new-applicant assessment
        errors = [e.value for e in at.error] + [str(x.value)[:200] for x in at.exception]
        print(f"  recorded {page:<40} {'ERRORS: ' + str(errors) if errors else 'ok'}")
        if errors:
            raise SystemExit(f"{page} has errors; fix before exporting.")


def main():
    import yaml
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gov = cfg["model_governance"]
    pkg = os.path.join(PROJECT_ROOT, gov["artifacts_dir"])

    print("Recording every page against the local platform...")
    run_pages()

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "queries"))
    manifest = {"_about": ("Public demo snapshot for the Streamlit dashboard (SENTINEL_DEMO=1). "
                           "Aggregates and model files only; no customer records. Built by "
                           "dashboard/tools/export_demo_data.py."),
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "model_version": gov["model_version"],
                "queries": [], "refused_queries": [], "reports": [], "excluded_reports": [],
                "model_files": []}

    # 1 · query snapshots
    seen = set()
    for q in read_jsonl("queries"):
        if q["key"] in seen:
            continue
        seen.add(q["key"])
        entry = {"key": q["key"], "sql": " ".join(q["sql"].split()), "rows": q["rows"],
                 "columns": q["columns"]}
        if "customer_id" in q["columns"]:
            manifest["refused_queries"].append({**entry, "reason": "customer rows"})
            continue
        shutil.copyfile(os.path.join(RECORD, "queries", f"{q['key']}.parquet"),
                        os.path.join(OUT, "queries", f"{q['key']}.parquet"))
        manifest["queries"].append(entry)

    # 2 · report files
    reports_src = os.path.join(PROJECT_ROOT, "reports")
    problems = []
    for rel in sorted({r["path"] for r in read_jsonl("reports")}):
        src = os.path.join(reports_src, rel)
        if rel.startswith(EXCLUDED_REPORT_PREFIXES):
            manifest["excluded_reports"].append({"path": rel, "reason": "per-customer"})
            continue
        if not os.path.exists(src):
            continue
        if rel.endswith(".csv") and "customer_id" in pd.read_csv(src, nrows=1).columns:
            problems.append(rel)
            continue
        if rel.endswith(".json"):
            with open(src, encoding="utf-8") as f:
                if has_customer_key(json.load(f)):
                    problems.append(rel)
                    continue
        dst = os.path.join(OUT, "reports", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, encoding="utf-8") as f:
            text = f.read()
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write(relative_paths(text))
        manifest["reports"].append(rel)
    if problems:
        raise SystemExit(f"Report files with customer-level fields, not exported: {problems}. "
                         "Decide how to handle them before publishing.")

    # 3 · model package (model files only) + held-out score quantiles
    os.makedirs(os.path.join(OUT, "model"))
    for name in MODEL_FILES:
        shutil.copyfile(os.path.join(pkg, name), os.path.join(OUT, "model", name))
        manifest["model_files"].append(name)
    with open(os.path.join(OUT, "model", "champion_metadata.json"), encoding="utf-8") as f:
        if has_customer_key(json.load(f)):
            raise SystemExit("champion_metadata.json contains customer-level fields.")
    scores = pd.read_csv(os.path.join(pkg, "test_predictions_stratified.csv"),
                         usecols=["proba_xgboost"])["proba_xgboost"].to_numpy()
    q = np.quantile(scores, np.linspace(0, 1, 1001))
    with open(os.path.join(OUT, "model", "heldout_score_quantiles.json"), "w", encoding="utf-8") as f:
        json.dump({"_about": "1,001 quantiles of the champion's held-out test scores, for the "
                             "'higher than X% of customers' figure. Replaces the per-customer file.",
                   "n_customers": int(len(scores)), "quantiles": [float(x) for x in q]}, f)
    manifest["model_files"].append("heldout_score_quantiles.json")

    # 4 · training ranges (aggregate)
    import scoring
    scoring.training_ranges().to_csv(os.path.join(OUT, "training_ranges.csv"))

    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    leaks = leaked_home_paths(OUT)
    if leaks:
        raise SystemExit(f"Local home-folder paths remain in: {leaks}. Not safe to publish.")
    shutil.rmtree(RECORD, ignore_errors=True)

    size = sum(os.path.getsize(os.path.join(d, n)) for d, _, fs in os.walk(OUT) for n in fs)
    print(f"\nWrote {OUT} ({size / 1e6:.1f} MB): {len(manifest['queries'])} query snapshots, "
          f"{len(manifest['reports'])} report files, {len(manifest['model_files'])} model files.")
    for r in manifest["refused_queries"]:
        print(f"  refused query ({r['reason']}): {r['sql'][:90]}")
    for r in manifest["excluded_reports"]:
        print(f"  excluded report ({r['reason']}): {r['path']}")


if __name__ == "__main__":
    main()

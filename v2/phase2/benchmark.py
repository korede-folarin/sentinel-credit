"""
Sentinel Credit v2, Phase 2: run the scaling benchmark (V2-04).

Plain English: for each dataset size, build the same feature store three
ways and time each build, then check every engine's answer against the
PostgreSQL reference cell by cell.

  1. PostgreSQL: load the Parquet files (a setup cost, timed separately),
     then run v1's SQL (read-only, pointed at the dataset's own schema).
  2. pandas: one Python process on the host.
  3. Spark 3.5: local mode in Docker, all of its cores. Files are first
     staged into a Docker volume (timed separately), so Spark reads local
     disk as pandas and PostgreSQL do, not a Windows folder share.

Each engine runs 3 times per size; the median is reported. Before any of
that, Spark and pandas must rebuild v1's own feature store exactly from
v1's raw data (the proof run).

Results: v2_scale.benchmark_runs, .setup_costs, .parity; MLflow experiment
sentinel_v2_phase2_spark (one run per engine and size, tagged with the
dataset_version); v2/reports/phase2/<run_label>.json.

  python v2/phase2/benchmark.py --sizes 50000 250000 1000000 2000000
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import pyarrow.parquet as pq
from psycopg2.extras import execute_values

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare  # noqa: E402
import postgres_engine  # noqa: E402
from common import (  # noqa: E402
    DATA_DIR,
    P2,
    PANDAS_OUT,
    RESULTS_DIR,
    ROOT,
    SPARK_OUT,
    V1,
    V1_EXPORT,
    dataset_dir,
    dataset_version,
    db_connect,
)

SPARK_IMAGE = "apache/spark:3.5.3-scala2.12-java17-python3-ubuntu"
SPARK_VOLUME = "sentinel_v2_spark_data"
SPARK_CORES = 8
SPARK_CONTAINER_MEMORY = "6g"
SPARK_DRIVER_MEMORY = "4g"
REPEATS = 3
PANDAS_TIMEOUT_S = 1800       # per pandas job (all repeats of one size)


def docker(args, **kw):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    return subprocess.run(["docker", *args], capture_output=True, text=True, env=env, **kw)


def spark_stage(dataset: str) -> float:
    t = time.perf_counter()
    r = docker(["run", "--rm", "-u", "0", "-v", f"{SPARK_VOLUME}:/data", "-v", f"{DATA_DIR}:/src:ro",
                SPARK_IMAGE, "sh", "-c",
                f"rm -rf /data/scaled/{dataset} /data/spark_out/{dataset} && mkdir -p /data/scaled "
                f"&& cp -r /src/{dataset} /data/scaled/"])
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return time.perf_counter() - t


def spark_run(dataset: str, repeats: int) -> dict:
    r = docker(["run", "--rm", "-u", "0", "--cpus", str(SPARK_CORES), "--memory", SPARK_CONTAINER_MEMORY,
                "-v", f"{SPARK_VOLUME}:/data", "-v", f"{HERE}:/opt/sentinel/phase2:ro", SPARK_IMAGE,
                "/opt/spark/bin/spark-submit", "--master", f"local[{SPARK_CORES}]",
                "--driver-memory", SPARK_DRIVER_MEMORY, "/opt/sentinel/phase2/spark_engine.py",
                "--dataset", dataset, "--repeats", str(repeats)])
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode or not lines:
        return {"failed": True, "detail": (r.stderr or r.stdout)[-1500:]}
    return json.loads(lines[-1])


def spark_fetch(dataset: str) -> str:
    os.makedirs(SPARK_OUT, exist_ok=True)
    r = docker(["run", "--rm", "-u", "0", "-v", f"{SPARK_VOLUME}:/data", "-v", f"{SPARK_OUT}:/out", SPARK_IMAGE,
                "sh", "-c", f"rm -rf /out/{dataset} && cp -r /data/spark_out/{dataset} /out/"])
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return os.path.join(SPARK_OUT, dataset)


def pandas_run(dataset: str, repeats: int) -> dict:
    out = os.path.join(PANDAS_OUT, dataset)
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "pandas_engine.py"), "--dataset", dataset,
                            "--out", out, "--repeats", str(repeats)], capture_output=True, text=True,
                           timeout=PANDAS_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        # Windows pages to disk rather than failing when memory runs out, so
        # a run that cannot fit shows up as one that does not finish.
        return {"failed": True, "detail": f"timed out after {PANDAS_TIMEOUT_S} s; output so far: "
                                          f"{(error.stdout or b'')[-500:]!r}"}
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode or not lines:
        tail = (r.stderr or r.stdout)[-1500:]
        return {"failed": True, "detail": tail, "partial_runs": r.stdout}
    result = json.loads(lines[-1])
    result["out"] = out
    return result


def row_counts(dataset: str) -> dict:
    folder = dataset_dir(dataset)
    counts = {}
    for t in ("customers", "transactions", "repayments"):
        path = os.path.join(folder, f"{t}.parquet")
        files = [path] if os.path.exists(path) else [os.path.join(folder, t, f)
                                                    for f in os.listdir(os.path.join(folder, t))]
        counts[t] = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    return counts


class Recorder:
    def __init__(self, label: str):
        self.label, self.runs, self.setup, self.parity = label, [], [], []

    def run(self, engine, dataset, n, counts, seconds_list, cores, peak=None, failed_detail=None):
        if failed_detail is not None:
            self.runs.append((self.label, engine, dataset, n, counts["transactions"], counts["repayments"], 1, None,
                              "failed", peak, cores, failed_detail[-1000:]))
            return
        for i, s in enumerate(seconds_list, start=1):
            self.runs.append((self.label, engine, dataset, n, counts["transactions"], counts["repayments"], i, s,
                              "ok", peak, cores, None))

    def save(self):
        conn = db_connect()
        try:
            with conn, conn.cursor() as cur:
                with open(os.path.join(ROOT, "v2", "sql", "phase2_scale.sql"), encoding="utf-8") as f:
                    cur.execute(f.read())
                if self.runs:
                    execute_values(cur, "INSERT INTO v2_scale.benchmark_runs (run_label, engine, dataset_version, "
                                        "n_customers, n_transactions, n_repayments, repeat, seconds, status, "
                                        "peak_memory_gb, cores, detail) VALUES %s ON CONFLICT DO NOTHING", self.runs)
                if self.setup:
                    execute_values(cur, "INSERT INTO v2_scale.setup_costs VALUES %s ON CONFLICT DO NOTHING",
                                   self.setup)
                if self.parity:
                    execute_values(cur, "INSERT INTO v2_scale.parity VALUES %s ON CONFLICT DO NOTHING", self.parity)
        finally:
            conn.close()


def check(rec, engine, dataset, reference_name, ref, candidate_path) -> dict:
    p = compare.compare(ref, compare.read_candidate(candidate_path))
    rec.parity.append((rec.label, engine, dataset, reference_name, p["rows_candidate"], p.get("columns_compared", 0),
                       p.get("cells_compared", 0), p.get("cells_mismatched", -1), p.get("max_abs_diff"), p["ok"]))
    print(f"    parity {engine}: {p.get('cells_mismatched')} of {p.get('cells_compared'):,} cells differ "
          f"-> {'OK' if p['ok'] else 'MISMATCH ' + str(p.get('columns_mismatched'))}", flush=True)
    return p


def log_mlflow(label, engine, dataset, n, counts, seconds_list, extra):
    import mlflow
    mlflow.set_tracking_uri(V1["mlflow"]["tracking_uri"])
    mlflow.set_experiment(P2["mlflow_experiment"])
    with mlflow.start_run(run_name=f"{engine}_{n}"):
        mlflow.set_tags({"dataset_version": dataset, "v2_phase": "phase2", "engine": engine, "run_label": label,
                         "model_version": "none (features only)"})
        mlflow.log_params({"n_customers": n, "n_transactions": counts["transactions"],
                           "n_repayments": counts["repayments"], "repeats": len(seconds_list),
                           **{k: v for k, v in extra.items() if not isinstance(v, (int, float)) or k == "cores"}})
        if seconds_list:
            mlflow.log_metrics({"median_s": statistics.median(seconds_list), "min_s": min(seconds_list),
                                "max_s": max(seconds_list)})
        mlflow.log_metrics({k: v for k, v in extra.items() if isinstance(v, (int, float)) and k != "cores"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=P2["sizes"])
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--skip-proof", action="store_true")
    parser.add_argument("--engines", nargs="+", default=["postgresql", "pandas", "spark"],
                        choices=["postgresql", "pandas", "spark"],
                        help="re-run some engines only; parity still uses the PostgreSQL table built earlier")
    args = parser.parse_args()
    label = f"phase2_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    rec = Recorder(label)
    summary = {"run_label": label, "sizes": {}}
    print(f"Run label: {label}", flush=True)

    if not args.skip_proof:
        print("== proof: rebuild v1's own feature store from v1's raw data", flush=True)
        ref = compare.reference(V1_EXPORT)
        spark_stage(V1_EXPORT)
        s = spark_run(V1_EXPORT, 1)
        p_s = check(rec, "spark", V1_EXPORT, "feature_store.customer_features (v1)", ref, spark_fetch(V1_EXPORT))
        pd_ = pandas_run(V1_EXPORT, 1)
        p_p = check(rec, "pandas", V1_EXPORT, "feature_store.customer_features (v1)", ref, pd_["out"])
        summary["proof"] = {"spark_ok": p_s["ok"], "pandas_ok": p_p["ok"], "cells": p_s.get("cells_compared"),
                            "spark_failed": s.get("failed", False)}
        if not (p_s["ok"] and p_p["ok"]):
            rec.save()
            sys.exit("Proof failed: an engine does not reproduce v1's feature store. Stopping.")

    for n in args.sizes:
        dataset = dataset_version(n)
        counts = row_counts(dataset)
        print(f"== {dataset}: {counts}", flush=True)
        result = {"rows": counts}

        if "postgresql" in args.engines:
            load = postgres_engine.load(n)
            for step, sec in load.items():
                rec.setup.append((label, "postgresql", dataset, step, sec))
            pg = [postgres_engine.build(n) for _ in range(args.repeats)]
            pg_s = [r["features_s"] for r in pg]
            rec.run("postgresql", dataset, n, counts, pg_s, cores=None)
            log_mlflow(label, "postgresql", dataset, n, counts, pg_s, {"load_s": sum(load.values())})
            result["postgresql"] = {"runs_s": pg_s, "load_s": load}
            print(f"    postgresql: {pg_s} s (load {sum(load.values()):.0f} s)", flush=True)
        ref = compare.reference(dataset)

        pdr = pandas_run(dataset, args.repeats) if "pandas" in args.engines else {"skipped": True}
        if pdr.get("skipped"):
            pass
        elif pdr.get("failed"):
            rec.run("pandas", dataset, n, counts, [], cores=1, failed_detail=pdr["detail"])
            result["pandas"] = {"failed": True, "detail": pdr["detail"][-400:]}
            print(f"    pandas FAILED: {pdr['detail'][-300:]}", flush=True)
        else:
            rec.run("pandas", dataset, n, counts, pdr["runs_s"], cores=1, peak=pdr["peak_memory_gb"])
            p = check(rec, "pandas", dataset, f"v2_scale.customer_features_n{n}", ref, pdr["out"])
            log_mlflow(label, "pandas", dataset, n, counts, pdr["runs_s"],
                       {"peak_memory_gb": pdr["peak_memory_gb"], "cells_mismatched": p.get("cells_mismatched", -1)})
            result["pandas"] = {"runs_s": pdr["runs_s"], "peak_memory_gb": pdr["peak_memory_gb"], "parity_ok": p["ok"]}
            print(f"    pandas: {pdr['runs_s']} s, peak {pdr['peak_memory_gb']} GB", flush=True)

        if "spark" not in args.engines:
            sr = {"skipped": True}
        else:
            stage_s = spark_stage(dataset)
            rec.setup.append((label, "spark", dataset, "stage_to_docker_volume_s", round(stage_s, 1)))
            sr = spark_run(dataset, args.repeats)
        if sr.get("skipped"):
            pass
        elif sr.get("failed"):
            rec.run("spark", dataset, n, counts, [], cores=SPARK_CORES, failed_detail=sr["detail"])
            result["spark"] = {"failed": True, "detail": sr["detail"][-400:]}
            print(f"    spark FAILED: {sr['detail'][-300:]}", flush=True)
        else:
            rec.setup.append((label, "spark", dataset, "session_start_s", sr["session_start_s"]))
            rec.run("spark", dataset, n, counts, sr["runs_s"], cores=SPARK_CORES)
            p = check(rec, "spark", dataset, f"v2_scale.customer_features_n{n}", ref, spark_fetch(dataset))
            log_mlflow(label, "spark", dataset, n, counts, sr["runs_s"],
                       {"session_start_s": sr["session_start_s"], "cells_mismatched": p.get("cells_mismatched", -1),
                        "cores": SPARK_CORES, "driver_memory": SPARK_DRIVER_MEMORY})
            result["spark"] = {"runs_s": sr["runs_s"], "session_start_s": sr["session_start_s"],
                               "stage_s": round(stage_s, 1), "parity_ok": p["ok"]}
            print(f"    spark: {sr['runs_s']} s (session {sr['session_start_s']} s, staging {stage_s:.0f} s)",
                  flush=True)
        summary["sizes"][n] = result
        rec.save()
        rec.runs, rec.setup, rec.parity = [], [], []
        del ref

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{label}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. v2/reports/phase2/{label}.json", flush=True)


if __name__ == "__main__":
    main()

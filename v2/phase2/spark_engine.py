"""
Sentinel Credit v2, Phase 2: the Spark engine (V2-04).

Plain English: build the feature store with Apache Spark (3.5, local mode in
Docker, all cores of one machine) from one dataset's Parquet files, and time
it. The query is feature_store_spark.sql, a marked transcription of v1's
SQL. Output: Parquet, compared with the PostgreSQL reference by compare.py.

Timing: Spark is lazy, so the timed step is read Parquet -> compute -> write
Parquet, measured around the write. Starting the Spark session (a JVM) is
timed separately, as a fixed cost paid once per job.

Runs inside the Spark container (see spark.ps1 / benchmark.py):
  spark-submit spark_engine.py --dataset <name> --repeats 3
"""

import argparse
import json
import os
import time

started = time.perf_counter()
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SENTINEL_SCALED_DIR", "/data/scaled")
OUT = os.environ.get("SENTINEL_SPARK_OUT", "/data/spark_out")

# [D1] v1 raw column scales (information_schema of raw.*): money 2 dp,
# ratios 4 dp, scores 3 dp. Everything else keeps its Parquet type.
DECIMALS = {
    "customers": {"annual_income_gbp": "14,2", "monthly_income_gbp": "14,2",
                  "existing_monthly_obligations_gbp": "14,2", "savings_balance_gbp": "14,2",
                  "vulnerability_score": "8,3"},
    "transactions": {"order_value_gbp": "14,2", "instalment_amount_gbp": "14,2",
                     "affordability_ratio": "10,4", "merchant_risk_multiplier": "8,3"},
    "repayments": {"instalment_amount_gbp": "14,2", "amount_paid_gbp": "14,2",
                   "outstanding_balance_gbp": "14,2"},
}


def register(spark, folder: str) -> None:
    for table, casts in DECIMALS.items():
        path = os.path.join(folder, f"{table}.parquet")
        frame = spark.read.parquet(path if os.path.exists(path) else os.path.join(folder, table))
        for col, scale in casts.items():
            frame = frame.withColumn(col, F.col(col).cast(f"decimal({scale})"))
        frame.createOrReplaceTempView(f"raw_{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    spark = (SparkSession.builder.appName(f"sentinel_v2_features_{args.dataset}")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "24"))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    session_s = time.perf_counter() - started
    with open(os.path.join(HERE, "feature_store_spark.sql"), encoding="utf-8") as f:
        query = f.read()

    out = os.path.join(OUT, args.dataset)
    runs = []
    for i in range(args.repeats):
        t = time.perf_counter()
        spark.catalog.clearCache()
        register(spark, os.path.join(DATA, args.dataset))
        result = spark.sql(query)
        result.write.mode("overwrite").parquet(out)
        runs.append(round(time.perf_counter() - t, 2))
        print(f"run {i + 1}: {runs[-1]} s", flush=True)
    rows = spark.read.parquet(out).count()
    conf = spark.sparkContext.getConf()
    report = {"engine": "spark", "dataset": args.dataset, "session_start_s": round(session_s, 2),
              "runs_s": runs, "rows": rows, "spark_version": spark.version,
              "master": conf.get("spark.master"), "driver_memory": conf.get("spark.driver.memory", "default"),
              "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
              "cores": spark.sparkContext.defaultParallelism}
    with open(os.path.join(out, "_timing.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))
    spark.stop()


if __name__ == "__main__":
    main()

"""
Sentinel Credit v2, Phase 2: the PostgreSQL baseline (V2-04).

Plain English: the reference answer. Load one scaled dataset into its own
schema, then build the feature store with v1's own SQL file
(sql/02_feature_store.sql, read-only), pointed at that schema. The other
engines (pandas, Spark) must reproduce this table exactly.

- load: copies the Parquet files into v2_scale_raw_n<N>.customers,
  .transactions, .repayments, with exactly v1's column types
  (CREATE TABLE ... (LIKE raw.<table>): v1's definitions are read, never
  changed). Timed separately: loading is not feature engineering.
- build: runs the rewritten v1 SQL in one transaction and times the
  feature table (CREATE TABLE AS) and the indexes separately.

  python v2/phase2/postgres_engine.py load --customers 250000
  python v2/phase2/postgres_engine.py build --customers 250000
"""

import argparse
import io
import os
import re
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_SCHEMA, dataset_dir, dataset_version, db_connect, raw_schema, v1_sql_for  # noqa: E402

TABLES = ("customers", "transactions", "repayments")


def load(n_customers: int) -> dict:
    schema = raw_schema(n_customers)
    folder = dataset_dir(dataset_version(n_customers))
    conn = db_connect()
    timings = {}
    try:
        with conn, conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"CREATE SCHEMA {schema}")
            for t in TABLES:
                cur.execute(f"CREATE TABLE {schema}.{t} (LIKE raw.{t})")
        for t in TABLES:
            path = os.path.join(folder, f"{t}.parquet")
            files = [path] if os.path.exists(path) else sorted(
                os.path.join(folder, t, f) for f in os.listdir(os.path.join(folder, t)))
            with conn.cursor() as cur:
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = %s "
                            "AND table_name = %s ORDER BY ordinal_position", (schema, t))
                columns = [r[0] for r in cur.fetchall()]
            started = time.perf_counter()
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=columns):
                    buf = io.StringIO()
                    batch.to_pandas()[columns].to_csv(buf, index=False, header=False)
                    buf.seek(0)
                    with conn.cursor() as cur:
                        cur.copy_expert(f"COPY {schema}.{t} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)", buf)
                    conn.commit()
            with conn, conn.cursor() as cur:
                cur.execute(f"ANALYZE {schema}.{t}")
            timings[f"load_{t}_s"] = round(time.perf_counter() - started, 1)
    finally:
        conn.close()
    return timings


def build(n_customers: int) -> dict:
    sql = v1_sql_for(n_customers)
    create = re.search(r"CREATE TABLE .*?;", sql, flags=re.S).group(0)
    indexes = re.findall(r"CREATE (?:UNIQUE )?INDEX [^;]+;", sql)
    target = f"{RESULTS_SCHEMA}.customer_features_n{n_customers}"
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {RESULTS_SCHEMA}")
            cur.execute(f"DROP TABLE IF EXISTS {target}")
            started = time.perf_counter()
            cur.execute(create)
            features_s = time.perf_counter() - started
            started = time.perf_counter()
            for statement in indexes:
                cur.execute(statement)
            indexes_s = time.perf_counter() - started
            cur.execute(f"SELECT count(*) FROM {target}")
            rows = cur.fetchone()[0]
    finally:
        conn.close()
    return {"features_s": round(features_s, 2), "indexes_s": round(indexes_s, 2), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["load", "build"])
    parser.add_argument("--customers", type=int, required=True)
    args = parser.parse_args()
    print((load if args.command == "load" else build)(args.customers))


if __name__ == "__main__":
    main()

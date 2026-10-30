"""
Sentinel Credit v2, Phase 2: copy v1's raw tables to Parquet, read-only (V2-04).

Plain English: before Spark is trusted on new data, it must rebuild v1's own
feature store exactly from v1's own raw data. This copies raw.customers,
raw.transactions and raw.repayments to Parquet, outside the repo
($SENTINEL_V2_DATA_DIR/scaled/v1_raw_export/), with SELECTs only. Numbers are
written the same way as the scaled datasets (64-bit floats; every raw value has at most
4 decimal places, so the shortest decimal form round-trips exactly).

  python v2/phase2/export_v1_raw.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V1_EXPORT, dataset_dir, db_connect  # noqa: E402


def main():
    out = dataset_dir(V1_EXPORT)
    os.makedirs(out, exist_ok=True)
    conn = db_connect()
    try:
        for table in ("customers", "transactions", "repayments"):
            frame = pd.read_sql(f"SELECT * FROM raw.{table}", conn)
            numeric = pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_schema = 'raw' "
                                  "AND table_name = %s AND data_type = 'numeric'", conn, params=(table,))
            for col in numeric["column_name"]:
                frame[col] = frame[col].astype(float)
            frame.to_parquet(os.path.join(out, f"{table}.parquet"), index=False)
            print(f"raw.{table}: {len(frame):,} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

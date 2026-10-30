"""
Sentinel Credit v2, Phase 2: parity check between feature-store builds (V2-04).

Plain English: is an engine's feature store the same as the reference,
customer by customer and column by column? Counts, dates, flags and text
must match exactly; decimal columns within 1e-9; NULL must match NULL.

Reference: PostgreSQL running v1's SQL. For v1's own data that is v1's
feature_store.customer_features itself (read-only); for a scaled dataset
it is v2_scale.customer_features_n<N>.

  python v2/phase2/compare.py --dataset v1_raw_export --candidate <parquet folder>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_SCHEMA, V1_EXPORT, db_connect  # noqa: E402

TOL = 1e-9


def reference(dataset: str) -> pd.DataFrame:
    table = ("feature_store.customer_features" if dataset == V1_EXPORT
             else f"{RESULTS_SCHEMA}.customer_features_n{int(dataset.rsplit('_n', 1)[1])}")
    conn = db_connect()
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)   # read-only
    finally:
        conn.close()


def read_candidate(path) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        return path
    files = sorted(glob.glob(os.path.join(path, "*.parquet"))) if os.path.isdir(path) else [path]
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def compare(ref: pd.DataFrame, cand: pd.DataFrame) -> dict:
    report = {"rows_reference": len(ref), "rows_candidate": len(cand)}
    missing_cols = [c for c in ref.columns if c not in cand.columns]
    report["missing_columns"] = missing_cols
    report["column_order_matches"] = list(cand.columns) == list(ref.columns)
    ref = ref.set_index("customer_id").sort_index()
    cand = cand.set_index("customer_id").sort_index()
    report["same_customers"] = bool(ref.index.equals(cand.index))
    if not report["same_customers"]:
        report["ok"] = False
        return report
    per_column, bad_columns = {}, []
    for col in ref.columns:
        if col in missing_cols:
            continue
        r, c = ref[col], cand[col]
        null_mismatch = int((r.isna() != c.isna()).sum())
        both = r.notna() & c.notna()
        numeric = pd.api.types.is_numeric_dtype(r) or r.dropna().map(type).astype(str).str.contains("Decimal").any()
        if numeric:
            diff = (pd.to_numeric(r[both]).astype(float) - pd.to_numeric(c[both]).astype(float)).abs()
            value_mismatch = int((diff > TOL).sum())
            max_diff = float(diff.max()) if len(diff) else 0.0
        else:
            rv = r[both].astype(str).str.slice(0, 10) if "date" in col else r[both].astype(str)
            cv = c[both].astype(str).str.slice(0, 10) if "date" in col else c[both].astype(str)
            value_mismatch = int((rv != cv).sum())
            max_diff = None
        per_column[col] = {"null_mismatch": null_mismatch, "value_mismatch": value_mismatch, "max_abs_diff": max_diff}
        if null_mismatch or value_mismatch:
            bad_columns.append(col)
    report.update({"columns_compared": len(per_column), "columns_mismatched": bad_columns,
                   "cells_compared": int(len(ref) * len(per_column)),
                   "cells_mismatched": int(sum(v["null_mismatch"] + v["value_mismatch"] for v in per_column.values())),
                   "max_abs_diff": float(np.nanmax([v["max_abs_diff"] or 0.0 for v in per_column.values()])),
                   "per_column": {k: v for k, v in per_column.items() if k in bad_columns}})
    report["ok"] = not missing_cols and not bad_columns and report["same_customers"]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    r = compare(reference(args.dataset), read_candidate(args.candidate))
    print(json.dumps({k: v for k, v in r.items() if k != "per_column"}, indent=2))
    if r.get("per_column"):
        print(json.dumps(r["per_column"], indent=2)[:3000])
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()

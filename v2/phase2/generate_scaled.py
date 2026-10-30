"""
Sentinel Credit v2, Phase 2: scaled synthetic datasets (V2-04).

Plain English: make bigger versions of the synthetic BNPL data, so the
feature store can be rebuilt with Spark at realistic laptop scale. It runs
the v1 generator's own functions, unchanged and imported read-only, with two
settings changed in memory only: the number of customers, and the random
seed (2026, owner's choice, so the data is clearly a new dataset and not a
superset of v1's customers). v1's config.yaml and code are never edited,
and v1's save step (which writes v1's CSVs and data dictionary) is never
called.

Chunks: the v1 generator holds every repayment row as a Python dictionary
(about 2 KB each), so 2 million customers (about 19 million rows) would
need roughly 40 GB of memory in one go. Customers are therefore generated
once, then applications and repayments in chunks of 100,000 customers, in
parallel processes. Chunk k uses seed 2026 + 1000 * k (the generator adds
+1 and +2 for its own steps, so chunks never share a random stream). Chunk
0 uses exactly the single-run seed, so a dataset of 100,000 customers or
fewer is identical to running the generator in one go. IDs are built from
customer IDs, which are generated once, so they are unique across chunks.

Output: Parquet in $SENTINEL_V2_DATA_DIR/scaled/<dataset_version>/ (outside the repo and
any synced folder; the repo-relative fallback v2/data/ is git-ignored):
customers.parquet, transactions/part-*.parquet, repayments/part-*.parquet,
and manifest.json (row counts, timings, exact settings).

  python v2/phase2/generate_scaled.py --customers 250000
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "data_generator"))

import bnpl_data_generator as gen  # noqa: E402  (v1 generator, read-only)

with open(os.path.join(ROOT, "v2", "config_v2.yaml"), encoding="utf-8") as f:
    P2 = yaml.safe_load(f)["phase2"]

CHUNK = 100_000
WORKERS = 4                      # about 2 GB of memory each at 100,000 customers


def dataset_version(n_customers: int) -> str:
    return f"{P2['dataset_prefix']}_seed{P2['seed']}_n{n_customers}"


def as_v1_types(frame):
    """Timestamp columns -> plain dates, as in v1's raw tables (type date).
    pandas would otherwise write nanosecond timestamps, which Spark 3.5
    cannot read ("Illegal Parquet type: INT64 (TIMESTAMP(NANOS,false))",
    found in the first benchmark run). Values are unchanged."""
    for col in frame.columns:
        if str(frame[col].dtype).startswith("datetime64"):
            frame[col] = frame[col].dt.date
    return frame


def retype_existing(n_customers: int) -> None:
    """Apply as_v1_types to an already generated customers.parquet."""
    path = os.path.join(_data_root(), "scaled", dataset_version(n_customers), "customers.parquet")
    as_v1_types(pd.read_parquet(path)).to_parquet(path, index=False)


def _data_root() -> str:
    return os.environ.get("SENTINEL_V2_DATA_DIR", os.path.join(ROOT, P2["data_dir"]))


def _use_seed(seed: int) -> None:
    # In-memory override only: the generator reads RANDOM_SEED at call time.
    gen.RANDOM_SEED = seed
    np.random.seed(seed)         # the generator also seeds the global RNG at import


def _chunk(args) -> dict:
    """Applications and repayments for one chunk of customers (own process)."""
    k, customers, out_dir = args
    _use_seed(int(P2["seed"]) + 1000 * k)
    t = time.perf_counter()
    transactions = gen.generate_transactions(customers)
    t_tx = time.perf_counter() - t
    t = time.perf_counter()
    repayments = gen.generate_repayments(customers, transactions)
    t_rep = time.perf_counter() - t
    as_v1_types(transactions).to_parquet(os.path.join(out_dir, "transactions", f"part-{k:04d}.parquet"), index=False)
    as_v1_types(repayments).to_parquet(os.path.join(out_dir, "repayments", f"part-{k:04d}.parquet"), index=False)
    return {"chunk": k, "transactions": len(transactions), "repayments": len(repayments),
            "approved": int(transactions["application_approved"].sum()),
            "transactions_s": round(t_tx, 1), "repayments_s": round(t_rep, 1)}


def generate(n_customers: int) -> dict:
    version = dataset_version(n_customers)
    out_dir = os.path.join(_data_root(), "scaled", version)
    for sub in ("transactions", "repayments"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    started = time.perf_counter()

    gen.GEN["n_customers"] = n_customers
    _use_seed(int(P2["seed"]))
    t = time.perf_counter()
    customers = gen.generate_customers()
    customers_s = time.perf_counter() - t
    as_v1_types(customers.copy()).to_parquet(os.path.join(out_dir, "customers.parquet"), index=False)

    jobs = [(k, customers.iloc[i:i + CHUNK].reset_index(drop=True), out_dir)
            for k, i in enumerate(range(0, n_customers, CHUNK))]
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as pool:
        chunks = list(pool.map(_chunk, jobs))

    manifest = {
        "dataset_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "data_generator/bnpl_data_generator.py (v1, unchanged)",
        "overrides": {"n_customers": n_customers, "random_seed": int(P2["seed"]),
                      "chunk_customers": CHUNK, "chunk_seed": "seed + 1000 * chunk", "workers": WORKERS},
        "rows": {"customers": len(customers), "transactions": sum(c["transactions"] for c in chunks),
                 "repayments": sum(c["repayments"] for c in chunks)},
        "approved_applications": sum(c["approved"] for c in chunks),
        "timings_s": {"customers": round(customers_s, 1), "total_wall": round(time.perf_counter() - started, 1)},
        "chunks": chunks,
        "bytes_on_disk": sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out_dir)
                             for f in fs if f.endswith(".parquet")),
        "note": "Synthetic data at scale for Phase 2 (features at scale only; no model is trained on it).",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--customers", type=int, required=True)
    parser.add_argument("--retype-only", action="store_true",
                        help="only rewrite an existing customers.parquet with v1's date type")
    args = parser.parse_args()
    if args.retype_only:
        retype_existing(args.customers)
        print(f"{dataset_version(args.customers)}: customers.parquet rewritten with plain dates")
        return
    m = generate(args.customers)
    print(f"{m['dataset_version']}: {m['rows']} in {m['timings_s']['total_wall']} s, "
          f"{m['bytes_on_disk'] / 1e6:.0f} MB on disk")


if __name__ == "__main__":
    main()

# Sentinel Credit v2, Phase 2: the feature store at scale

**The question:** v1 builds its feature store with one SQL script over 50,000 customers. Does
the same logic hold at millions of rows, can Spark give exactly the same answers, and when
does Spark pay off on one machine? Full reasoning and results: **V2-04** in the
[v2 Engineering Decision Log](../../docs/v2_engineering_decision_log.md).

**The data:** v1's own synthetic generator, unchanged, run at 50k, 250k, 1M and 2M customers
with a new seed (2026). This is synthetic data at scale, not real customers. No model is
trained on it.

## Results (real runs, 30th October 2026)

Every engine matched PostgreSQL (v1's own SQL) in **every cell at every size**: 0 of
126 million cells different at 2 million customers. Spark and pandas also rebuilt v1's own
feature store exactly.

Median seconds to build the feature store:

| Customers | Repayment rows | PostgreSQL | pandas (1 core) | Spark (8 cores, Docker) |
|---:|---:|---:|---:|---:|
| 50,000 | 0.5M | 2.7 | **2.2** | 24.6 |
| 250,000 | 2.4M | 18.9 | **12.9** | 49.2 |
| 1,000,000 | 9.7M | 82.5 | **69.9** | 190.7 |
| 2,000,000 | 19.4M | **211** | 302 (9.1 GB) | 354 |

- **Spark did not pay off by 2 million customers on one laptop.** It was slowest at every
  size, but scaled best: from 1M to 2M, Spark took 1.9× longer, pandas 4.3×.
- **pandas was close to the machine's memory** at 2M (9.1 GB of 15.4 GB).

## Files

| File | What it does |
|---|---|
| `generate_scaled.py` | Scaled datasets from v1's generator (in-memory overrides, chunked, parallel) |
| `export_v1_raw.py` | v1's raw tables to Parquet, read-only, for the exactness proof |
| `postgres_engine.py` | Loads a dataset into its own schema; runs v1's SQL against it (refuses to touch v1) |
| `pandas_engine.py` | The feature store in pandas, with PostgreSQL-exact rounding |
| `spark_engine.py`, `feature_store_spark.sql` | The feature store in Spark SQL (marked transcription of v1's SQL) |
| `compare.py` | Cell-by-cell parity against the PostgreSQL reference |
| `benchmark.py` | The full run: proof, then every size and engine, 3 times each |
| `common.py` | Settings, paths, the guarded rewrite of v1's SQL |
| `../sql/phase2_scale.sql` | Results schema `v2_scale` |
| `../../tests/test_v2_phase2.py` | Unit tests |

## Run it

Needs the v1 PostgreSQL database, Docker (for Spark) and `SENTINEL_DB_PASSWORD`. Keep the
data folder **outside OneDrive** or any synced folder (about 1 GB):

```powershell
$env:SENTINEL_V2_DATA_DIR = "C:\Users\<you>\sentinel_v2_data"
python v2/phase2/generate_scaled.py --customers 250000
python v2/phase2/export_v1_raw.py
python v2/phase2/benchmark.py --sizes 50000 250000
```

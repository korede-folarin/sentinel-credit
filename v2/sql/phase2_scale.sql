-- Sentinel Credit v2, Phase 2: feature engineering at scale, results (V2-04).
-- A new schema. The scaled datasets themselves live in v2_scale_raw_n<N>
-- schemas and their PostgreSQL feature stores in v2_scale.customer_features_n<N>;
-- nothing in raw, feature_store, reporting, v2_results or v2_stream is touched.

CREATE SCHEMA IF NOT EXISTS v2_scale;

-- One row per timed run of one engine on one dataset.
CREATE TABLE IF NOT EXISTS v2_scale.benchmark_runs (
    run_label        text        NOT NULL,
    engine           text        NOT NULL CHECK (engine IN ('postgresql', 'pandas', 'spark')),
    dataset_version  text        NOT NULL,
    n_customers      integer     NOT NULL,
    n_transactions   bigint,
    n_repayments     bigint,
    repeat           integer     NOT NULL,
    seconds          double precision,          -- NULL when the run failed
    status           text        NOT NULL CHECK (status IN ('ok', 'failed')),
    peak_memory_gb   double precision,
    cores            integer,
    detail           text,
    recorded_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_label, engine, dataset_version, repeat)
);

-- Fixed costs kept out of the feature-engineering timings (loading data
-- into PostgreSQL, staging files for Spark, starting a Spark session).
CREATE TABLE IF NOT EXISTS v2_scale.setup_costs (
    run_label        text        NOT NULL,
    engine           text        NOT NULL,
    dataset_version  text        NOT NULL,
    step             text        NOT NULL,
    seconds          double precision NOT NULL,
    PRIMARY KEY (run_label, engine, dataset_version, step)
);

-- Parity of each engine's feature store with the PostgreSQL reference.
CREATE TABLE IF NOT EXISTS v2_scale.parity (
    run_label          text    NOT NULL,
    engine             text    NOT NULL,
    dataset_version    text    NOT NULL,
    reference          text    NOT NULL,
    rows_compared      bigint  NOT NULL,
    columns_compared   integer NOT NULL,
    cells_compared     bigint  NOT NULL,
    cells_mismatched   bigint  NOT NULL,
    max_abs_diff       double precision,
    ok                 boolean NOT NULL,
    PRIMARY KEY (run_label, engine, dataset_version)
);

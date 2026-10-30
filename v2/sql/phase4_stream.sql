-- Sentinel Credit v2, Phase 4: streaming results (V2-03).
-- A new schema; nothing in raw, feature_store, reporting or v2_results is
-- touched. Every row carries the replay run_id. Simulated stream only.

CREATE SCHEMA IF NOT EXISTS v2_stream;

-- One row per replay: what was asked for, what the simulator sent.
-- mode is always 'shadow': decisions are logged, never returned to anyone.
CREATE TABLE IF NOT EXISTS v2_stream.replay_runs (
    run_id              text PRIMARY KEY,
    status              text NOT NULL CHECK (status IN ('requested', 'running', 'finished', 'failed')),
    requested_by        text,
    requested_at        timestamptz NOT NULL DEFAULT now(),
    rate_eps            integer NOT NULL,
    n_customers         integer NOT NULL,
    seed                integer NOT NULL,
    duplicate_rate      numeric NOT NULL,
    late_rate           numeric NOT NULL,
    late_max_s          integer NOT NULL,
    n_events            integer,
    n_messages_sent     integer,
    n_duplicates_sent   integer,
    n_late_sent         integer,
    started_at          timestamptz,
    stream_end_at       timestamptz,
    finished_at         timestamptz,
    mode                text NOT NULL DEFAULT 'shadow' CHECK (mode = 'shadow'),
    dataset_version     text NOT NULL,
    model_version       text NOT NULL,
    note                text
);

-- One row per application scored in the stream (point-in-time features).
-- Primary key (run_id, event_id): a re-delivered output is ignored.
CREATE TABLE IF NOT EXISTS v2_stream.scored_events (
    run_id                            text NOT NULL,
    event_id                          text NOT NULL,
    customer_id                       text NOT NULL,
    transaction_id                    text NOT NULL,
    hist_date                         date NOT NULL,
    event_time_ms                     bigint NOT NULL,
    sent_ms                           bigint NOT NULL,
    processed_ms                      bigint NOT NULL,
    stored_ms                         bigint NOT NULL,
    late                              boolean NOT NULL,
    delayed_profile                   boolean NOT NULL,
    monthly_income_gbp                numeric,
    existing_monthly_obligations_gbp  numeric,
    savings_balance_gbp               numeric,
    credit_score                      numeric,
    months_credit_history             numeric,
    num_credit_accounts               numeric,
    num_defaults_24m                  numeric,
    thin_file_flag                    numeric,
    max_affordability_ratio           numeric,
    avg_order_value_gbp               numeric,
    avg_n_instalments                 numeric,
    pct_high_risk_category            numeric,
    pct_medium_risk_category          numeric,
    pd                                double precision NOT NULL,
    decision                          text NOT NULL,
    hist_approved                     integer NOT NULL,
    model_version                     text NOT NULL,
    PRIMARY KEY (run_id, event_id)
);

-- Latest state per customer per run (compared with the v1 feature store).
-- seq orders updates: an older update arriving later never overwrites.
CREATE TABLE IF NOT EXISTS v2_stream.customer_state (
    run_id                            text NOT NULL,
    customer_id                       text NOT NULL,
    seq                               integer NOT NULL,
    monthly_income_gbp                numeric,
    existing_monthly_obligations_gbp  numeric,
    savings_balance_gbp               numeric,
    credit_score                      numeric,
    months_credit_history             numeric,
    num_credit_accounts               numeric,
    num_defaults_24m                  numeric,
    thin_file_flag                    numeric,
    max_affordability_ratio           numeric,
    avg_order_value_gbp               numeric,
    avg_n_instalments                 numeric,
    pct_high_risk_category            numeric,
    pct_medium_risk_category          numeric,
    updated_ms                        bigint NOT NULL,
    PRIMARY KEY (run_id, customer_id)
);

-- Every processed event (registrations and applications), for latency,
-- throughput, completeness and where the time goes. late = arrived behind
-- the watermark.
CREATE TABLE IF NOT EXISTS v2_stream.processed_events (
    run_id        text NOT NULL,
    event_id      text NOT NULL,
    kind          text NOT NULL,
    event_time_ms bigint NOT NULL,
    sent_ms       bigint NOT NULL,
    features_ms   bigint NOT NULL,   -- left the per-customer feature step
    processed_ms  bigint NOT NULL,   -- left the scoring step
    stored_ms     bigint NOT NULL,   -- committed by the sink
    late          boolean NOT NULL,
    PRIMARY KEY (run_id, event_id)
);

-- Duplicates Flink recognised and dropped (copy = which send it was).
CREATE TABLE IF NOT EXISTS v2_stream.duplicates_dropped (
    run_id        text NOT NULL,
    event_id      text NOT NULL,
    copy          integer NOT NULL,
    processed_ms  bigint NOT NULL,
    PRIMARY KEY (run_id, event_id, copy)
);

-- Consumer lag samples taken by the simulator while it sends.
CREATE TABLE IF NOT EXISTS v2_stream.lag_samples (
    run_id          text NOT NULL,
    sampled_ms      bigint NOT NULL,
    produced        bigint NOT NULL,
    flink_committed bigint,
    stored          bigint,
    PRIMARY KEY (run_id, sampled_ms)
);

-- Results of the checks (parity, latency, throughput, completeness, drift).
CREATE TABLE IF NOT EXISTS v2_stream.run_metrics (
    run_id       text NOT NULL,
    metric       text NOT NULL,
    value        double precision,
    detail       text,
    computed_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, metric)
);

CREATE TABLE IF NOT EXISTS v2_stream.feature_drift (
    run_id      text NOT NULL,
    comparison  text NOT NULL,
    feature     text NOT NULL,
    psi         double precision,
    band        text,
    PRIMARY KEY (run_id, comparison, feature)
);

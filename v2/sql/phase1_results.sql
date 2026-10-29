-- Sentinel Credit v2, Phase 1 results (v2 Engineering Decision Log, V2-02).
-- Plain English: durable tables for the PyTorch challenger's results, in a
-- schema of their own. Nothing in raw, feature_store or reporting (v1) is
-- read for writing or altered. Rows are appended per run, keyed by run_label,
-- so results are never overwritten.

CREATE SCHEMA IF NOT EXISTS v2_results;

-- One row per model per split: champion and challenger side by side.
CREATE TABLE IF NOT EXISTS v2_results.model_comparison (
    run_label             text        NOT NULL,
    created_at            timestamptz NOT NULL DEFAULT now(),
    split                 text        NOT NULL CHECK (split IN ('stratified', 'temporal')),
    model                 text        NOT NULL,
    role                  text        NOT NULL CHECK (role IN ('champion', 'challenger')),
    mlflow_run_id         text,
    dataset_version       text        NOT NULL,
    n_test                integer     NOT NULL,
    n_defaults            integer     NOT NULL,
    gini                  numeric(7,4) NOT NULL,
    auc                   numeric(7,4) NOT NULL,
    ks                    numeric(7,4) NOT NULL,
    pr_auc                numeric(7,4) NOT NULL,
    brier                 numeric(7,4) NOT NULL,
    ece                   numeric(7,4) NOT NULL,
    business_cutoff       numeric(7,4) NOT NULL,
    share_sent_to_review  numeric(7,4) NOT NULL,
    recall_at_cutoff      numeric(7,4) NOT NULL,
    precision_at_cutoff   numeric(7,4) NOT NULL,
    bad_rate_not_reviewed numeric(7,4),
    PRIMARY KEY (run_label, split, model)
);

-- Error rates by segment at the business cutoff (Fairness and Segments page).
CREATE TABLE IF NOT EXISTS v2_results.segment_metrics (
    run_label           text        NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now(),
    split               text        NOT NULL,
    model               text        NOT NULL,
    segment             text        NOT NULL,
    segment_value       text        NOT NULL,
    n_customers         integer     NOT NULL,
    n_defaults          integer     NOT NULL,
    actual_default_rate numeric(7,4) NOT NULL,
    avg_predicted_pd    numeric(7,4) NOT NULL,
    share_sent_to_review numeric(7,4) NOT NULL,
    false_negative_rate numeric(7,4) NOT NULL,
    false_positive_rate numeric(7,4) NOT NULL,
    PRIMARY KEY (run_label, split, model, segment, segment_value)
);

-- Autoencoder: default rate by reconstruction-error decile (1 = most normal).
CREATE TABLE IF NOT EXISTS v2_results.anomaly_deciles (
    run_label            text        NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now(),
    split                text        NOT NULL,
    error_decile         integer     NOT NULL CHECK (error_decile BETWEEN 1 AND 10),
    n_customers          integer     NOT NULL,
    n_defaults           integer     NOT NULL,
    default_rate         numeric(7,4) NOT NULL,
    mean_reconstruction_error numeric(12,6) NOT NULL,
    mean_champion_pd     numeric(7,4) NOT NULL,
    PRIMARY KEY (run_label, split, error_decile)
);

-- Feature attributions: the challenger's (Captum) next to the champion's (SHAP).
CREATE TABLE IF NOT EXISTS v2_results.feature_attributions (
    run_label       text        NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    model           text        NOT NULL,
    method          text        NOT NULL,
    feature         text        NOT NULL,
    mean_abs_attribution numeric(14,8) NOT NULL,
    share_of_total  numeric(7,4) NOT NULL,
    rank            integer     NOT NULL,
    PRIMARY KEY (run_label, model, feature)
);

-- The promotion evaluation and Approval Gate outcome, logged whatever it is.
CREATE TABLE IF NOT EXISTS v2_results.gate_outcomes (
    run_label               text        PRIMARY KEY,
    evaluated_at            timestamptz NOT NULL DEFAULT now(),
    challenger_model        text        NOT NULL,
    champion_run_stratified text        NOT NULL,
    champion_run_temporal   text        NOT NULL,
    challenger_run_stratified text      NOT NULL,
    challenger_run_temporal text        NOT NULL,
    delta_gini_stratified   numeric(7,4) NOT NULL,
    delta_gini_temporal     numeric(7,4) NOT NULL,
    min_gini_delta          numeric(7,4),
    material_win_both_splits boolean    NOT NULL,
    approval_gate_status    text        NOT NULL,
    promotion_recommended   boolean     NOT NULL,
    promoted                boolean     NOT NULL DEFAULT false CHECK (promoted = false),
    outcome                 text        NOT NULL,
    detail                  jsonb       NOT NULL
);

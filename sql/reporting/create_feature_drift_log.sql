-- Sentinel Credit, Feature Drift Log
--
-- Purpose
-- One row per model feature per pipeline/monitor_drift.py run,
-- appended by log_drift_history(). Gives drift monitoring a
-- history; reports/monitoring/feature_drift.csv is overwritten
-- every run and holds only the latest result.
--
-- run_id groups the rows of one run (same value is written to
-- monitoring_run_metadata.json and the MLflow run's tags).
--
-- The 14 rows logged on 2026-08-13 23:41 predate run_id and keep
-- run_id = NULL deliberately: they are a real run, but no id was
-- recorded for it, and one is not invented after the fact. Group
-- them by logged_at.
--
-- Limitation (see docs/audit_findings_2026-10-12.md, A2/A3): the
-- reference and comparison populations are both fixed snapshots,
-- so repeated runs on unchanged data log identical values. Rows
-- accumulating over time are NOT a drift trend.
--
-- Safe to re-run.

CREATE SCHEMA IF NOT EXISTS reporting;

CREATE TABLE IF NOT EXISTS reporting.feature_drift_log (
    feature TEXT,
    psi NUMERIC,
    psi_band TEXT,
    reference_null_rate NUMERIC,
    comparison_null_rate NUMERIC,
    null_rate_delta NUMERIC,
    reference_mean NUMERIC,
    comparison_mean NUMERIC,
    mean_shift_sigma NUMERIC,
    out_of_range_rate NUMERIC,
    alerts TEXT,
    any_alert BOOLEAN,
    logged_at TIMESTAMPTZ,
    run_id TEXT
);

-- Migration for the table as first created (without run_id).
ALTER TABLE reporting.feature_drift_log
    ADD COLUMN IF NOT EXISTS run_id TEXT;

CREATE INDEX IF NOT EXISTS feature_drift_log_run_id_idx
    ON reporting.feature_drift_log (run_id);

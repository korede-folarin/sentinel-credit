-- Sentinel Credit, Feature Drift by Origination Cohort
--
-- Purpose
-- PSI per model feature for each origination quarter of held-out
-- customers (the stratified test partition: never seen in training)
-- against the champion's training reference. Written by
-- pipeline/monitor_drift.py (write_cohort_drift) on every run.
--
-- Why this exists: reporting.feature_drift_log compares two fixed
-- snapshots, so every run logs identical values and its rows are
-- not a trend (audit A3). This table gives drift a REAL time axis —
-- when customers originated — computed from actual cohorts. It is
-- not a run history, and not backfilled history: each run replaces
-- the whole table with its own computation (run_id, computed_at
-- identify which run produced it).
--
-- Reading it:
--   scored = false      cohort smaller than MIN_COHORT_N (200); psi NULL
--   psi_noise_floor     PSI expected from sampling alone at these sample
--                       sizes, ~ (bins - 1) * (1/n + 1/N_reference).
--                       A psi near this value is indistinguishable from
--                       noise.
--   cohort_bad_rate     observed default rate of the cohort
--
-- Informational only: does not feed the alert count or the approval
-- gate. Decision 25.
--
-- Safe to re-run.

CREATE SCHEMA IF NOT EXISTS reporting;

CREATE TABLE IF NOT EXISTS reporting.feature_drift_by_cohort (
    cohort_period    TEXT NOT NULL,     -- e.g. '2024Q3'
    n_customers      INTEGER NOT NULL,
    scored           BOOLEAN NOT NULL,
    feature          TEXT NOT NULL,
    psi              NUMERIC,
    psi_band         TEXT,
    psi_noise_floor  NUMERIC,
    cohort_bad_rate  NUMERIC,
    run_id           TEXT NOT NULL,
    computed_at      TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (cohort_period, feature)
);

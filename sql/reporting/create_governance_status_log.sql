-- Sentinel Credit, Governance Status Log
--
-- Purpose
-- A small, separate table, new, does not touch or modify any
-- existing view or table in this project. Every time
-- pipeline/approval_gate.py runs, it writes one new row here
-- with the real result it just computed. Power BI reads the
-- latest row through reporting.governance_status_latest, live,
-- via the same DirectQuery connection already used for every
-- other reporting view, so the gate badge reflects the actual
-- production decision, not a manually typed value.
--
-- This is additive only. Nothing in create_reporting_views.sql
-- changes. approval_gate.py's actual decision logic, the three
-- checks it already runs, is unchanged, this only adds where
-- the final answer gets saved.

CREATE SCHEMA IF NOT EXISTS reporting;

CREATE TABLE IF NOT EXISTS reporting.governance_status_log (
    id SERIAL PRIMARY KEY,
    gate_status text NOT NULL,
    evaluated_at timestamptz NOT NULL,
    n_alerts integer,
    n_features_total integer,
    monitoring_age_hours numeric,
    max_monitoring_age_hours integer,
    created_at timestamptz DEFAULT now()
);

-- Status is derived at READ time, not only at write time.
--
-- A logged PASS is only true while the monitoring evidence behind
-- it is fresh. The evidence's age now is its age when the gate ran
-- (monitoring_age_hours) plus the time since the gate ran. Once
-- that exceeds the gate's own max_monitoring_age_hours (stored on
-- the row, from config.yaml approval_gate), gate_status reads
-- STALE instead of PASS. No second threshold is introduced here,
-- so this view and approval_gate.py cannot disagree about what
-- "too old" means. Power BI queries this view live (DirectQuery),
-- so the badge flips on its own when the pipeline stops writing.
--
-- A logged BLOCK stays BLOCK: it already withholds approval, and
-- age cannot make it less blocking. is_stale still reports age.
--
-- gate_status keeps its original column name so existing report
-- bindings keep working; the as-logged value is
-- recorded_gate_status.
--
-- Time basis: evaluated_at is timestamptz (an absolute instant) and
-- is compared against now(), so rows from the UTC Airflow container
-- and the Europe/London Windows host are on one timeline. Converted
-- from naive timestamps by
-- sql/reporting/migrations/2026_10_13_timestamptz.sql.
CREATE OR REPLACE VIEW reporting.governance_status_latest AS
WITH latest AS (
    SELECT
        *,
        monitoring_age_hours
            + EXTRACT(EPOCH FROM (now() - evaluated_at)) / 3600.0
            AS evidence_age_hours_now
    FROM reporting.governance_status_log
    ORDER BY evaluated_at DESC
    LIMIT 1
)
SELECT
    CASE
        WHEN gate_status = 'PASS'
             AND evidence_age_hours_now > max_monitoring_age_hours
        THEN 'STALE'
        ELSE gate_status
    END AS gate_status,
    evaluated_at,
    n_alerts,
    n_features_total,
    monitoring_age_hours,
    max_monitoring_age_hours,
    gate_status AS recorded_gate_status,
    ROUND(evidence_age_hours_now::numeric, 1) AS evidence_age_hours_now,
    COALESCE(evidence_age_hours_now > max_monitoring_age_hours, false)
        AS is_stale
FROM latest;

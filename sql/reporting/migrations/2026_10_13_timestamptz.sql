-- Sentinel Credit, one-off migration: naive TIMESTAMP -> TIMESTAMPTZ
-- for reporting.governance_status_log and reporting.feature_drift_log.
-- See Decision 24 (Addendum) and docs/audit_findings_2026-10-12.md
-- A15/A17.
--
-- Why: rows were written as naive local times by two writers in
-- different zones — the Windows host (Europe/London) and the
-- Airflow container (UTC). Stored without a zone, the two cannot be
-- ordered or compared correctly. After this migration every row is
-- an absolute instant, and writers send offset-aware values.
--
-- Each existing row is converted from the zone it was ACTUALLY
-- written in, established from evidence, not assumed:
--   governance_status_log, ids 1-3: approval_gate.py run on the
--     Windows host (gate never ran in the container before this
--     migration) -> Europe/London.
--   created_at: column DEFAULT now() evaluated by the server, whose
--     TimeZone is Europe/London -> Europe/London.
--   feature_drift_log, run_id NULL (2026-08-13 23:41): host run;
--     matches feature_drift.csv's local mtime -> Europe/London.
--   feature_drift_log, the three container runs below: Airflow run
--     records (scheduled__2026-08-14, scheduled__2026-10-12,
--     scheduled__2026-10-13) confirm UTC -> UTC.
--   feature_drift_log, drift_20261013T005435_21a21fce: host run
--     -> Europe/London.
--
-- Guarded: each ALTER runs only while the column is still naive,
-- so re-running this file is a no-op. Run in the same transaction
-- as sql/reporting/create_governance_status_log.sql, which
-- recreates the view dropped here (a view blocks ALTER TYPE on the
-- columns it reads).

DROP VIEW IF EXISTS reporting.governance_status_latest;

DO $$
BEGIN
    IF (SELECT data_type FROM information_schema.columns
        WHERE table_schema = 'reporting'
          AND table_name = 'governance_status_log'
          AND column_name = 'evaluated_at') = 'timestamp without time zone'
    THEN
        ALTER TABLE reporting.governance_status_log
            ALTER COLUMN evaluated_at TYPE timestamptz
                USING evaluated_at AT TIME ZONE 'Europe/London',
            ALTER COLUMN created_at TYPE timestamptz
                USING created_at AT TIME ZONE 'Europe/London';
    END IF;

    IF (SELECT data_type FROM information_schema.columns
        WHERE table_schema = 'reporting'
          AND table_name = 'feature_drift_log'
          AND column_name = 'logged_at') = 'timestamp without time zone'
    THEN
        ALTER TABLE reporting.feature_drift_log
            ALTER COLUMN logged_at TYPE timestamptz
                USING CASE
                    WHEN run_id IN (
                        'drift_20261012T235906_e20c19e5',
                        'drift_20261012T235907_63e586e9',
                        'drift_20261013T000012_ae0c3fbf'
                    ) THEN logged_at AT TIME ZONE 'UTC'
                    ELSE logged_at AT TIME ZONE 'Europe/London'
                END;
    END IF;
END
$$;

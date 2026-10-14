-- Sentinel Credit: column inventory for the tables that would feed the
-- six-page model-risk report structure. Run this once, paste the full
-- output back, and each page can be mapped to "have it" / "needs work"
-- with certainty instead of inference from table names alone.

SELECT
    table_schema,
    table_name,
    column_name,
    data_type,
    ordinal_position
FROM information_schema.columns
WHERE table_schema IN ('feature_store', 'reporting', 'public')
  AND table_name IN (
        'feature_drift',
        'model_comparison_stratified',
        'segment_summary',
        'default_rate_monthly',
        'originations_monthly',
        'champion_predictions',
        'customer_risk_snapshot'
  )
ORDER BY table_schema, table_name, ordinal_position;

-- Optional follow-up, run separately if any of the above tables aren't found:
-- lists every table Power BI could see, across all three schemas, so you
-- can spot naming mismatches (e.g. table lives under a schema not guessed above).
--
-- SELECT table_schema, table_name
-- FROM information_schema.tables
-- WHERE table_schema IN ('feature_store', 'reporting', 'public')
-- ORDER BY table_schema, table_name;
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema IN ('feature_store', 'reporting', 'public')
ORDER BY table_schema, table_name;



SELECT
    table_schema,
    table_name,
    column_name,
    data_type,
    ordinal_position
FROM information_schema.columns
WHERE table_schema = 'reporting'
  AND table_name IN ('governance_status_latest', 'governance_status_log')
ORDER BY table_name, ordinal_position;
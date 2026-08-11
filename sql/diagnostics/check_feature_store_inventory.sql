-- ============================================================
-- FEATURE STORE INVENTORY
--
-- Lists every column in feature_store.customer_features with
-- its type, null rate, and distinct-value count, so the full
-- candidate set is visible rather than only the current
-- MODEL_FEATURES allowlist.
--
-- Run from the project root:
--   psql -U postgres -d sentinel_credit -f sql/diagnostics/check_feature_store_inventory.sql
-- ============================================================

\echo '=== ALL COLUMNS, WITH TYPE ==='
SELECT
    ordinal_position AS pos,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'feature_store'
  AND table_name   = 'customer_features'
ORDER BY ordinal_position;


\echo ''
\echo '=== COLUMN COUNT ==='
SELECT COUNT(*) AS total_columns
FROM information_schema.columns
WHERE table_schema = 'feature_store'
  AND table_name   = 'customer_features';


-- Null rate and cardinality for every column, computed
-- dynamically so this does not need updating when the schema
-- changes. Runs on the modelable population only, since that
-- is what training would see.
\echo ''
\echo '=== NULL RATE AND CARDINALITY (modelable population) ==='
DO $$
DECLARE
    col   record;
    n_tot bigint;
    n_null bigint;
    n_dist bigint;
BEGIN
    SELECT COUNT(*) INTO n_tot
    FROM feature_store.customer_features
    WHERE has_outcome = 1;

    RAISE NOTICE '% -30s | % -10s | % -10s | % -12s',
                 'column', 'null_pct', 'distinct', 'note';
    RAISE NOTICE '%', repeat('-', 70);

    FOR col IN
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'feature_store'
          AND table_name   = 'customer_features'
        ORDER BY ordinal_position
    LOOP
        EXECUTE format(
            'SELECT COUNT(*) FILTER (WHERE %I IS NULL), COUNT(DISTINCT %I)
             FROM feature_store.customer_features WHERE has_outcome = 1',
            col.column_name, col.column_name
        ) INTO n_null, n_dist;

        RAISE NOTICE '% -30s | % -10s | % -10s | %',
            col.column_name,
            ROUND(100.0 * n_null / NULLIF(n_tot, 0), 2)::text || '%',
            n_dist::text,
            CASE
                WHEN n_dist = 1 THEN 'CONSTANT - no signal possible'
                WHEN n_null = n_tot THEN 'ALL NULL'
                WHEN n_dist = 2 THEN 'binary'
                ELSE ''
            END;
    END LOOP;
END $$;

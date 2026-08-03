-- ============================================================
-- MATURITY COLLAPSE DIAGNOSTIC
--
-- Only 4,871 of 50,000 customers (9.7%) qualify for the fully
-- mature A/B cohort (last transaction on or before 2023-11-07).
-- Three questions this answers:
--
--   1. WHERE do last transactions actually fall? If most sit
--      after the cutoff, the simulation horizon is the
--      binding constraint, not the causal design.
--   2. WOULD extending data_end_date fix it? Shows how the
--      cohort grows as the cutoff moves later.
--   3. IS the differential exclusion (A 2,343 vs B 2,528)
--      driven by origination timing or by treatment?
-- ============================================================

-- Q1: Distribution of last transaction dates by quarter.
--     The cutoff is 2023-11-07 — everything after is excluded.
\echo '=== Q1: WHERE DO LAST TRANSACTIONS FALL? ==='
SELECT
    DATE_TRUNC('quarter', last_transaction_date)::date AS quarter,
    COUNT(*) AS n_customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct,
    COUNT(*) FILTER (
        WHERE last_transaction_date <= DATE '2023-11-07'
    ) AS n_in_cohort
FROM feature_store.customer_features
WHERE last_transaction_date IS NOT NULL
GROUP BY 1
ORDER BY 1;


-- Q2: How many customers would qualify at different cutoffs?
--     Each row shows the cohort size if data_end_date were
--     extended by that many months (cutoff = data_end + N
--     months - 420 days).
\echo ''
\echo '=== Q2: WOULD EXTENDING THE SIMULATION HELP? ==='
WITH extensions AS (
    SELECT generate_series(0, 18, 3) AS extra_months
),
cutoffs AS (
    SELECT
        extra_months,
        (DATE '2024-12-31' + (extra_months || ' months')::interval
         - INTERVAL '420 days')::date AS cutoff
    FROM extensions
)
SELECT
    c.extra_months,
    c.cutoff AS maturity_cutoff,
    COUNT(cf.customer_id) AS n_eligible,
    ROUND(100.0 * COUNT(cf.customer_id) / 50000, 2) AS pct_of_population
FROM cutoffs c
LEFT JOIN feature_store.customer_features cf
    ON cf.last_transaction_date IS NOT NULL
   AND cf.last_transaction_date <= c.cutoff
GROUP BY c.extra_months, c.cutoff
ORDER BY c.extra_months;


-- Q3a: Is the differential exclusion driven by treatment or by
--      origination timing? Compares transaction COUNT and SPAN
--      by arm — if B customers take more loans (they should,
--      given higher approval), their last transaction falls
--      later mechanically.
\echo ''
\echo '=== Q3a: TRANSACTION BEHAVIOUR BY ARM (full population) ==='
SELECT
    ab_test_group,
    COUNT(*) AS n_customers,
    ROUND(AVG(n_approved), 3) AS avg_approved_loans,
    ROUND(AVG(transaction_span_days), 1) AS avg_span_days,
    MIN(first_transaction_date) AS earliest_first,
    ROUND(AVG(last_transaction_date - first_transaction_date), 1)
        AS avg_days_first_to_last
FROM feature_store.customer_features
WHERE last_transaction_date IS NOT NULL
GROUP BY ab_test_group
ORDER BY ab_test_group;


-- Q3b: Origination timing by arm. If assignment was random,
--      first_transaction_date distributions should match. Any
--      difference here is a generator artefact, not treatment.
\echo ''
\echo '=== Q3b: ORIGINATION TIMING BY ARM (should NOT differ) ==='
SELECT
    ab_test_group,
    MIN(first_transaction_date) AS earliest,
    ROUND(AVG(first_transaction_date - DATE '2022-01-01'), 1)
        AS avg_days_from_window_start,
    MAX(first_transaction_date) AS latest
FROM feature_store.customer_features
WHERE first_transaction_date IS NOT NULL
GROUP BY ab_test_group
ORDER BY ab_test_group;


-- Q3c: Cohort composition by arm, and what share of each arm
--      survives the cutoff.
\echo ''
\echo '=== Q3c: WHO SURVIVES THE CUTOFF, BY ARM ==='
SELECT
    ab_test_group,
    COUNT(*) AS n_total,
    COUNT(*) FILTER (
        WHERE last_transaction_date <= DATE '2023-11-07'
    ) AS n_in_cohort,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE last_transaction_date <= DATE '2023-11-07'
    ) / COUNT(*), 2) AS pct_surviving
FROM feature_store.customer_features
WHERE last_transaction_date IS NOT NULL
GROUP BY ab_test_group
ORDER BY ab_test_group;

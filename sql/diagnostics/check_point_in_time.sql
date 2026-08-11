-- ============================================================
-- POINT-IN-TIME VERIFICATION
--
-- Checks whether the nine features flagged conditional_as_of_
-- logic (avg_n_instalments, pct_high_risk_category,
-- bnpl_exposure_to_monthly_income_ratio, n_applications,
-- avg_order_value_gbp, pct_late_payments, pct_medium_risk_
-- category, max_affordability_ratio, n_late_payments) are
-- computed from a customer's FULL lifetime activity or from
-- activity available at a defined decision point.
--
-- Method: the empirical test below checks whether these
-- values correlate with how much exposure window a customer
-- has AFTER their first transaction. A feature genuinely fixed
-- at application time would not correlate with that; a
-- lifetime aggregate would, since it keeps incorporating
-- activity the application-time version could not have known
-- about.
--
-- Run from the project root:
--   psql -U postgres -d sentinel_credit -f sql/diagnostics/check_point_in_time.sql
-- ============================================================

\echo '=== Q1: SOURCE SQL REMINDER ==='
\echo 'Trace the CTEs in sql/02_feature_store.sql for each flagged'
\echo 'column. A column built with COUNT/AVG/SUM over ALL of a'
\echo 'customer''s raw.transactions or raw.repayments rows, with no'
\echo 'date filter relative to a specific application, is a'
\echo 'lifetime aggregate, not point-in-time.'
\echo ''

\echo '=== Q2: DO THE FLAGGED FEATURES CORRELATE WITH POST-FIRST-TRANSACTION ACTIVITY? ==='
WITH exposure_window AS (
    SELECT
        customer_id,
        (last_transaction_date - first_transaction_date) AS days_of_activity_after_first,
        n_applications,
        avg_n_instalments,
        pct_high_risk_category,
        pct_medium_risk_category,
        bnpl_exposure_to_monthly_income_ratio,
        avg_order_value_gbp,
        pct_late_payments,
        max_affordability_ratio,
        n_late_payments
    FROM feature_store.customer_features
    WHERE has_outcome = 1
      AND first_transaction_date IS NOT NULL
      AND last_transaction_date IS NOT NULL
)
SELECT 'n_applications' AS feature, corr(days_of_activity_after_first, n_applications) AS r FROM exposure_window
UNION ALL
SELECT 'avg_n_instalments', corr(days_of_activity_after_first, avg_n_instalments) FROM exposure_window
UNION ALL
SELECT 'pct_high_risk_category', corr(days_of_activity_after_first, pct_high_risk_category) FROM exposure_window
UNION ALL
SELECT 'pct_medium_risk_category', corr(days_of_activity_after_first, pct_medium_risk_category) FROM exposure_window
UNION ALL
SELECT 'bnpl_exposure_to_monthly_income_ratio', corr(days_of_activity_after_first, bnpl_exposure_to_monthly_income_ratio) FROM exposure_window
UNION ALL
SELECT 'avg_order_value_gbp', corr(days_of_activity_after_first, avg_order_value_gbp) FROM exposure_window
UNION ALL
SELECT 'pct_late_payments', corr(days_of_activity_after_first, pct_late_payments) FROM exposure_window
UNION ALL
SELECT 'max_affordability_ratio', corr(days_of_activity_after_first, max_affordability_ratio) FROM exposure_window
UNION ALL
SELECT 'n_late_payments', corr(days_of_activity_after_first, n_late_payments) FROM exposure_window
ORDER BY 2 DESC NULLS LAST;

\echo ''
\echo '=== Q3: MEAN FEATURE VALUE BY EXPOSURE WINDOW LENGTH ==='
SELECT
    WIDTH_BUCKET(last_transaction_date - first_transaction_date, 0, 700, 7) AS exposure_bucket,
    COUNT(*) AS n,
    ROUND(AVG(n_applications), 3) AS avg_n_applications,
    ROUND(AVG(avg_order_value_gbp), 2) AS avg_order_value,
    ROUND(AVG(bnpl_exposure_to_monthly_income_ratio), 3) AS avg_bnpl_ratio,
    ROUND(AVG(avg_n_instalments), 3) AS avg_instalments
FROM feature_store.customer_features
WHERE has_outcome = 1
  AND first_transaction_date IS NOT NULL
  AND last_transaction_date IS NOT NULL
GROUP BY 1
ORDER BY 1;

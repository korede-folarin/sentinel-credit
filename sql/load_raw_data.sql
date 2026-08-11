-- ============================================================
-- SENTINEL CREDIT — RAW DATA LOAD
--
-- Columns named explicitly on every \copy. The generator
-- appends outcome columns in a second pass, so CSV column
-- order can diverge from schema order — naming them is what
-- prevents a silent misalignment (Decision 10).
--
-- Run from the project root:
--   psql -U postgres -d sentinel_credit -f sql/load_raw_data.sql
-- ============================================================

\echo 'Loading raw.customers...'
\copy raw.customers (customer_id, registration_date, age, age_bracket, gender, income_band, annual_income_gbp, monthly_income_gbp, employment_status, existing_monthly_obligations_gbp, savings_balance_gbp, thin_file_flag, credit_score, months_credit_history, num_credit_accounts, num_defaults_24m, mental_health_flag, vulnerability_health, vulnerability_life_event, vulnerability_resilience, vulnerability_capability, vulnerability_score, is_vulnerable, device_type, acquisition_channel, ab_test_group) FROM 'data_generator/output/customers.csv' WITH (FORMAT csv, HEADER true)

\echo 'Loading raw.transactions...'
\copy raw.transactions (transaction_id, customer_id, transaction_date, merchant_category, order_value_gbp, n_instalments, instalment_frequency, instalment_amount_gbp, affordability_ratio, fca_affordability_flag, application_approved, merchant_risk_multiplier, ab_test_group) FROM 'data_generator/output/transactions.csv' WITH (FORMAT csv, HEADER true)

\echo 'Loading raw.repayments...'
\copy raw.repayments (repayment_id, transaction_id, customer_id, instalment_number, instalment_amount_gbp, due_date, payment_status, days_late, days_past_due, arrears_depth, amount_paid_gbp, outstanding_balance_gbp, ab_test_group, merchant_category, default_flag, outcome_observed, max_dpd, reached_sicr, cycles_delinquent, terminal_state) FROM 'data_generator/output/repayments.csv' WITH (FORMAT csv, HEADER true)

\echo ''
\echo '=== ROW COUNTS (expect 50,000 / 174,421 / 483,579) ==='
SELECT
    (SELECT COUNT(*) FROM raw.customers)    AS customers,
    (SELECT COUNT(*) FROM raw.transactions) AS transactions,
    (SELECT COUNT(*) FROM raw.repayments)   AS repayments;

\echo ''
\echo '=== CENSORING CHECK (expect 0 censored) ==='
SELECT
    COUNT(*) AS total_repayments,
    SUM(CASE WHEN outcome_observed = 0 THEN 1 ELSE 0 END) AS censored,
    ROUND(100.0 * SUM(CASE WHEN outcome_observed = 0 THEN 1 ELSE 0 END)
          / COUNT(*), 4) AS pct_censored
FROM raw.repayments;

-- Sentinel Credit, Reporting Views
--
-- Purpose
-- These views exist so Power BI, and any other reporting tool,
-- can connect DIRECTLY to the live database, the same way a real
-- production risk reporting stack would. They are read only,
-- built on the existing raw and feature_store schemas, and
-- change nothing about the modelling pipeline.
--
-- portfolio_monthly_summary mirrors the exact origination and
-- cohort default rate logic already built and verified in
-- forecasting/build_portfolio_timeseries.py, as a SQL view rather
-- than a Python script, so DirectQuery mode in Power BI reflects
-- live data without needing a Python refresh step in between.
--
-- segment_summary and customer_risk_snapshot expose
-- feature_store.customer_features for portfolio and fairness
-- reporting, at the same governance boundary already established
-- there: is_vulnerable is exposed as a label only, never presented
-- as something the model was trained on.
--
-- See docs/engineering_decision_log.md, Decision 12 addendum,
-- Decision 22.

CREATE SCHEMA IF NOT EXISTS reporting;

-- Origination volume by month, approved transactions only.
CREATE OR REPLACE VIEW reporting.originations_monthly AS
SELECT
    date_trunc('month', transaction_date)::date AS month,
    count(*) AS n_originations,
    sum(order_value_gbp) AS total_value_gbp,
    avg(order_value_gbp) AS avg_order_value_gbp
FROM raw.transactions
WHERE application_approved = 1
GROUP BY 1
ORDER BY 1;

-- Default rate by ORIGINATION cohort month, resolved outcomes
-- only. A default surfacing several months after origination is
-- attributed back to the month the transaction happened, not the
-- month the default was recorded, matching
-- build_portfolio_timeseries.py's logic exactly.
CREATE OR REPLACE VIEW reporting.default_rate_monthly AS
WITH per_transaction AS (
    SELECT
        transaction_id,
        max(default_flag) AS defaulted,
        min(outcome_observed) AS fully_observed
    FROM raw.repayments
    GROUP BY transaction_id
)
SELECT
    date_trunc('month', t.transaction_date)::date AS month,
    count(*) AS n_resolved,
    sum(p.defaulted) AS n_defaulted,
    round(sum(p.defaulted)::numeric / count(*), 6) AS default_rate
FROM raw.transactions t
JOIN per_transaction p ON p.transaction_id = t.transaction_id
WHERE t.application_approved = 1
  AND p.fully_observed = 1
GROUP BY 1
ORDER BY 1;

-- Portfolio segment summary, for fairness and portfolio reporting.
-- is_vulnerable is exposed as a comparison label only, consistent
-- with every other place in this project, never as a feature.
CREATE OR REPLACE VIEW reporting.segment_summary AS
SELECT
    thin_file_flag,
    is_vulnerable,
    count(*) AS n_customers,
    round(avg(target_default)::numeric, 6) AS actual_default_rate,
    round(avg(credit_score)::numeric, 1) AS avg_credit_score,
    round(avg(savings_balance_gbp)::numeric, 2) AS avg_savings_balance_gbp,
    -- Join key to customer_risk_snapshot for Power BI's segment
    -- relationship. First applied via
    -- sql/reporting/maintenance/fix_segment_key_final.sql; folded in here so this file
    -- rebuilds the live view exactly.
    thin_file_flag::text || '-' || is_vulnerable::text AS segment_key
FROM feature_store.customer_features
WHERE has_outcome = 1
GROUP BY thin_file_flag, is_vulnerable
ORDER BY thin_file_flag, is_vulnerable;

-- Customer level snapshot for drillthrough, restricted to the
-- governed model features plus identifiers and outcome, not the
-- full 64 column feature store. Deliberately excludes any feature
-- that is not in the governed model, so a Power BI user cannot
-- accidentally build a report on a leaked or excluded field.
--
-- KNOWN EXCEPTION (Decision 27): pct_late_payments was removed from
-- the model in v1.1 for leakage but is still in this view, because
-- dropping a column breaks any Power BI visual bound to it. To be
-- removed once the report is confirmed not to use it (audit A26).
CREATE OR REPLACE VIEW reporting.customer_risk_snapshot AS
SELECT
    customer_id,
    thin_file_flag,
    is_vulnerable,
    target_default,
    has_outcome,
    monthly_income_gbp,
    existing_monthly_obligations_gbp,
    savings_balance_gbp,
    credit_score,
    months_credit_history,
    num_credit_accounts,
    num_defaults_24m,
    max_affordability_ratio,
    avg_order_value_gbp,
    avg_n_instalments,
    pct_high_risk_category,
    pct_medium_risk_category,
    pct_late_payments,
    first_transaction_date,
    last_transaction_date,
    -- Same key as segment_summary.segment_key (see above).
    thin_file_flag::text || '-' || is_vulnerable::text AS segment_key
FROM feature_store.customer_features
WHERE has_outcome = 1;

-- Grants, adjust the role name to whatever Power BI's connection
-- actually authenticates as in your environment.
-- GRANT USAGE ON SCHEMA reporting TO powerbi_reader;
-- GRANT SELECT ON ALL TABLES IN SCHEMA reporting TO powerbi_reader;

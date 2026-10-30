-- Sentinel Credit v2, Phase 2: the v1 feature store in Spark SQL (V2-04).
--
-- A transcription of sql/02_feature_store.sql (v1, unchanged) for Spark 3.5.
-- The logic, column names and order are the same. The only differences are
-- what Spark needs to give PostgreSQL's exact answers:
--   [D1] Inputs are exact decimals (views raw_customers etc., cast in
--        spark_engine.py to the v1 raw scales: money 2 dp, ratios 4 dp,
--        vulnerability score 3 dp), as PostgreSQL's numeric columns are.
--   [D2] Every rounded average averages a DECIMAL(30,12) (AVG(CAST(...))).
--        Spark's AVG of an integer is a double, and of a small decimal it
--        rounds to only 4 more places; either could round a value
--        differently from PostgreSQL at a tie. PostgreSQL averages integers
--        and numerics exactly.
--   [D3] PostgreSQL's x::numeric is written CAST(x AS DECIMAL(30,12)).
--   [D4] date - date (an integer number of days in PostgreSQL) is DATEDIFF.
--   [D5] No DROP / CREATE / INDEX: the result is written to Parquet.
-- Checked, not assumed: Spark's output is compared with PostgreSQL's,
-- column by column, for v1's own data and every scaled dataset.

WITH customer_default AS (
    SELECT
        customer_id,
        MAX(default_flag)     AS default_flag,
        MIN(outcome_observed) AS all_observed
    FROM raw_repayments
    GROUP BY customer_id
),

application_behaviour AS (
    SELECT
        customer_id,
        COUNT(*)                                                         AS n_applications,
        SUM(application_approved)                                        AS n_approved,
        COUNT(*) - SUM(application_approved)                             AS n_declined,
        ROUND(AVG(CAST(application_approved AS DECIMAL(30,12))), 4)      AS approval_rate,             -- [D3]
        ROUND(AVG(CAST(affordability_ratio AS DECIMAL(30,12))), 4)       AS avg_affordability_ratio,   -- [D2]
        ROUND(MAX(affordability_ratio), 4)                               AS max_affordability_ratio,
        SUM(fca_affordability_flag)                                      AS n_affordability_flags,
        ROUND(AVG(CAST(fca_affordability_flag AS DECIMAL(30,12))), 4)    AS pct_affordability_flagged  -- [D3]
    FROM raw_transactions
    GROUP BY customer_id
),

exposure AS (
    SELECT
        customer_id,
        COUNT(*)                                                         AS n_transactions,
        ROUND(SUM(order_value_gbp), 2)                                   AS total_order_value_gbp,
        ROUND(AVG(CAST(order_value_gbp AS DECIMAL(30,12))), 2)           AS avg_order_value_gbp,       -- [D2]
        ROUND(MAX(order_value_gbp), 2)                                   AS max_order_value_gbp,
        ROUND(AVG(CAST(n_instalments AS DECIMAL(30,12))), 2)             AS avg_n_instalments,         -- [D2]
        ROUND(SUM(instalment_amount_gbp), 2)                             AS total_instalment_exposure_gbp,
        COUNT(DISTINCT merchant_category)                                AS n_merchant_categories,
        ROUND(AVG(CAST(CASE WHEN merchant_category IN ('Travel','Electronics')
             THEN 1 ELSE 0 END AS DECIMAL(30,12))), 4)                   AS pct_high_risk_category,    -- [D2]
        ROUND(AVG(CAST(CASE WHEN merchant_category IN ('Fashion & Clothing','Home','Other')
             THEN 1 ELSE 0 END AS DECIMAL(30,12))), 4)                   AS pct_medium_risk_category,  -- [D2]
        ROUND(AVG(CAST(CASE WHEN merchant_category IN ('Sports','Health & Beauty','Food & Grocery')
             THEN 1 ELSE 0 END AS DECIMAL(30,12))), 4)                   AS pct_low_risk_category,     -- [D2]
        MIN(transaction_date)                                            AS first_transaction_date,
        MAX(transaction_date)                                            AS last_transaction_date
    FROM raw_transactions
    WHERE application_approved = 1
    GROUP BY customer_id
),

repayment_behaviour AS (
    SELECT
        customer_id,
        COUNT(*)                                                         AS n_instalments_total,
        COUNT(DISTINCT transaction_id)                                   AS n_transactions_with_repayments,
        SUM(CASE WHEN payment_status = 'Late' THEN 1 ELSE 0 END)         AS n_late_payments,
        ROUND(AVG(CAST(CASE WHEN payment_status = 'Late' THEN 1 ELSE 0 END
             AS DECIMAL(30,12))), 4)                                     AS pct_late_payments,         -- [D2]
        ROUND(AVG(CAST(CASE WHEN days_late > 0 THEN days_late END
             AS DECIMAL(30,12))), 2)                                     AS avg_days_late,             -- [D2]
        MAX(days_late)                                                   AS max_days_late,
        SUM(days_late)                                                   AS total_days_late,
        COUNT(DISTINCT CASE WHEN payment_status = 'Late'
              THEN transaction_id END)                                   AS n_transactions_with_late_payment,
        MAX(max_dpd)                                                     AS max_dpd_ever,
        ROUND(AVG(CAST(max_dpd AS DECIMAL(30,12))), 2)                   AS avg_max_dpd,               -- [D2]
        SUM(CASE WHEN reached_sicr = 1 THEN 1 ELSE 0 END)                AS n_reached_sicr,
        ROUND(AVG(CAST(reached_sicr AS DECIMAL(30,12))), 4)              AS pct_reached_sicr,          -- [D3]
        MAX(cycles_delinquent)                                           AS max_cycles_delinquent,
        ROUND(AVG(CAST(cycles_delinquent AS DECIMAL(30,12))), 2)         AS avg_cycles_delinquent,     -- [D2]
        MAX(arrears_depth)                                               AS max_arrears_depth,
        ROUND(AVG(CAST(NULLIF(arrears_depth, 0) AS DECIMAL(30,12))), 2)  AS avg_arrears_depth_when_behind, -- [D2]
        COUNT(DISTINCT CASE WHEN terminal_state = 'CURRENT'
              THEN transaction_id END)                                   AS n_terminal_current,
        COUNT(DISTINCT CASE WHEN terminal_state = 'DELINQUENT'
              THEN transaction_id END)                                   AS n_terminal_delinquent,
        COUNT(DISTINCT CASE WHEN terminal_state = 'DEFAULT'
              THEN transaction_id END)                                   AS n_terminal_default,
        ROUND(AVG(CAST(instalment_amount_gbp AS DECIMAL(30,12))), 2)     AS avg_instalment_amount_gbp, -- [D2]
        ROUND(SUM(instalment_amount_gbp), 2)                             AS total_scheduled_repayment_gbp
    FROM raw_repayments
    GROUP BY customer_id
)

SELECT
    c.customer_id,

    COALESCE(cd.default_flag, 0) AS target_default,
    CASE
        WHEN cd.customer_id IS NOT NULL
             AND (cd.default_flag = 1 OR cd.all_observed = 1)
        THEN 1
        ELSE 0
    END AS has_outcome,

    c.age,
    c.annual_income_gbp,
    c.monthly_income_gbp,
    c.existing_monthly_obligations_gbp,
    c.savings_balance_gbp,
    c.credit_score,
    c.months_credit_history,
    c.num_credit_accounts,
    c.num_defaults_24m,
    c.thin_file_flag,

    c.vulnerability_score,
    c.is_vulnerable,

    COALESCE(ab.n_applications, 0)        AS n_applications,
    COALESCE(ab.n_approved, 0)            AS n_approved,
    COALESCE(ab.n_declined, 0)            AS n_declined,
    ab.approval_rate,
    ab.avg_affordability_ratio,
    ab.max_affordability_ratio,
    COALESCE(ab.n_affordability_flags, 0) AS n_affordability_flags,
    ab.pct_affordability_flagged,

    COALESCE(ex.n_transactions, 0)                AS n_transactions,
    COALESCE(ex.total_order_value_gbp, 0)         AS total_order_value_gbp,
    ex.avg_order_value_gbp,
    ex.max_order_value_gbp,
    ex.avg_n_instalments,
    COALESCE(ex.total_instalment_exposure_gbp, 0) AS total_instalment_exposure_gbp,
    COALESCE(ex.n_merchant_categories, 0)         AS n_merchant_categories,
    ex.pct_high_risk_category,
    ex.pct_medium_risk_category,
    ex.pct_low_risk_category,

    COALESCE(rb.n_instalments_total, 0)              AS n_instalments_total,
    COALESCE(rb.n_transactions_with_repayments, 0)   AS n_transactions_with_repayments,
    COALESCE(rb.n_late_payments, 0)                  AS n_late_payments,
    rb.pct_late_payments,
    rb.avg_days_late,
    rb.max_days_late,
    COALESCE(rb.total_days_late, 0)                  AS total_days_late,
    COALESCE(rb.n_transactions_with_late_payment, 0) AS n_transactions_with_late_payment,
    rb.max_dpd_ever,
    rb.avg_max_dpd,
    COALESCE(rb.n_reached_sicr, 0)                   AS n_reached_sicr,
    rb.pct_reached_sicr,
    rb.max_cycles_delinquent,
    rb.avg_cycles_delinquent,
    rb.max_arrears_depth,
    rb.avg_arrears_depth_when_behind,
    COALESCE(rb.n_terminal_current, 0)               AS n_terminal_current,
    COALESCE(rb.n_terminal_delinquent, 0)            AS n_terminal_delinquent,
    COALESCE(rb.n_terminal_default, 0)               AS n_terminal_default,
    rb.avg_instalment_amount_gbp,
    COALESCE(rb.total_scheduled_repayment_gbp, 0)    AS total_scheduled_repayment_gbp,

    ROUND(c.existing_monthly_obligations_gbp
          / NULLIF(c.monthly_income_gbp, 0), 4) AS debt_to_income_ratio,

    ROUND(COALESCE(ex.total_order_value_gbp, 0)
          / NULLIF(c.monthly_income_gbp, 0), 4)
          AS bnpl_exposure_to_monthly_income_ratio,

    ex.first_transaction_date,
    ex.last_transaction_date,
    DATEDIFF(ex.last_transaction_date, ex.first_transaction_date)
          AS transaction_span_days,                                                                    -- [D4]
    DATEDIFF(DATE '2024-12-31', ex.last_transaction_date)
          AS days_since_last_transaction,                                                              -- [D4]

    c.age_bracket,
    c.income_band,
    c.employment_status,
    c.ab_test_group

FROM raw_customers c
LEFT JOIN customer_default      cd ON c.customer_id = cd.customer_id
LEFT JOIN application_behaviour ab ON c.customer_id = ab.customer_id
LEFT JOIN exposure              ex ON c.customer_id = ex.customer_id
LEFT JOIN repayment_behaviour   rb ON c.customer_id = rb.customer_id

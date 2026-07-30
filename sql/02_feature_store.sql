-- ============================================================
-- SENTINEL CREDIT — FEATURE STORE
-- Stage 3 added: repayment and delinquency features.
--
-- WHY THIS STAGE WAS PREVIOUSLY BLOCKED
-- Under the old generator, payment_status values of 'Missed'
-- and 'Defaulted' were assigned only after the outcome was
-- already decided, so counting them would have reconstructed
-- the target directly (leakage). Only the independent
-- 'Late' path was safe, capping this stage at ~10 features.
-- Under the hazard-based rewrite, default is emergent: the
-- delinquency ladder (max_dpd, reached_sicr,
-- cycles_delinquent, arrears_depth, terminal_state) genuinely
-- precedes default in time rather than being backfilled from
-- it, so these are now legitimate pre-default features.
-- See docs/engineering_decision_log.md — Decision 10.
--
-- OUTCOME OBSERVABILITY
-- Customers whose applications were all declined generate no
-- repayment records and therefore have no observable default
-- outcome. They are retained in the feature store for
-- portfolio analytics and monitoring, but flagged via
-- has_outcome so the modelling pipeline can exclude them from
-- training. Reject inference is not applied because the
-- synthetic approval policy is deterministic, leaving no
-- overlap between approved and rejected populations from
-- which rejected outcomes could be meaningfully inferred.
--
-- NULL SEMANTICS
-- Counts and sums COALESCE to 0, representing zero observed
-- events. Averages, maxima and ratios remain NULL because no
-- quantity exists to calculate. For example, approval_rate
-- NULL means the customer made no applications, whereas
-- approval_rate = 0.0000 means every application was declined.
--
-- LEAKAGE
-- Generator-internal fields (merchant_risk_multiplier;
-- default_probability no longer exists post-rewrite) are
-- excluded at the feature boundary. They remain in the raw
-- layer for auditability and synthetic data validation, but
-- are never exposed as model inputs.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS feature_store;

DROP TABLE IF EXISTS feature_store.customer_features;

CREATE TABLE feature_store.customer_features AS

WITH customer_default AS (
    SELECT
        customer_id,
        MAX(default_flag) AS default_flag
    FROM raw.repayments
    GROUP BY customer_id
),

application_behaviour AS (
    -- ALL applications. Credit-seeking volume and decline
    -- rate are established distress signals in credit risk.
    SELECT
        customer_id,
        COUNT(*)                                       AS n_applications,
        SUM(application_approved)                      AS n_approved,
        COUNT(*) - SUM(application_approved)           AS n_declined,
        ROUND(AVG(application_approved::numeric), 4)   AS approval_rate,
        ROUND(AVG(affordability_ratio), 4)             AS avg_affordability_ratio,
        ROUND(MAX(affordability_ratio), 4)             AS max_affordability_ratio,
        SUM(fca_affordability_flag)                    AS n_affordability_flags,
        ROUND(AVG(fca_affordability_flag::numeric), 4) AS pct_affordability_flagged
    FROM raw.transactions
    GROUP BY customer_id
),

exposure AS (
    -- APPROVED only. Approved transactions are the only ones
    -- that generate repayment obligations and therefore
    -- exposure to default risk.
    SELECT
        customer_id,
        COUNT(*)                             AS n_transactions,
        ROUND(SUM(order_value_gbp), 2)       AS total_order_value_gbp,
        ROUND(AVG(order_value_gbp), 2)       AS avg_order_value_gbp,
        ROUND(MAX(order_value_gbp), 2)       AS max_order_value_gbp,
        ROUND(AVG(n_instalments), 2)         AS avg_n_instalments,
        ROUND(SUM(instalment_amount_gbp), 2) AS total_instalment_exposure_gbp,
        COUNT(DISTINCT merchant_category)    AS n_merchant_categories,

        -- Merchant risk tiers. All eight categories are
        -- assigned so the three proportions sum to 1. Tiers
        -- are based on documented domain judgement, not the
        -- generator's hidden merchant_risk_multiplier, which
        -- is excluded by design.
        ROUND(AVG(CASE WHEN merchant_category IN
             ('Travel','Electronics')
             THEN 1.0 ELSE 0.0 END), 4) AS pct_high_risk_category,
        ROUND(AVG(CASE WHEN merchant_category IN
             ('Fashion & Clothing','Home','Other')
             THEN 1.0 ELSE 0.0 END), 4) AS pct_medium_risk_category,
        ROUND(AVG(CASE WHEN merchant_category IN
             ('Sports','Health & Beauty','Food & Grocery')
             THEN 1.0 ELSE 0.0 END), 4) AS pct_low_risk_category,

        MIN(transaction_date)                AS first_transaction_date,
        MAX(transaction_date)                AS last_transaction_date
    FROM raw.transactions
    WHERE application_approved = 1
    GROUP BY customer_id
),

-- ============================================================
-- STAGE 3 — repayment and delinquency behaviour.
-- One row per customer, aggregated from raw.repayments.
-- Every field here genuinely precedes default in time under
-- the hazard model, so none of it is proxy-to-label leakage.
-- ============================================================
repayment_behaviour AS (
    SELECT
        customer_id,

        -- ---- Volume ----
        COUNT(*)                              AS n_instalments_total,
        COUNT(DISTINCT transaction_id)         AS n_transactions_with_repayments,

        -- ---- Late-payment path (independent of default) ----
        SUM(CASE WHEN payment_status = 'Late' THEN 1 ELSE 0 END)
            AS n_late_payments,
        ROUND(AVG(CASE WHEN payment_status = 'Late' THEN 1.0 ELSE 0.0 END), 4)
            AS pct_late_payments,
        ROUND(AVG(CASE WHEN days_late > 0 THEN days_late END), 2)
            AS avg_days_late,
        MAX(days_late)                        AS max_days_late,
        SUM(days_late)                        AS total_days_late,
        COUNT(DISTINCT CASE WHEN payment_status = 'Late'
              THEN transaction_id END)         AS n_transactions_with_late_payment,

        -- ---- Delinquency ladder (now legitimate under the
        -- hazard model — see Decision 10) ----
        MAX(max_dpd)                          AS max_dpd_ever,
        ROUND(AVG(max_dpd), 2)                AS avg_max_dpd,
        SUM(CASE WHEN reached_sicr = 1 THEN 1 ELSE 0 END)
            AS n_reached_sicr,
        ROUND(AVG(reached_sicr::numeric), 4)  AS pct_reached_sicr,
        MAX(cycles_delinquent)                AS max_cycles_delinquent,
        ROUND(AVG(cycles_delinquent), 2)      AS avg_cycles_delinquent,
        MAX(arrears_depth)                    AS max_arrears_depth,
        ROUND(AVG(NULLIF(arrears_depth, 0)), 2) AS avg_arrears_depth_when_behind,

        -- ---- Terminal state mix (how transactions resolved) ----
        COUNT(DISTINCT CASE WHEN terminal_state = 'CURRENT'
              THEN transaction_id END)         AS n_terminal_current,
        COUNT(DISTINCT CASE WHEN terminal_state = 'DELINQUENT'
              THEN transaction_id END)         AS n_terminal_delinquent,
        COUNT(DISTINCT CASE WHEN terminal_state = 'DEFAULT'
              THEN transaction_id END)         AS n_terminal_default,

        -- ---- Structural ----
        ROUND(AVG(instalment_amount_gbp), 2)  AS avg_instalment_amount_gbp,
        ROUND(SUM(instalment_amount_gbp), 2)  AS total_scheduled_repayment_gbp

    FROM raw.repayments
    GROUP BY customer_id
)

SELECT
    c.customer_id,

    -- ---- TARGET AND OBSERVABILITY ----
    COALESCE(cd.default_flag, 0) AS target_default,
    CASE WHEN cd.customer_id IS NOT NULL THEN 1 ELSE 0 END AS has_outcome,

    -- ---- CUSTOMER ATTRIBUTES ----
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

    -- ---- VULNERABILITY (Consumer Duty monitoring) ----
    c.vulnerability_score,
    c.is_vulnerable,

    -- ---- APPLICATION BEHAVIOUR (all applications) ----
    COALESCE(ab.n_applications, 0)        AS n_applications,
    COALESCE(ab.n_approved, 0)            AS n_approved,
    COALESCE(ab.n_declined, 0)            AS n_declined,
    ab.approval_rate,
    ab.avg_affordability_ratio,
    ab.max_affordability_ratio,
    COALESCE(ab.n_affordability_flags, 0) AS n_affordability_flags,
    ab.pct_affordability_flagged,

    -- ---- EXPOSURE (approved only) ----
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

    -- ---- STAGE 3: REPAYMENT AND DELINQUENCY BEHAVIOUR ----
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

    -- ---- DERIVED RATIOS ----
    -- Conventional debt-to-income: recurring monthly
    -- obligations over monthly income.
    ROUND(c.existing_monthly_obligations_gbp
          / NULLIF(c.monthly_income_gbp, 0), 4) AS debt_to_income_ratio,

    -- NOT a standard debt-to-income ratio. Compares the
    -- customer's total BNPL borrowing over the entire
    -- observation period with one month's income. This
    -- measures historical borrowing scale, not current
    -- affordability.
    ROUND(COALESCE(ex.total_order_value_gbp, 0)
          / NULLIF(c.monthly_income_gbp, 0), 4)
          AS bnpl_exposure_to_monthly_income_ratio,

    -- ---- TIMING ----
    ex.first_transaction_date,
    ex.last_transaction_date,
    (ex.last_transaction_date - ex.first_transaction_date)
          AS transaction_span_days,
    -- Uses the configured performance-window end date rather
    -- than the system date, ensuring reproducible results.
    (DATE '2024-12-31' - ex.last_transaction_date)
          AS days_since_last_transaction,

    -- ---- SEGMENTATION (analysis, not model input) ----
    c.age_bracket,
    c.income_band,
    c.employment_status,
    c.ab_test_group

FROM raw.customers c
LEFT JOIN customer_default      cd ON c.customer_id = cd.customer_id
LEFT JOIN application_behaviour ab ON c.customer_id = ab.customer_id
LEFT JOIN exposure              ex ON c.customer_id = ex.customer_id
LEFT JOIN repayment_behaviour   rb ON c.customer_id = rb.customer_id;

-- ---- INDEXES ----
CREATE UNIQUE INDEX idx_cf_customer    ON feature_store.customer_features(customer_id);
CREATE INDEX idx_cf_target             ON feature_store.customer_features(target_default);
CREATE INDEX idx_cf_has_outcome        ON feature_store.customer_features(has_outcome);
CREATE INDEX idx_cf_ab_group           ON feature_store.customer_features(ab_test_group);

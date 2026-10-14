CREATE OR REPLACE VIEW reporting.customer_risk_snapshot AS
SELECT
    customer_id,
    thin_file_flag,
    is_vulnerable,
    thin_file_flag::text || '-' || is_vulnerable::text AS segment_key,
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
    last_transaction_date
FROM feature_store.customer_features
WHERE has_outcome = 1;

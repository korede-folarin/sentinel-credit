-- Tests whether target_default=0 customers are structurally
-- forced to be "older" (earlier in the origination window)
-- than target_default=1 customers, due to the asymmetric
-- maturity requirement in has_outcome (defaulters bypass the
-- maturity check; non-defaulters must satisfy it).

SELECT
    target_default,
    COUNT(*) AS n,
    ROUND(AVG(days_since_last_transaction), 1) AS avg_days_since_last_transaction,
    ROUND(AVG(first_transaction_date - DATE '2022-01-01'), 1) AS avg_days_from_window_start,
    MIN(first_transaction_date) AS earliest_first_transaction,
    MAX(first_transaction_date) AS latest_first_transaction
FROM feature_store.customer_features
WHERE has_outcome = 1
GROUP BY target_default;

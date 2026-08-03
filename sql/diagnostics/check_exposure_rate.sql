-- Tests total_order_value_gbp and n_approved for the same
-- artifact found in days_since_last_transaction: are these
-- just proportional to how much time each group had
-- available to transact, or is there a real difference in
-- RATE (value/count per day of exposure) beyond that?
--
-- If the per-day rate is similar between target_default=0
-- and target_default=1, the raw IV is mostly an artifact of
-- exposure time (same mechanism as days_since_last_transaction).
-- If the rate itself differs meaningfully, there's genuine
-- signal here independent of the maturity-path selection
-- effect.

SELECT
    target_default,
    COUNT(*) AS n,
    ROUND(AVG(total_order_value_gbp), 2)      AS avg_total_order_value_gbp,
    ROUND(AVG(n_approved), 2)                 AS avg_n_approved,
    ROUND(AVG(transaction_span_days), 1)      AS avg_transaction_span_days,
    ROUND(
        AVG(total_order_value_gbp) / NULLIF(AVG(GREATEST(transaction_span_days, 1)), 0),
        4
    ) AS order_value_per_day_of_exposure,
    ROUND(
        AVG(n_approved) / NULLIF(AVG(GREATEST(transaction_span_days, 1)), 0),
        6
    ) AS n_approved_per_day_of_exposure
FROM feature_store.customer_features
WHERE has_outcome = 1
GROUP BY target_default;

-- How many transactions and repayments actually sit behind
-- the 18,694 corrected modelable customers.

SELECT
    (SELECT COUNT(*) FROM feature_store.customer_features WHERE has_outcome = 1) AS modelable_customers,
    (SELECT COUNT(*) FROM raw.transactions t
        JOIN feature_store.customer_features cf ON t.customer_id = cf.customer_id
        WHERE cf.has_outcome = 1) AS transactions,
    (SELECT COUNT(*) FROM raw.repayments r
        JOIN feature_store.customer_features cf ON r.customer_id = cf.customer_id
        WHERE cf.has_outcome = 1) AS repayment_instalments,
    (SELECT SUM(target_default) FROM feature_store.customer_features WHERE has_outcome = 1) AS defaults;

-- Checks what percentage of the current has_outcome
-- population would remain modelable if has_outcome required
-- full maturity (outcome_observed), not just "has any
-- repayment record". See engineering_decision_log.md
-- Decision 12 addendum, 30th July 2026.

WITH customer_outcome AS (
    SELECT
        customer_id,
        MAX(default_flag) AS defaulted,
        MIN(outcome_observed) AS all_observed
    FROM raw.repayments
    GROUP BY customer_id
)
SELECT
    COUNT(*) AS current_has_outcome_population,
    SUM(CASE WHEN defaulted = 1 OR all_observed = 1 THEN 1 ELSE 0 END) AS correctly_modelable,
    ROUND(
        100.0 * SUM(CASE WHEN defaulted = 1 OR all_observed = 1 THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS pct_correctly_modelable
FROM customer_outcome;

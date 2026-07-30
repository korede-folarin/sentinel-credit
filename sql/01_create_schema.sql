-- ============================================================
-- SENTINEL CREDIT — RAW SCHEMA (v2, hazard-based data)
-- Rebuilt for the emergent-default repayment model.
-- Default is no longer a drawn probability; it emerges at
-- 90 DPD. New columns carry the delinquency ladder and
-- outcome observability. default_probability is removed.
-- Constraints encode business rules; foreign keys guarantee
-- every instalment traces to a real order and customer.
-- ============================================================

DROP SCHEMA IF EXISTS raw CASCADE;
CREATE SCHEMA raw;
CREATE SCHEMA IF NOT EXISTS feature_store;

-- ---- CUSTOMERS -------------------------------------------
CREATE TABLE raw.customers (
    customer_id                       TEXT PRIMARY KEY,
    registration_date                 DATE    NOT NULL,
    age                               INTEGER NOT NULL CHECK (age BETWEEN 18 AND 85),
    age_bracket                       TEXT    NOT NULL,
    gender                            TEXT,
    income_band                       TEXT    NOT NULL,
    annual_income_gbp                 NUMERIC(12,2) NOT NULL CHECK (annual_income_gbp >= 0),
    monthly_income_gbp                NUMERIC(12,2) CHECK (monthly_income_gbp >= 0),
    employment_status                 TEXT,
    existing_monthly_obligations_gbp  NUMERIC(12,2) CHECK (existing_monthly_obligations_gbp >= 0),
    savings_balance_gbp               NUMERIC(12,2) CHECK (savings_balance_gbp >= 0),
    thin_file_flag                    INTEGER NOT NULL CHECK (thin_file_flag IN (0,1)),
    credit_score                      INTEGER CHECK (credit_score BETWEEN 300 AND 999),
    months_credit_history             INTEGER CHECK (months_credit_history >= 0),
    num_credit_accounts               INTEGER CHECK (num_credit_accounts >= 0),
    num_defaults_24m                  INTEGER CHECK (num_defaults_24m >= 0),
    mental_health_flag                INTEGER CHECK (mental_health_flag IN (0,1)),
    vulnerability_health              INTEGER CHECK (vulnerability_health IN (0,1)),
    vulnerability_life_event          INTEGER CHECK (vulnerability_life_event IN (0,1)),
    vulnerability_resilience          INTEGER CHECK (vulnerability_resilience IN (0,1)),
    vulnerability_capability          INTEGER CHECK (vulnerability_capability IN (0,1)),
    vulnerability_score               NUMERIC(6,3) CHECK (vulnerability_score >= 0),
    is_vulnerable                     INTEGER NOT NULL CHECK (is_vulnerable IN (0,1)),
    device_type                       TEXT,
    acquisition_channel               TEXT,
    ab_test_group                     TEXT CHECK (ab_test_group IN ('A_strict','B_relaxed'))
);

-- ---- TRANSACTIONS ----------------------------------------
CREATE TABLE raw.transactions (
    transaction_id            TEXT PRIMARY KEY,
    customer_id               TEXT NOT NULL REFERENCES raw.customers(customer_id),
    transaction_date          DATE NOT NULL,
    merchant_category         TEXT NOT NULL,
    order_value_gbp           NUMERIC(12,2) NOT NULL CHECK (order_value_gbp >= 0),
    n_instalments             INTEGER NOT NULL CHECK (n_instalments > 0),
    instalment_frequency      TEXT CHECK (instalment_frequency IN ('monthly')),
    instalment_amount_gbp     NUMERIC(12,2) CHECK (instalment_amount_gbp >= 0),
    affordability_ratio       NUMERIC(8,4)  CHECK (affordability_ratio >= 0),
    fca_affordability_flag    INTEGER CHECK (fca_affordability_flag IN (0,1)),
    application_approved      INTEGER NOT NULL CHECK (application_approved IN (0,1)),
    -- Generator-internal. Excluded from features (leakage).
    merchant_risk_multiplier  NUMERIC(6,3) CHECK (merchant_risk_multiplier >= 0),
    ab_test_group             TEXT CHECK (ab_test_group IN ('A_strict','B_relaxed'))
);

-- ---- REPAYMENTS ------------------------------------------
CREATE TABLE raw.repayments (
    repayment_id             TEXT PRIMARY KEY,
    transaction_id           TEXT NOT NULL REFERENCES raw.transactions(transaction_id),
    customer_id              TEXT NOT NULL REFERENCES raw.customers(customer_id),
    instalment_number        INTEGER NOT NULL CHECK (instalment_number > 0),
    instalment_amount_gbp    NUMERIC(12,2) CHECK (instalment_amount_gbp >= 0),
    due_date                 DATE NOT NULL,
    payment_status           TEXT NOT NULL CHECK (
                                 payment_status IN
                                 ('Paid','Late','Missed','Defaulted')),
    days_late                INTEGER CHECK (days_late >= 0),
    days_past_due            INTEGER CHECK (days_past_due >= 0),
    arrears_depth            INTEGER CHECK (arrears_depth >= 0),
    amount_paid_gbp          NUMERIC(12,2) CHECK (amount_paid_gbp >= 0),
    outstanding_balance_gbp  NUMERIC(12,2) CHECK (outstanding_balance_gbp >= 0),
    -- ---- OUTCOME (emergent) ----
    default_flag             INTEGER NOT NULL CHECK (default_flag IN (0,1)),
    outcome_observed         INTEGER NOT NULL CHECK (outcome_observed IN (0,1)),
    max_dpd                  INTEGER CHECK (max_dpd >= 0),
    reached_sicr             INTEGER CHECK (reached_sicr IN (0,1)),
    cycles_delinquent        INTEGER CHECK (cycles_delinquent >= 0),
    terminal_state           TEXT CHECK (
                                 terminal_state IN
                                 ('CURRENT','DELINQUENT','DEFAULT')),
    ab_test_group            TEXT CHECK (ab_test_group IN ('A_strict','B_relaxed')),
    merchant_category        TEXT
);

-- ---- INDEXES ---------------------------------------------
CREATE INDEX idx_transactions_customer  ON raw.transactions(customer_id);
CREATE INDEX idx_repayments_transaction ON raw.repayments(transaction_id);
CREATE INDEX idx_repayments_customer    ON raw.repayments(customer_id);
-- Outcome observability is filtered on constantly when
-- building the modelling population.
CREATE INDEX idx_repayments_observed    ON raw.repayments(outcome_observed);
-- ============================================================
-- SENTINEL CREDIT — RAW SCHEMA
-- Raw source tables loaded from generator output. Constraints
-- encode business rules at the database level; foreign keys
-- guarantee every instalment traces to a real order and
-- customer. Raw and feature_store schemas kept separate for
-- data lineage.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS raw;
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
    ab_test_group                     TEXT
);

-- ---- TRANSACTIONS ----------------------------------------
CREATE TABLE raw.transactions (
    transaction_id            TEXT PRIMARY KEY,
    customer_id               TEXT NOT NULL REFERENCES raw.customers(customer_id),
    transaction_date          DATE NOT NULL,
    merchant_category         TEXT NOT NULL,
    order_value_gbp           NUMERIC(12,2) NOT NULL CHECK (order_value_gbp >= 0),
    n_instalments             INTEGER NOT NULL CHECK (n_instalments > 0),
    instalment_frequency      TEXT,
    instalment_amount_gbp     NUMERIC(12,2) CHECK (instalment_amount_gbp >= 0),
    affordability_ratio       NUMERIC(8,4)  CHECK (affordability_ratio >= 0),
    fca_affordability_flag    INTEGER CHECK (fca_affordability_flag IN (0,1)),
    application_approved      INTEGER NOT NULL CHECK (application_approved IN (0,1)),
    merchant_risk_multiplier  NUMERIC(6,3) CHECK (merchant_risk_multiplier >= 0),
    ab_test_group             TEXT
);

-- ---- REPAYMENTS ------------------------------------------
CREATE TABLE raw.repayments (
    repayment_id             TEXT PRIMARY KEY,
    transaction_id           TEXT NOT NULL REFERENCES raw.transactions(transaction_id),
    customer_id              TEXT NOT NULL REFERENCES raw.customers(customer_id),
    instalment_number        INTEGER NOT NULL CHECK (instalment_number > 0),
    instalment_amount_gbp    NUMERIC(12,2) CHECK (instalment_amount_gbp >= 0),
    due_date                 DATE NOT NULL,
    payment_status           TEXT NOT NULL,
    days_late                INTEGER CHECK (days_late >= 0),
    amount_paid_gbp          NUMERIC(12,2) CHECK (amount_paid_gbp >= 0),
    outstanding_balance_gbp  NUMERIC(12,2),
    default_flag             INTEGER NOT NULL CHECK (default_flag IN (0,1)),
    default_probability      NUMERIC(6,4) CHECK (default_probability BETWEEN 0 AND 1),
    ab_test_group            TEXT,
    merchant_category        TEXT
);

-- ---- INDEXES ---------------------------------------------
CREATE INDEX idx_transactions_customer  ON raw.transactions(customer_id);
CREATE INDEX idx_repayments_transaction ON raw.repayments(transaction_id);
CREATE INDEX idx_repayments_customer    ON raw.repayments(customer_id);
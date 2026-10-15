# Sentinel Credit — Database Design

## Purpose

The database stores the synthetic BNPL portfolio in a form that is auditable, query-efficient, and safe for feature engineering. It is divided into two schemas:

* **`raw`** – an immutable representation of the generator output.
* **`feature_store`** – model-ready features derived from the raw data.

Source data and derived features remain physically separated, allowing every model feature to be traced back to immutable raw records. This separation preserves lineage, simplifies auditing, and prevents accidental leakage between data generation and modelling.

---

## Grain

Each table represents one clearly defined real-world entity.

* **`customers`** – one row per customer, keyed by `customer_id`.
* **`transactions`** – one row per BNPL order, keyed by `transaction_id`.
* **`repayments`** – one row per scheduled instalment, keyed by `repayment_id`.

Maintaining a single grain per table ensures each fact is stored only once and prevents ambiguity during feature engineering.

---

# How Default is Represented

This is the defining characteristic of the schema.

Default is **not** generated as a probability or assigned before repayment behaviour occurs. Instead, it emerges naturally from customer repayment behaviour.

An instalment enters arrears when unpaid. Days past due accumulate against the oldest outstanding obligation, and a transaction reaches default once that obligation exceeds **90 days past due**, consistent with the **CRR/EBA regulatory definition of default**.

The schema therefore stores the **observed outcome of a repayment process**, not a predetermined label.

Rather than relying on a single default indicator, the `repayments` table records the customer's delinquency journey.

Behavioural fields include:

* `days_past_due`
* `arrears_depth`
* `max_dpd`
* `cycles_delinquent`
* `reached_sicr`
* `terminal_state`

`reached_sicr` records whether the transaction ever crossed **30 days past due**, the IFRS 9 Significant Increase in Credit Risk (SICR) threshold. It is an early-warning indicator, not a default event.

These variables represent behavioural signals that genuinely precede default, allowing the model to learn from realistic repayment behaviour rather than synthetic target probabilities.

---

# Observation Window

Every transaction is observed over a defined performance window following origination.

Behavioural variables such as `days_past_due`, `max_dpd`, and `cycles_delinquent` evolve throughout this period before a final outcome can be assigned.

Not every transaction reaches the end of this observation window before the dataset closes. Rather than discarding these records, they are retained as **right-censored observations**, preserving the temporal structure of the portfolio while preventing label leakage.

This distinction separates transactions with genuinely observed outcomes from those whose repayment journey is still unfolding.

---

# Outcome Observability and Right-Censoring

Transactions originated late in the simulation may not have completed the performance period required to observe a 90-day default.

Their outcome is **unknown**, not negative.

The `outcome_observed` flag records this explicitly.

> **Note (15th October 2026).** Since the observation horizon was extended (Decision 16, `data_end_date` 2025-12-31), every approved loan reaches an observed outcome: right-censoring is 0%. The flag and the handling above remain, as a safeguard.

A transaction has an observed outcome only if:

* it defaulted, or
* its final instalment plus the full 90-day observation window elapsed within the simulation period.

`default_flag` remains `NOT NULL` and defaults to `0` for censored observations as a storage convention.

However, modelling decisions are governed by `outcome_observed`.

A transaction with:

```text
default_flag = 0
outcome_observed = 0
```

means:

> **Outcome not yet known**

rather than

> **Customer performed successfully.**

The feature store retains censored observations for portfolio analytics, while the modelling pipeline filters training data to observed outcomes only.

---

# Normalisation

The raw layer follows **Third Normal Form (3NF)** with one deliberate exception.

Every non-key attribute depends only on its primary key, ensuring each fact is stored once and eliminating update anomalies.

Because every table uses a single-column primary key, Second Normal Form is satisfied automatically.

The only intentional denormalisation occurs within the `repayments` table.

Although `customer_id` and `merchant_category` are derivable through `transaction_id`, they are stored directly to support efficient customer-level repayment aggregation without repeatedly joining three large tables.

This small increase in storage cost substantially reduces query complexity during feature engineering and reflects a common optimisation used in production feature stores.

The denormalisation is intentional, documented, and limited.

---

# Keys

Primary keys use **structured natural identifiers** generated directly by the simulation.

Examples include:

* `T-{customer_id}-{n}` for transactions
* `R-{transaction_id}-{n}` for repayments

Each identifier occupies a single primary-key column while preserving traceability back to the generator output.

Customer identifiers are hashed pseudonyms to support GDPR-style pseudonymisation.

A production system might instead introduce surrogate integer keys for join performance, but retaining natural identifiers during development improves auditability and debugging.

---

# Referential Integrity

Foreign-key constraints guarantee relational consistency.

Every transaction must belong to an existing customer.

Every repayment must belong to an existing transaction.

The database therefore rejects orphaned records during loading rather than relying on downstream validation.

Traceability from any repayment back to its originating transaction and customer is enforced by the relational model itself.

---

# Data Types and Constraints

Financial values are stored using exact `NUMERIC(12,2)` types rather than floating-point numbers, ensuring monetary calculations remain precise.

Dates are stored as `DATE`, supporting interval calculations required for delinquency tracking, ageing, and forecasting.

Business rules are enforced through database constraints.

Flags are limited to valid binary values.

Ages and credit scores are constrained to realistic ranges.

Enumerated fields are restricted to valid business values:

* `payment_status`: Paid, Late, Missed, Defaulted
* `terminal_state`: CURRENT, DELINQUENT, DEFAULT
* `instalment_frequency`: Monthly
* `ab_test_group`: A_strict, B_relaxed

Invalid records are rejected during ingestion rather than discovered downstream.

---

# Indexing

Indexes reflect the dominant query patterns used during feature engineering.

Foreign-key columns are indexed because repayment and transaction records are frequently aggregated to customer level.

`outcome_observed` is also indexed because virtually every modelling query filters to transactions with observed outcomes. Indexing this column avoids repeated full-table scans during feature generation and training dataset construction.

Indexes are selected to optimise real workload patterns rather than applied by default.

---

# Label and Leakage Handling

`default_flag` on the `repayments` table is the modelling target.

`merchant_risk_multiplier` is an internal generator parameter used solely to synthesise repayment behaviour.

It is retained in the raw schema for auditability but deliberately excluded from the feature store so it can never become a model input.

Leakage is therefore prevented at the **feature boundary**, not the storage boundary.

This reflects an important engineering principle:

> **Leakage is a modelling constraint, not a data-retention rule.**

The previous probability-based generator stored a `default_probability` field.

That field no longer exists.

Under the hazard-based design, default emerges from repayment behaviour rather than being sampled from a probability distribution, so there is no hidden probability available to leak into the model.

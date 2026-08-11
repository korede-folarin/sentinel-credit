"""
Sentinel Credit — Portfolio Time-Series Construction

Purpose
-------
Builds the monthly portfolio-level series forecasting is actually
run against: origination volume/value and cohort default rate.
Nothing here touches customer-level features or the credit-risk
model — this is a separate, portfolio-level view built directly
from raw.transactions and raw.repayments.

Two series, two different grains, joined carefully rather than
naively:

originations_monthly: transactions grouped by month of
transaction_date, restricted to application_approved = 1 — a
declined application was never an origination and would inflate
volume with events that generated no exposure.

default_rate_monthly: THIS IS NOT "defaults recorded in month X".
default_flag and outcome_observed live on raw.repayments, at
instalment grain, and a default recorded in July from a March
origination belongs to March's cohort risk, not July's — attributing
it to the recording month would produce a noisy series describing
when defaults happen to surface rather than how risky each
origination cohort actually was. Every repayment row is joined
back to its transaction to recover the ORIGINATION month, and only
outcome_observed = 1 customers contribute — the same right-
censoring discipline used everywhere else in this project (Decision
10/12), applied here at cohort level: an origination cohort's
default rate is only meaningful once enough time has passed for
outcomes to resolve, and a partially-resolved recent cohort would
understate its own eventual rate if unresolved rows were dropped
silently rather than filtered explicitly.

This script does not forecast anything — it only builds and
validates the two series. forecasting/forecast_portfolio.py
consumes its output.

See docs/engineering_decision_log.md — Decision 10 (right-
censoring), Decision 12 (reject inference / outcome_observed).
"""

import argparse
import os
import sys
import warnings

import pandas as pd
import psycopg2
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "forecasting")

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB = CONFIG["database"]


def load_data(db_host: str) -> tuple:
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    conn = psycopg2.connect(
        host=db_host, port=DB["port"], dbname=DB["name"],
        user="postgres", password=password,
    )
    try:
        transactions = pd.read_sql(
            """
            SELECT transaction_id, customer_id, transaction_date,
                   order_value_gbp, application_approved
            FROM raw.transactions
            """,
            conn,
        )
        repayments = pd.read_sql(
            """
            SELECT r.transaction_id, r.customer_id, r.default_flag,
                   r.outcome_observed
            FROM raw.repayments r
            """,
            conn,
        )
    finally:
        conn.close()
    return transactions, repayments


def build_originations_monthly(transactions: pd.DataFrame) -> pd.DataFrame:
    approved = transactions[transactions["application_approved"] == 1].copy()
    approved["month"] = pd.to_datetime(approved["transaction_date"]).dt.to_period("M").dt.to_timestamp()

    monthly = approved.groupby("month").agg(
        n_originations=("transaction_id", "count"),
        total_value_gbp=("order_value_gbp", "sum"),
        avg_order_value_gbp=("order_value_gbp", "mean"),
    ).reset_index()

    return monthly.sort_values("month").reset_index(drop=True)


def build_default_rate_monthly(
    transactions: pd.DataFrame, repayments: pd.DataFrame
) -> pd.DataFrame:
    """
    Cohort default rate by ORIGINATION month, resolved-outcome
    customers only. A customer with multiple transactions
    contributes to each transaction's cohort separately here,
    since risk is being measured at the origination-event level
    (how risky was this month's business), not the customer level
    used elsewhere in the modelling pipeline.
    """
    approved = transactions[transactions["application_approved"] == 1][
        ["transaction_id", "transaction_date"]
    ].copy()
    approved["origination_month"] = (
        pd.to_datetime(approved["transaction_date"]).dt.to_period("M").dt.to_timestamp()
    )

    # One row per transaction: did ANY instalment on it default, and
    # was the outcome actually resolved for it. A transaction with
    # any unresolved instalment is treated as unresolved overall —
    # consistent with never inferring an outcome that hasn't happened.
    per_transaction = repayments.groupby("transaction_id").agg(
        defaulted=("default_flag", "max"),
        fully_observed=("outcome_observed", "min"),
    ).reset_index()

    merged = approved.merge(per_transaction, on="transaction_id", how="left")

    resolved = merged[merged["fully_observed"] == 1].copy()
    unresolved_count = len(merged) - len(resolved)

    monthly = resolved.groupby("origination_month").agg(
        n_resolved=("transaction_id", "count"),
        n_defaulted=("defaulted", "sum"),
    ).reset_index()
    monthly["default_rate"] = monthly["n_defaulted"] / monthly["n_resolved"]

    monthly = monthly.rename(columns={"origination_month": "month"})
    return monthly.sort_values("month").reset_index(drop=True), unresolved_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-host", default=None,
        help="Overrides config.yaml database.host (e.g. host.docker.internal in a container).",
    )
    args = parser.parse_args()

    db_host = args.db_host or os.environ.get("SENTINEL_DB_HOST", DB["host"])

    print("=" * 62)
    print("SENTINEL CREDIT — PORTFOLIO TIME-SERIES CONSTRUCTION")
    print("=" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transactions, repayments = load_data(db_host)
    print(f"Loaded {len(transactions):,} transactions, {len(repayments):,} repayment rows")

    originations = build_originations_monthly(transactions)
    default_rate, unresolved_count = build_default_rate_monthly(transactions, repayments)

    print("\n" + "-" * 62)
    print("ORIGINATIONS, MONTHLY")
    print("-" * 62)
    print(originations.to_string(index=False))

    print("\n" + "-" * 62)
    print("DEFAULT RATE BY ORIGINATION COHORT, MONTHLY (resolved outcomes only)")
    print("-" * 62)
    print(default_rate.to_string(index=False))
    print(f"\n  {unresolved_count:,} transactions excluded — outcome not yet "
          f"fully resolved for at least one instalment.")

    if len(default_rate) < 2:
        print(
            "\n  WARNING: fewer than 2 resolved monthly cohorts. Forecasting "
            "against this series would not be meaningful yet."
        )

    originations_path = os.path.join(OUTPUT_DIR, "originations_monthly.csv")
    default_rate_path = os.path.join(OUTPUT_DIR, "default_rate_monthly.csv")
    originations.to_csv(originations_path, index=False)
    default_rate.to_csv(default_rate_path, index=False)

    print(f"\nSaved:")
    print(f"  {originations_path}")
    print(f"  {default_rate_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()

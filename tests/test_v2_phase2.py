"""
v2 Phase 2 (V2-04): feature engineering at scale.

- v1's SQL can only ever be run against v2 schemas (its first statement
  drops feature_store.customer_features, so this matters);
- the pandas engine rounds exactly as PostgreSQL does and follows the
  feature store's NULL rules;
- the Spark SQL transcription and the pandas engine produce v1's columns in
  v1's order.
Full-scale parity (every cell against PostgreSQL) is checked by
v2/phase2/benchmark.py and recorded in V2-04.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, "v2", "phase2"))
import common  # noqa: E402
import pandas_engine as pe  # noqa: E402


def select_columns(sql: str) -> list:
    """Output column names of the final SELECT, in order."""
    block = sql[sql.rindex("\nSELECT"):]
    block = block[:re.search(r"\nFROM\s+\w+", block).start()]
    block = "\n".join(line.split("--", 1)[0] for line in block.splitlines())
    names = []
    for item in re.split(r",\s*\n", block[len("\nSELECT"):]):
        item = item.strip()
        m = re.search(r"\bAS\s+(\w+)\s*$", item) or re.fullmatch(r"\w+\.(\w+)", item)
        names.append(m.group(1))
    return names


def v1_sql() -> str:
    with open(common.V1_FEATURE_SQL, encoding="utf-8") as f:
        return f.read()


def test_rewritten_v1_sql_only_touches_v2_schemas():
    sql = common.v1_sql_for(250000)
    code = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
    assert "feature_store" not in code
    assert not re.search(r"(?<![\w.])raw\.", code)
    assert "DROP TABLE IF EXISTS v2_scale.customer_features_n250000;" in code
    assert "FROM v2_scale_raw_n250000.repayments" in code
    assert "idx_cf_customer_n250000" in code


def test_rewrite_refuses_sql_that_still_names_v1(tmp_path, monkeypatch):
    bad = tmp_path / "bad.sql"
    bad.write_text("DROP TABLE IF EXISTS feature_store.other_table;\nSELECT 1 FROM raw.customers;\n",
                   encoding="utf-8")
    monkeypatch.setattr(common, "V1_FEATURE_SQL", str(bad))
    with pytest.raises(RuntimeError, match="v1 schema"):
        common.v1_sql_for(50000)


def test_spark_sql_and_pandas_produce_v1_columns_in_order():
    v1_columns = select_columns(v1_sql())
    with open(os.path.join(ROOT, "v2", "phase2", "feature_store_spark.sql"), encoding="utf-8") as f:
        assert select_columns(f.read()) == v1_columns
    out = pe.build(*tiny_dataset())
    assert list(out.columns) == v1_columns


def test_round_half_up_matches_postgresql_not_float_rounding():
    num = pd.Series([75125, 1, 2, 5])            # 75125 / 1000 = 75.125 exactly
    den = pd.Series([1000, 8, 3, 0])
    out = pe.round_half_up(num, den, 2)
    assert out[0] == 75.13                       # PostgreSQL: 75.13 (float rounding gives 75.12)
    assert round(75.125, 2) == 75.12
    assert out[1] == 0.13                        # 0.125 -> 0.13
    assert out[2] == 0.67
    assert np.isnan(out[3])                      # NULLIF(den, 0)
    with pytest.raises(ValueError):
        pe.round_half_up(pd.Series([-1]), pd.Series([2]), 2)


def tiny_dataset():
    customers = pd.DataFrame({
        "customer_id": ["A", "B"], "age": [30, 40], "annual_income_gbp": [30000.0, 24000.0],
        "monthly_income_gbp": [2500.0, 2000.0], "existing_monthly_obligations_gbp": [400.0, 0.0],
        "savings_balance_gbp": [1000.0, 50.0], "credit_score": [700, 500], "months_credit_history": [60, 3],
        "num_credit_accounts": [4, 0], "num_defaults_24m": [0, 1], "thin_file_flag": [0, 1],
        "vulnerability_score": [0.1, 0.5], "is_vulnerable": [0, 1], "age_bracket": ["25-34", "35-44"],
        "income_band": ["Mid", "Low"], "employment_status": ["Employed", "Unemployed"], "ab_test_group": ["A", "B"]})
    tx = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"], "customer_id": ["A", "A", "B"],
        "transaction_date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-02-01"]).date,
        "merchant_category": ["Travel", "Home", "Sports"], "order_value_gbp": [100.0, 50.25, 300.0],
        "n_instalments": [4, 3, 6], "instalment_amount_gbp": [25.0, 16.75, 50.0],
        "affordability_ratio": [0.2, 0.35, 0.9], "fca_affordability_flag": [0, 0, 1],
        "application_approved": [1, 1, 0]})
    rep = pd.DataFrame({
        "repayment_id": ["R1", "R2", "R3"], "transaction_id": ["T1", "T1", "T2"], "customer_id": ["A", "A", "A"],
        "instalment_amount_gbp": [25.0, 25.0, 16.75], "payment_status": ["Paid", "Late", "Paid"],
        "days_late": [0, 10, 0], "max_dpd": [0, 10, 0], "reached_sicr": [0, 0, 0], "cycles_delinquent": [0, 1, 0],
        "arrears_depth": [0, 1, 0], "terminal_state": ["CURRENT", "CURRENT", "CURRENT"],
        "default_flag": [0, 0, 0], "outcome_observed": [1, 1, 1]})
    return customers, tx, rep


def test_pandas_engine_follows_the_feature_store_rules():
    out = pe.build(*tiny_dataset()).set_index("customer_id")
    a, b = out.loc["A"], out.loc["B"]
    assert a["avg_order_value_gbp"] == 75.13                 # (100 + 50.25) / 2 = 75.125, half up
    assert a["max_affordability_ratio"] == 0.35
    assert a["pct_high_risk_category"] == 0.5
    assert a["n_late_payments"] == 1 and a["avg_days_late"] == 10.0
    assert a["has_outcome"] == 1 and a["target_default"] == 0
    assert a["transaction_span_days"] == 60
    # B: one declined application, no repayments
    assert b["n_applications"] == 1 and b["n_approved"] == 0 and b["n_declined"] == 1
    assert b["approval_rate"] == 0.0 and b["max_affordability_ratio"] == 0.9
    assert pd.isna(b["avg_order_value_gbp"])                 # averages stay NULL
    assert b["n_transactions"] == 0 and b["total_order_value_gbp"] == 0   # counts and sums become 0
    assert b["has_outcome"] == 0 and b["n_instalments_total"] == 0
    assert b["debt_to_income_ratio"] == 0.0

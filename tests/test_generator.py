"""
Synthetic data generator: the repayment state machine (v1 Decisions 1, 10).

Plain English: a customer's repayments move between CURRENT, DELINQUENT and
DEFAULT. A payment always clears the OLDEST unpaid instalment first (FIFO),
and default is never assigned up front: it happens when the oldest unpaid
instalment reaches 90 days past due (DPD), the CRR/EBA definition.

How: the generator has two random decisions (does this instalment get missed,
and how does a customer in arrears respond). The tests pin those decisions so
the outcome is exact, then check the generator's own output row by row.
"""

import pandas as pd
import pytest

from conftest import load_module

gen = load_module("bnpl_data_generator", "data_generator/bnpl_data_generator.py")


def one_application(n_instalments=6, start="2023-01-02", amount=100.0):
    customers = pd.DataFrame([{
        "customer_id": "C1", "savings_balance_gbp": 500.0, "monthly_income_gbp": 2000.0,
        "existing_monthly_obligations_gbp": 300.0, "employment_status": "Employed FT",
        "thin_file_flag": 0, "is_vulnerable": 0, "credit_score": 650,
    }])
    transactions = pd.DataFrame([{
        "transaction_id": "T1", "customer_id": "C1", "application_approved": 1,
        "n_instalments": n_instalments, "instalment_amount_gbp": amount,
        "transaction_date": pd.Timestamp(start), "merchant_risk_multiplier": 1.0,
        "affordability_ratio": 0.2, "ab_test_group": "A_strict", "merchant_category": "Home",
    }])
    return customers, transactions


def run(monkeypatch, n_instalments, miss, actions):
    """miss: always/never miss when CURRENT. actions: what the customer does
    in arrears each cycle ('full', 'partial', 'none'), in order."""
    seq = iter(actions)
    monkeypatch.setattr(gen, "miss_hazard", lambda cust, tx, base: 1.0 if miss else 0.0)
    monkeypatch.setattr(gen, "resolve_arrears", lambda *a, **k: next(seq, "none"))
    customers, transactions = one_application(n_instalments)
    return gen.generate_repayments(customers, transactions).sort_values("instalment_number")


def test_default_triggers_at_exactly_90_dpd(monkeypatch):
    # Miss the first instalment and never pay: DPD rises 0, 30, 60, 90 ...
    rows = run(monkeypatch, 6, miss=True, actions=[])
    dpd = rows["days_past_due"].tolist()
    status = rows["payment_status"].tolist()
    assert dpd[:4] == [0, 30, 60, 90]
    assert status[:3] == ["Missed", "Missed", "Missed"]      # 60 DPD is not yet default
    assert status[3] == "Defaulted"                          # 90 DPD is
    assert set(rows["terminal_state"]) == {"DEFAULT"}
    assert set(rows["default_flag"]) == {1}


def test_default_is_absorbing(monkeypatch):
    # Even a 'full' payment offered after default does not cure it.
    rows = run(monkeypatch, 6, miss=True, actions=["none", "none", "full", "full"])
    status = rows["payment_status"].tolist()
    assert status[3:] == ["Defaulted", "Defaulted", "Defaulted"]


def test_payments_clear_the_oldest_instalment_first(monkeypatch):
    # Miss instalment 1, then a partial payment at instalment 2: it clears
    # instalment 1 (the oldest), so instalment 2 becomes the oldest unpaid
    # and DPD at instalment 3 is 30 days (its age), not 60 and not reset to 0.
    rows = run(monkeypatch, 6, miss=True, actions=["partial", "full"])
    r2, r3 = rows.iloc[1], rows.iloc[2]
    assert r2["payment_status"] == "Late" and r2["days_late"] == 30
    assert r2["arrears_depth"] == 1
    assert r3["days_past_due"] == 30
    # A 'full' payment clears every arrear plus the current instalment.
    assert r3["payment_status"] == "Late" and r3["amount_paid_gbp"] == pytest.approx(200.0)
    assert r3["arrears_depth"] == 0 and r3["outstanding_balance_gbp"] == 0


def test_never_missing_never_defaults(monkeypatch):
    rows = run(monkeypatch, 12, miss=False, actions=[])
    assert set(rows["default_flag"]) == {0}
    assert set(rows["terminal_state"]) == {"CURRENT"}
    assert rows["max_dpd"].max() == 0
    assert set(rows["payment_status"]) <= {"Paid", "Late"}


def test_default_can_happen_after_the_schedule_ends(monkeypatch):
    # A 3-instalment plan with all instalments missed reaches only 60 DPD in
    # the schedule; collections continue, and default lands at 90 DPD after it.
    rows = run(monkeypatch, 3, miss=True, actions=[])
    assert "Defaulted" not in rows["payment_status"].tolist()
    assert set(rows["terminal_state"]) == {"DEFAULT"}
    assert set(rows["default_flag"]) == {1}
    assert rows["max_dpd"].iloc[0] >= 90


def test_sicr_flag_follows_its_dpd_threshold(monkeypatch):
    rows = run(monkeypatch, 6, miss=True, actions=[])
    assert set(rows["reached_sicr"]) == {int(rows["max_dpd"].iloc[0] >= gen.GEN["sicr_dpd_threshold"])}


def test_affordability_ratio_formula():
    # (existing obligations + new instalment) / monthly income, FCA CONC 5.2A
    assert gen.compute_affordability_ratio(2000.0, 300.0, 600.0, 3) == pytest.approx((300 + 200) / 2000, abs=1e-4)

"""
v2 Phase 4 (V2-03): the streaming feature state.

Pure-Python tests of v2/streaming/stream_features.py, the logic Flink runs
for every customer. They need no Flink, Kafka or database (except the last
test, which replays real customers against the v1 feature store locally).
"""

import os
import random
import sys
from decimal import Decimal

import pytest

from conftest import ROOT, needs_db

sys.path.insert(0, os.path.join(ROOT, "v2", "streaming"))
import stream_features as sf  # noqa: E402

PROFILE = {"monthly_income_gbp": "2500.00", "existing_monthly_obligations_gbp": "400.00",
           "savings_balance_gbp": "1200.50", "credit_score": "640", "months_credit_history": "48",
           "num_credit_accounts": "3", "num_defaults_24m": "0", "thin_file_flag": "0"}


def registration(order=0, t=0):
    return {"type": "registration", "event_id": "reg-C1", "customer_id": "C1", "event_time_ms": t,
            "order": order, **PROFILE}


def application(n, value, inst, category, approved, aff, t=None):
    return {"type": "application", "event_id": f"app-T{n}", "customer_id": "C1", "transaction_id": f"T{n}",
            "event_time_ms": n * 1000 if t is None else t, "order": n, "order_value_gbp": value,
            "n_instalments": inst, "merchant_category": category, "application_approved": approved,
            "affordability_ratio": aff}


def test_feature_order_matches_the_champion():
    from models.model_features import MODEL_FEATURES
    assert sf.FEATURES == MODEL_FEATURES


def test_risk_tiers_match_serving():
    from serving import features
    assert sf.HIGH_RISK_CATEGORIES == features.HIGH_RISK_CATEGORIES
    assert sf.MEDIUM_RISK_CATEGORIES == features.MEDIUM_RISK_CATEGORIES
    assert sf.PROFILE_FIELDS == features.PROFILE_FIELDS


def test_final_state_follows_the_feature_store_sql():
    st = sf.CustomerState()
    sf.process_event(st, registration())
    sf.process_event(st, application(1, "100.00", 4, "Travel", 1, "0.2000"))
    sf.process_event(st, application(2, "50.25", 3, "Home", 1, "0.3500"))
    sf.process_event(st, application(3, "900.00", 6, "Electronics", 0, "0.9000"))   # declined
    f = sf.current_features(st)
    assert f["max_affordability_ratio"] == Decimal("0.9000")      # MAX over ALL applications
    assert f["avg_order_value_gbp"] == Decimal("75.13")           # (100 + 50.25) / 2 = 75.125, half up
    assert f["avg_n_instalments"] == Decimal("3.50")              # approved only
    assert f["pct_high_risk_category"] == Decimal("0.5000")
    assert f["pct_medium_risk_category"] == Decimal("0.5000")
    assert f["credit_score"] == Decimal("640")


def test_no_approved_application_gives_null_behaviour_features():
    st = sf.CustomerState()
    sf.process_event(st, registration())
    sf.process_event(st, application(1, "300.00", 3, "Sports", 0, "0.4100"))
    f = sf.current_features(st)
    assert f["max_affordability_ratio"] == Decimal("0.4100")
    assert all(f[k] is None for k in sf.BEHAVIOUR_FIELDS[1:])
    assert sf.to_float_row(f)[-1] != sf.to_float_row(f)[-1]      # NaN, as the v1 API passes it


def test_point_in_time_score_sees_history_plus_this_order_not_its_outcome():
    st = sf.CustomerState()
    sf.process_event(st, registration())
    sf.process_event(st, application(1, "100.00", 4, "Travel", 1, "0.2000"))
    out = sf.process_event(st, application(2, "300.00", 2, "Sports", 0, "0.5000"))[0]
    pit = out["features"]
    assert pit["avg_order_value_gbp"] == Decimal("200.00")        # (100 + 300) / 2: candidate order included
    assert pit["max_affordability_ratio"] == Decimal("0.5000")
    assert pit["pct_high_risk_category"] == Decimal("0.5000")
    after = out["state"]                                          # declined, so the averages do not move
    assert after["avg_order_value_gbp"] == Decimal("100.00")


def test_first_application_matches_the_v1_api_new_customer_features():
    from serving.features import build_new_customer_features
    value, inst = Decimal("240.00"), 4
    profile = {k: Decimal(v) for k, v in PROFILE.items()}
    aff = (profile["existing_monthly_obligations_gbp"] + value / inst) / profile["monthly_income_gbp"]
    st = sf.CustomerState()
    sf.process_event(st, registration())
    pit = sf.process_event(st, application(1, str(value), inst, "Electronics", 1, str(round(aff, 4))))[0]["features"]
    api = build_new_customer_features({k: float(v) for k, v in profile.items()}, float(value), inst, "Electronics")
    assert sf.to_float_row(pit) == pytest.approx([api[f] for f in sf.FEATURES], abs=1e-9)


def test_application_before_profile_is_held_then_scored():
    st = sf.CustomerState()
    assert sf.process_event(st, application(1, "100.00", 4, "Travel", 1, "0.2000")) == []
    results = sf.process_event(st, registration(), late=True)
    assert [r["kind"] for r in results] == ["registration", "application"]
    assert results[1]["delayed_profile"] is True
    assert st.pending == []


def test_final_state_does_not_depend_on_arrival_order():
    events = [application(n, f"{random.Random(n).uniform(20, 900):.2f}", random.Random(n).choice([3, 4, 6]),
                          random.Random(n).choice(sorted(sf.HIGH_RISK_CATEGORIES | sf.MEDIUM_RISK_CATEGORIES
                                                          | {"Sports"})),
                          n % 3 != 0, f"{random.Random(-n).uniform(0.05, 0.9):.4f}") for n in range(1, 30)]
    reference = sf.CustomerState()
    for e in [registration()] + events:
        sf.process_event(reference, e)
    for seed in range(5):
        shuffled = [registration()] + events
        random.Random(seed).shuffle(shuffled)
        st = sf.CustomerState()
        for e in shuffled:
            sf.process_event(st, e, late=True)
        assert sf.current_features(st) == sf.current_features(reference)


@needs_db
def test_replayed_history_equals_the_v1_feature_store():
    """200 real customers, every event applied in a shuffled order: the final
    state must equal feature_store.customer_features exactly."""
    import pandas as pd

    from v2.phase1.common import db_connect
    profile_sql = ", ".join(f"{f}::text AS {f}" for f in sf.PROFILE_FIELDS)
    conn = db_connect()
    try:
        customers = pd.read_sql(f"SELECT customer_id, {profile_sql} FROM raw.customers "
                                "ORDER BY md5(customer_id) LIMIT 200", conn)
        ids = customers["customer_id"].tolist()
        apps = pd.read_sql("SELECT transaction_id, customer_id, merchant_category, order_value_gbp::text AS "
                           "order_value_gbp, n_instalments, affordability_ratio::text AS affordability_ratio, "
                           "application_approved FROM raw.transactions WHERE customer_id = ANY(%s)",
                           conn, params=(ids,))
        expected = pd.read_sql(f"SELECT customer_id, {', '.join(sf.FEATURES)} FROM feature_store.customer_features "
                               "WHERE customer_id = ANY(%s)", conn, params=(ids,)).set_index("customer_id")
    finally:
        conn.close()
    events = [{"type": "registration", "event_id": f"reg-{r['customer_id']}", "event_time_ms": 0, "order": 0, **r}
              for r in customers.to_dict("records")]
    events += [{"type": "application", "event_id": f"app-{r['transaction_id']}", "event_time_ms": 0, "order": 0, **r}
               for r in apps.to_dict("records")]
    random.Random(7).shuffle(events)
    states = {}
    for e in events:
        sf.process_event(states.setdefault(e["customer_id"], sf.CustomerState()), e, late=True)
    for cid, st in states.items():
        got = sf.current_features(st)
        for f in sf.FEATURES:
            want = expected.at[cid, f]
            if pd.isna(want):
                assert got[f] is None, (cid, f)
            else:
                assert float(got[f]) == pytest.approx(float(want), abs=1e-9), (cid, f)

"""
Sentinel Credit v2, Phase 4: per-customer streaming feature state (V2-03).

Plain English: as each event arrives for a customer, keep a small running
summary (the "state") so the champion's 13 features are always up to date,
without re-reading the customer's whole history. Flink holds one of these
per customer (keyed state); this module is pure Python so it can be unit
tested without Flink, Kafka or a database.

Two kinds of event, both replayed from v1's raw tables (read-only):
- registration: the customer's profile (raw.customers), the 8 profile
  features;
- application: one BNPL application (raw.transactions).

Feature definitions match sql/02_feature_store.sql exactly:
- max_affordability_ratio: ROUND(MAX(affordability_ratio), 4) over ALL
  applications;
- avg_order_value_gbp, avg_n_instalments: ROUND(AVG(...), 2) over APPROVED
  applications only;
- pct_high_risk_category, pct_medium_risk_category: ROUND(AVG(1 or 0), 4)
  over APPROVED applications only;
- no approved application -> NULL behaviour features (the SQL LEFT JOIN).
Money and ratios are exact decimals rounded half away from zero, as
PostgreSQL's numeric ROUND does, so the final state can be compared with the
v1 feature store to 1e-9.

Point-in-time scoring (what the model sees when an application arrives):
the profile, plus everything BEFORE this application, plus this application
itself as the candidate order. This is how the v1 API builds a first-time
applicant's features (serving/features.py: the averages equal this order),
extended to customers with history. The historical approve/decline flag of
the current application is never used to score it; it only updates the
state afterwards.

State updates are sums, counts and a maximum, so the FINAL state does not
depend on arrival order. That is what makes late events safe: they are
applied as soon as they arrive, and only their own point-in-time score is
flagged as "late".
"""

from decimal import ROUND_HALF_UP, Decimal

PROFILE_FIELDS = [
    "monthly_income_gbp",
    "existing_monthly_obligations_gbp",
    "savings_balance_gbp",
    "credit_score",
    "months_credit_history",
    "num_credit_accounts",
    "num_defaults_24m",
    "thin_file_flag",
]
BEHAVIOUR_FIELDS = [
    "max_affordability_ratio",
    "avg_order_value_gbp",
    "avg_n_instalments",
    "pct_high_risk_category",
    "pct_medium_risk_category",
]
# Same order as models/model_features.py MODEL_FEATURES (checked by tests).
FEATURES = PROFILE_FIELDS + BEHAVIOUR_FIELDS

# Same tiers as sql/02_feature_store.sql and serving/features.py.
HIGH_RISK_CATEGORIES = {"Travel", "Electronics"}
MEDIUM_RISK_CATEGORIES = {"Fashion & Clothing", "Home", "Other"}

_ZERO = Decimal(0)


def _round(value: Decimal, places: int) -> Decimal:
    """PostgreSQL numeric ROUND: half away from zero."""
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def _dec(value) -> Decimal:
    # Events carry numbers as strings so no binary rounding creeps in.
    return value if isinstance(value, Decimal) else Decimal(str(value))


class CustomerState:
    """Everything Flink keeps for one customer (pickled into keyed state)."""

    def __init__(self):
        self.profile = None          # dict of the 8 profile values, once registered
        self.n_applications = 0
        self.max_affordability = None
        self.n_approved = 0
        self.sum_order_value = _ZERO
        self.sum_instalments = 0
        self.n_high = 0
        self.n_medium = 0
        self.seen = set()            # event ids already accepted (de-duplication)
        self.last_event_ms = -1      # newest event time applied (lateness check)
        self.pending = []            # applications that arrived before the profile
        self.seq = 0                 # events applied so far (orders state updates)


def behaviour_features(max_aff, n_approved, sum_order, sum_inst, n_high, n_medium) -> dict:
    """The 5 behaviour features from running totals, rounded as the SQL rounds."""
    out = {"max_affordability_ratio": None if max_aff is None else _round(max_aff, 4)}
    if n_approved == 0:
        out.update({f: None for f in BEHAVIOUR_FIELDS[1:]})
        return out
    n = Decimal(n_approved)
    out["avg_order_value_gbp"] = _round(sum_order / n, 2)
    out["avg_n_instalments"] = _round(Decimal(sum_inst) / n, 2)
    out["pct_high_risk_category"] = _round(Decimal(n_high) / n, 4)
    out["pct_medium_risk_category"] = _round(Decimal(n_medium) / n, 4)
    return out


def current_features(state: CustomerState) -> dict:
    """The customer's features after every event applied so far (what the v1
    feature store holds once the whole history has been applied)."""
    profile = state.profile or {f: None for f in PROFILE_FIELDS}
    return {**profile, **behaviour_features(
        state.max_affordability, state.n_approved, state.sum_order_value,
        state.sum_instalments, state.n_high, state.n_medium)}


def point_in_time_features(state: CustomerState, application: dict) -> dict:
    """Features at the moment this application arrives: history so far plus
    this order as the candidate (its historical outcome is not used)."""
    aff = _dec(application["affordability_ratio"])
    category = application["merchant_category"]
    max_aff = aff if state.max_affordability is None else max(state.max_affordability, aff)
    return {**state.profile, **behaviour_features(
        max_aff,
        state.n_approved + 1,
        state.sum_order_value + _dec(application["order_value_gbp"]),
        state.sum_instalments + int(application["n_instalments"]),
        state.n_high + (category in HIGH_RISK_CATEGORIES),
        state.n_medium + (category in MEDIUM_RISK_CATEGORIES))}


def apply_application(state: CustomerState, application: dict) -> None:
    """Fold one application into the running totals, as the SQL aggregates do."""
    aff = _dec(application["affordability_ratio"])
    state.n_applications += 1
    state.max_affordability = aff if state.max_affordability is None else max(state.max_affordability, aff)
    if int(application["application_approved"]) == 1:
        category = application["merchant_category"]
        state.n_approved += 1
        state.sum_order_value += _dec(application["order_value_gbp"])
        state.sum_instalments += int(application["n_instalments"])
        state.n_high += category in HIGH_RISK_CATEGORIES
        state.n_medium += category in MEDIUM_RISK_CATEGORIES
    state.seq += 1


def apply_registration(state: CustomerState, registration: dict) -> None:
    state.profile = {f: _dec(registration[f]) for f in PROFILE_FIELDS}
    state.seq += 1


def process_event(state: CustomerState, event: dict, late: bool = False) -> list:
    """
    Apply one accepted (not duplicate) event in arrival order and return what
    happened, for the caller to score and emit:
      {"kind": "registration", "event": ..., "state": {...}}
      {"kind": "application", "event": ..., "features": {...}, "state": {...},
       "late": bool, "delayed_profile": bool}
    An application that arrives before its customer's profile is held and
    processed when the profile arrives (delayed_profile = True).
    """
    if event["type"] == "registration":
        apply_registration(state, event)
        results = [{"kind": "registration", "event": event, "late": late,
                    "state": current_features(state), "seq": state.seq}]
        held, state.pending = state.pending, []
        for app, app_late in sorted(held, key=lambda item: order_key(item[0])):
            results.append(_score_application(state, app, app_late, delayed_profile=True))
        return results
    if state.profile is None:
        state.pending.append((event, late))
        return []
    return [_score_application(state, event, late, delayed_profile=False)]


def _score_application(state, app, late, delayed_profile) -> dict:
    features = point_in_time_features(state, app)
    apply_application(state, app)
    return {"kind": "application", "event": app, "features": features, "late": late,
            "delayed_profile": delayed_profile, "state": current_features(state), "seq": state.seq}


def order_key(event: dict) -> tuple:
    """Event-time order; ties broken by the replay's historical order."""
    return (int(event["event_time_ms"]), int(event["order"]))


def to_float_row(features: dict) -> list:
    """Model input in FEATURES order; missing becomes NaN, as the v1 API passes it."""
    return [float("nan") if features[f] is None else float(features[f]) for f in FEATURES]


def to_json_values(features: dict) -> dict:
    """Exact decimal strings for audit and parity (None stays None)."""
    return {f: None if features[f] is None else str(features[f]) for f in FEATURES}

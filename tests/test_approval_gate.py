"""
The Approval Gate (v1 Decisions 23-24).

Plain English: before a model can be promoted, the gate checks the latest
monitoring evidence. It PASSES only if that evidence exists, is recent
enough, and shows no feature drift alerts. The portfolio bad-rate shift is
reported but deliberately does not block.
"""

from datetime import datetime, timedelta, timezone

from conftest import load_module

gate = load_module("approval_gate", "pipeline/approval_gate.py")


def evidence(age_hours=1.0, alerts=0, total=13, delta=0.001):
    return {"generated_at": (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat(),
            "n_features_with_alerts": alerts, "n_features_total": total,
            "target_rate_shift": {"absolute_delta": delta}}


def test_no_evidence_blocks():
    assert gate.evaluate_gate(None)["gate_status"] == "BLOCK"


def test_fresh_evidence_with_no_alerts_passes():
    result = gate.evaluate_gate(evidence())
    assert result["gate_status"] == "PASS"
    assert result["n_alerts"] == 0 and result["n_features_total"] == 13


def test_stale_evidence_blocks():
    result = gate.evaluate_gate(evidence(age_hours=gate.MAX_MONITORING_AGE_HOURS + 1))
    assert result["gate_status"] == "BLOCK"
    assert any(c["check"] == "monitoring_evidence_recent" and c["passed"] is False for c in result["checks"])


def test_drift_alerts_block():
    assert gate.evaluate_gate(evidence(alerts=2))["gate_status"] == "BLOCK"


def test_bad_rate_shift_is_reported_but_never_blocks():
    result = gate.evaluate_gate(evidence(delta=0.25))
    assert result["gate_status"] == "PASS"
    info = [c for c in result["checks"] if c["passed"] is None]
    assert len(info) == 1 and "+0.2500" in info[0]["detail"]


def test_naive_timestamps_are_read_as_local_time():
    # Evidence written before v1 Decision 24 carried no UTC offset.
    naive = dict(evidence(), generated_at=(datetime.now() - timedelta(hours=1)).isoformat())
    result = gate.evaluate_gate(naive)
    assert result["gate_status"] == "PASS" and 0.9 < result["monitoring_age_hours"] < 1.1

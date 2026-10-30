"""
Sentinel Credit v2, Phase 4: checks on one replay run (V2-03).

Plain English: after a replay, answer five questions with real numbers:
1. Completeness: did every event get processed exactly once? Were the
   injected duplicates dropped, and how many events arrived late?
2. Parity: does each customer's final streaming state equal the v1 feature
   store row, feature by feature, to within 1e-9?
3. Speed: how long from send to processed (latency, p50/p95/p99), how many
   events per second (throughput), and how far behind the job fell (lag)?
4. Scoring parity: does the champion inside Flink give the same probability
   as the champion on this machine, for the same features?
5. Drift: how do the point-in-time features the stream scored compare with
   the population the champion was trained on (v1's PSI calculation)?
It also records the v1 Approval Gate status, read-only. The stream runs in
shadow mode whatever the gate says: nothing is returned to an applicant.

Reads v2_stream and, read-only, feature_store.customer_features and the v1
model package. Writes only v2_stream.run_metrics, v2_stream.feature_drift
and v2/reports/phase4/<run_id>_check.json.

  python v2/streaming/stream_check.py --run-id stream_100eps_...
  python v2/streaming/stream_check.py --latest
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "pipeline"), os.path.join(ROOT, "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from stream_common import P4, SCHEMA, V1, db_connect  # noqa: E402
from stream_features import FEATURES  # noqa: E402

REPORTS = os.path.join(ROOT, "v2", "reports", "phase4")
TOL = float(P4["parity_tolerance"])


def q(conn, sql, params=None) -> pd.DataFrame:
    return pd.read_sql(sql, conn, params=params)


def pct(series: pd.Series, p: float) -> float:
    return float(np.percentile(series, p)) if len(series) else float("nan")


def completeness(conn, run_id) -> dict:
    run = q(conn, f"SELECT * FROM {SCHEMA}.replay_runs WHERE run_id = %s", (run_id,)).iloc[0]
    counts = q(conn, f"""SELECT
        (SELECT count(*) FROM {SCHEMA}.processed_events WHERE run_id = %(r)s) AS processed,
        (SELECT count(*) FROM {SCHEMA}.processed_events WHERE run_id = %(r)s AND late) AS processed_late,
        (SELECT count(*) FROM {SCHEMA}.scored_events WHERE run_id = %(r)s) AS scored,
        (SELECT count(*) FROM {SCHEMA}.scored_events WHERE run_id = %(r)s AND delayed_profile) AS delayed_profile,
        (SELECT count(*) FROM {SCHEMA}.duplicates_dropped WHERE run_id = %(r)s AND copy = 1) AS duplicates_dropped,
        (SELECT count(*) FROM {SCHEMA}.customer_state WHERE run_id = %(r)s) AS customers""",
                {"r": run_id}).iloc[0]
    return {
        "rate_eps": int(run["rate_eps"]), "n_customers": int(run["n_customers"]),
        "events": int(run["n_events"]), "messages_sent": int(run["n_messages_sent"]),
        "duplicates_sent": int(run["n_duplicates_sent"]), "late_sent": int(run["n_late_sent"]),
        "processed": int(counts["processed"]), "processed_late": int(counts["processed_late"]),
        "scored_applications": int(counts["scored"]), "delayed_profile": int(counts["delayed_profile"]),
        "duplicates_dropped": int(counts["duplicates_dropped"]), "customers_with_state": int(counts["customers"]),
        "all_processed_once": bool(counts["processed"] == run["n_events"]),
        "all_duplicates_dropped": bool(counts["duplicates_dropped"] == run["n_duplicates_sent"]),
        "stream_seconds": float((run["stream_end_at"] - run["started_at"]).total_seconds()),
    }


def parity(conn, run_id) -> dict:
    cols = ", ".join(f"s.{f} AS s_{f}, v.{f} AS v_{f}" for f in FEATURES)
    df = q(conn, f"""SELECT s.customer_id, {cols}
        FROM {SCHEMA}.customer_state s
        LEFT JOIN feature_store.customer_features v ON v.customer_id = s.customer_id
        WHERE s.run_id = %s""", (run_id,))
    per_feature, mismatched = {}, set()
    for f in FEATURES:
        s = pd.to_numeric(df[f"s_{f}"], errors="coerce")
        v = pd.to_numeric(df[f"v_{f}"], errors="coerce")
        both_null = s.isna() & v.isna()
        diff = (s - v).abs()
        bad = ~both_null & (s.isna() | v.isna() | (diff > TOL))
        mismatched |= set(df.loc[bad, "customer_id"])
        per_feature[f] = {"max_abs_diff": float(diff.max()) if diff.notna().any() else 0.0,
                          "mismatches": int(bad.sum()), "both_null": int(both_null.sum())}
    return {"customers_compared": len(df), "customers_matching": len(df) - len(mismatched),
            "customers_mismatched": len(mismatched), "tolerance": TOL, "per_feature": per_feature,
            "example_mismatches": sorted(mismatched)[:10]}


def speed(conn, run_id) -> dict:
    ev = q(conn, f"SELECT kind, event_time_ms, sent_ms, features_ms, processed_ms, stored_ms, late "
                 f"FROM {SCHEMA}.processed_events WHERE run_id = %s", (run_id,))
    on_time = ev[~ev["late"]]
    flink = on_time["processed_ms"] - on_time["sent_ms"]
    e2e = on_time["stored_ms"] - on_time["sent_ms"]
    seconds = (ev["processed_ms"] // 1000).value_counts()
    # Throughput over the main stream only: late events trickle in up to 10
    # minutes later and would make the run look slow.
    span = (on_time["processed_ms"].max() - on_time["processed_ms"].min()) / 1000
    lag = q(conn, f"SELECT * FROM {SCHEMA}.lag_samples WHERE run_id = %s ORDER BY sampled_ms", (run_id,))
    return {
        "on_time_events": len(on_time),
        "latency_send_to_processed_ms": {"p50": pct(flink, 50), "p95": pct(flink, 95), "p99": pct(flink, 99),
                                         "max": float(flink.max())},
        "latency_send_to_stored_ms": {"p50": pct(e2e, 50), "p95": pct(e2e, 95), "p99": pct(e2e, 99),
                                      "max": float(e2e.max())},
        # Where the time goes (on-time events, medians): waiting for the
        # watermark to put events in order, the scoring step, the sink.
        "stage_median_ms": {
            "event_time_to_features": pct(on_time["features_ms"] - on_time["event_time_ms"], 50),
            "features_to_scored": pct(on_time["processed_ms"] - on_time["features_ms"], 50),
            "scored_to_stored": pct(on_time["stored_ms"] - on_time["processed_ms"], 50)},
        "late_events_send_to_processed_ms_p50": pct(ev.loc[ev["late"], "processed_ms"]
                                                     - ev.loc[ev["late"], "sent_ms"], 50),
        "throughput_mean_eps": float(len(on_time) / span) if span > 0 else float("nan"),
        "throughput_peak_1s_eps": int(seconds.max()),
        "lag_max_sent_not_stored": int((lag["produced"] - lag["stored"]).max()) if len(lag) else None,
        "lag_max_sent_not_committed_by_flink": (int((lag["produced"] - lag["flink_committed"]).max())
                                                if lag["flink_committed"].notna().any() else None),
        "lag_samples": len(lag),
    }


def scoring_parity(conn, run_id) -> dict:
    import sklearn
    import xgboost
    if (sklearn.__version__, xgboost.__version__) != ("1.5.0", "2.0.3"):
        return {"skipped": f"needs scikit-learn 1.5.0 and xgboost 2.0.3 (the champion's), found "
                           f"{sklearn.__version__} and {xgboost.__version__}"}
    import joblib
    model = joblib.load(os.path.join(ROOT, "dashboard", "demo_data", "model", "champion_calibrated.joblib"))
    df = q(conn, f"SELECT {', '.join(FEATURES)}, pd, decision FROM {SCHEMA}.scored_events WHERE run_id = %s",
           (run_id,))
    X = df[FEATURES].apply(pd.to_numeric).astype(float)
    host = model.predict_proba(X)[:, 1]
    cutoff = float(V1["model"]["business_cutoff"])
    host_decision = np.where(host >= cutoff, "flag_high_risk", "pass")
    return {"scored": len(df), "max_abs_pd_diff": float(np.abs(host - df["pd"].to_numpy()).max()),
            "decision_disagreements": int((host_decision != df["decision"].to_numpy()).sum()),
            "flagged_share_point_in_time": float((df["decision"] == "flag_high_risk").mean())}


def drift(conn, run_id) -> list:
    """PSI of the stream's point-in-time features against v1's training population."""
    import monitor_drift as v1_drift  # v1 PSI code, read-only
    full = v1_drift.load_full_population()
    reference = v1_drift.load_reference_population(full)
    stream = q(conn, f"SELECT {', '.join(FEATURES)} FROM {SCHEMA}.scored_events WHERE run_id = %s", (run_id,))
    # Final state: only the modelable customers (the population the champion
    # was trained from), so the comparison is like for like. Without this,
    # customers with no approved application (NULL averages, as in the v1
    # feature store) showed as a false "material shift".
    modelable = set(full.loc[full[v1_drift.FILTER_COLUMN] == v1_drift.FILTER_VALUE, "customer_id"].astype(str))
    final = q(conn, f"SELECT customer_id, {', '.join(FEATURES)} FROM {SCHEMA}.customer_state WHERE run_id = %s",
              (run_id,))
    final = final[final["customer_id"].isin(modelable)]
    rows = []
    for comparison, frame in (("point_in_time_vs_v1_training", stream), ("final_state_vs_v1_training", final)):
        for f in FEATURES:
            psi = v1_drift.population_stability_index(reference[f].astype(float),
                                                      pd.to_numeric(frame[f]).astype(float), v1_drift.PSI_BINS)
            rows.append((run_id, comparison, f, psi, v1_drift.psi_band(psi)))
    return rows


def approval_gate() -> dict:
    import approval_gate as v1_gate  # read-only: evaluate, never record
    gate = v1_gate.evaluate_gate(v1_gate.load_monitoring_evidence())
    return {"gate_status": gate["gate_status"], "mode": P4["mode"],
            "meaning": "Shadow mode: stream decisions are logged only. The gate governs promotion, which "
                       "this phase never attempts; a BLOCK keeps the stream in shadow mode."}


def save(conn, run_id, report, drift_rows):
    flat = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(value, (bool, int, float, np.integer, np.floating)) and value is not None:
            flat.append((run_id, prefix, float(value), None))
        elif value is not None:
            flat.append((run_id, prefix, None, json.dumps(value) if not isinstance(value, str) else value))

    walk("", report)
    with conn, conn.cursor() as cur:
        execute_values(cur, f"""INSERT INTO {SCHEMA}.run_metrics (run_id, metric, value, detail) VALUES %s
            ON CONFLICT (run_id, metric) DO UPDATE SET value = EXCLUDED.value, detail = EXCLUDED.detail,
            computed_at = now()""", flat)
        if drift_rows:
            execute_values(cur, f"""INSERT INTO {SCHEMA}.feature_drift VALUES %s
                ON CONFLICT (run_id, comparison, feature) DO UPDATE SET psi = EXCLUDED.psi, band = EXCLUDED.band""",
                           drift_rows)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, f"{run_id}_check.json"), "w", encoding="utf-8") as f:
        keys = ("run_id", "comparison", "feature", "psi", "band")
        json.dump({"run_id": run_id, **report, "drift": [dict(zip(keys, r, strict=True)) for r in drift_rows]},
                  f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--skip-drift", action="store_true")
    args = parser.parse_args()
    conn = db_connect()
    try:
        run_id = args.run_id
        if args.latest or not run_id:
            run_id = q(conn, f"SELECT run_id FROM {SCHEMA}.replay_runs WHERE status = 'finished' "
                             "ORDER BY finished_at DESC LIMIT 1").iloc[0, 0]
        report = {"completeness": completeness(conn, run_id), "parity": parity(conn, run_id),
                  "speed": speed(conn, run_id), "scoring_parity": scoring_parity(conn, run_id),
                  "approval_gate": approval_gate()}
        drift_rows = [] if args.skip_drift else drift(conn, run_id)
        save(conn, run_id, report, drift_rows)
    finally:
        conn.close()

    c, p, s = report["completeness"], report["parity"], report["speed"]
    print(f"{run_id}: {c['processed']:,}/{c['events']:,} events processed once; "
          f"duplicates dropped {c['duplicates_dropped']}/{c['duplicates_sent']}; "
          f"late (behind watermark) {c['processed_late']}")
    print(f"  parity: {p['customers_matching']:,}/{p['customers_compared']:,} customers match the v1 feature store")
    print(f"  latency send->processed p50/p95/p99 ms: {s['latency_send_to_processed_ms']['p50']:.0f} / "
          f"{s['latency_send_to_processed_ms']['p95']:.0f} / {s['latency_send_to_processed_ms']['p99']:.0f}; "
          f"throughput mean {s['throughput_mean_eps']:.1f}/s, peak {s['throughput_peak_1s_eps']}/s")
    print(f"  scoring parity: {report['scoring_parity']}")
    print(f"  gate: {report['approval_gate']['gate_status']} (mode {report['approval_gate']['mode']})")
    ok = c["all_processed_once"] and c["all_duplicates_dropped"] and p["customers_mismatched"] == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

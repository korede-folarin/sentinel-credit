"""
Sentinel Credit v2, Phase 4: results sink (V2-03).

Plain English: reads what the Flink job produced and stores it twice:
- in PostgreSQL, schema v2_stream, for the checks and reporting;
- in a local audit folder laid out the way an S3 bucket would be
  (bucket=.../stream/run_id=.../dt=YYYY-MM-DD/part-....jsonl), append-only.

Idempotent writes: Flink's output is at-least-once, so after a restart some
results can arrive twice. Every table has a natural key (run_id + event_id,
or run_id + customer_id with a sequence number), and the writes are
"insert, or ignore if already there" / "update only if newer". Kafka read
positions are committed only after the database commit, so a crash between
the two replays a batch instead of losing it.
"""

import json
import os
import time
from datetime import datetime, timezone

from confluent_kafka import Consumer
from psycopg2.extras import execute_values
from stream_common import P4, SCHEMA, db_connect
from stream_features import FEATURES

AUDIT_ROOT = os.environ.get("SENTINEL_AUDIT_DIR", "/audit")
BUCKET = "bucket=sentinel-v2-audit"
BATCH_MAX = 2000
BATCH_WAIT_S = 0.5


def _latest(rows, key):
    """Keep one row per key (the highest seq), so one INSERT never touches a row twice."""
    best = {}
    for r in rows:
        k = key(r)
        if k not in best or r["seq"] > best[k]["seq"]:
            best[k] = r
    return list(best.values())


def _unique(rows, key):
    return list({key(r): r for r in rows}.values())


def write_audit(records, stored_ms):
    by_run = {}
    for r in records:
        by_run.setdefault(r["run_id"], []).append(r)
    dt = datetime.fromtimestamp(stored_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    for run_id, rows in by_run.items():
        folder = os.path.join(AUDIT_ROOT, BUCKET, "stream", f"run_id={run_id}", f"dt={dt}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"part-{stored_ms}-{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def store(conn, records):
    stored_ms = int(time.time() * 1000)
    processed, scored, states, dups = [], [], [], []
    for r in records:
        if r["kind"] == "duplicate":
            dups.append((r["run_id"], r["event_id"], int(r["copy"]), r["processed_ms"]))
            continue
        processed.append(r)
        states.append(r)
        if r["kind"] == "application":
            scored.append(r)

    with conn, conn.cursor() as cur:
        if processed:
            execute_values(cur, f"INSERT INTO {SCHEMA}.processed_events VALUES %s ON CONFLICT DO NOTHING",
                           [(r["run_id"], r["event_id"], r["kind"], r["event_time_ms"], r["sent_ms"],
                             r["features_ms"], r["processed_ms"], stored_ms, r["late"])
                            for r in _unique(processed, lambda r: (r["run_id"], r["event_id"]))])
        if scored:
            execute_values(cur, f"""INSERT INTO {SCHEMA}.scored_events
                (run_id, event_id, customer_id, transaction_id, hist_date, event_time_ms, sent_ms, processed_ms,
                 stored_ms, late, delayed_profile, {", ".join(FEATURES)}, pd, decision, hist_approved,
                 model_version) VALUES %s ON CONFLICT DO NOTHING""",
                [(r["run_id"], r["event_id"], r["customer_id"], r["transaction_id"], r["hist_date"],
                  r["event_time_ms"], r["sent_ms"], r["processed_ms"], stored_ms, r["late"],
                  r["delayed_profile"], *[r["features"][f] for f in FEATURES], r["pd"], r["decision"],
                  r["hist_approved"], r["model_version"])
                 for r in _unique(scored, lambda r: (r["run_id"], r["event_id"]))])
        if states:
            execute_values(cur, f"""INSERT INTO {SCHEMA}.customer_state
                (run_id, customer_id, seq, {", ".join(FEATURES)}, updated_ms) VALUES %s
                ON CONFLICT (run_id, customer_id) DO UPDATE SET seq = EXCLUDED.seq,
                {", ".join(f"{f} = EXCLUDED.{f}" for f in FEATURES)}, updated_ms = EXCLUDED.updated_ms
                WHERE {SCHEMA}.customer_state.seq < EXCLUDED.seq""",
                [(r["run_id"], r["customer_id"], r["seq"], *[r["state"][f] for f in FEATURES], r["processed_ms"])
                 for r in _latest(states, lambda r: (r["run_id"], r["customer_id"]))])
        if dups:
            execute_values(cur, f"INSERT INTO {SCHEMA}.duplicates_dropped VALUES %s ON CONFLICT DO NOTHING",
                           list({d[:3]: d for d in dups}.values()))
    write_audit(records, stored_ms)


def main():
    consumer = Consumer({"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"),
                         "group.id": "sentinel-v2-sink", "enable.auto.commit": False,
                         "auto.offset.reset": "earliest"})
    consumer.subscribe([P4["kafka"]["scores_topic"]])
    conn = db_connect()
    print("sink running")
    try:
        while True:
            messages = consumer.consume(num_messages=BATCH_MAX, timeout=BATCH_WAIT_S)
            records = []
            for m in messages:
                if m.error():
                    print(f"kafka error: {m.error()}")
                    continue
                records.append(json.loads(m.value()))
            if not records:
                continue
            try:
                store(conn, records)
            except Exception:
                conn.close()
                conn = db_connect()      # reconnect; the batch is re-read because offsets were not committed
                raise
            consumer.commit(asynchronous=False)
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    main()

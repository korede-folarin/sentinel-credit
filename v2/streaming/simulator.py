"""
Sentinel Credit v2, Phase 4: event-stream simulator (V2-03).

Plain English: turns the v1 synthetic history into a live-looking stream.
It reads chosen customers' profiles and applications from v1's raw tables
(read-only), puts them in the order they happened, and sends them to Kafka
at a fixed rate (10, 100 or 1,000 events per second). To test the job the
way real streams misbehave, it also:
- sends 1% of events twice (duplicates);
- delivers 2% of events late, by 1 second to 10 minutes;
- samples consumer lag every 2 seconds while it sends;
- sends a heartbeat to every Kafka partition every 250 ms, stamped with the
  replay clock. Without it, the watermark stops when the last event is
  sent, and the final 2 seconds of events would wait in Flink forever
  (found in the first smoke run: 140 of 241 events processed).

This is a simulated stream of synthetic data: no real customers, no live
traffic. Each event's time is the replay clock (start + position / rate),
so history plays in its true order, compressed.

Commands (inside the tools container; see stream.ps1):
  python simulator.py init                       create Kafka topics and the v2_stream schema
  python simulator.py run --rate 100 --customers 4000
  python simulator.py request --rate 100 --customers 4000 --by airflow
  python simulator.py agent                      run requested replays (used by the Airflow DAG)
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd
from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from stream_common import BOOTSTRAP, P4, SCHEMA, db_connect
from stream_features import PROFILE_FIELDS

SQL_FILE = os.environ.get("SENTINEL_STREAM_SQL", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql", "phase4_stream.sql"))
SIM = P4["simulator"]
EVENTS_TOPIC = P4["kafka"]["events_topic"]
FLINK_GROUP = "sentinel-v2-stream-scorer"


def now_ms() -> int:
    return int(time.time() * 1000)


def init():
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    wanted = [NewTopic(t, num_partitions=int(P4["kafka"]["partitions"]), replication_factor=1)
              for t in (P4["kafka"]["events_topic"], P4["kafka"]["scores_topic"])]
    existing = set(admin.list_topics(timeout=10).topics)
    todo = [t for t in wanted if t.topic not in existing]
    for topic, future in (admin.create_topics(todo).items() if todo else []):
        future.result()
        print(f"created topic {topic}")
    with open(SQL_FILE, encoding="utf-8") as f:
        ddl = f.read()
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(ddl)
    finally:
        conn.close()
    print(f"schema {SCHEMA} ready")


def load_events(n_customers: int, seed: int) -> list:
    """Chosen customers' full histories, in the order they happened."""
    profile_sql = ", ".join(f"{f}::text AS {f}" for f in PROFILE_FIELDS)
    conn = db_connect()
    try:
        customers = pd.read_sql(
            f"SELECT customer_id, registration_date, {profile_sql} FROM raw.customers ORDER BY customer_id", conn)
        if n_customers < len(customers):
            customers = customers.sample(n=n_customers, random_state=seed).sort_values("customer_id")
        ids = customers["customer_id"].tolist()
        apps = pd.read_sql(
            "SELECT transaction_id, customer_id, transaction_date, merchant_category, "
            "order_value_gbp::text AS order_value_gbp, n_instalments, "
            "affordability_ratio::text AS affordability_ratio, application_approved "
            "FROM raw.transactions WHERE customer_id = ANY(%s)", conn, params=(ids,))
    finally:
        conn.close()

    events = []
    for row in customers.itertuples(index=False):
        e = {"type": "registration", "event_id": f"reg-{row.customer_id}", "customer_id": row.customer_id,
             "hist_date": str(row.registration_date)}
        e.update({f: getattr(row, f) for f in PROFILE_FIELDS})
        events.append(((row.registration_date, 0, row.customer_id), e))
    for row in apps.itertuples(index=False):
        e = {"type": "application", "event_id": f"app-{row.transaction_id}", "customer_id": row.customer_id,
             "transaction_id": row.transaction_id, "hist_date": str(row.transaction_date),
             "merchant_category": row.merchant_category, "order_value_gbp": row.order_value_gbp,
             "n_instalments": int(row.n_instalments), "affordability_ratio": row.affordability_ratio,
             "application_approved": int(row.application_approved)}
        events.append(((row.transaction_date, 1, row.transaction_id), e))
    events.sort(key=lambda item: item[0])
    return [e for _, e in events]


def schedule(n_events: int, rate: float, start_ms: int, rng: random.Random, late_max_s: float) -> list:
    """(deliver_ms, copy, index) for every send, including duplicates and late ones."""
    sends, n_late, n_dup = [], 0, 0
    for i in range(n_events):
        event_ms = start_ms + round(i * 1000 / rate)
        deliver = event_ms
        if rng.random() < float(SIM["late_rate"]):
            deliver += round(rng.uniform(float(SIM["late_min_s"]), late_max_s) * 1000)
            n_late += 1
        sends.append((deliver, 0, i))
        if rng.random() < float(SIM["duplicate_rate"]):
            sends.append((deliver + round(rng.uniform(0, float(SIM["duplicate_max_delay_s"])) * 1000), 1, i))
            n_dup += 1
    sends.sort()
    return sends, n_late, n_dup


def topic_end_offsets(consumer) -> int:
    meta = consumer.list_topics(EVENTS_TOPIC, timeout=10).topics[EVENTS_TOPIC]
    total = 0
    for p in meta.partitions:
        _, high = consumer.get_watermark_offsets(TopicPartition(EVENTS_TOPIC, p), timeout=10)
        total += high
    return total


def flink_committed(consumer) -> int | None:
    meta = consumer.list_topics(EVENTS_TOPIC, timeout=10).topics[EVENTS_TOPIC]
    group = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": FLINK_GROUP, "enable.auto.commit": False})
    try:
        parts = group.committed([TopicPartition(EVENTS_TOPIC, p) for p in meta.partitions], timeout=10)
    finally:
        group.close()
    offsets = [tp.offset for tp in parts if tp.offset >= 0]
    return sum(offsets) if offsets else None


def heartbeat(producer, stop: threading.Event, run_id: str, cursor: dict):
    """Keep event time moving on every partition, even when no events are due.
    Never ahead of the next unsent event (cursor), so a sender running behind
    schedule cannot make on-time events look late."""
    partitions = int(P4["kafka"]["partitions"])
    while not stop.wait(0.25):
        t = min(now_ms(), cursor["next_ms"])
        for p in range(partitions):
            producer.produce(EVENTS_TOPIC, partition=p, timestamp=t,
                             value=json.dumps({"type": "heartbeat", "run_id": run_id, "event_time_ms": t}))
        producer.poll(0)


def sample_lag(run_id: str, base_offset: int, base_committed: int, stop: threading.Event):
    """Every 2 s: sent so far, Flink's committed read position (moves at each
    checkpoint), and outputs stored by the sink, all relative to this run."""
    probe = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "sentinel-v2-lag-probe"})
    conn = db_connect()
    try:
        while not stop.wait(2.0):
            produced = topic_end_offsets(probe) - base_offset
            committed = flink_committed(probe)
            with conn, conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {SCHEMA}.processed_events WHERE run_id = %s", (run_id,))
                stored = cur.fetchone()[0]
                cur.execute(f"INSERT INTO {SCHEMA}.lag_samples VALUES (%s, %s, %s, %s, %s) "
                            "ON CONFLICT DO NOTHING",
                            (run_id, now_ms(), produced,
                             None if committed is None else committed - base_committed, stored))
    finally:
        probe.close()
        conn.close()


def run(run_id: str, rate: float, n_customers: int, seed: int, requested_by: str = "manual",
        late_max_s: float | None = None, wait_s: int = 300):
    late_max_s = float(SIM["late_max_s"]) if late_max_s is None else late_max_s
    conn = db_connect()
    with conn, conn.cursor() as cur:
        cur.execute(f"""INSERT INTO {SCHEMA}.replay_runs
            (run_id, status, requested_by, rate_eps, n_customers, seed, duplicate_rate, late_rate,
             late_max_s, dataset_version, model_version)
            VALUES (%s, 'running', %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET status = 'running'""",
                    (run_id, requested_by, int(rate), n_customers, seed, SIM["duplicate_rate"], SIM["late_rate"],
                     int(late_max_s), P4["dataset_version"], P4["model_version"]))
    try:
        events = load_events(n_customers, seed)
        rng = random.Random(seed)
        start = now_ms() + 3000
        sends, n_late, n_dup = schedule(len(events), rate, start, rng, late_max_s)
        print(f"{run_id}: {len(events):,} events for {n_customers:,} customers at {rate:g}/s; "
              f"{n_dup} duplicates, {n_late} late; stream ends about "
              f"{(sends[-1][0] - start) / 1000:.0f} s after start")

        producer = Producer({"bootstrap.servers": BOOTSTRAP, "enable.idempotence": True, "acks": "all",
                             "linger.ms": 5})
        probe = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "sentinel-v2-lag-probe"})
        base_offset, base_committed = topic_end_offsets(probe), flink_committed(probe) or 0
        probe.close()
        if base_committed < base_offset:
            # Flink's read position must be at the end of the topic before a
            # new run, otherwise the lag figures would include old runs.
            print(f"note: Flink committed offset {base_committed} is behind topic end {base_offset}")
        with conn, conn.cursor() as cur:
            cur.execute(f"UPDATE {SCHEMA}.replay_runs SET started_at = now(), n_events = %s WHERE run_id = %s",
                        (len(events), run_id))

        stop = threading.Event()
        sampler = threading.Thread(target=sample_lag, args=(run_id, base_offset, base_committed, stop), daemon=True)
        sampler.start()
        beating = threading.Event()
        cursor = {"next_ms": start}
        beater = threading.Thread(target=heartbeat, args=(producer, beating, run_id, cursor), daemon=True)
        beater.start()
        for deliver_ms, copy, i in sends:
            cursor["next_ms"] = deliver_ms
            wait = deliver_ms - now_ms()
            if wait > 0:
                time.sleep(wait / 1000)
            e = events[i]
            msg = dict(e, run_id=run_id, order=i, copy=copy, event_time_ms=start + round(i * 1000 / rate),
                       sent_ms=now_ms())
            producer.produce(EVENTS_TOPIC, key=e["customer_id"], value=json.dumps(msg),
                             timestamp=msg["event_time_ms"])
            producer.poll(0)
        cursor["next_ms"] = float("inf")
        producer.flush(30)
        with conn, conn.cursor() as cur:
            cur.execute(f"""UPDATE {SCHEMA}.replay_runs SET stream_end_at = now(), n_messages_sent = %s,
                            n_duplicates_sent = %s, n_late_sent = %s WHERE run_id = %s""",
                        (len(sends), n_dup, n_late, run_id))

        # Wait until every distinct event has been processed and stored.
        deadline = time.time() + wait_s
        stored = 0
        while time.time() < deadline:
            with conn, conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {SCHEMA}.processed_events WHERE run_id = %s", (run_id,))
                stored = cur.fetchone()[0]
            if stored >= len(events):
                break
            time.sleep(2)
        time.sleep(4)            # one more lag sample after completion
        stop.set()
        beating.set()
        sampler.join(10)
        beater.join(5)
        producer.flush(10)
        note = None if stored >= len(events) else f"incomplete after {wait_s}s: {stored}/{len(events)} stored"
        with conn, conn.cursor() as cur:
            cur.execute(f"UPDATE {SCHEMA}.replay_runs SET status = 'finished', finished_at = now(), note = %s "
                        "WHERE run_id = %s", (note, run_id))
        print(f"{run_id}: finished, {stored:,}/{len(events):,} events stored" + (f" ({note})" if note else ""))
    except Exception as error:
        with conn, conn.cursor() as cur:
            cur.execute(f"UPDATE {SCHEMA}.replay_runs SET status = 'failed', finished_at = now(), note = %s "
                        "WHERE run_id = %s", (f"{type(error).__name__}: {error}"[:500], run_id))
        raise
    finally:
        conn.close()


def request(run_id: str, rate: float, n_customers: int, seed: int, requested_by: str):
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(f"""INSERT INTO {SCHEMA}.replay_runs
                (run_id, status, requested_by, rate_eps, n_customers, seed, duplicate_rate, late_rate,
                 late_max_s, dataset_version, model_version)
                VALUES (%s, 'requested', %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (run_id, requested_by, int(rate), n_customers, seed, SIM["duplicate_rate"],
                         SIM["late_rate"], int(SIM["late_max_s"]), P4["dataset_version"], P4["model_version"]))
    finally:
        conn.close()
    print(f"requested {run_id}")


def agent(poll_s: float = 5.0):
    """Runs replays that someone (the Airflow DAG) asked for, one at a time."""
    print("replay agent waiting for requests")
    while True:
        conn = db_connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(f"""UPDATE {SCHEMA}.replay_runs SET status = 'running'
                    WHERE run_id = (SELECT run_id FROM {SCHEMA}.replay_runs WHERE status = 'requested'
                                    ORDER BY requested_at LIMIT 1 FOR UPDATE SKIP LOCKED)
                    RETURNING run_id, rate_eps, n_customers, seed, requested_by""")
                claimed = cur.fetchone()
        finally:
            conn.close()
        if claimed:
            run_id, rate, n_customers, seed, by = claimed
            try:
                run(run_id, rate, n_customers, seed, by)
            except Exception as error:   # recorded on the run row; keep serving requests
                print(f"{run_id} failed: {error}", file=sys.stderr)
        else:
            time.sleep(poll_s)


def default_run_id(rate: float) -> str:
    return f"stream_{int(rate)}eps_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("agent")
    for name in ("run", "request"):
        p = sub.add_parser(name)
        p.add_argument("--rate", type=float, required=True, help="events per second")
        p.add_argument("--customers", type=int, required=True, help="customers whose full history is replayed")
        p.add_argument("--seed", type=int, default=int(SIM["seed"]))
        p.add_argument("--run-id")
        p.add_argument("--by", default="manual")
        p.add_argument("--late-max-s", type=float, help="override the configured 600 s (smoke tests only)")
    args = parser.parse_args()
    if args.command == "init":
        init()
    elif args.command == "agent":
        agent()
    elif args.command == "run":
        run(args.run_id or default_run_id(args.rate), args.rate, args.customers, args.seed, args.by, args.late_max_s)
    else:
        request(args.run_id or default_run_id(args.rate), args.rate, args.customers, args.seed, args.by)


if __name__ == "__main__":
    main()

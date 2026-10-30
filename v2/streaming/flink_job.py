"""
Sentinel Credit v2, Phase 4: the PyFlink streaming job (V2-03).

Plain English: read every event from Kafka, keep each customer's features
up to date, and score each application with the unchanged v1.1 champion as
soon as it arrives. Results go back to Kafka for the sink to store. Shadow mode: nothing here answers an applicant.

Two steps (operators):
1. Features, per customer (keyed state). Events are grouped by
   (run, customer); each customer has one CustomerState
   (stream_features.py), saved in every checkpoint.
   - Order: every event is applied the moment it arrives. Kafka keeps one
     customer's events in order (one key, one partition), so on-time
     events are already in the order they happened.
   - Event time and watermarks: each event carries the time it happened
     (the Kafka record timestamp). The watermark says "no events older than
     this are still expected" (newest time seen minus 2 seconds).
   - Late events: behind the watermark, or older than an event already
     applied for this customer. They are applied too, since the final state
     does not depend on order, and flagged "late": their own point-in-time
     score may include events that happened after them.
   - Why no buffering: the first design held each on-time event until the
     watermark passed it (one event-time timer per event). Measured on this
     laptop it topped out at about 340 events/s and added a 2.4 s wait; this
     design removes both (V2-03).
   - De-duplication: an event id already accepted for this customer is
     dropped and reported, not applied twice.
2. Scoring, in small batches. One champion call costs about the same for
   1 row as for hundreds (measured: 9 ms for 1 row, 15 ms for 500, with
   XGBoost on one thread), so applications are collected for up to 100 ms
   or 500 rows and scored together. The waiting rows are kept in Flink
   state, so a crash cannot lose them.

Checkpoints (every 10 s) save all state and the Kafka read positions
together; after a crash the job resumes from the last checkpoint. Output is
at-least-once; the sink's writes are idempotent, so a repeated output
changes nothing.

Run inside the Flink cluster:
  flink run -d -py /opt/sentinel/streaming/flink_job.py
"""

import json
import os
import time
import zlib

import yaml
from pyflink.common import Duration, Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import (
    CheckpointingMode,
    ExternalizedCheckpointCleanup,
    KeyedProcessFunction,
    RuntimeContext,
    StreamExecutionEnvironment,
)
from pyflink.datastream.connectors.base import DeliveryGuarantee
from pyflink.datastream.connectors.kafka import (
    KafkaOffsetResetStrategy,
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)
from pyflink.datastream.state import ListStateDescriptor, ValueStateDescriptor

SENTINEL = os.environ.get("SENTINEL_HOME", "/opt/sentinel")
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")

with open(os.path.join(SENTINEL, "config_v2.yaml"), encoding="utf-8") as f:
    P4 = yaml.safe_load(f)["phase4"]

# Keys for the scoring step. Flink places keys on workers by hash: 4 keys
# landed on only 2 of 8 workers (those two became the bottleneck). 8 keys
# with the cheaper array input below (V2-03).
SCORING_BUCKETS = 8
BATCH_MAX_ROWS = 500
BATCH_MAX_WAIT_MS = 100


def now_ms() -> int:
    return int(time.time() * 1000)


class CustomerFeatures(KeyedProcessFunction):
    """Step 1: per-customer state, lateness by event time, de-duplication."""

    def open(self, runtime_context: RuntimeContext):
        import stream_features as sf
        self.sf = sf
        self.state = runtime_context.get_state(
            ValueStateDescriptor("customer_state", Types.PICKLED_BYTE_ARRAY()))

    def process_element(self, event, ctx: "KeyedProcessFunction.Context"):
        st = self.state.value() or self.sf.CustomerState()
        if event["event_id"] in st.seen:
            yield {"kind": "duplicate", "run_id": event["run_id"], "event_id": event["event_id"],
                   "customer_id": event["customer_id"], "copy": event.get("copy", 0),
                   "sent_ms": event["sent_ms"], "processed_ms": now_ms()}
            return
        st.seen.add(event["event_id"])
        event_time = int(event["event_time_ms"])
        late = event_time <= ctx.timer_service().current_watermark() or event_time < st.last_event_ms
        st.last_event_ms = max(st.last_event_ms, event_time)
        results = self.sf.process_event(st, event, late=late)
        self.state.update(st)
        yield from self._records(results)

    def _records(self, results):
        sf = self.sf
        for r in results:
            e = r["event"]
            out = {"kind": r["kind"], "run_id": e["run_id"], "event_id": e["event_id"],
                   "customer_id": e["customer_id"], "event_time_ms": e["event_time_ms"],
                   "hist_date": e["hist_date"], "sent_ms": e["sent_ms"], "late": r["late"],
                   "seq": r["seq"], "state": sf.to_json_values(r["state"]), "features_ms": now_ms()}
            if r["kind"] == "application":
                out.update({"transaction_id": e["transaction_id"],
                            "hist_approved": int(e["application_approved"]),
                            "delayed_profile": r["delayed_profile"],
                            "features": sf.to_json_values(r["features"])})
            yield out


class BatchScore(KeyedProcessFunction):
    """Step 2: score applications with the champion in micro-batches."""

    def open(self, runtime_context: RuntimeContext):
        import warnings

        import joblib
        import numpy as np
        import stream_features as sf

        # The champion is given a plain array in its feature order (checked
        # below): about 3 times faster per call than a DataFrame, identical
        # probabilities (measured, V2-03). That triggers sklearn's "no feature
        # names" notice, silenced here.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        self.sf, self.np = sf, np
        self.pending = runtime_context.get_list_state(
            ListStateDescriptor("pending_scores", Types.PICKLED_BYTE_ARRAY()))
        # A running count, so adding a row never re-reads the waiting rows.
        # (The first version re-read the whole list on every application:
        # quadratic in batch size, and the job fell behind at 100+ events/s.)
        self.count = runtime_context.get_state(ValueStateDescriptor("pending_count", Types.INT()))
        self.timer_at = runtime_context.get_state(ValueStateDescriptor("flush_timer", Types.LONG()))
        model_dir = os.path.join(SENTINEL, "model")
        self.model = joblib.load(os.path.join(model_dir, "champion_calibrated.joblib"))
        # One thread per call: with many tiny calls, XGBoost's thread start-up
        # cost dominates (measured 32 ms per row with all threads, 9 ms with
        # one). Probabilities are identical; only the speed changes.
        for calibrated in self.model.calibrated_classifiers_:
            calibrated.estimator.set_params(n_jobs=1)
        with open(os.path.join(model_dir, "feature_list.json"), encoding="utf-8") as f:
            if json.load(f)["features"] != sf.FEATURES:
                raise RuntimeError("Champion feature list differs from the stream's features")
        with open(os.path.join(SENTINEL, "config.yaml"), encoding="utf-8") as f:
            v1 = yaml.safe_load(f)
        self.cutoff = float(v1["model"]["business_cutoff"])
        self.model_version = v1["model_governance"]["model_version"]

    def process_element(self, record, ctx: "KeyedProcessFunction.Context"):
        if record["kind"] != "application":
            record["processed_ms"] = now_ms()
            yield json.dumps(record)
            return
        self.pending.add(record)
        n = (self.count.value() or 0) + 1
        self.count.update(n)
        if n >= BATCH_MAX_ROWS:
            yield from self._flush(list(self.pending.get()), ctx)
        elif self.timer_at.value() is None:
            at = ctx.timer_service().current_processing_time() + BATCH_MAX_WAIT_MS
            ctx.timer_service().register_processing_time_timer(at)
            self.timer_at.update(at)

    def on_timer(self, timestamp: int, ctx: "KeyedProcessFunction.OnTimerContext"):
        self.timer_at.clear()
        rows = list(self.pending.get())
        if rows:
            yield from self._flush(rows, ctx)

    def _flush(self, rows, ctx):
        sf = self.sf
        matrix = self.np.array([sf.to_float_row(r["features"]) for r in rows], dtype=float)
        probabilities = self.model.predict_proba(matrix)[:, 1]
        self.pending.clear()
        self.count.clear()
        if self.timer_at.value() is not None:
            ctx.timer_service().delete_processing_time_timer(self.timer_at.value())
            self.timer_at.clear()
        done = now_ms()
        for r, p in zip(rows, probabilities, strict=False):
            p = float(p)
            r.update({"pd": p, "decision": "flag_high_risk" if p >= self.cutoff else "pass",  # labels as serving/app.py
                      "model_version": self.model_version, "batch_rows": len(rows), "processed_ms": done})
            yield json.dumps(r)


def scoring_bucket(record) -> int:
    return zlib.crc32(record["customer_id"].encode()) % SCORING_BUCKETS


def main():
    # Latency settings (bundle time, buffer timeout) are in flink-conf/config.yaml.
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(int(P4["flink"]["parallelism"]))
    env.enable_checkpointing(int(P4["flink"]["checkpoint_interval_ms"]), CheckpointingMode.EXACTLY_ONCE)
    env.get_checkpoint_config().set_externalized_checkpoint_cleanup(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION)

    source = (KafkaSource.builder()
              .set_bootstrap_servers(BOOTSTRAP)
              .set_topics(P4["kafka"]["events_topic"])
              .set_group_id("sentinel-v2-stream-scorer")
              .set_starting_offsets(KafkaOffsetsInitializer.committed_offsets(KafkaOffsetResetStrategy.EARLIEST))
              .set_value_only_deserializer(SimpleStringSchema())
              .build())
    # Event time = the Kafka record timestamp, which the simulator sets to
    # the moment the event happened. No idleness timeout: the simulator's
    # heartbeats keep every partition active, and under backpressure the
    # timeout let the watermark run ahead of unread events (stream_1000eps_r1).
    watermarks = WatermarkStrategy.for_bounded_out_of_orderness(
        Duration.of_millis(int(P4["flink"]["watermark_out_of_orderness_ms"])))
    sink = (KafkaSink.builder()
            .set_bootstrap_servers(BOOTSTRAP)
            .set_record_serializer(KafkaRecordSerializationSchema.builder()
                                   .set_topic(P4["kafka"]["scores_topic"])
                                   .set_value_serialization_schema(SimpleStringSchema())
                                   .build())
            .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
            .build())

    (env.from_source(source, watermarks, "sentinel-events")
        .map(json.loads, output_type=Types.PICKLED_BYTE_ARRAY()).name("parse")
        .filter(lambda e: e["type"] != "heartbeat").name("drop-heartbeats")   # they only move the watermark
        .key_by(lambda e: f"{e['run_id']}|{e['customer_id']}", key_type=Types.STRING())
        .process(CustomerFeatures(), output_type=Types.PICKLED_BYTE_ARRAY()).name("customer-features")
        .key_by(scoring_bucket, key_type=Types.INT())
        .process(BatchScore(), output_type=Types.STRING()).name("champion-batch-score")
        .sink_to(sink).name("scores-to-kafka"))
    env.execute("sentinel_v2_stream_scorer")


if __name__ == "__main__":
    main()

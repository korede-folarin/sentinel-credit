# Sentinel Credit v2 Phase 4: Flink 1.20 cluster image with PyFlink (V2-03).
# Plain English: the official Flink 1.20 runtime, plus Python 3.11 (the same
# Python as the project) and the exact scikit-learn/XGBoost versions the
# champion was saved with, plus the Kafka connector.
FROM flink:1.20.0-java17 AS flink

FROM python:3.11-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless curl procps \
 && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    FLINK_HOME=/opt/flink \
    PATH=/opt/flink/bin:$PATH \
    PYTHONPATH=/opt/sentinel/streaming \
    PYTHONUNBUFFERED=1 \
    SENTINEL_HOME=/opt/sentinel

COPY --from=flink /opt/flink /opt/flink
# PyFlink runtime jar, and the Kafka connector from Maven Central.
RUN cp /opt/flink/opt/flink-python-1.20.0.jar /opt/flink/lib/ \
 && curl -fsSL -o /opt/flink/lib/flink-sql-connector-kafka-3.3.0-1.20.jar \
    https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.3.0-1.20/flink-sql-connector-kafka-3.3.0-1.20.jar

COPY requirements-flink.txt /tmp/requirements-flink.txt
RUN pip install --no-cache-dir -r /tmp/requirements-flink.txt
COPY flink-conf/config.yaml /opt/flink/conf/config.yaml

RUN useradd --uid 9999 --create-home flink \
 && mkdir -p /checkpoints /opt/sentinel \
 && chown -R flink:flink /opt/flink /checkpoints /opt/sentinel
USER flink
WORKDIR /opt/flink

# Sentinel Credit v2 Phase 4: simulator and sink image (V2-03).
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/sentinel/streaming SENTINEL_HOME=/opt/sentinel \
    SENTINEL_STREAM_SQL=/opt/sentinel/sql/phase4_stream.sql
COPY requirements-tools.txt /tmp/requirements-tools.txt
RUN pip install --no-cache-dir -r /tmp/requirements-tools.txt \
 && useradd --uid 10001 --create-home sentinel \
 && mkdir -p /audit && chown sentinel /audit
USER sentinel
WORKDIR /opt/sentinel/streaming

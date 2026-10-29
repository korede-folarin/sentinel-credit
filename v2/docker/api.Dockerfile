# Sentinel Credit scoring API image (v2 V2-01).
# Plain English: packages the existing v1 FastAPI service, unchanged, with the
# committed copy of the v1.1 champion, so it runs anywhere Docker runs.
# The model folder is chosen with SENTINEL_MODEL_DIR (default behaviour of
# serving/app.py is unchanged when it is not set). No credentials are baked
# in: database access for existing customers needs SENTINEL_DB_HOST and
# SENTINEL_DB_PASSWORD supplied at run time.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SENTINEL_MODEL_DIR=/app/model

COPY v2/docker/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt

COPY config.yaml /app/config.yaml
COPY serving /app/serving
COPY dashboard/demo_data/model /app/model

RUN useradd --create-home --uid 10001 sentinel
USER sentinel

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

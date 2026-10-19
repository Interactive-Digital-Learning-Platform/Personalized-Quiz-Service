# syntax=docker/dockerfile:1

# ── Stage 1: dependencies + RAG models ────────────────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

# Dependencies first, on their own layer: they only reinstall when pyproject.toml / uv.lock change.
# torch comes from the CPU-only index pinned in pyproject.toml (no multi-GB CUDA wheels).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Bake the embedding / rerank / BM25 models into the image (see scripts/download_models.py). Only the few modules
# that load them are copied here -- not the whole app -- so an ordinary code change doesn't re-download ~600 MB of
# models; only a change to these files (e.g. a different model name in config.py) does. The runtime stage copies
# the full app on its own.
COPY app/__init__.py ./app/__init__.py
COPY app/core/__init__.py app/core/config.py ./app/core/
COPY app/schemas/__init__.py app/schemas/retrieval.py ./app/schemas/
COPY app/services/__init__.py app/services/embedding_service.py app/services/rerank_service.py \
     app/services/sparse_embedding_service.py ./app/services/
COPY scripts/download_models.py ./scripts/download_models.py
RUN /app/.venv/bin/python scripts/download_models.py


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: the models are already in the image -- never reach out to Hugging Face.
# ENVIRONMENT: production unless overridden (development turns on /docs, CORS *, debug logging and auth bypass).
# RUN_MIGRATIONS: run `alembic upgrade head` on start; set "false" when running several copies and run it once yourself.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ENVIRONMENT=production \
    RUN_MIGRATIONS=true \
    WEB_CONCURRENCY=1

# Non-root user with a real, writable home (the base image's system users get /nonexistent).
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app --no-create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/.cache /app/.cache
COPY --chown=appuser:appgroup alembic.ini ./
COPY --chown=appuser:appgroup alembic ./alembic
COPY --chown=appuser:appgroup app ./app

USER appuser

EXPOSE 8000

# Models load at startup (~20 s), hence the start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["sh", "-c", "if [ \"$RUN_MIGRATIONS\" = \"true\" ]; then alembic upgrade head; fi && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1}"]

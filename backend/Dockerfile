FROM ghcr.io/astral-sh/uv:0.10.12 AS uv

FROM python:3.12-slim AS builder

ARG UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_INDEX_URL=${UV_INDEX_URL}

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev --no-install-project --no-default-groups

FROM python:3.12-slim AS runtime

ARG APT_MIRROR=http://deb.debian.org/debian

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/backend \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UPLOAD_DIR=/app/uploads \
    AUDIT_LOG_FILE=/app/logs/audit.log

WORKDIR /app

RUN apt_mirror="${APT_MIRROR%/}" \
    && sed -i \
        -e "s|http://deb.debian.org/debian|${apt_mirror}|g" \
        -e "s|http://deb.debian.org/debian-security|${apt_mirror}-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app backend /app/backend

RUN mkdir -p /app/uploads /app/logs /app/data/bm25 /app/backend/data/evidence-review /app/backend/evaluation/results \
    && chown -R app:app /app/uploads /app/logs /app/data/bm25 /app/backend/data/evidence-review /app/backend/evaluation/results

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health/live', timeout=3).read()"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]

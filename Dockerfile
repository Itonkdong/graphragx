# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install locked third-party dependencies before copying the source so this
# expensive layer remains cached while application files change.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen


FROM python:3.11-slim-bookworm AS runtime

ENV HOME=/tmp/graphragx-home \
    HF_HOME=/cache/huggingface \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_DIR=/app/wandb

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        fonts-dejavu-core \
        libgomp1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

# These paths are writable by the configurable Compose UID/GID and are either
# bind-mounted or backed by a named cache volume at runtime.
RUN mkdir -p \
        /app/data \
        /app/metadata \
        /app/.experiment-runs \
        /app/wandb \
        /cache/huggingface \
        /tmp/graphragx-home \
    && chmod -R a+rwX \
        /app/data \
        /app/metadata \
        /app/.experiment-runs \
        /app/wandb \
        /cache/huggingface \
        /tmp/graphragx-home

ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["graphragx-experiments", "experiments/example.toml"]

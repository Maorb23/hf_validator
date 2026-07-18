# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:/workspace/.venv/bin:${PATH}"
ENV UV_LINK_MODE=copy
ENV HF_MODEL_WATCH_PREBUILT_ENV=1

COPY pyproject.toml uv.lock README.md /workspace/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra finetune --no-install-project

COPY . /workspace

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra finetune

RUN uv run --no-sync python -m hf_model_watch.runtime.capabilities --build-check

CMD ["bash", "scripts/run_model_worker_job.sh"]

# Overlay only the optimizer package; preserve the approved serving runtime.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY --link llm_optimizer /opt/infiro-worker/llm_optimizer
ENV PYTHONPATH=/opt/infiro-worker:${PYTHONPATH}

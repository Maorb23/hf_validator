# Build only public inputs. Runtime credentials must never be build arguments.
FROM node:24-bookworm-slim AS frontend
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/astro.config.mjs ./
COPY web/src ./src
COPY web/public ./public
ARG SITE_URL
RUN npm run build

FROM python:3.12-slim-bookworm
ENV INFIRO_ENV=production PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/infiro/src:/opt/infiro/web \
    INFIRO_PRIVATE_ROOT=/data/private INFIRO_DATABASE=/data/private/platform.sqlite3 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
WORKDIR /opt/infiro
COPY web/requirements-django.txt web/requirements-platform.txt web/requirements-railway.txt ./web/
COPY vendor/llm-optimizer ./vendor/llm-optimizer
RUN sha256sum -c vendor/llm-optimizer/SHA256SUMS \
    && pip install --no-cache-dir -r web/requirements-railway.txt \
    && pip check
COPY src ./src
COPY config ./config
COPY web/manage.py ./web/manage.py
COPY web/infiro_site ./web/infiro_site
COPY web/infiro_platform ./web/infiro_platform
COPY deploy ./deploy
COPY --from=frontend /build/web/dist ./web/dist
# Railway mounts /data before this runs. Never migrate during image build/pre-deploy.
CMD ["python", "deploy/web_start.py"]

#!/usr/bin/env bash
set -euo pipefail

emit_trace_event() {
  local payload="$1"
  echo "TRACE_EVENT ${payload}"
}

decode_b64_file() {
  local value="$1"
  local destination="$2"
  printf "%s" "${!value}" | base64 --decode > "${destination}"
}

RUN_ID="${RUN_ID:-worker-$(date -u +%Y%m%d-%H%M%S)}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
HF_TOKEN="${HF_TOKEN:-}"
CONFIG_B64="${CONFIG_B64:-}"
PAYLOAD_JSON_B64="${PAYLOAD_JSON_B64:-}"
AWS_S3_BUCKET="${AWS_S3_BUCKET:-}"
AWS_S3_PREFIX="${AWS_S3_PREFIX:-hf-model-watch}"
AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-hf-model-watch}"
PREBUILT_ENV="${HF_MODEL_WATCH_PREBUILT_ENV:-0}"
DRY_RUN_VALUE="${DRY_RUN:-${HF_MODEL_WATCH_WORKER_DRY_RUN:-0}}"
DEFAULT_WORKSPACE_DIR="/workspace"
if [[ "${DRY_RUN_VALUE}" =~ ^(1|true|yes|on)$ ]]; then
  DEFAULT_WORKSPACE_DIR=".worker_workspace"
fi
WORKSPACE_DIR="${HF_MODEL_WATCH_WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
OUTPUT_DIR="${HF_MODEL_WATCH_WORKER_OUTPUT_DIR:-${WORKSPACE_DIR}/output}"
PAYLOAD_JSON="${HF_MODEL_WATCH_PAYLOAD_JSON:-${WORKSPACE_DIR}/payload.json}"

is_dry_run() {
  [[ "${DRY_RUN_VALUE}" =~ ^(1|true|yes|on)$ ]]
}

is_prebuilt_env() {
  [[ "${PREBUILT_ENV}" =~ ^(1|true|yes|on)$ ]]
}

emit_trace_event '{"event_type":"worker_bootstrap","stage":"bootstrap","status":"started","message":"model worker bootstrap started"}'

if command -v apt-get >/dev/null 2>&1 && ! is_dry_run && ! is_prebuilt_env; then
  apt-get update
  apt-get install -y git curl python3 python3-pip ca-certificates
  emit_trace_event '{"event_type":"system_dependencies","stage":"bootstrap","status":"completed","message":"system dependencies installed"}'
else
  emit_trace_event '{"event_type":"system_dependencies","stage":"bootstrap","status":"skipped","message":"system dependency install skipped"}'
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  emit_trace_event '{"event_type":"python_env_ready","stage":"bootstrap","status":"failed","message":"uv is not available"}'
  exit 1
fi

if ! mkdir -p "${WORKSPACE_DIR}" >/dev/null 2>&1; then
  WORKSPACE_DIR="$(pwd)/.worker_workspace"
  OUTPUT_DIR="${HF_MODEL_WATCH_WORKER_OUTPUT_DIR:-${WORKSPACE_DIR}/output}"
  PAYLOAD_JSON="${HF_MODEL_WATCH_PAYLOAD_JSON:-${WORKSPACE_DIR}/payload.json}"
  mkdir -p "${WORKSPACE_DIR}"
fi
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/cache/hf" "${OUTPUT_DIR}/cache/uv"
export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${OUTPUT_DIR}/cache/uv}"
export HF_TOKEN
export WANDB_API_KEY
export WANDB_PROJECT
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_ENDPOINT_URL_S3
export HF_MODEL_WATCH_OBJECT_STORE_DRY_RUN="${HF_MODEL_WATCH_OBJECT_STORE_DRY_RUN:-true}"
export RUN_ID
export MODEL_ID
export MODEL_REVISION
export OUTPUT_DIR
export PAYLOAD_JSON
export AWS_S3_BUCKET
export AWS_S3_PREFIX

if [[ -n "${CONFIG_B64}" ]]; then
  mkdir -p config
  decode_b64_file CONFIG_B64 "${WORKSPACE_DIR}/config_payload.json"
  emit_trace_event '{"event_type":"config_sync","stage":"bootstrap","status":"completed","message":"CONFIG_B64 decoded"}'
else
  emit_trace_event '{"event_type":"config_sync","stage":"bootstrap","status":"skipped","message":"CONFIG_B64 missing; using repository config files"}'
fi

if [[ -z "${PAYLOAD_JSON_B64}" && -z "${MODEL_ID}" ]] && is_dry_run; then
  MODEL_ID="sshleifer/tiny-gpt2"
  MODEL_REVISION="${MODEL_REVISION:-dryrun}"
  HF_MODEL_WATCH_WORKER_DRY_RUN="true"
  AWS_S3_BUCKET="${AWS_S3_BUCKET:-dry-run-bucket}"
  export MODEL_ID
  export MODEL_REVISION
  export HF_MODEL_WATCH_WORKER_DRY_RUN
  export AWS_S3_BUCKET
fi

if [[ -n "${PAYLOAD_JSON_B64}" ]]; then
  decode_b64_file PAYLOAD_JSON_B64 "${PAYLOAD_JSON}"
elif [[ -n "${MODEL_ID}" ]]; then
  WORKER_DRY_RUN_JSON="false"
  if [[ "${HF_MODEL_WATCH_WORKER_DRY_RUN:-false}" =~ ^(1|true|yes|on)$ ]]; then
    WORKER_DRY_RUN_JSON="true"
  fi
  S3_ENABLED_JSON="false"
  if [[ -n "${AWS_S3_BUCKET}" ]]; then
    S3_ENABLED_JSON="true"
  fi
  S3_DRY_RUN_JSON="true"
  if [[ "${HF_MODEL_WATCH_OBJECT_STORE_DRY_RUN}" =~ ^(0|false|no|off)$ ]]; then
    S3_DRY_RUN_JSON="false"
  fi
  cat > "${PAYLOAD_JSON}" <<JSON
{
  "run_id": "${RUN_ID}",
  "model_id": "${MODEL_ID}",
  "revision": "${MODEL_REVISION:-}",
  "config": {
    "worker": {"dry_run": ${WORKER_DRY_RUN_JSON}},
    "output_dir": "${OUTPUT_DIR}",
    "storage": {
      "local_artifact_dir": "${OUTPUT_DIR}",
      "s3": {
        "enabled": ${S3_ENABLED_JSON},
        "bucket": "${AWS_S3_BUCKET}",
        "prefix": "${AWS_S3_PREFIX}",
        "endpoint_url": "${AWS_ENDPOINT_URL_S3}",
        "dry_run": ${S3_DRY_RUN_JSON}
      }
    }
  },
  "artifact_prefix": "${AWS_S3_PREFIX}",
  "skip_finetune": false,
  "finetune_smoke_only": false
}
JSON
else
  emit_trace_event '{"event_type":"worker_payload","stage":"bootstrap","status":"failed","message":"PAYLOAD_JSON_B64 or MODEL_ID is required"}'
  exit 1
fi

emit_trace_event '{"event_type":"worker_payload","stage":"bootstrap","status":"completed","message":"payload ready"}'

if is_prebuilt_env; then
  emit_trace_event '{"event_type":"python_env_ready","stage":"bootstrap","status":"completed","message":"using immutable prebuilt Python environment"}'
else
  uv sync --locked --no-dev --extra finetune
  emit_trace_event '{"event_type":"python_env_ready","stage":"bootstrap","status":"completed","message":"uv sync complete with finetune dependencies"}'
fi

if ! is_dry_run; then
  uv run --no-sync python -m hf_model_watch.runtime.capabilities --require-gpu
fi

if uv run --no-sync hf-watch worker --payload-json "${PAYLOAD_JSON}"; then
  emit_trace_event '{"event_type":"worker_finished","stage":"worker","status":"completed","message":"worker command completed"}'
else
  emit_trace_event '{"event_type":"worker_finished","stage":"worker","status":"failed","message":"worker command failed"}'
  exit 1
fi

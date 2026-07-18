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
SKIP_SYSTEM_DEPS="${HF_MODEL_WATCH_SKIP_SYSTEM_DEPS:-0}"

DRY_RUN="${DRY_RUN:-${HF_MODEL_WATCH_DRY_RUN:-0}}"
CONFIG_DIR="${HF_MODEL_WATCH_CONFIG_DIR:-${1:-config}}"
NB_PROFILE_NAME="${NB_PROFILE_NAME:-hf-model-watch-controller}"
NB_PROJECT_ID="${NB_PROJECT_ID:-}"
NEBIUS_CONFIG_YAML_B64="${NEBIUS_CONFIG_YAML_B64:-}"
NEBIUS_CONFIG_PATH="${NEBIUS_CONFIG_PATH:-$HOME/.nebius/config.yaml}"
NEBIUS_SERVICE_ACCOUNT_KEY_B64="${NEBIUS_SERVICE_ACCOUNT_KEY_B64:-}"
SA_ID="${SA_ID:-${NEBIUS_SERVICE_ACCOUNT_ID:-}}"
PUBLIC_KEY_ID="${PUBLIC_KEY_ID:-${NEBIUS_PUBLIC_KEY_ID:-}}"
NEBIUS_PRIVATE_KEY_B64="${NEBIUS_PRIVATE_KEY_B64:-}"
NEBIUS_PRIVATE_KEY_PATH="${NEBIUS_PRIVATE_KEY_PATH:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$(pwd)/.uv-cache}"
export UV_CACHE_DIR

is_dry_run() {
  [[ "${DRY_RUN}" =~ ^(1|true|yes|on)$ ]]
}

emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"started","message":"prefect controller bootstrap started"}'

if command -v apt-get >/dev/null 2>&1 && ! is_dry_run && [[ ! "${SKIP_SYSTEM_DEPS}" =~ ^(1|true|yes|on)$ ]]; then
  apt-get update
  apt-get install -y git curl python3 python3-pip ca-certificates
  emit_trace_event '{"event_type":"system_dependencies","stage":"bootstrap","status":"completed","message":"system dependencies installed"}'
else
  emit_trace_event '{"event_type":"system_dependencies","stage":"bootstrap","status":"skipped","message":"system dependency install skipped"}'
fi

if ! command -v uv >/dev/null 2>&1; then
  if is_dry_run; then
    emit_trace_event '{"event_type":"python_env_ready","stage":"bootstrap","status":"failed","message":"uv missing in dry-run environment"}'
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  emit_trace_event '{"event_type":"python_env_ready","stage":"bootstrap","status":"failed","message":"uv is not available"}'
  exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
  emit_trace_event "{\"event_type\":\"controller_config\",\"stage\":\"bootstrap\",\"status\":\"failed\",\"message\":\"config directory not found\",\"metadata\":{\"config_dir\":\"${CONFIG_DIR}\"}}"
  exit 1
fi

if is_dry_run; then
  emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"skipped","message":"dry-run mode; skipping Nebius service-account auth"}'
else
  export PATH="/root/.nebius/bin:$HOME/.nebius/bin:$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"
  if ! command -v nebius >/dev/null 2>&1; then
    emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"started","message":"installing nebius CLI"}'
    curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
    export PATH="/root/.nebius/bin:$HOME/.nebius/bin:$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"
  fi
  if ! command -v nebius >/dev/null 2>&1; then
    emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"nebius CLI unavailable","metadata":{"failure_category":"controller_auth_failed"}}'
    exit 1
  fi

  mkdir -p "$(dirname "${NEBIUS_CONFIG_PATH}")"
  if [[ -n "${NEBIUS_CONFIG_YAML_B64}" ]]; then
    decode_b64_file NEBIUS_CONFIG_YAML_B64 "${NEBIUS_CONFIG_PATH}"
    chmod 600 "${NEBIUS_CONFIG_PATH}" >/dev/null 2>&1 || true
    emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius config injected"}'
  elif [[ -n "${NB_PROJECT_ID}" && -n "${SA_ID}" && -n "${PUBLIC_KEY_ID}" ]]; then
    PRIVATE_KEY_PATH="${NEBIUS_PRIVATE_KEY_PATH}"
    if [[ -z "${PRIVATE_KEY_PATH}" && -n "${NEBIUS_PRIVATE_KEY_B64}" ]]; then
      PRIVATE_KEY_PATH="/tmp/hf-model-watch-nebius-sa.pem"
      decode_b64_file NEBIUS_PRIVATE_KEY_B64 "${PRIVATE_KEY_PATH}"
      chmod 600 "${PRIVATE_KEY_PATH}" >/dev/null 2>&1 || true
    fi
    if [[ -z "${PRIVATE_KEY_PATH}" || ! -f "${PRIVATE_KEY_PATH}" ]]; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"missing service-account private key","metadata":{"failure_category":"controller_auth_failed"}}'
      exit 1
    fi
    if nebius profile create --endpoint api.nebius.cloud --service-account-id "${SA_ID}" --public-key-id "${PUBLIC_KEY_ID}" --private-key-file "${PRIVATE_KEY_PATH}" --profile "${NB_PROFILE_NAME}" --parent-id "${NB_PROJECT_ID}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius service-account keypair profile created"}'
    else
      nebius profile update "${NB_PROFILE_NAME}" --endpoint api.nebius.cloud --parent-id "${NB_PROJECT_ID}" --service-account-id "${SA_ID}" --public-key-id "${PUBLIC_KEY_ID}" --private-key-file "${PRIVATE_KEY_PATH}"
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius service-account keypair profile updated"}'
    fi
  elif [[ -n "${NB_PROJECT_ID}" && -n "${NEBIUS_SERVICE_ACCOUNT_KEY_B64}" ]]; then
    KEY_PATH="/tmp/hf-model-watch-nebius-sa.json"
    decode_b64_file NEBIUS_SERVICE_ACCOUNT_KEY_B64 "${KEY_PATH}"
    chmod 600 "${KEY_PATH}" >/dev/null 2>&1 || true
    if nebius profile create --profile "${NB_PROFILE_NAME}" --endpoint api.nebius.cloud --federation-endpoint auth.nebius.com --parent-id "${NB_PROJECT_ID}" --auth-method service-account-key --service-account-key-path "${KEY_PATH}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius service-account json profile created"}'
    else
      nebius profile update "${NB_PROFILE_NAME}" --parent-id "${NB_PROJECT_ID}" --auth-method service-account-key --service-account-key-path "${KEY_PATH}"
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius service-account json profile updated"}'
    fi
  else
    emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"Nebius service-account credentials missing","metadata":{"failure_category":"controller_auth_failed"}}'
    exit 1
  fi
fi

uv sync
emit_trace_event '{"event_type":"python_env_ready","stage":"bootstrap","status":"completed","message":"uv sync complete"}'

FLOW_ARGS=(hf-watch prefect-flow --config-dir "${CONFIG_DIR}")
if is_dry_run; then
  FLOW_ARGS+=(--dry-run)
fi

emit_trace_event '{"event_type":"controller_run","stage":"orchestration","status":"started","message":"starting hf-watch prefect-flow"}'
uv run "${FLOW_ARGS[@]}"
emit_trace_event '{"event_type":"controller_run","stage":"orchestration","status":"completed","message":"hf-watch prefect-flow completed"}'

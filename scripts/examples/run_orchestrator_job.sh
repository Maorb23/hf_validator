set -euo pipefail

emit_trace_event() {
  local payload="$1"
  echo "TRACE_EVENT ${payload}"
}

REPO_URL="${AGENTIC_CONTROLLER_REPO_URL:-}"
REPO_REF="${AGENTIC_CONTROLLER_REPO_REF:-}"
PROJECT_BUNDLE_B64="${AGENTIC_PROJECT_BUNDLE_B64:-}"
CONFIG_PATH="${AGENTIC_CONTROLLER_CONFIG_PATH:-config.yaml}"
RUN_NAME="${AGENTIC_CONTROLLER_RUN_NAME:-}"
NB_PROFILE_NAME="${NB_PROFILE_NAME:-${SA_PROFILE_NAME:-agentic-controller}}"
NB_PROJECT_ID="${NB_PROJECT_ID:-}"
NEBIUS_SERVICE_ACCOUNT_KEY_B64="${NEBIUS_SERVICE_ACCOUNT_KEY_B64:-}"
NEBIUS_CONFIG_YAML_B64="${NEBIUS_CONFIG_YAML_B64:-}"
NEBIUS_CONFIG_PATH="${NEBIUS_CONFIG_PATH:-/root/.nebius/config.yaml}"
SA_ID="${SA_ID:-${NEBIUS_SERVICE_ACCOUNT_ID:-}}"
PUBLIC_KEY_ID="${PUBLIC_KEY_ID:-${NEBIUS_PUBLIC_KEY_ID:-}}"
PRIVATE_KEY_PATH="${PRIVATE_KEY_PATH:-${NEBIUS_PRIVATE_KEY_PATH:-}}"
NEBIUS_PRIVATE_KEY_B64="${NEBIUS_PRIVATE_KEY_B64:-}"

emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"started","message":"controller bootstrap started"}'

apt-get update
apt-get install -y git curl python3 python3-pip ca-certificates tar

WORKDIR="/workspace/agentic-controller"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"

if [[ -n "${REPO_URL}" ]]; then
  emit_trace_event '{"event_type":"controller_source","stage":"bootstrap","status":"started","message":"using repository clone source"}'
  git clone "${REPO_URL}" "${WORKDIR}"
  cd "${WORKDIR}"

  if [[ -n "${REPO_REF}" ]]; then
    git fetch --all --tags
    git checkout "${REPO_REF}"
  fi

  emit_trace_event '{"event_type":"controller_source","stage":"bootstrap","status":"completed","message":"repository source ready"}'
else
  emit_trace_event '{"event_type":"controller_source","stage":"bootstrap","status":"started","message":"using inline bundled source"}'
  if [[ -z "${PROJECT_BUNDLE_B64}" ]]; then
    emit_trace_event '{"event_type":"controller_source","stage":"bootstrap","status":"failed","message":"AGENTIC_PROJECT_BUNDLE_B64 is required when repo URL is not provided"}'
    exit 1
  fi

  printf "%s" "${PROJECT_BUNDLE_B64}" | base64 --decode > /tmp/agentic_bundle.tar.gz
  tar -xzf /tmp/agentic_bundle.tar.gz -C "${WORKDIR}"
  cd "${WORKDIR}"

  emit_trace_event '{"event_type":"controller_source","stage":"bootstrap","status":"completed","message":"inline bundled source extracted"}'
fi

if [[ ! -f "requirements.txt" ]]; then
  emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"failed","message":"requirements.txt missing in controller workspace"}'
  exit 1
fi

python3 -m pip install --no-cache-dir -r requirements.txt

export PATH="/root/.nebius/bin:$HOME/.nebius/bin:$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"

if ! command -v nebius >/dev/null 2>&1; then
  emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"started","message":"installing nebius CLI"}'
  curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
  export PATH="/root/.nebius/bin:$HOME/.nebius/bin:$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"
fi

if ! command -v nebius >/dev/null 2>&1; then
  emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"failed","message":"nebius CLI not available in controller job"}'
  exit 1
fi

emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"completed","message":"nebius CLI available in controller job"}'

mkdir -p "$(dirname "${NEBIUS_CONFIG_PATH}")"
if [[ -n "${NEBIUS_CONFIG_YAML_B64}" ]]; then
  emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"started","message":"injecting nebius config from NEBIUS_CONFIG_YAML_B64"}'
  if printf "%s" "${NEBIUS_CONFIG_YAML_B64}" | base64 --decode > "${NEBIUS_CONFIG_PATH}"; then
    chmod 600 "${NEBIUS_CONFIG_PATH}" >/dev/null 2>&1 || true
    emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius config injected"}'
  else
    emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"failed to decode NEBIUS_CONFIG_YAML_B64"}'
    exit 1
  fi
fi

if [[ -n "${NB_PROJECT_ID}" ]]; then
  emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"started","message":"configuring nebius profile"}'

  if [[ -z "${PRIVATE_KEY_PATH}" && -n "${NEBIUS_PRIVATE_KEY_B64}" ]]; then
    PRIVATE_KEY_PATH="/tmp/nebius-service-account-private-key.pem"
    if ! printf "%s" "${NEBIUS_PRIVATE_KEY_B64}" | base64 --decode > "${PRIVATE_KEY_PATH}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"failed to decode NEBIUS_PRIVATE_KEY_B64"}'
      exit 1
    fi
    chmod 600 "${PRIVATE_KEY_PATH}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${SA_ID}" && -n "${PUBLIC_KEY_ID}" ]]; then
    if [[ -z "${PRIVATE_KEY_PATH}" || ! -f "${PRIVATE_KEY_PATH}" ]]; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"missing PRIVATE_KEY_PATH/NEBIUS_PRIVATE_KEY_PATH file for service-account profile auth"}'
      exit 1
    fi

    PROFILE_CREATE_ARGS=(
      --endpoint api.nebius.cloud
      --service-account-id "${SA_ID}"
      --public-key-id "${PUBLIC_KEY_ID}"
      --private-key-file "${PRIVATE_KEY_PATH}"
      --profile "${NB_PROFILE_NAME}"
      --parent-id "${NB_PROJECT_ID}"
    )

    if nebius profile create "${PROFILE_CREATE_ARGS[@]}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius profile created (service account keypair auth)"}'
    elif nebius profile update "${NB_PROFILE_NAME}" --endpoint api.nebius.cloud --parent-id "${NB_PROJECT_ID}" --service-account-id "${SA_ID}" --public-key-id "${PUBLIC_KEY_ID}" --private-key-file "${PRIVATE_KEY_PATH}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius profile updated (service account keypair auth)"}'
    else
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"failed to create or update nebius profile with service-account-id/public-key-id/private-key-file"}'
      exit 1
    fi
  elif [[ -n "${NEBIUS_SERVICE_ACCOUNT_KEY_B64}" ]]; then
    PROFILE_CREATE_ARGS=(
      --profile "${NB_PROFILE_NAME}"
      --endpoint api.nebius.cloud
      --federation-endpoint auth.nebius.com
      --parent-id "${NB_PROJECT_ID}"
    )

    KEY_PATH="/tmp/nebius-service-account-key.json"
    if ! printf "%s" "${NEBIUS_SERVICE_ACCOUNT_KEY_B64}" | base64 --decode > "${KEY_PATH}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"failed to decode NEBIUS_SERVICE_ACCOUNT_KEY_B64"}'
      exit 1
    fi

    PROFILE_CREATE_ARGS+=(
      --auth-method service-account-key
      --service-account-key-path "${KEY_PATH}"
    )

    if nebius profile create "${PROFILE_CREATE_ARGS[@]}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius profile created"}'
    elif nebius profile update "${NB_PROFILE_NAME}" --parent-id "${NB_PROJECT_ID}" --auth-method service-account-key --service-account-key-path "${KEY_PATH}"; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"nebius profile updated"}'
    else
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"failed to create or update nebius profile with service account key"}'
      exit 1
    fi
  elif [[ -f "${NEBIUS_CONFIG_PATH}" ]]; then
    if nebius profile list >/dev/null 2>&1; then
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"completed","message":"using existing nebius profile from injected/existing config"}'
    else
      emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"injected nebius config exists but no usable profile was found"}'
      exit 1
    fi
  else
    emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"failed","message":"controller auth unavailable; set service account env vars (SA_ID/PUBLIC_KEY_ID/PRIVATE_KEY_PATH), NEBIUS_SERVICE_ACCOUNT_KEY_B64, or provide NEBIUS_CONFIG_YAML_B64"}'
    exit 1
  fi
else
  emit_trace_event '{"event_type":"controller_auth","stage":"bootstrap","status":"skipped","message":"NB_PROJECT_ID not set; skipping profile setup"}'
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  emit_trace_event "{\"event_type\":\"controller_bootstrap\",\"stage\":\"bootstrap\",\"status\":\"failed\",\"message\":\"config path not found: ${CONFIG_PATH}\"}"
  exit 1
fi

emit_trace_event '{"event_type":"controller_bootstrap","stage":"bootstrap","status":"completed","message":"controller dependencies ready"}'

CMD=(python3 -m agentic.cli --config "${CONFIG_PATH}" run-local)
if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--name "${RUN_NAME}")
fi

emit_trace_event '{"event_type":"controller_run","stage":"orchestrator","status":"started","message":"starting inner run-local orchestration"}'
"${CMD[@]}"
emit_trace_event '{"event_type":"controller_run","stage":"orchestrator","status":"completed","message":"inner run-local orchestration completed"}'

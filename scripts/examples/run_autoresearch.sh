

set -euo pipefail

emit_trace_event() {
  local payload="$1"
  echo "TRACE_EVENT ${payload}"
}

RUN_ID="${AGENTIC_RUN_ID:-manual-$(date -u +%Y%m%d-%H%M%S)}"
AWS_S3_BUCKET="${AWS_S3_BUCKET:-your-aws-bucket-placeholder}"
AWS_S3_PREFIX="${AWS_S3_PREFIX:-autoresearch/models}"
AWS_S3_ENDPOINT_URL="${AWS_ENDPOINT_URL_S3:-${AWS_ENDPOINT_URL:-}}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/workspace/output}"
AGENT_PROGRAM_MD_B64="${AGENT_PROGRAM_MD_B64:-}"
AGENT_PROGRAM_MD_GZ_B64="${AGENT_PROGRAM_MD_GZ_B64:-}"
AGENT_PROGRAM_MD_S3_URI="${AGENT_PROGRAM_MD_S3_URI:-}"
AGENT_SELECTED_MODEL="${AGENT_SELECTED_MODEL:-}"
AGENT_TRAIN_PY_B64="${AGENT_TRAIN_PY_B64:-}"
AGENT_TRAIN_PY_GZ_B64="${AGENT_TRAIN_PY_GZ_B64:-}"
AGENT_TRAIN_PY_S3_URI="${AGENT_TRAIN_PY_S3_URI:-}"
AGENT_TRAIN_PY_SHA256="${AGENT_TRAIN_PY_SHA256:-}"

sanitize_optional_env() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo ""
    return
  fi
  if [[ "${value}" == *'${'* ]] || [[ "${value}" == *'%'*'%'* ]]; then
    echo ""
    return
  fi
  echo "${value}"
}

AGENT_PROGRAM_MD_B64="$(sanitize_optional_env "${AGENT_PROGRAM_MD_B64}")"
AGENT_PROGRAM_MD_GZ_B64="$(sanitize_optional_env "${AGENT_PROGRAM_MD_GZ_B64}")"
AGENT_PROGRAM_MD_S3_URI="$(sanitize_optional_env "${AGENT_PROGRAM_MD_S3_URI}")"
AGENT_TRAIN_PY_B64="$(sanitize_optional_env "${AGENT_TRAIN_PY_B64}")"
AGENT_TRAIN_PY_GZ_B64="$(sanitize_optional_env "${AGENT_TRAIN_PY_GZ_B64}")"
AGENT_TRAIN_PY_S3_URI="$(sanitize_optional_env "${AGENT_TRAIN_PY_S3_URI}")"
AGENT_TRAIN_PY_SHA256="$(sanitize_optional_env "${AGENT_TRAIN_PY_SHA256}")"
AWS_S3_ENDPOINT_URL="$(sanitize_optional_env "${AWS_S3_ENDPOINT_URL}")"

emit_trace_event '{"event_type":"job_bootstrap","stage":"bootstrap","status":"started","message":"bootstrap started"}'

apt-get update
apt-get install -y git curl python3 python3-pip
python3 -m pip install --no-cache-dir awscli wandb boto3

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

emit_trace_event '{"event_type":"environment_ready","stage":"bootstrap","status":"completed","message":"system dependencies installed"}'

git clone https://github.com/karpathy/autoresearch /workspace/repo
cd /workspace/repo
emit_trace_event '{"event_type":"repo_cloned","stage":"bootstrap","status":"completed","message":"karpathy/autoresearch cloned"}'

if [[ -n "${AGENT_PROGRAM_MD_S3_URI}" ]]; then
  AWS_DOWNLOAD_CMD=(aws s3 cp "${AGENT_PROGRAM_MD_S3_URI}" /workspace/repo/program.md)
  if [[ -n "${AWS_S3_ENDPOINT_URL}" ]]; then
    AWS_DOWNLOAD_CMD=(aws --endpoint-url "${AWS_S3_ENDPOINT_URL}" s3 cp "${AGENT_PROGRAM_MD_S3_URI}" /workspace/repo/program.md)
  fi

  if "${AWS_DOWNLOAD_CMD[@]}"; then
    emit_trace_event '{"event_type":"program_sync","stage":"bootstrap","status":"completed","message":"program.md downloaded from AGENT_PROGRAM_MD_S3_URI"}'
  else
    emit_trace_event '{"event_type":"program_sync","stage":"bootstrap","status":"failed","message":"failed to download program.md from AGENT_PROGRAM_MD_S3_URI"}'
    exit 1
  fi
elif [[ -n "${AGENT_PROGRAM_MD_GZ_B64}" ]]; then
  if printf "%s" "${AGENT_PROGRAM_MD_GZ_B64}" | base64 --decode | gzip -dc > /workspace/repo/program.md; then
    emit_trace_event '{"event_type":"program_sync","stage":"bootstrap","status":"completed","message":"program.md synced from controller program.MD (gzip payload)"}'
  else
    emit_trace_event '{"event_type":"program_sync","stage":"bootstrap","status":"failed","message":"failed to decode AGENT_PROGRAM_MD_GZ_B64"}'
    exit 1
  fi
elif [[ -n "${AGENT_PROGRAM_MD_B64}" ]]; then
  if printf "%s" "${AGENT_PROGRAM_MD_B64}" | base64 --decode > /workspace/repo/program.md; then
    emit_trace_event '{"event_type":"program_sync","stage":"bootstrap","status":"completed","message":"program.md synced from controller program.MD"}'
  else
    emit_trace_event '{"event_type":"program_sync","stage":"bootstrap","status":"failed","message":"failed to decode AGENT_PROGRAM_MD_B64"}'
    exit 1
  fi
else
  emit_trace_event '{"event_type":"program_sync","stage":"bootstrap","status":"skipped","message":"AGENT_PROGRAM_MD_B64 missing; using upstream program.md"}'
fi

if [[ -n "${AGENT_TRAIN_PY_S3_URI}" ]]; then
  AWS_DOWNLOAD_CMD=(aws s3 cp "${AGENT_TRAIN_PY_S3_URI}" /workspace/repo/train.py)
  if [[ -n "${AWS_S3_ENDPOINT_URL}" ]]; then
    AWS_DOWNLOAD_CMD=(aws --endpoint-url "${AWS_S3_ENDPOINT_URL}" s3 cp "${AGENT_TRAIN_PY_S3_URI}" /workspace/repo/train.py)
  fi

  if "${AWS_DOWNLOAD_CMD[@]}"; then
    emit_trace_event '{"event_type":"code_sync","stage":"train","status":"completed","message":"train.py downloaded from AGENT_TRAIN_PY_S3_URI"}'
  else
    emit_trace_event '{"event_type":"code_sync","stage":"train","status":"failed","message":"failed to download train.py from AGENT_TRAIN_PY_S3_URI"}'
    exit 1
  fi
elif [[ -n "${AGENT_TRAIN_PY_GZ_B64}" ]]; then
  if printf "%s" "${AGENT_TRAIN_PY_GZ_B64}" | base64 --decode | gzip -dc > /workspace/repo/train.py; then
    emit_trace_event '{"event_type":"code_sync","stage":"train","status":"completed","message":"train.py updated from orchestrator candidate (gzip payload)"}'
  else
    emit_trace_event '{"event_type":"code_sync","stage":"train","status":"failed","message":"failed to decode AGENT_TRAIN_PY_GZ_B64"}'
    exit 1
  fi
elif [[ -n "${AGENT_TRAIN_PY_B64}" ]]; then
  if printf "%s" "${AGENT_TRAIN_PY_B64}" | base64 --decode > /workspace/repo/train.py; then
    emit_trace_event '{"event_type":"code_sync","stage":"train","status":"completed","message":"train.py updated from orchestrator candidate"}'
  else
    emit_trace_event '{"event_type":"code_sync","stage":"train","status":"failed","message":"failed to decode AGENT_TRAIN_PY_B64"}'
    exit 1
  fi
else
  emit_trace_event '{"event_type":"code_sync","stage":"train","status":"skipped","message":"no AGENT_TRAIN_PY_B64 provided; using repository train.py"}'
fi

APPLIED_TRAIN_PY_SHA256="$(sha256sum /workspace/repo/train.py | awk '{print $1}')"
if [[ -n "${AGENT_TRAIN_PY_SHA256}" ]]; then
  if [[ "${AGENT_TRAIN_PY_SHA256}" == "${APPLIED_TRAIN_PY_SHA256}" ]]; then
    emit_trace_event "{\"event_type\":\"code_sync\",\"stage\":\"train\",\"status\":\"completed\",\"message\":\"train.py hash verified\",\"metadata\":{\"expected_sha256\":\"${AGENT_TRAIN_PY_SHA256}\",\"actual_sha256\":\"${APPLIED_TRAIN_PY_SHA256}\"}}"
  else
    emit_trace_event "{\"event_type\":\"code_sync\",\"stage\":\"train\",\"status\":\"failed\",\"message\":\"train.py hash mismatch after sync\",\"metadata\":{\"expected_sha256\":\"${AGENT_TRAIN_PY_SHA256}\",\"actual_sha256\":\"${APPLIED_TRAIN_PY_SHA256}\"}}"
    exit 1
  fi
else
  emit_trace_event "{\"event_type\":\"code_sync\",\"stage\":\"train\",\"status\":\"completed\",\"message\":\"train.py hash captured\",\"metadata\":{\"actual_sha256\":\"${APPLIED_TRAIN_PY_SHA256}\"}}"
fi

if [[ -n "${AGENT_SELECTED_MODEL}" ]]; then
  python3 <<'PY'
import json
import os

print("TRACE_EVENT " + json.dumps({
    "event_type": "routing_config",
    "stage": "bootstrap",
    "status": "completed",
    "message": "selected routing model received from orchestrator",
    "metadata": {"selected_model": os.environ["AGENT_SELECTED_MODEL"]},
}))
PY
fi

uv sync
emit_trace_event '{"event_type":"python_env_ready","stage":"bootstrap","status":"completed","message":"uv sync complete"}'

emit_trace_event '{"event_type":"prepare","stage":"prepare","status":"started","message":"prepare.py started"}'
CACHE_DIR="${HOME}/.cache/autoresearch"
if [[ -f "${CACHE_DIR}/tokenizer/tokenizer.pkl" ]] && find "${CACHE_DIR}/data" -maxdepth 1 -name 'shard_*.parquet' -print -quit 2>/dev/null | grep -q .; then
  emit_trace_event '{"event_type":"prepare","stage":"prepare","status":"skipped","message":"cache already present; skipping prepare.py"}'
else
  if uv run prepare.py --num-shards 1; then
    emit_trace_event '{"event_type":"prepare","stage":"prepare","status":"completed","message":"prepare.py finished"}'
  else
    emit_trace_event '{"event_type":"prepare","stage":"prepare","status":"failed","message":"prepare.py failed"}'
    exit 1
  fi
fi

emit_trace_event '{"event_type":"train","stage":"train","status":"started","message":"single train.py run started"}'
if uv run train.py > /workspace/repo/run.log 2>&1; then
  emit_trace_event '{"event_type":"train","stage":"train","status":"completed","message":"train.py finished"}'
else
  emit_trace_event '{"event_type":"train","stage":"train","status":"failed","message":"train.py failed"}'
  exit 1
fi

emit_trace_event '{"event_type":"eval","stage":"eval","status":"started","message":"post-train eval extraction started"}'
python3 <<'PY'
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

repo_dir = Path("/workspace/repo")
out_dir = Path("/workspace/output")
out_dir.mkdir(parents=True, exist_ok=True)

run_log = repo_dir / "run.log"
text = run_log.read_text(encoding="utf-8", errors="ignore") if run_log.exists() else ""

val_match = re.search(r"^val_bpb:\s*([0-9]*\.?[0-9]+)", text, flags=re.MULTILINE)
peak_match = re.search(r"^peak_vram_mb:\s*([0-9]*\.?[0-9]+)", text, flags=re.MULTILINE)
step_match = re.findall(r"(?:step|iter|iteration)\s*[=:]?\s*(\d+)", text, flags=re.IGNORECASE)
loss_match = re.findall(r"loss\s*[=:]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)

val_bpb = float(val_match.group(1)) if val_match else None
peak_vram_mb = float(peak_match.group(1)) if peak_match else None
last_step = int(step_match[-1]) if step_match else None
last_loss = float(loss_match[-1]) if loss_match else None
quality_score = (1.0 / val_bpb) if isinstance(val_bpb, float) and val_bpb > 0 else None

metrics = {
    "quality_score": quality_score,
    "val_bpb": val_bpb,
    "peak_vram_mb": peak_vram_mb,
    "last_step": last_step,
    "last_loss": last_loss,
}

(out_dir / "eval").mkdir(parents=True, exist_ok=True)
metrics_path = out_dir / "eval" / "latest_metrics.json"
metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

if run_log.exists():
    shutil.copyfile(run_log, out_dir / "run.log")

train_path = repo_dir / "train.py"
if train_path.exists():
    shutil.copyfile(train_path, out_dir / "train.py")

event = {
    "event_type": "eval",
    "stage": "eval",
    "status": "completed",
    "message": "post-train eval extraction completed",
    "metrics": metrics,
    "artifacts": {
        "metrics_file": str(metrics_path),
        "run_log": str(out_dir / "run.log"),
        "train_py": str(out_dir / "train.py"),
    },
}
print("TRACE_EVENT " + json.dumps(event))
PY

if [[ "${AWS_S3_BUCKET}" == *"placeholder"* ]]; then
  emit_trace_event '{"event_type":"artifact_upload","stage":"artifact_upload","status":"skipped","message":"AWS bucket placeholder detected"}'
else
  if [[ ! -d "${MODEL_OUTPUT_DIR}" ]]; then
    emit_trace_event "{\"event_type\":\"artifact_upload\",\"stage\":\"artifact_upload\",\"status\":\"failed\",\"message\":\"missing model directory ${MODEL_OUTPUT_DIR}\"}"
    exit 1
  fi

  if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    emit_trace_event '{"event_type":"artifact_upload","stage":"artifact_upload","status":"skipped","message":"AWS credentials missing; skipping upload"}'
    emit_trace_event '{"event_type":"job_finished","stage":"orchestrator","status":"completed","message":"run_autoresearch.sh completed"}'
    exit 0
  fi

  DEST="s3://${AWS_S3_BUCKET}/${AWS_S3_PREFIX}/${RUN_ID}/"
  emit_trace_event "{\"event_type\":\"artifact_upload\",\"stage\":\"artifact_upload\",\"status\":\"started\",\"message\":\"uploading to ${DEST}\"}"

  AWS_UPLOAD_CMD=(aws s3 cp --recursive "${MODEL_OUTPUT_DIR}" "${DEST}")
  if [[ -n "${AWS_S3_ENDPOINT_URL}" ]]; then
    AWS_UPLOAD_CMD=(aws --endpoint-url "${AWS_S3_ENDPOINT_URL}" s3 cp --recursive "${MODEL_OUTPUT_DIR}" "${DEST}")
  fi

  if "${AWS_UPLOAD_CMD[@]}"; then
    emit_trace_event "{\"event_type\":\"artifact_upload\",\"stage\":\"artifact_upload\",\"status\":\"completed\",\"message\":\"upload completed\",\"artifacts\":{\"destination\":\"${DEST}\"}}"
  else
    emit_trace_event "{\"event_type\":\"artifact_upload\",\"stage\":\"artifact_upload\",\"status\":\"failed\",\"message\":\"upload failed\",\"artifacts\":{\"destination\":\"${DEST}\"}}"
    exit 1
  fi
fi

emit_trace_event '{"event_type":"job_finished","stage":"orchestrator","status":"completed","message":"run_autoresearch.sh completed"}'

set -euo pipefail

AGENT_PROGRAM_MD_B64="${AGENT_PROGRAM_MD_B64:-}"
AGENT_SELECTED_MODEL="${AGENT_SELECTED_MODEL:-}"

apt-get update
apt-get install -y git curl python3 python3-pip

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/karpathy/autoresearch /workspace/repo
cd /workspace/repo

if [[ -n "${AGENT_PROGRAM_MD_B64}" ]]; then
    printf "%s" "${AGENT_PROGRAM_MD_B64}" | base64 --decode > /workspace/repo/program.md
fi

if [[ -n "${AGENT_SELECTED_MODEL}" ]]; then
    echo "Selected routing model: ${AGENT_SELECTED_MODEL}"
fi

uv sync

uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

CACHE_DIR="${HOME}/.cache/autoresearch"
if [[ -f "${CACHE_DIR}/tokenizer/tokenizer.pkl" ]] && find "${CACHE_DIR}/data" -maxdepth 1 -name 'shard_*.parquet' -print -quit 2>/dev/null | grep -q .; then
    echo "Using existing autoresearch cache; skipping prepare.py"
else
    uv run prepare.py --num-shards 1
fi

uv run train.py > /workspace/repo/run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" /workspace/repo/run.log || true
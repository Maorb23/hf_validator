#!/usr/bin/env bash
set -euo pipefail

# Stage only the installed Python package and its lightweight dependencies for
# the NanoClaw container. This intentionally does not copy .env, git metadata,
# artifacts, or the repository working tree.
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/linux/path/to/runtime" >&2
  exit 2
fi

runtime_root="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$runtime_root" in
  /*) ;;
  *) echo "Runtime path must be absolute and Linux-native." >&2; exit 2 ;;
esac

mkdir -p "$runtime_root"
python3 -m venv "$runtime_root/.venv"
"$runtime_root/.venv/bin/python" -m pip install --upgrade pip
"$runtime_root/.venv/bin/python" -m pip install -r "$repo_root/requirements_cpu.txt"
"$runtime_root/.venv/bin/python" -m pip install --no-deps "$repo_root"
mkdir -p "$runtime_root/config"
install -m 0644 "$repo_root/config/recommender.yaml" "$runtime_root/config/recommender.yaml"
install -m 0644 "$repo_root/config/scan.yaml" "$runtime_root/config/scan.yaml"
install -m 0644 "$repo_root/config/scoring.yaml" "$runtime_root/config/scoring.yaml"
chmod -R a+rX "$runtime_root"

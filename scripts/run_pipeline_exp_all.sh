#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

shopt -s nullglob
configs=(input/pipeline/exp/*.yml)

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No pipeline configs found under input/pipeline/exp" >&2
  exit 1
fi

for config in "${configs[@]}"; do
  echo "[pipeline] $config"
  uv run src/cli.py pipeline --config "$config"
done
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

run_pipeline() {
  local config="$1"
  echo "[pipeline] $config"
  python -m src.cli pipeline --config "$config"
}

run_pipeline input/pipeline/logistic_n5000_compare_t_auto_cn.yml
run_pipeline input/pipeline/logistic_n5000_compare_t_auto_rn.yml

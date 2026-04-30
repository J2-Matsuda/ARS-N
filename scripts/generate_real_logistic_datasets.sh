#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "input/generate_data/real_logistic/epsilon_normalized.yml"
  "input/generate_data/real_logistic/real_sim.yml"
  "input/generate_data/real_logistic/rcv1_binary.yml"
  "input/generate_data/real_logistic/gisette.yml"
)

for config in "${CONFIGS[@]}"; do
  echo "[generate_real_logistic] ${config}"
  python -m src.cli generate --config "${config}"
done

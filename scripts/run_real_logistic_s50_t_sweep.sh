#!/usr/bin/env bash
set -euo pipefail

python -m src.cli pipeline --config input/pipeline/real_logistic_s50_t_sweep.yml

#!/usr/bin/env bash
set -euo pipefail

python -m src.cli pipeline --config input/pipeline/real_logistic_all.yml

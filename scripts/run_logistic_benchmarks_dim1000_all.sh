#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

run_pipeline() {
  local config="$1"
  echo "[pipeline] $config"
  python -m src.cli pipeline --config "$config"
}

# run_pipeline input/pipeline/logistic_benchmarks_dim1000/00_baseline_dense_reduced_from_original.yml
# run_pipeline input/pipeline/logistic_benchmarks_dim1000/01_baseline_dense_quick.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/02_overlap_nonseparable.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/03_interaction_misspecified.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/04_categorical_mixed.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/05_imbalanced_1pct.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/06_p_gt_n_sparse_beta.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/07_correlated_ill_conditioned.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/08_heavy_tail_outlier.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/09_sparse_x.yml
run_pipeline input/pipeline/logistic_benchmarks_dim1000/10_near_separable.yml

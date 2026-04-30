# Logistic Benchmarks Dim1000

Benchmark dataset-generation configs for synthetic logistic regression problems with final feature dimension capped at 1000.

Notes:
- `00_baseline_dense_reduced_from_original.yml` is the reduced version of the user's original config, with `d=1000` instead of `d=10000`.
- `04_categorical_mixed.yml` uses `final_dim = 1000` exactly because `988 + (3 + 4 + 5) = 1000`.
- `07_correlated_ill_conditioned.yml` uses `d=500` intentionally because dense Toeplitz covariance generation is more expensive.
- `09_sparse_x.yml` must not be changed to `covariance_type: toeplitz`; sparse feature generation is intended to stay with identity covariance.
- `06_p_gt_n_sparse_beta.yml` is the capped replacement for the earlier `p >> n` idea with `d=5000`, reduced to `d=1000` to satisfy the dimension cap.

| file | purpose | n | final_dim | main knobs | output npz |
|---|---|---:|---:|---|---|
| `00_baseline_dense_reduced_from_original.yml` | Reduced version of the original dense baseline | 50000 | 1000 | dense Gaussian, `feature_scale=1.0`, no extra misspecification | `data/generated/logistic_benchmarks_dim1000/logistic_baseline_dense_reduced_from_original.npz` |
| `01_baseline_dense_quick.yml` | Quick dense baseline for repeated optimizer runs | 5000 | 1000 | dense Gaussian, smaller `n` | `data/generated/logistic_benchmarks_dim1000/logistic_baseline_dense_quick.npz` |
| `02_overlap_nonseparable.yml` | Harder, less separable benchmark | 5000 | 1000 | `feature_scale=2.0`, `beta_scale=0.15`, `label_flip_prob=0.05` | `data/generated/logistic_benchmarks_dim1000/logistic_overlap_nonseparable.npz` |
| `03_interaction_misspecified.yml` | True model includes interactions, training remains linear | 5000 | 1000 | interaction pairs `(0,1)`, `(2,3)`, `(4,5)`, `interaction_scale=1.0` | `data/generated/logistic_benchmarks_dim1000/logistic_interaction_misspecified.npz` |
| `04_categorical_mixed.yml` | Continuous plus categorical one-hot features | 5000 | 1000 | `d=988`, categorical cardinalities `[3,4,5]`, `categorical_effect_scale=0.5` | `data/generated/logistic_benchmarks_dim1000/logistic_categorical_mixed.npz` |
| `05_imbalanced_1pct.yml` | Strong class imbalance around 1% positive | 20000 | 1000 | `class_balance=target_positive_rate`, `target_positive_rate=0.01` | `data/generated/logistic_benchmarks_dim1000/logistic_imbalanced_1pct.npz` |
| `06_p_gt_n_sparse_beta.yml` | Capped high-dimensional sparse-beta regime | 200 | 1000 | `p > n`, `sparse_beta=true`, `num_nonzero=20`, `beta_scale=0.2` | `data/generated/logistic_benchmarks_dim1000/logistic_p_gt_n_sparse_beta.npz` |
| `07_correlated_ill_conditioned.yml` | Dense correlated / ill-conditioned benchmark | 3000 | 500 | `covariance_type=toeplitz`, `cov_rho=0.95` | `data/generated/logistic_benchmarks_dim1000/logistic_correlated_ill_conditioned.npz` |
| `08_heavy_tail_outlier.yml` | Heavy-tailed features with outliers | 5000 | 1000 | `feature_distribution=student_t`, `t_df=3.0`, outliers at 1% with scale 20 | `data/generated/logistic_benchmarks_dim1000/logistic_heavy_tail_outlier.npz` |
| `09_sparse_x.yml` | Sparse feature matrix benchmark for large-scale methods | 10000 | 1000 | `sparse_X=true`, `x_density=0.01`, `sparse_beta=true`, identity covariance only | `data/generated/logistic_benchmarks_dim1000/logistic_sparse_x.npz` |
| `10_near_separable.yml` | Large-signal, almost separable regime | 5000 | 1000 | `beta_scale=3.0`, zero label noise | `data/generated/logistic_benchmarks_dim1000/logistic_near_separable.npz` |

# Lambda Sweep Data

Generate from raw data:

```bash
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_lam1e-1.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_lam1e-5.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep/usps_softmax_l2_lam1e-1.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep/usps_softmax_l2_lam1e-5.yml
```

Preferred clone flow:

```bash
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/epsilon_logistic_l2.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/usps_softmax_l2.yml

python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep_clone/epsilon_logistic_l2_clone_lam1e-1.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep_clone/epsilon_logistic_l2_clone_lam1e-5.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep_clone/usps_softmax_l2_clone_lam1e-1.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/lambda_sweep_clone/usps_softmax_l2_clone_lam1e-5.yml
```

Verify:

```bash
python scripts/verify_generated_problem.py data/generated/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_m10000_lam1e-1.npz
python scripts/verify_generated_problem.py data/generated/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_m10000_lam1e-5.npz
python scripts/verify_generated_problem.py data/generated/real_strongly_convex/lambda_sweep/usps_softmax_l2_m7291_lam1e-1.npz
python scripts/verify_generated_problem.py data/generated/real_strongly_convex/lambda_sweep/usps_softmax_l2_m7291_lam1e-5.npz
```

Compare same-data invariance:

```bash
python scripts/compare_npz_same_data.py data/generated/real_strongly_convex/epsilon_logistic_l2_m10000.npz data/generated/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_m10000_lam1e-1.npz
python scripts/compare_npz_same_data.py data/generated/real_strongly_convex/epsilon_logistic_l2_m10000.npz data/generated/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_m10000_lam1e-5.npz
python scripts/compare_npz_same_data.py data/generated/real_strongly_convex/usps_softmax_l2_m7291.npz data/generated/real_strongly_convex/lambda_sweep/usps_softmax_l2_m7291_lam1e-1.npz
python scripts/compare_npz_same_data.py data/generated/real_strongly_convex/usps_softmax_l2_m7291.npz data/generated/real_strongly_convex/lambda_sweep/usps_softmax_l2_m7291_lam1e-5.npz
```

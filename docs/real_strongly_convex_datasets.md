# Real Strongly Convex Datasets

## Auto-download

- MNIST softmax can be downloaded automatically from the LIBSVM multiclass collection.
- Mediamill exp1 can also be downloaded automatically from the LIBSVM multi-label collection when using the provided benchmark YAML.
- The current PPI benchmark YAML uses the LIBSVM `ppi_deepwalk.svm.bz2` file.

## Manual placement for multi-label data

If you prefer to place raw files manually, use these paths:

- `data/raw/multilabel/mediamill_exp1.txt`
- `data/raw/multilabel/PPI.txt`

Expected format:

- `source_format: multilabel_libsvm`
- the first token is a comma-separated label list
- the remaining tokens are `feature_index:value`

Example:

```text
1,5,10 3:0.2 10:1.0 120:0.5
```

If your local files are zero-based, set:

```yaml
index_base: 0
label_index_base: 0
```

If your local files are already stored as `.npz`, switch to:

```yaml
source_format: npz
raw_source: path/to/your_multilabel_dataset.npz
```

Supported feature keys in `.npz`:

- `A`
- `X`
- `A_data/A_indices/A_indptr/A_shape`
- `X_data/X_indices/X_indptr/X_shape`

Supported label keys in `.npz`:

- `Y`
- `y_multilabel`
- `Y_data/Y_indices/Y_indptr/Y_shape`

## Additional benchmark commands

```bash
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/additional/mnist_softmax_l2_m30000.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/additional/mediamill_multilabel_l2_m30993.yml
python -m src.cli generate_data --config input/generate_data/real_strongly_convex/additional/ppi_multilabel_l2_m50000.yml
```

```bash
python scripts/verify_generated_problem.py data/generated/real_strongly_convex/additional/mnist_softmax_l2_m30000.npz
python scripts/verify_generated_problem.py data/generated/real_strongly_convex/additional/mediamill_multilabel_l2_m30993.npz
python scripts/verify_generated_problem.py data/generated/real_strongly_convex/additional/ppi_multilabel_l2_m50000.npz
```

```bash
python scripts/compare_npz_problem_metadata.py \
  data/generated/real_strongly_convex/mnist_softmax_l2_m60000.npz \
  data/generated/real_strongly_convex/additional/mnist_softmax_l2_m30000.npz
```

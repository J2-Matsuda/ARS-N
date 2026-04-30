# Real Logistic Datasets

These configs convert raw LIBSVM/SVMLight classification files into the repo's `.npz` format for `LogisticRegressionProblem` with L2 regularization.

Expected raw files:

| config | raw_source | saved npz | n_features |
|---|---|---|---:|
| `epsilon_normalized.yml` | `data/raw/libsvm/epsilon_normalized.bz2` | `data/generated/real_logistic/epsilon_normalized.npz` | 2000 |
| `real_sim.yml` | `data/raw/libsvm/real-sim.bz2` | `data/generated/real_logistic/real_sim.npz` | 20958 |
| `rcv1_binary.yml` | `data/raw/libsvm/rcv1_train.binary.bz2` | `data/generated/real_logistic/rcv1_binary.npz` | 47236 |
| `gisette.yml` | `data/raw/libsvm/gisette_scale.bz2` | `data/generated/real_logistic/gisette.npz` | 5000 |

The loader also accepts `.gz` and `.bz2` paths if you edit `raw_source` accordingly. These configs have `download_if_missing: true`, so the raw `.bz2` files are downloaded from the LIBSVM dataset page when missing. Labels must be in `{0, 1}` or `{-1, +1}`. Feature indices are configured as one-based (`index_base: 1`) because standard LIBSVM files use one-based indexing.

Examples:

```bash
python -m src.cli generate --config input/generate_data/real_logistic/epsilon_normalized.yml
python -m src.cli optimize --config input/optimize/real_logistic/epsilon_normalized/ars_rn.yml
```

For a quick conversion smoke test on a subset, temporarily uncomment `max_rows` in a generate config.

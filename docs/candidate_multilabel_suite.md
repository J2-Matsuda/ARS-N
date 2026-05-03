# Candidate Multilabel Suite

This suite is meant to find real-data problems where ARS-CN is often faster than RS-CN.

The design follows the strongest existing positive case, PPI multilabel logistic regression:
low-dimensional graph embeddings, many labels, L2 regularization, and zero initialization.
The generated problems vary the real dataset, embedding method, sample size, regularization,
random seed, and ARS-CN anchor parameters.

## Datasets

- `ppi_line`: 121 labels, 128 features, sample sizes 20000, 50000. PPI graph node embeddings; same task as the current PPI winner, different embedding geometry.
- `ppi_node2vec`: 121 labels, 128 features, sample sizes 20000, 50000. PPI graph node embeddings; a second natural embedding variant for robustness.
- `flickr_deepwalk`: 195 labels, 128 features, sample sizes 20000, 50000. Social-network image/user graph labels with low-dimensional graph embeddings and many labels.
- `flickr_line`: 195 labels, 128 features, sample sizes 20000, 50000. Flickr graph labels with LINE embeddings; useful if anchor benefits depend on embedding spectra.
- `flickr_node2vec`: 195 labels, 128 features, sample sizes 20000, 50000. Flickr graph labels with node2vec embeddings; same natural task, different feature geometry.
- `delicious`: 983 labels, 500 features, sample sizes 8000, 16000. Bookmark tagging with many labels; increases optimization dimension without synthetic labels.
- `bibtex`: 159 labels, 1836 features, sample sizes 4000, 7395. Publication tagging with sparse text-like features; a moderate high-dimensional multilabel test.

## Generated Problem Count

The script creates 42 dataset-generation configs. For each generated problem it creates
21 optimization configs.

## Recommended First Pass

Run only these first if compute time is limited:

- `ppi_line_m50000_lam1p0em3`
- `ppi_node2vec_m50000_lam1p0em3`
- `flickr_deepwalk_m50000_lam1p0em3`
- `flickr_line_m50000_lam1p0em3`
- `flickr_node2vec_m50000_lam1p0em3`
- `delicious_m16000_lam1p0em3`

Compare time to `grad_norm <= 1e-5` and `grad_norm <= 1e-4`.
The best early ARS-CN settings from the existing runs are usually `s=100, r=100, T=50`
and `s=100, r=100, T=100`.

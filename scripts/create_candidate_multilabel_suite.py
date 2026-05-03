from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATE_ROOT = PROJECT_ROOT / "input" / "generate_data" / "candidate_multilabel"
OPTIMIZE_ROOT = PROJECT_ROOT / "input" / "optimize" / "candidate_multilabel"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot" / "candidate_multilabel"

BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel"

SEEDS = (0, 1, 2)
REG_LAMBDAS = ("1.0e-2", "1.0e-3", "1.0e-4")
ARS_CONFIGS = (
    (50, 50, 50),
    (100, 50, 100),
    (100, 100, 100),
    (200, 100, 100),
)
RS_SUBSPACE_DIMS = (50, 100, 200)
TOL = "1.0e-5"
ARS_MAX_ITER = 1000
RS_MAX_ITER = 10000
FIRST_PASS_SLUGS = (
    "ppi_line_m50000_lam1p0em3",
    "ppi_node2vec_m50000_lam1p0em3",
    "flickr_deepwalk_m50000_lam1p0em3",
    "flickr_line_m50000_lam1p0em3",
    "flickr_node2vec_m50000_lam1p0em3",
    "delicious_m16000_lam1p0em3",
)

DATASETS = (
    {
        "key": "ppi_line",
        "dataset_name": "ppi-line",
        "file": "ppi_line.svm.bz2",
        "n_features": 128,
        "num_labels": 121,
        "sample_sizes": (20000, 50000),
        "why": "PPI graph node embeddings; same task as the current PPI winner, different embedding geometry.",
    },
    {
        "key": "ppi_node2vec",
        "dataset_name": "ppi-node2vec",
        "file": "ppi_node2vec.svm.bz2",
        "n_features": 128,
        "num_labels": 121,
        "sample_sizes": (20000, 50000),
        "why": "PPI graph node embeddings; a second natural embedding variant for robustness.",
    },
    {
        "key": "flickr_deepwalk",
        "dataset_name": "flickr-deepwalk",
        "file": "flickr_deepwalk.svm.bz2",
        "n_features": 128,
        "num_labels": 195,
        "sample_sizes": (20000, 50000),
        "why": "Social-network image/user graph labels with low-dimensional graph embeddings and many labels.",
    },
    {
        "key": "flickr_line",
        "dataset_name": "flickr-line",
        "file": "flickr_line.svm.bz2",
        "n_features": 128,
        "num_labels": 195,
        "sample_sizes": (20000, 50000),
        "why": "Flickr graph labels with LINE embeddings; useful if anchor benefits depend on embedding spectra.",
    },
    {
        "key": "flickr_node2vec",
        "dataset_name": "flickr-node2vec",
        "file": "flickr_node2vec.svm.bz2",
        "n_features": 128,
        "num_labels": 195,
        "sample_sizes": (20000, 50000),
        "why": "Flickr graph labels with node2vec embeddings; same natural task, different feature geometry.",
    },
    {
        "key": "delicious",
        "dataset_name": "delicious",
        "file": "delicious.bz2",
        "n_features": 500,
        "num_labels": 983,
        "sample_sizes": (8000, 16000),
        "why": "Bookmark tagging with many labels; increases optimization dimension without synthetic labels.",
    },
    {
        "key": "bibtex",
        "dataset_name": "bibtex",
        "file": "bibtex.bz2",
        "n_features": 1836,
        "num_labels": 159,
        "sample_sizes": (4000, 7395),
        "why": "Publication tagging with sparse text-like features; a moderate high-dimensional multilabel test.",
    },
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _slug(dataset_key: str, sample_size: int, reg_lambda: str) -> str:
    lam = reg_lambda.replace(".", "p").replace("-", "m")
    return f"{dataset_key}_m{sample_size}_lam{lam}"


def _generated_path(slug: str) -> str:
    return f"data/generated/candidate_multilabel/{slug}.npz"


def _raw_path(dataset: dict[str, object]) -> str:
    return f"data/raw/multilabel/candidates/{dataset['file']}"


def _generate_config(dataset: dict[str, object], sample_size: int, reg_lambda: str) -> str:
    slug = _slug(str(dataset["key"]), sample_size, reg_lambda)
    return f"""task: generate_data
run_name: {slug}
seed: 0

problem:
  type: multilabel_logistic
  source_format: multilabel_libsvm
  raw_source: {_raw_path(dataset)}
  download_if_missing: true
  download_if_corrupt: true
  download_url: {BASE_URL}/{dataset['file']}
  dataset_name: {dataset['dataset_name']}
  n_features: {dataset['n_features']}
  num_labels: {dataset['num_labels']}
  index_base: 1
  label_index_base: 1
  add_bias: true
  reg_lambda: {reg_lambda}
  regularize_bias: true
  sample_size: {sample_size}
  sample_seed: 0

save:
  path: {_generated_path(slug)}
"""


def _problem_block(slug: str) -> str:
    return f"""problem:
  type: multilabel_logistic
  source: {_generated_path(slug)}

initialization:
  type: zeros
"""


def _stagnation_block() -> str:
    return """  stop_on_grad_norm_stagnation: true
  grad_norm_stagnation_patience: 50
  grad_norm_stagnation_rtol: 1.0e-12
  grad_norm_stagnation_atol: 1.0e-12
"""


def _sketch_block() -> str:
    return """  sketch:
    mode: operator
    block_size: 512
    dtype: float64
"""


def _footer(run_name: str) -> str:
    return f"""log:
  enabled: true
  csv_path: output/results/candidate_multilabel/{run_name}/{run_name}.csv
  save_everytime: true

save_meta:
  enabled: true
  meta_path: output/results/candidate_multilabel/{run_name}/{run_name}.json
  resolved_config_path: output/results/candidate_multilabel/{run_name}/{run_name}.resolved.yml
"""


def _rs_cn_config(slug: str, s: int, seed: int) -> str:
    run_name = f"rs_cn_{slug}_s{s}_seed{seed}"
    return f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(slug)}
optimizer:
  type: rs_cn
  max_iter: {RS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {s}
  sigma0: 1.0
  sigma_min: 1.0e-8
  sigma_max: 1.0e8
  eta1: 0.05
  eta2: 0.9
  gamma1: 1.5
  gamma2: 2.0
  solver: lanczos
  exact_tol: 1.0e-12
  krylov_tol: 1.0e-8
  solve_each_i_th_krylov_space: 1
  keep_Q_matrix_in_memory: false
  verbose: true
  print_every: 10
{_sketch_block()}
{_footer(run_name)}"""


def _ars_cn_config(slug: str, s: int, t: int, r: int, seed: int) -> str:
    run_name = f"ars_cn_{slug}_s{s}_r{r}_t{t}_seed{seed}"
    return f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(slug)}
optimizer:
  type: ars_cn
  max_iter: {ARS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {s}
  sigma0: 1.0
  sigma_min: 1.0e-8
  sigma_max: 1.0e8
  eta1: 0.05
  eta2: 0.9
  gamma1: 1.5
  gamma2: 2.0
  solver: lanczos
  exact_tol: 1.0e-10
  krylov_tol: 1.0e-8
  solve_each_i_th_krylov_space: 1
  keep_Q_matrix_in_memory: false
  verbose: true
  print_every: 10
{_sketch_block()}  rk:
    mode: default
    T: {t}
    r: {r}
    seed_offset: {seed * 1_000_003}

{_footer(run_name)}"""


def _pipeline_config(slugs: list[str]) -> str:
    steps: list[str] = []
    for slug in slugs:
        steps.append(
            f"""  - command: generate_data
    config: input/generate_data/candidate_multilabel/{slug}.yml"""
        )
        for seed in SEEDS:
            for s, t, r in ARS_CONFIGS:
                steps.append(
                    f"""  - command: optimize
    config: input/optimize/candidate_multilabel/{slug}/ars_cn_s{s}_r{r}_t{t}_seed{seed}.yml"""
                )
            for s in RS_SUBSPACE_DIMS:
                steps.append(
                    f"""  - command: optimize
    config: input/optimize/candidate_multilabel/{slug}/rs_cn_s{s}_seed{seed}.yml"""
                )
        steps.append(
            f"""  - command: plot
    config: input/plot/candidate_multilabel/{slug}_grad_norm.yml"""
        )

    return f"""task: pipeline
pipeline_name: candidate_multilabel_screen

steps:
{chr(10).join(steps)}
"""


def _first_pass_pipeline_config(slugs: list[str]) -> str:
    available = set(slugs)
    selected = [slug for slug in FIRST_PASS_SLUGS if slug in available]
    steps: list[str] = []
    for slug in selected:
        steps.append(
            f"""  - command: generate_data
    config: input/generate_data/candidate_multilabel/{slug}.yml"""
        )
        for s, t, r in ARS_CONFIGS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/candidate_multilabel/{slug}/ars_cn_s{s}_r{r}_t{t}_seed0.yml"""
            )
        for s in RS_SUBSPACE_DIMS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/candidate_multilabel/{slug}/rs_cn_s{s}_seed0.yml"""
            )
        steps.append(
            f"""  - command: plot
    config: input/plot/candidate_multilabel/{slug}_grad_norm.yml"""
        )

    return f"""task: pipeline
pipeline_name: candidate_multilabel_first_pass

steps:
{chr(10).join(steps)}
"""


def _plot_config(slug: str) -> str:
    items: list[str] = []
    for seed in SEEDS:
        for s, t, r in ARS_CONFIGS:
            run_name = f"ars_cn_{slug}_s{s}_r{r}_t{t}_seed{seed}"
            items.append(
                f"""  - path: output/results/candidate_multilabel/{run_name}/{run_name}.csv
    label: "ARS-CN s={s}, r={r}, T={t}, seed={seed}"
    linestyle: solid
"""
            )
        for s in RS_SUBSPACE_DIMS:
            run_name = f"rs_cn_{slug}_s{s}_seed{seed}"
            items.append(
                f"""  - path: output/results/candidate_multilabel/{run_name}/{run_name}.csv
    label: "RS-CN s={s}, seed={seed}"
    linestyle: dashed
"""
            )

    return f"""task: plot
plot_name: candidate_multilabel_{slug}_grad_norm

inputs:
{''.join(items)}

plot:
  x: cumulative_time
  y: grad_norm
  xscale: linear
  yscale: log
  title: "candidate multilabel: {slug}"
  xlabel: cumulative_time
  ylabel: grad_norm
  grid: true
  skip_missing: true

save:
  path: output/plots/candidate_multilabel/{slug}_grad_norm.png
  dpi: 180
"""


def _readme(slugs: list[str]) -> str:
    dataset_lines = []
    for dataset in DATASETS:
        sizes = ", ".join(str(value) for value in dataset["sample_sizes"])
        dataset_lines.append(
            f"- `{dataset['key']}`: {dataset['num_labels']} labels, "
            f"{dataset['n_features']} features, sample sizes {sizes}. {dataset['why']}"
        )

    return f"""# Candidate Multilabel Suite

This suite is meant to find real-data problems where ARS-CN is often faster than RS-CN.

The design follows the strongest existing positive case, PPI multilabel logistic regression:
low-dimensional graph embeddings, many labels, L2 regularization, and zero initialization.
The generated problems vary the real dataset, embedding method, sample size, regularization,
random seed, and ARS-CN anchor parameters.

## Datasets

{chr(10).join(dataset_lines)}

## Generated Problem Count

The script creates {len(slugs)} dataset-generation configs. For each generated problem it creates
{len(SEEDS) * (len(ARS_CONFIGS) + len(RS_SUBSPACE_DIMS))} optimization configs.

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
"""


def main() -> None:
    slugs: list[str] = []
    for dataset in DATASETS:
        for sample_size in dataset["sample_sizes"]:
            for reg_lambda in REG_LAMBDAS:
                slug = _slug(str(dataset["key"]), int(sample_size), reg_lambda)
                slugs.append(slug)
                _write(GENERATE_ROOT / f"{slug}.yml", _generate_config(dataset, int(sample_size), reg_lambda))
                problem_dir = OPTIMIZE_ROOT / slug
                for seed in SEEDS:
                    for s, t, r in ARS_CONFIGS:
                        _write(problem_dir / f"ars_cn_s{s}_r{r}_t{t}_seed{seed}.yml", _ars_cn_config(slug, s, t, r, seed))
                    for s in RS_SUBSPACE_DIMS:
                        _write(problem_dir / f"rs_cn_s{s}_seed{seed}.yml", _rs_cn_config(slug, s, seed))
                _write(PLOT_ROOT / f"{slug}_grad_norm.yml", _plot_config(slug))

    _write(PIPELINE_ROOT / "candidate_multilabel_screen.yml", _pipeline_config(slugs))
    _write(PIPELINE_ROOT / "candidate_multilabel_first_pass.yml", _first_pass_pipeline_config(slugs))
    _write(PROJECT_ROOT / "docs" / "candidate_multilabel_suite.md", _readme(slugs))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATE_ROOT = PROJECT_ROOT / "input" / "generate_data" / "exp"
OPTIMIZE_BASE = PROJECT_ROOT / "input" / "optimize" / "exp"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline" / "exp"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot" / "exp"

SEEDS = range(5)
SUBSPACE_DIMS = (100, 200)
TOL = "1.0e-5"
GD_MAX_ITER = 100000
RS_MAX_ITER = 10000
ARS_MAX_ITER = 1000
AUTO_RK_TOL = "0.31622776601683794"
AUTO_T_MAX = 500

EXPERIMENTS = (
    {
        "pipeline_short": "exp_1",
        "experiment_name": "exp_1_gisette_reg1e-3_mfull",
        "dataset_key": "gisette_reg1e-3_mfull",
        "problem_type": "logistic",
        "dataset_source": "data/generated/exp/gisette_reg1e-3_mfull.npz",
        "plot_title": "exp_1: gisette reg=1e-3 m=full",
        "generate_config": """task: generate_data
run_name: gisette_reg1e-3_mfull
seed: 0

problem:
  type: logistic
  source_format: libsvm
  raw_source: data/raw/libsvm/gisette_scale.bz2
  download_if_missing: true
  download_url: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
  n_features: 5000
  index_base: 1
  reg_lambda: 1.0e-3
  # sample_size:
  # sample_seed:

save:
  path: data/generated/exp/gisette_reg1e-3_mfull.npz
""",
    },
    {
        "pipeline_short": "exp_2",
        "experiment_name": "exp_2_usps_reg0_mfull",
        "dataset_key": "usps_reg0_mfull",
        "problem_type": "softmax",
        "dataset_source": "data/generated/real_noreg_convex/usps_noreg.npz",
        "plot_title": "exp_2: usps reg=0 m=full",
        "generate_config": """task: generate_data
run_name: usps_reg0_mfull
seed: 0

problem:
  type: softmax
  source_format: libsvm
  raw_source: data/raw/libsvm/usps.bz2
  download_if_missing: true
  download_if_corrupt: true
  download_url: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
  n_features: 256
  num_classes: 10
  index_base: 1
  add_bias: true
  reg_lambda: 0.0
  regularize_bias: true
  sample_size: 7291
  sample_seed: 0

save:
  path: data/generated/real_noreg_convex/usps_noreg.npz
""",
    },
    {
        "pipeline_short": "exp_3",
        "experiment_name": "exp_3_ppi_reg0_mfull",
        "dataset_key": "ppi_reg0_mfull",
        "problem_type": "multilabel_logistic",
        "dataset_source": "data/generated/real_noreg_convex/ppi_noreg.npz",
        "plot_title": "exp_3: ppi reg=0 m=full",
        "generate_config": """task: generate_data
run_name: ppi_reg0_mfull
seed: 0

problem:
  type: multilabel_logistic
  source_format: multilabel_libsvm
  raw_source: data/raw/multilabel/ppi_deepwalk.svm.bz2
  download_if_missing: true
  download_if_corrupt: true
  download_url: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/ppi_deepwalk.svm.bz2
  dataset_name: ppi
  n_features: 128
  num_labels: 121
  index_base: 1
  label_index_base: 1
  add_bias: true
  reg_lambda: 0.0
  regularize_bias: true
  sample_size: 54958
  sample_seed: 0

save:
  path: data/generated/real_noreg_convex/ppi_noreg.npz
""",
    },
    {
        "pipeline_short": "exp_4",
        "experiment_name": "exp_4_unfair_reg0_mfull",
        "dataset_key": "unfair_reg0_mfull",
        "problem_type": "mlp_multilabel_logistic",
        "dataset_source": "data/generated/real_nonconvex/unfair_tos_mlp_multilabel_h1_m5532.npz",
        "plot_title": "exp_4: unfair-tos shared-mlp h=1",
        "initialization_type": "random_normal",
        "initialization_scale": "1.0e-2",
        "generate_config": """task: generate_data
run_name: unfair_tos_mlp_multilabel_h1
seed: 0

problem:
  type: mlp_multilabel_logistic
  dataset_name: UNFAIR-ToS
  source_format: lexglue_unfair_tos_tfidf
  raw_source: data/raw/lexglue/unfair_tos
  download_if_missing: true
  n_features: 6290
  num_labels: 8
  sample_size: 5532
  sample_seed: 0
  add_bias: true
  hidden_width: 1
  activation: tanh
  reg_lambda: 1.0e-3
  init_scale: 1.0e-2
  regularize_bias: true
  loss_average: sample_label

save:
  path: data/generated/real_nonconvex/unfair_tos_mlp_multilabel_h1_m5532.npz

expected:
  raw_feature_dim: 6290
  augmented_feature_dim: 6291
  num_labels: 8
  hidden_width: 1
  optimization_dim: 6307
  sample_size: 5532
""",
    },
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _result_path(run_name: str) -> str:
    return f"output/results/{run_name}/{run_name}.csv"


def _meta_block(run_name: str) -> str:
    return f"""log:
  enabled: true
  csv_path: {_result_path(run_name)}
  save_everytime: true

save_meta:
  enabled: true
  meta_path: output/meta/{run_name}.json
  resolved_config_path: output/meta/{run_name}.resolved.yml
"""


def _problem_block(problem_type: str, dataset_source: str) -> str:
    initialization_type = "zeros"
    initialization_scale = None
    for experiment in EXPERIMENTS:
        if experiment["problem_type"] == problem_type and experiment["dataset_source"] == dataset_source:
            initialization_type = str(experiment.get("initialization_type", "zeros"))
            initialization_scale = experiment.get("initialization_scale")
            break

    initialization_lines = ["initialization:", f"  type: {initialization_type}"]
    if initialization_type == "random_normal" and initialization_scale is not None:
        initialization_lines.append(f"  scale: {initialization_scale}")
    initialization_block = "\n".join(initialization_lines)

    return f"""problem:
  type: {problem_type}
  source: {dataset_source}

{initialization_block}
"""


def _line_search_block(indent: str = "  ") -> str:
    return f"""{indent}line_search:
{indent}  enabled: true
{indent}  alpha0: 1.0
{indent}  c1: 1.0e-4
{indent}  beta: 0.5
{indent}  max_iter: 25
"""


def _sketch_block(indent: str = "  ") -> str:
    return f"""{indent}sketch:
{indent}  mode: operator
{indent}  block_size: 512
{indent}  dtype: float64
"""


def _stagnation_block() -> str:
    return """  stop_on_grad_norm_stagnation: true
  grad_norm_stagnation_patience: 50
  grad_norm_stagnation_rtol: 1.0e-12
  grad_norm_stagnation_atol: 1.0e-12
"""


def _gd_config(experiment: dict[str, str]) -> str:
    run_name = f"gd_{experiment['dataset_key']}"
    return f"""task: optimize
run_name: {run_name}
seed: 0

{_problem_block(experiment['problem_type'], experiment['dataset_source'])}optimizer:
  type: gd
  max_iter: {GD_MAX_ITER}
  tol: {TOL}
  verbose: true
  print_every: 1000
{_line_search_block()}{_meta_block(run_name)}"""


def _rs_cn_config(experiment: dict[str, str], s: int, seed: int) -> str:
    run_name = f"rs_cn_{experiment['dataset_key']}_s{s}_seed{seed}"
    return f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(experiment['problem_type'], experiment['dataset_source'])}optimizer:
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
{_sketch_block()}{_meta_block(run_name)}"""


def _ars_cn_t100_config(experiment: dict[str, str], s: int, seed: int) -> str:
    run_name = f"ars_cn_{experiment['dataset_key']}_s{s}_t100_seed{seed}"
    return f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(experiment['problem_type'], experiment['dataset_source'])}optimizer:
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
    T: 100
    r: {s}
    seed_offset: {seed * 1_000_003}

{_meta_block(run_name)}"""


def _ars_cn_tauto_config(experiment: dict[str, str], s: int, seed: int) -> str:
    run_name = f"ars_cn_{experiment['dataset_key']}_s{s}_tauto_10m0p5_seed{seed}"
    return f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(experiment['problem_type'], experiment['dataset_source'])}optimizer:
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
    mode: T_auto
    T: {AUTO_T_MAX}
    r: {s}
    rk_tol: {AUTO_RK_TOL}
    seed_offset: {seed * 1_000_003}

{_meta_block(run_name)}"""


def _rs_rn_config(experiment: dict[str, str], s: int, seed: int) -> str:
    run_name = f"rs_rn_{experiment['dataset_key']}_s{s}_seed{seed}"
    return f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(experiment['problem_type'], experiment['dataset_source'])}optimizer:
  type: rs_rn
  max_iter: {RS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {s}
  verbose: true
  print_every: 20
{_sketch_block()}  diag_shift:
    c1: 2.0
    c2: 1.0
    gamma: 0.5
{_line_search_block()}{_meta_block(run_name)}"""


def _aggregate_block(paths: list[str]) -> str:
    lines = "\n".join(f"      - {path}" for path in paths)
    return f"""paths:
{lines}
    aggregate:
      center: mean
      band: minmax
      alpha: 0.16"""


def _plot_inputs_yaml(experiment: dict[str, str]) -> str:
    dataset_key = experiment["dataset_key"]
    blocks: list[str] = [
        f"""  - path: {_result_path(f"gd_{dataset_key}")}
    label: "GD"
    color: "#111111"
    linestyle: solid"""
    ]

    for s, color in ((100, "#212529"), (200, "#495057")):
        paths = [_result_path(f"rs_cn_{dataset_key}_s{s}_seed{seed}") for seed in SEEDS]
        blocks.append(
            f"""  - {_aggregate_block(paths)}
    label: "RS-CN s={s}"
    color: "{color}"
    linestyle: dashdot"""
        )

    for s, color in ((100, "#1971c2"), (200, "#0b7285")):
        paths = [_result_path(f"ars_cn_{dataset_key}_s{s}_t100_seed{seed}") for seed in SEEDS]
        blocks.append(
            f"""  - {_aggregate_block(paths)}
    label: "ARS-CN s=r={s}, T=100"
    color: "{color}"
    linestyle: solid"""
        )

    for s, color in ((100, "#862e9c"), (200, "#5f3dc4")):
        paths = [_result_path(f"ars_cn_{dataset_key}_s{s}_tauto_10m0p5_seed{seed}") for seed in SEEDS]
        blocks.append(
            f"""  - {_aggregate_block(paths)}
    label: "ARS-CN s=r={s}, T_auto (T<=500, rk_tol=10^-0.5)"
    color: "{color}"
    linestyle: dashed"""
        )

    for s, color in ((100, "#d9480f"), (200, "#e8590c")):
        paths = [_result_path(f"rs_rn_{dataset_key}_s{s}_seed{seed}") for seed in SEEDS]
        blocks.append(
            f"""  - {_aggregate_block(paths)}
    label: "RS-RN s={s}"
    color: "{color}"
    linestyle: dotted"""
        )

    return "\n".join(blocks)


def _plot_config(experiment: dict[str, str], y_key: str, ylabel: str, suffix: str) -> str:
    experiment_name = experiment["experiment_name"]
    return f"""task: plot
plot_name: {experiment_name}_{suffix}

inputs:
{_plot_inputs_yaml(experiment)}

plot:
  x: cumulative_time
  y: {y_key}
  xscale: linear
  yscale: log
  title: "{experiment['plot_title']}"
  xlabel: cumulative_time
  ylabel: {ylabel}
  grid: true
  skip_missing: true

save:
  path: output/plots/exp/{experiment_name}_{suffix}.png
  dpi: 180
"""


def _pipeline_config(experiment: dict[str, str]) -> str:
    experiment_name = experiment["experiment_name"]
    dataset_key = experiment["dataset_key"]
    pipeline_name = experiment["pipeline_short"]

    steps: list[str] = [
        "  - command: generate_data",
        f"    config: input/generate_data/exp/{dataset_key}.yml",
        "  - command: optimize",
        f"    config: input/optimize/exp/{experiment_name}/gd.yml",
    ]

    for s in SUBSPACE_DIMS:
        for seed in SEEDS:
            steps.extend(
                [
                    "  - command: optimize",
                    f"    config: input/optimize/exp/{experiment_name}/rs_cn_s{s}_seed{seed}.yml",
                ]
            )
    for s in SUBSPACE_DIMS:
        for seed in SEEDS:
            steps.extend(
                [
                    "  - command: optimize",
                    f"    config: input/optimize/exp/{experiment_name}/ars_cn_s{s}_t100_seed{seed}.yml",
                ]
            )
    for s in SUBSPACE_DIMS:
        for seed in SEEDS:
            steps.extend(
                [
                    "  - command: optimize",
                    f"    config: input/optimize/exp/{experiment_name}/ars_cn_s{s}_tauto_10m0p5_seed{seed}.yml",
                ]
            )
    for s in SUBSPACE_DIMS:
        for seed in SEEDS:
            steps.extend(
                [
                    "  - command: optimize",
                    f"    config: input/optimize/exp/{experiment_name}/rs_rn_s{s}_seed{seed}.yml",
                ]
            )

    steps.extend(
        [
            "  - command: plot",
            f"    config: input/plot/exp/{experiment_name}_grad_norm.yml",
            "  - command: plot",
            f"    config: input/plot/exp/{experiment_name}_function_value.yml",
        ]
    )

    return f"task: pipeline\npipeline_name: {pipeline_name}\n\nsteps:\n" + "\n".join(steps) + "\n"


def _write_experiment(experiment: dict[str, str]) -> None:
    experiment_name = experiment["experiment_name"]
    dataset_key = experiment["dataset_key"]
    optimize_root = OPTIMIZE_BASE / experiment_name

    _write(GENERATE_ROOT / f"{dataset_key}.yml", experiment["generate_config"])
    _write(optimize_root / "gd.yml", _gd_config(experiment))

    for s in SUBSPACE_DIMS:
        for seed in SEEDS:
            _write(optimize_root / f"rs_cn_s{s}_seed{seed}.yml", _rs_cn_config(experiment, s, seed))
            _write(optimize_root / f"ars_cn_s{s}_t100_seed{seed}.yml", _ars_cn_t100_config(experiment, s, seed))
            _write(
                optimize_root / f"ars_cn_s{s}_tauto_10m0p5_seed{seed}.yml",
                _ars_cn_tauto_config(experiment, s, seed),
            )
            _write(optimize_root / f"rs_rn_s{s}_seed{seed}.yml", _rs_rn_config(experiment, s, seed))

    _write(PLOT_ROOT / f"{experiment_name}_grad_norm.yml", _plot_config(experiment, "grad_norm", "grad_norm", "grad_norm"))
    _write(
        PLOT_ROOT / f"{experiment_name}_function_value.yml",
        _plot_config(experiment, "f", "function_value", "function_value"),
    )
    pipeline_content = _pipeline_config(experiment)
    _write(PIPELINE_ROOT / f"{experiment['pipeline_short']}.yml", pipeline_content)
    _write(PIPELINE_ROOT / f"{experiment_name}.yml", pipeline_content)

    if experiment["pipeline_short"] == "exp_1":
        _write(GENERATE_ROOT / "gisette_reg1e-5_mfull.yml", experiment["generate_config"])
        _write(PIPELINE_ROOT / "exp_1_gisette_reg1e-5_mfull.yml", pipeline_content)
    if experiment["pipeline_short"] == "exp_4":
        _write(GENERATE_ROOT / "unfair_reg0_mfull.yml", experiment["generate_config"])
        _write(PIPELINE_ROOT / "exp_4_unfair_reg0_mfull.yml", pipeline_content)


def main() -> None:
    for experiment in EXPERIMENTS:
        _write_experiment(experiment)


if __name__ == "__main__":
    main()

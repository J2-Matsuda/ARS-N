from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZE_ROOT = PROJECT_ROOT / "input" / "optimize" / "real_logistic_compare"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot"

DATASETS = (
    ("gisette", "gisette", "data/generated/real_logistic/gisette.npz"),
    ("epsilon", "epsilon_normalized", "data/generated/real_logistic/epsilon_normalized.npz"),
    ("real_sim", "real_sim", "data/generated/real_logistic/real_sim.npz"),
)
SEEDS = (0, 1, 2, 3, 4)
STOCHASTIC_SUBSPACE_DIM = 100

GD_MAX_ITER = 100000
AGD_MAX_ITER = 100000
RS_MAX_ITER = 10000
ARS_CN_MAX_ITER = 1000
TOL = "1.0e-5"

ARS_CN_FIXED_T = 100
ARS_CN_T_AUTO_MAX = 500
ARS_CN_T_AUTO_TOLS = (
    ("0p1", "0.1", "0.1"),
    ("10m0p5", "0.31622776601683794", "10^-0.5"),
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _result_csv(run_name: str) -> str:
    return f"output/results/{run_name}/{run_name}.csv"


def _optimize_paths(run_name: str) -> str:
    return f"""log:
  enabled: true
  csv_path: output/results/{run_name}.csv
  save_everytime: true

save_meta:
  enabled: true
  meta_path: output/meta/{run_name}.json
  resolved_config_path: output/meta/{run_name}.resolved.yml
"""


def _problem_block(source: str) -> str:
    return f"""problem:
  type: logistic
  source: {source}

initialization:
  type: zeros
"""


def _stagnation_block() -> str:
    return """  stop_on_grad_norm_stagnation: true
  grad_norm_stagnation_patience: 50
  grad_norm_stagnation_rtol: 1.0e-12
  grad_norm_stagnation_atol: 1.0e-12
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


def _seed_offset(seed: int) -> int:
    return seed * 1_000_003


def _gd_config(dataset_slug: str, source: str) -> tuple[str, str]:
    run_name = f"gd_real_logistic_{dataset_slug}"
    content = f"""task: optimize
run_name: {run_name}
seed: 0

{_problem_block(source)}
optimizer:
  type: gd
  max_iter: {GD_MAX_ITER}
  tol: {TOL}
  verbose: true
  print_every: 1000
{_line_search_block()}
{_optimize_paths(run_name)}"""
    return run_name, content


def _agd_config(dataset_slug: str, source: str) -> tuple[str, str]:
    run_name = f"agd_unknown_real_logistic_{dataset_slug}"
    content = f"""task: optimize
run_name: {run_name}
seed: 0

{_problem_block(source)}
optimizer:
  type: agd_unknown
  max_iter: {AGD_MAX_ITER}
  tol: {TOL}
  verbose: true
  print_every: 1000
  backtracking:
    enabled: true
    L0: 1.0
    eta: 2.0
    max_iter: 50
    reuse_previous_L: true
  restart:
    enabled: true
    objective_increase: true
    misaligned_momentum: true

{_optimize_paths(run_name)}"""
    return run_name, content


def _rs_rn_config(dataset_slug: str, source: str, seed: int) -> tuple[str, str]:
    run_name = f"rs_rn_real_logistic_{dataset_slug}_s100_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(source)}
optimizer:
  type: rs_rn
  max_iter: {RS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {STOCHASTIC_SUBSPACE_DIM}
  verbose: true
  print_every: 20
{_sketch_block()}  diag_shift:
    c1: 2.0
    c2: 1.0
    gamma: 0.5
{_line_search_block()}
{_optimize_paths(run_name)}"""
    return run_name, content


def _rs_cn_config(dataset_slug: str, source: str, seed: int) -> tuple[str, str]:
    run_name = f"rs_cn_real_logistic_{dataset_slug}_s100_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(source)}
optimizer:
  type: rs_cn
  max_iter: {RS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {STOCHASTIC_SUBSPACE_DIM}
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
{_optimize_paths(run_name)}"""
    return run_name, content


def _ars_cn_fixed_config(dataset_slug: str, source: str, seed: int) -> tuple[str, str]:
    run_name = f"ars_cn_real_logistic_{dataset_slug}_s100_t100_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(source)}
optimizer:
  type: ars_cn
  max_iter: {ARS_CN_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {STOCHASTIC_SUBSPACE_DIM}
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
    T: {ARS_CN_FIXED_T}
    r: {STOCHASTIC_SUBSPACE_DIM}
    seed_offset: {_seed_offset(seed)}

{_optimize_paths(run_name)}"""
    return run_name, content


def _ars_cn_tauto_config(
    dataset_slug: str,
    source: str,
    seed: int,
    tol_slug: str,
    rk_tol_value: str,
) -> tuple[str, str]:
    run_name = f"ars_cn_real_logistic_{dataset_slug}_s100_tauto_{tol_slug}_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block(source)}
optimizer:
  type: ars_cn
  max_iter: {ARS_CN_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {STOCHASTIC_SUBSPACE_DIM}
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
    T: {ARS_CN_T_AUTO_MAX}
    r: {STOCHASTIC_SUBSPACE_DIM}
    rk_tol: {rk_tol_value}
    seed_offset: {_seed_offset(seed)}

{_optimize_paths(run_name)}"""
    return run_name, content


def _aggregate_block(paths: list[str], label: str, color: str, linestyle: str) -> str:
    joined_paths = "\n".join(f"      - {path}" for path in paths)
    return f"""  - paths:
{joined_paths}
    label: "{label}"
    color: "{color}"
    linestyle: {linestyle}
    aggregate:
      center: mean
      band: minmax
      alpha: 0.22
"""


def _single_block(path: str, label: str, color: str, linestyle: str) -> str:
    return f"""  - path: {path}
    label: "{label}"
    color: "{color}"
    linestyle: {linestyle}
"""


def _panel_block(title: str, body: str) -> str:
    indented_body = "\n".join(f"    {line}" if line else "" for line in body.rstrip().splitlines())
    return f"""  - title: "{title}"
    inputs:
{indented_body}
"""


def _rn_plot_config() -> str:
    panel_blocks: list[str] = []
    for dataset_label, dataset_slug, _source in DATASETS:
        body = ""
        body += _single_block(
            _result_csv(f"gd_real_logistic_{dataset_slug}"),
            "GD",
            "#111111",
            "solid",
        )
        body += _single_block(
            _result_csv(f"agd_unknown_real_logistic_{dataset_slug}"),
            "AGD-Unknown",
            "#1c7ed6",
            "dashdot",
        )
        body += _aggregate_block(
            [
                _result_csv(f"rs_rn_real_logistic_{dataset_slug}_s100_seed{seed}")
                for seed in SEEDS
            ],
            "RS-RN s=100",
            "#e8590c",
            "dashed",
        )
        panel_blocks.append(_panel_block(dataset_label, body))

    joined_panels = "\n".join(panel_blocks)
    return f"""task: plot
plot_name: real_logistic_compare_rn

panels:
{joined_panels}

plot:
  x: cumulative_time
  y: grad_norm
  xscale: linear
  yscale: log
  xlabel: cumulative_time
  ylabel: grad_norm
  grid: true
  figure_title: "real logistic: RN comparison"
  shared_legend: true
  shared_legend_location: lower center
  shared_legend_ncol: 3
  layout:
    ncols: 3
    sharey: true
    width: 18.0
    height: 4.6

save:
  path: output/plots/real_logistic_compare_rn.png
  dpi: 180
"""


def _cn_plot_config() -> str:
    panel_blocks: list[str] = []
    for dataset_label, dataset_slug, _source in DATASETS:
        body = ""
        body += _single_block(
            _result_csv(f"gd_real_logistic_{dataset_slug}"),
            "GD",
            "#111111",
            "solid",
        )
        body += _single_block(
            _result_csv(f"agd_unknown_real_logistic_{dataset_slug}"),
            "AGD-Unknown",
            "#1c7ed6",
            "dashdot",
        )
        body += _aggregate_block(
            [
                _result_csv(f"rs_cn_real_logistic_{dataset_slug}_s100_seed{seed}")
                for seed in SEEDS
            ],
            "RS-CN s=100",
            "#495057",
            "dashdot",
        )
        body += _aggregate_block(
            [
                _result_csv(f"ars_cn_real_logistic_{dataset_slug}_s100_t100_seed{seed}")
                for seed in SEEDS
            ],
            "ARS-CN s=r=100, T=100",
            "#d9480f",
            "solid",
        )
        for tol_slug, _rk_tol_value, rk_tol_label in ARS_CN_T_AUTO_TOLS:
            body += _aggregate_block(
                [
                    _result_csv(
                        f"ars_cn_real_logistic_{dataset_slug}_s100_tauto_{tol_slug}_seed{seed}"
                    )
                    for seed in SEEDS
                ],
                f"ARS-CN s=r=100, T_auto (T<=500, rk_tol={rk_tol_label})",
                "#2b8a3e" if tol_slug == "0p1" else "#9c36b5",
                "dashed" if tol_slug == "0p1" else "dotted",
            )
        panel_blocks.append(_panel_block(dataset_label, body))

    joined_panels = "\n".join(panel_blocks)
    return f"""task: plot
plot_name: real_logistic_compare_cn

panels:
{joined_panels}

plot:
  x: cumulative_time
  y: grad_norm
  xscale: linear
  yscale: log
  xlabel: cumulative_time
  ylabel: grad_norm
  grid: true
  figure_title: "real logistic: CN comparison"
  shared_legend: true
  shared_legend_location: lower center
  shared_legend_ncol: 3
  layout:
    ncols: 3
    sharey: true
    width: 18.0
    height: 4.8

save:
  path: output/plots/real_logistic_compare_cn.png
  dpi: 180
"""


def _rn_pipeline_config(optimize_paths: list[str]) -> str:
    steps: list[str] = []
    for dataset_label, dataset_slug, _source in DATASETS:
        steps.append(
            f"""  - command: generate
    config: input/generate_data/real_logistic/{dataset_slug}.yml"""
        )
        steps.append(
            f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/gd.yml"""
        )
        steps.append(
            f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/agd_unknown.yml"""
        )
        for seed in SEEDS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/rs_rn_s100_seed{seed}.yml"""
            )
    steps.append(
        """  - command: plot
    config: input/plot/real_logistic_compare_rn.yml"""
    )
    joined_steps = "\n".join(steps)
    return f"""task: pipeline
pipeline_name: real_logistic_compare_rn

steps:
{joined_steps}
"""


def _cn_pipeline_config() -> str:
    steps: list[str] = []
    for _dataset_label, dataset_slug, _source in DATASETS:
        steps.append(
            f"""  - command: generate
    config: input/generate_data/real_logistic/{dataset_slug}.yml"""
        )
        steps.append(
            f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/gd.yml"""
        )
        steps.append(
            f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/agd_unknown.yml"""
        )
        for seed in SEEDS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/rs_cn_s100_seed{seed}.yml"""
            )
        for seed in SEEDS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/ars_cn_s100_t100_seed{seed}.yml"""
            )
        for tol_slug, _rk_tol_value, _rk_tol_label in ARS_CN_T_AUTO_TOLS:
            for seed in SEEDS:
                steps.append(
                    f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/ars_cn_s100_tauto_{tol_slug}_seed{seed}.yml"""
                )
    steps.append(
        """  - command: plot
    config: input/plot/real_logistic_compare_cn.yml"""
    )
    joined_steps = "\n".join(steps)
    return f"""task: pipeline
pipeline_name: real_logistic_compare_cn

steps:
{joined_steps}
"""


def main() -> None:
    optimize_paths: list[str] = []

    for _dataset_label, dataset_slug, source in DATASETS:
        dataset_dir = OPTIMIZE_ROOT / dataset_slug

        run_name, content = _gd_config(dataset_slug, source)
        path = dataset_dir / "gd.yml"
        _write(path, content)
        optimize_paths.append(str(path))

        run_name, content = _agd_config(dataset_slug, source)
        path = dataset_dir / "agd_unknown.yml"
        _write(path, content)
        optimize_paths.append(str(path))

        for seed in SEEDS:
            run_name, content = _rs_rn_config(dataset_slug, source, seed)
            path = dataset_dir / f"rs_rn_s100_seed{seed}.yml"
            _write(path, content)
            optimize_paths.append(str(path))

            run_name, content = _rs_cn_config(dataset_slug, source, seed)
            path = dataset_dir / f"rs_cn_s100_seed{seed}.yml"
            _write(path, content)
            optimize_paths.append(str(path))

            run_name, content = _ars_cn_fixed_config(dataset_slug, source, seed)
            path = dataset_dir / f"ars_cn_s100_t100_seed{seed}.yml"
            _write(path, content)
            optimize_paths.append(str(path))

            for tol_slug, rk_tol_value, _rk_tol_label in ARS_CN_T_AUTO_TOLS:
                run_name, content = _ars_cn_tauto_config(
                    dataset_slug,
                    source,
                    seed,
                    tol_slug,
                    rk_tol_value,
                )
                path = dataset_dir / f"ars_cn_s100_tauto_{tol_slug}_seed{seed}.yml"
                _write(path, content)
                optimize_paths.append(str(path))

    _write(PIPELINE_ROOT / "real_logistic_compare_rn.yml", _rn_pipeline_config(optimize_paths))
    _write(PIPELINE_ROOT / "real_logistic_compare_cn.yml", _cn_pipeline_config())
    _write(PLOT_ROOT / "real_logistic_compare_rn.yml", _rn_plot_config())
    _write(PLOT_ROOT / "real_logistic_compare_cn.yml", _cn_plot_config())


if __name__ == "__main__":
    main()

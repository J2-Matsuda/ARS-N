from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZE_ROOT = PROJECT_ROOT / "input" / "optimize" / "real_strongly_convex_compare"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot"

PROBLEMS = (
    {
        "name": "usps",
        "label": "usps",
        "type": "softmax",
        "generate_config": "input/generate_data/real_strongly_convex/usps_softmax_l2.yml",
        "source": "data/generated/real_strongly_convex/usps_softmax_l2_m7291.npz",
    },
    {
        "name": "epsilon",
        "label": "epsilon",
        "type": "logistic",
        "generate_config": "input/generate_data/real_strongly_convex/epsilon_logistic_l2.yml",
        "source": "data/generated/real_strongly_convex/epsilon_logistic_l2_m10000.npz",
    },
    {
        "name": "blogcatalog",
        "label": "blogcatalog",
        "type": "multilabel_logistic",
        "generate_config": "input/generate_data/real_strongly_convex/blogcatalog_multilabel_l2.yml",
        "source": "data/generated/real_strongly_convex/blogcatalog_multilabel_l2_m10312.npz",
    },
    {
        "name": "mnist",
        "label": "mnist",
        "type": "softmax",
        "generate_config": "input/generate_data/real_strongly_convex/mnist_softmax_l2.yml",
        "source": "data/generated/real_strongly_convex/mnist_softmax_l2_m60000.npz",
    },
    {
        "name": "ppi",
        "label": "ppi",
        "type": "multilabel_logistic",
        "generate_config": "input/generate_data/real_strongly_convex/ppi_multilabel_l2.yml",
        "source": "data/generated/real_strongly_convex/ppi_multilabel_l2_m50000.npz",
    },
    {
        "name": "epsilon_lam1e-1",
        "label": "epsilon (lambda=1e-1)",
        "type": "logistic",
        "generate_config": "input/generate_data/real_strongly_convex/additional/epsilon_logistic_l2_clone_lam1e-1.yml",
        "source": "data/generated/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_m10000_lam1e-1.npz",
    },
    {
        "name": "epsilon_lam1e-5",
        "label": "epsilon (lambda=1e-5)",
        "type": "logistic",
        "generate_config": "input/generate_data/real_strongly_convex/additional/epsilon_logistic_l2_clone_lam1e-5.yml",
        "source": "data/generated/real_strongly_convex/lambda_sweep/epsilon_logistic_l2_m10000_lam1e-5.npz",
    },
    {
        "name": "usps_lam1e-1",
        "label": "usps (lambda=1e-1)",
        "type": "softmax",
        "generate_config": "input/generate_data/real_strongly_convex/additional/usps_softmax_l2_clone_lam1e-1.yml",
        "source": "data/generated/real_strongly_convex/lambda_sweep/usps_softmax_l2_m7291_lam1e-1.npz",
    },
    {
        "name": "usps_lam1e-5",
        "label": "usps (lambda=1e-5)",
        "type": "softmax",
        "generate_config": "input/generate_data/real_strongly_convex/additional/usps_softmax_l2_clone_lam1e-5.yml",
        "source": "data/generated/real_strongly_convex/lambda_sweep/usps_softmax_l2_m7291_lam1e-5.npz",
    },
    {
        "name": "mnist_m30000",
        "label": "mnist (m=30000)",
        "type": "softmax",
        "generate_config": "input/generate_data/real_strongly_convex/additional/mnist_softmax_l2_m30000.yml",
        "source": "data/generated/real_strongly_convex/additional/mnist_softmax_l2_m30000.npz",
    },
    {
        "name": "mnist_m10000",
        "label": "mnist (m=10000)",
        "type": "softmax",
        "generate_config": "input/generate_data/real_strongly_convex/additional/mnist_softmax_l2_m10000.yml",
        "source": "data/generated/real_strongly_convex/additional/mnist_softmax_l2_m10000.npz",
    },
    {
        "name": "mediamill",
        "label": "mediamill exp1",
        "type": "multilabel_logistic",
        "generate_config": "input/generate_data/real_strongly_convex/additional/mediamill_multilabel_l2_m30993.yml",
        "source": "data/generated/real_strongly_convex/additional/mediamill_multilabel_l2_m30993.npz",
    },
    {
        "name": "ppi_additional",
        "label": "ppi (additional)",
        "type": "multilabel_logistic",
        "generate_config": "input/generate_data/real_strongly_convex/additional/ppi_multilabel_l2_m50000.yml",
        "source": "data/generated/real_strongly_convex/additional/ppi_multilabel_l2_m50000.npz",
    },
)

SEED = 0
SUBSPACE_DIMS = (50, 100, 200)
ARS_T_VALUES = (50, 100, 200)
ARS_MAX_ITER = 1000
RS_MAX_ITER = 10000
FIRST_ORDER_MAX_ITER = 100000
TOL = "1.0e-5"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _optimize_footer(run_name: str) -> str:
    return f"""log:
  enabled: true
  csv_path: output/results/{run_name}.csv
  save_everytime: true

save_meta:
  enabled: true
  meta_path: output/meta/{run_name}.json
  resolved_config_path: output/meta/{run_name}.resolved.yml
"""


def _problem_block(problem_type: str, source: str) -> str:
    return f"""problem:
  type: {problem_type}
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


def _result_path(run_name: str) -> str:
    return f"output/results/{run_name}/{run_name}.csv"


def _gd_config(problem: dict[str, str]) -> str:
    run_name = f"gd_{problem['name']}_strongly_convex"
    return f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(problem['type'], problem['source'])}
optimizer:
  type: gd
  max_iter: {FIRST_ORDER_MAX_ITER}
  tol: {TOL}
  verbose: true
  print_every: 1000
{_line_search_block()}{_optimize_footer(run_name)}"""


def _agd_config(problem: dict[str, str]) -> str:
    run_name = f"agd_unknown_{problem['name']}_strongly_convex"
    return f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(problem['type'], problem['source'])}
optimizer:
  type: agd_unknown
  max_iter: {FIRST_ORDER_MAX_ITER}
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

{_optimize_footer(run_name)}"""


def _rs_rn_config(problem: dict[str, str], s: int) -> str:
    run_name = f"rs_rn_{problem['name']}_strongly_convex_s{s}_seed{SEED}"
    return f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(problem['type'], problem['source'])}
optimizer:
  type: rs_rn
  max_iter: {RS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {SEED}
  subspace_dim: {s}
  verbose: true
  print_every: 20
{_sketch_block()}  diag_shift:
    c1: 2.0
    c2: 1.0
    gamma: 0.5
{_line_search_block()}{_optimize_footer(run_name)}"""


def _rs_cn_config(problem: dict[str, str], s: int) -> str:
    run_name = f"rs_cn_{problem['name']}_strongly_convex_s{s}_seed{SEED}"
    return f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(problem['type'], problem['source'])}
optimizer:
  type: rs_cn
  max_iter: {RS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {SEED}
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
{_optimize_footer(run_name)}"""


def _ars_cn_config(problem: dict[str, str], s: int, T: int) -> str:
    run_name = f"ars_cn_{problem['name']}_strongly_convex_s{s}_t{T}_seed{SEED}"
    return f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(problem['type'], problem['source'])}
optimizer:
  type: ars_cn
  max_iter: {ARS_MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {SEED}
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
    T: {T}
    r: {s}
    seed_offset: {_seed_offset(SEED)}

{_optimize_footer(run_name)}"""


def _plot_item(path: str, label: str, color: str, linestyle: str) -> str:
    return f"""  - path: {path}
    label: "{label}"
    color: "{color}"
    linestyle: {linestyle}
"""


def _plot_config(
    problem: dict[str, str],
    *,
    y_key: str,
    suffix: str,
    ylabel: str,
    yscale: str,
) -> str:
    items: list[str] = []
    ars_colors = {50: "#2b8a3e", 100: "#1c7ed6", 200: "#9c36b5"}
    t_linestyles = {50: "dotted", 100: "solid", 200: "dashed"}
    for s in SUBSPACE_DIMS:
        for T in ARS_T_VALUES:
            items.append(
                _plot_item(
                    _result_path(f"ars_cn_{problem['name']}_strongly_convex_s{s}_t{T}_seed0"),
                    f"ARS-CN s=r={s}, T={T}",
                    ars_colors[s],
                    t_linestyles[T],
                )
            )

    rs_cn_colors = {50: "#adb5bd", 100: "#6c757d", 200: "#212529"}
    for s in SUBSPACE_DIMS:
        items.append(
            _plot_item(
                _result_path(f"rs_cn_{problem['name']}_strongly_convex_s{s}_seed0"),
                f"RS-CN s={s}",
                rs_cn_colors[s],
                "dashdot",
            )
        )

    items.append(_plot_item(_result_path(f"gd_{problem['name']}_strongly_convex"), "GD", "#111111", "solid"))
    items.append(
        _plot_item(
            _result_path(f"agd_unknown_{problem['name']}_strongly_convex"),
            "AGD",
            "#d9480f",
            "dashdot",
        )
    )

    rs_rn_colors = {50: "#ffa94d", 100: "#fd7e14", 200: "#e8590c"}
    for s in SUBSPACE_DIMS:
        items.append(
            _plot_item(
                _result_path(f"rs_rn_{problem['name']}_strongly_convex_s{s}_seed0"),
                f"RS-RN s={s}",
                rs_rn_colors[s],
                "dashed",
            )
        )

    body = "".join(items)
    return f"""task: plot
plot_name: real_strongly_convex_compare_{problem['name']}_{suffix}

inputs:
{body}

plot:
  x: cumulative_time
  y: {y_key}
  xscale: linear
  yscale: {yscale}
  title: "real strongly convex: {problem['label']}"
  xlabel: cumulative_time
  ylabel: {ylabel}
  grid: true
  skip_missing: true

save:
  path: output/plots/real_strongly_convex_compare_{problem['name']}_{suffix}.png
  dpi: 180
"""


def _pipeline_config() -> str:
    steps: list[str] = []
    for problem in PROBLEMS:
        steps.append(
            f"""  - command: generate_data
    config: {problem['generate_config']}"""
        )
        for s in SUBSPACE_DIMS:
            for T in ARS_T_VALUES:
                steps.append(
                    f"""  - command: optimize
    config: input/optimize/real_strongly_convex_compare/{problem['name']}/ars_cn_s{s}_t{T}_seed0.yml"""
                )
        for s in SUBSPACE_DIMS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/real_strongly_convex_compare/{problem['name']}/rs_cn_s{s}_seed0.yml"""
            )
        steps.append(
            f"""  - command: optimize
    config: input/optimize/real_strongly_convex_compare/{problem['name']}/gd.yml"""
        )
        steps.append(
            f"""  - command: optimize
    config: input/optimize/real_strongly_convex_compare/{problem['name']}/agd_unknown.yml"""
        )
        for s in SUBSPACE_DIMS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/real_strongly_convex_compare/{problem['name']}/rs_rn_s{s}_seed0.yml"""
            )
        steps.append(
            f"""  - command: plot
    config: input/plot/real_strongly_convex_compare_{problem['name']}_grad_norm.yml"""
        )
        steps.append(
            f"""  - command: plot
    config: input/plot/real_strongly_convex_compare_{problem['name']}_function_value.yml"""
        )

    joined_steps = "\n".join(steps)
    return f"""task: pipeline
pipeline_name: real_strongly_convex_compare

steps:
{joined_steps}
"""


def main() -> None:
    for problem in PROBLEMS:
        problem_dir = OPTIMIZE_ROOT / problem["name"]
        _write(problem_dir / "gd.yml", _gd_config(problem))
        _write(problem_dir / "agd_unknown.yml", _agd_config(problem))

        for s in SUBSPACE_DIMS:
            for T in ARS_T_VALUES:
                _write(problem_dir / f"ars_cn_s{s}_t{T}_seed0.yml", _ars_cn_config(problem, s, T))
            _write(problem_dir / f"rs_cn_s{s}_seed0.yml", _rs_cn_config(problem, s))
            _write(problem_dir / f"rs_rn_s{s}_seed0.yml", _rs_rn_config(problem, s))

        _write(
            PLOT_ROOT / f"real_strongly_convex_compare_{problem['name']}_grad_norm.yml",
            _plot_config(
                problem,
                y_key="grad_norm",
                suffix="grad_norm",
                ylabel="grad_norm",
                yscale="log",
            ),
        )
        _write(
            PLOT_ROOT / f"real_strongly_convex_compare_{problem['name']}_function_value.yml",
            _plot_config(
                problem,
                y_key="f",
                suffix="function_value",
                ylabel="function_value",
                yscale="linear",
            ),
        )

    _write(PIPELINE_ROOT / "real_strongly_convex_compare.yml", _pipeline_config())


if __name__ == "__main__":
    main()

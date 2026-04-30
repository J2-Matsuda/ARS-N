from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZE_ROOT = PROJECT_ROOT / "input" / "optimize" / "logistic_n5000_compare"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot"

DATA_SOURCE = "data/generated/logistic_n5000.npz"
SEEDS = (0, 1, 2, 3, 4)
SUBSPACE_DIMS = (10, 50, 100)
ARS_CONDITIONS = (
    (10, 100),
    (50, 50),
    (50, 100),
    (50, 200),
    (100, 100),
)
T_AUTO_CAPS = {
    10: 100,
    50: 200,
    100: 100,
}
RK_T_AUTO_TOL = 0.1

MAX_ITER = 10000
TOL = "1.0e-5"


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


def _problem_block() -> str:
    return f"""problem:
  type: logistic
  source: {DATA_SOURCE}

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
{indent}  block_size: 256
{indent}  dtype: float64
"""


def _seed_offset(seed: int) -> int:
    return seed * 1_000_003


def _rn_config() -> tuple[str, str]:
    run_name = "rn_logistic_n5000_compare"
    content = f"""task: optimize
run_name: {run_name}
seed: 0

{_problem_block()}
optimizer:
  type: rn
  max_iter: {MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  verbose: true
  print_every: 10
  diag_shift:
    c1: 2.0
    c2: 1.0
    delta: 0.5
{_line_search_block()}
{_optimize_paths(run_name)}"""
    return run_name, content


def _cn_config() -> tuple[str, str]:
    run_name = "cn_logistic_n5000_compare"
    content = f"""task: optimize
run_name: {run_name}
seed: 0

{_problem_block()}
optimizer:
  type: cn
  variant: arc
  solver: exact
  max_iter: {MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  sigma0: 1.0
  sigma_min: 1.0e-8
  sigma_max: 1.0e8
  eta1: 0.05
  eta2: 0.9
  gamma1: 1.5
  gamma2: 2.0
  exact_tol: 1.0e-10
  verbose: true
  print_every: 10

{_optimize_paths(run_name)}"""
    return run_name, content


def _ars_rn_config(s: int, t: int, seed: int) -> tuple[str, str]:
    run_name = f"ars_rn_logistic_n5000_compare_s{s}_t{t}_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block()}
optimizer:
  type: ars_rn
  max_iter: {MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {s}
  verbose: true
  print_every: 10
  diag_shift:
    lambda_factor: 2.0
    grad_factor: 1.0
    grad_exponent: 0.5
    min_eta: 1.0e-8
{_line_search_block()}{_sketch_block()}  rk:
    T: {t}
    r: {s}
    seed_offset: {_seed_offset(seed)}

{_optimize_paths(run_name)}"""
    return run_name, content


def _ars_cn_config(s: int, t: int, seed: int) -> tuple[str, str]:
    run_name = f"ars_cn_logistic_n5000_compare_s{s}_t{t}_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block()}
optimizer:
  type: ars_cn
  max_iter: {MAX_ITER}
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
    T: {t}
    r: {s}
    seed_offset: {_seed_offset(seed)}

{_optimize_paths(run_name)}"""
    return run_name, content


def _ars_rn_tauto_config(s: int, t_cap: int, seed: int) -> tuple[str, str]:
    run_name = f"ars_rn_logistic_n5000_compare_s{s}_tauto_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block()}
optimizer:
  type: ars_rn
  max_iter: {MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {s}
  verbose: true
  print_every: 10
  diag_shift:
    lambda_factor: 2.0
    grad_factor: 1.0
    grad_exponent: 0.5
    min_eta: 1.0e-8
{_line_search_block()}{_sketch_block()}  rk:
    mode: T_auto
    T: {t_cap}
    r: {s}
    rk_tol: {RK_T_AUTO_TOL}
    seed_offset: {_seed_offset(seed)}

{_optimize_paths(run_name)}"""
    return run_name, content


def _ars_cn_tauto_config(s: int, t_cap: int, seed: int) -> tuple[str, str]:
    run_name = f"ars_cn_logistic_n5000_compare_s{s}_tauto_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block()}
optimizer:
  type: ars_cn
  max_iter: {MAX_ITER}
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
    T: {t_cap}
    r: {s}
    rk_tol: {RK_T_AUTO_TOL}
    seed_offset: {_seed_offset(seed)}

{_optimize_paths(run_name)}"""
    return run_name, content


def _rs_rn_config(s: int, seed: int) -> tuple[str, str]:
    run_name = f"rs_rn_logistic_n5000_compare_s{s}_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block()}
optimizer:
  type: rs_rn
  max_iter: {MAX_ITER}
  tol: {TOL}
{_stagnation_block()}  seed: {seed}
  subspace_dim: {s}
  verbose: true
  print_every: 10
{_sketch_block()}  diag_shift:
    c1: 2.0
    c2: 1.0
    gamma: 0.5
{_line_search_block()}
{_optimize_paths(run_name)}"""
    return run_name, content


def _rs_cn_config(s: int, seed: int) -> tuple[str, str]:
    run_name = f"rs_cn_logistic_n5000_compare_s{s}_seed{seed}"
    content = f"""task: optimize
run_name: {run_name}
seed: {seed}

{_problem_block()}
optimizer:
  type: rs_cn
  max_iter: {MAX_ITER}
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
{_optimize_paths(run_name)}"""
    return run_name, content


def _plot_group(method: str, label: str, run_names: list[str], color: str, linestyle: str) -> str:
    paths = "\n".join(f"      - {_result_csv(run_name)}" for run_name in run_names)
    return f"""  - paths:
{paths}
    label: "{label}"
    color: "{color}"
    linestyle: {linestyle}
    aggregate:
      center: mean
      band: minmax
      alpha: 0.16
"""


def _single_plot_input(label: str, run_name: str, color: str, linestyle: str) -> str:
    return f"""  - path: {_result_csv(run_name)}
    label: {label}
    color: "{color}"
    linestyle: {linestyle}
"""


def _plot_config(
    family: str,
    deterministic_run: str,
    groups: list[tuple[str, str, list[str], str, str]],
    deterministic_label: str | None = None,
    title: str | None = None,
) -> str:
    family_upper = family.upper()
    baseline_label = deterministic_label or family_upper
    plot_title = title or f"logistic n5000: {family_upper} comparison"
    inputs = [_single_plot_input(baseline_label, deterministic_run, "#111111", "dashdot")]
    inputs.extend(_plot_group(method, label, run_names, color, linestyle) for method, label, run_names, color, linestyle in groups)
    return f"""task: plot
plot_name: logistic_n5000_compare_{family}

inputs:
{''.join(inputs)}
plot:
  x: iter
  y: grad_norm
  xscale: linear
  yscale: log
  title: "{plot_title}"
  xlabel: iteration
  ylabel: grad_norm
  grid: true

save:
  path: output/plots/logistic_n5000_compare_{family}.png
  dpi: 180
"""


def _pipeline_config(name: str, optimize_configs: list[str], plot_config: str) -> str:
    optimize_steps = "\n".join(
        f"  - command: optimize\n    config: {config}" for config in optimize_configs
    )
    return f"""task: pipeline
pipeline_name: {name}

steps:
{optimize_steps}
  - command: plot
    config: {plot_config}
"""


def main() -> None:
    rn_run, rn_content = _rn_config()
    cn_run, cn_content = _cn_config()
    rn_configs = ["input/optimize/logistic_n5000_compare/rn/rn.yml"]
    cn_configs = ["input/optimize/logistic_n5000_compare/cn/cn.yml"]

    _write(OPTIMIZE_ROOT / "rn" / "rn.yml", rn_content)
    _write(OPTIMIZE_ROOT / "cn" / "cn.yml", cn_content)

    rn_groups: list[tuple[str, str, list[str], str, str]] = []
    cn_groups: list[tuple[str, str, list[str], str, str]] = []

    rs_colors = {10: "#6c757d", 50: "#495057", 100: "#adb5bd"}
    rs_styles = {10: "dashed", 50: "dashdot", 100: "dotted"}
    for s in SUBSPACE_DIMS:
        rs_rn_runs: list[str] = []
        rs_cn_runs: list[str] = []
        for seed in SEEDS:
            rs_rn_run, rs_rn_content = _rs_rn_config(s, seed)
            rs_cn_run, rs_cn_content = _rs_cn_config(s, seed)
            rn_path = OPTIMIZE_ROOT / "rn" / f"rs_rn_s{s}_seed{seed}.yml"
            cn_path = OPTIMIZE_ROOT / "cn" / f"rs_cn_s{s}_seed{seed}.yml"
            _write(rn_path, rs_rn_content)
            _write(cn_path, rs_cn_content)
            rn_configs.append(f"input/optimize/logistic_n5000_compare/rn/{rn_path.name}")
            cn_configs.append(f"input/optimize/logistic_n5000_compare/cn/{cn_path.name}")
            rs_rn_runs.append(rs_rn_run)
            rs_cn_runs.append(rs_cn_run)
        rn_groups.append(("rs_rn", f"RS-RN s={s}", rs_rn_runs, rs_colors[s], rs_styles[s]))
        cn_groups.append(("rs_cn", f"RS-CN s={s}", rs_cn_runs, rs_colors[s], rs_styles[s]))

    ars_colors = {
        (10, 100): "#d9480f",
        (50, 50): "#2b8a3e",
        (50, 100): "#1971c2",
        (50, 200): "#9c36b5",
        (100, 100): "#c92a2a",
    }
    ars_styles = {
        (10, 100): "solid",
        (50, 50): "dashed",
        (50, 100): "solid",
        (50, 200): "dotted",
        (100, 100): "dashdot",
    }
    for s, t in ARS_CONDITIONS:
        ars_rn_runs: list[str] = []
        ars_cn_runs: list[str] = []
        for seed in SEEDS:
            ars_rn_run, ars_rn_content = _ars_rn_config(s, t, seed)
            ars_cn_run, ars_cn_content = _ars_cn_config(s, t, seed)
            rn_path = OPTIMIZE_ROOT / "rn" / f"ars_rn_s{s}_t{t}_seed{seed}.yml"
            cn_path = OPTIMIZE_ROOT / "cn" / f"ars_cn_s{s}_t{t}_seed{seed}.yml"
            _write(rn_path, ars_rn_content)
            _write(cn_path, ars_cn_content)
            rn_configs.append(f"input/optimize/logistic_n5000_compare/rn/{rn_path.name}")
            cn_configs.append(f"input/optimize/logistic_n5000_compare/cn/{cn_path.name}")
            ars_rn_runs.append(ars_rn_run)
            ars_cn_runs.append(ars_cn_run)
        color = ars_colors[(s, t)]
        linestyle = ars_styles[(s, t)]
        rn_groups.append(("ars_rn", f"ARS-RN s=r={s}, T={t}", ars_rn_runs, color, linestyle))
        cn_groups.append(("ars_cn", f"ARS-CN s=r={s}, T={t}", ars_cn_runs, color, linestyle))

    rn_plot = _plot_config("rn", rn_run, rn_groups)
    cn_plot = _plot_config("cn", cn_run, cn_groups)
    _write(PLOT_ROOT / "logistic_n5000_compare_rn.yml", rn_plot)
    _write(PLOT_ROOT / "logistic_n5000_compare_cn.yml", cn_plot)

    _write(
        PIPELINE_ROOT / "logistic_n5000_compare_rn.yml",
        _pipeline_config(
            "logistic_n5000_compare_rn",
            rn_configs,
            "input/plot/logistic_n5000_compare_rn.yml",
        ),
    )
    _write(
        PIPELINE_ROOT / "logistic_n5000_compare_cn.yml",
        _pipeline_config(
            "logistic_n5000_compare_cn",
            cn_configs,
            "input/plot/logistic_n5000_compare_cn.yml",
        ),
    )

    rn_tauto_configs = ["input/optimize/logistic_n5000_compare/rn/rn.yml"]
    cn_tauto_configs = ["input/optimize/logistic_n5000_compare/cn/cn.yml"]
    rn_tauto_groups: list[tuple[str, str, list[str], str, str]] = []
    cn_tauto_groups: list[tuple[str, str, list[str], str, str]] = []

    for s in SUBSPACE_DIMS:
        rs_rn_runs = [f"rs_rn_logistic_n5000_compare_s{s}_seed{seed}" for seed in SEEDS]
        rs_cn_runs = [f"rs_cn_logistic_n5000_compare_s{s}_seed{seed}" for seed in SEEDS]
        rn_tauto_configs.extend(
            f"input/optimize/logistic_n5000_compare/rn/rs_rn_s{s}_seed{seed}.yml"
            for seed in SEEDS
        )
        cn_tauto_configs.extend(
            f"input/optimize/logistic_n5000_compare/cn/rs_cn_s{s}_seed{seed}.yml"
            for seed in SEEDS
        )
        rn_tauto_groups.append(("rs_rn", f"RS-RN s={s}", rs_rn_runs, rs_colors[s], rs_styles[s]))
        cn_tauto_groups.append(("rs_cn", f"RS-CN s={s}", rs_cn_runs, rs_colors[s], rs_styles[s]))

    tauto_colors = {10: "#d9480f", 50: "#1971c2", 100: "#c92a2a"}
    tauto_styles = {10: "solid", 50: "dashed", 100: "dashdot"}
    for s in SUBSPACE_DIMS:
        t_cap = T_AUTO_CAPS[s]
        ars_rn_runs: list[str] = []
        ars_cn_runs: list[str] = []
        for seed in SEEDS:
            ars_rn_run, ars_rn_content = _ars_rn_tauto_config(s, t_cap, seed)
            ars_cn_run, ars_cn_content = _ars_cn_tauto_config(s, t_cap, seed)
            rn_path = OPTIMIZE_ROOT / "rn" / f"ars_rn_s{s}_tauto_seed{seed}.yml"
            cn_path = OPTIMIZE_ROOT / "cn" / f"ars_cn_s{s}_tauto_seed{seed}.yml"
            _write(rn_path, ars_rn_content)
            _write(cn_path, ars_cn_content)
            rn_tauto_configs.append(f"input/optimize/logistic_n5000_compare/rn/{rn_path.name}")
            cn_tauto_configs.append(f"input/optimize/logistic_n5000_compare/cn/{cn_path.name}")
            ars_rn_runs.append(ars_rn_run)
            ars_cn_runs.append(ars_cn_run)
        rn_tauto_groups.append(
            (
                "ars_rn_tauto",
                f"ARS-RN s=r={s}, T_auto (T<={t_cap})",
                ars_rn_runs,
                tauto_colors[s],
                tauto_styles[s],
            )
        )
        cn_tauto_groups.append(
            (
                "ars_cn_tauto",
                f"ARS-CN s=r={s}, T_auto (T<={t_cap})",
                ars_cn_runs,
                tauto_colors[s],
                tauto_styles[s],
            )
        )

    rn_tauto_plot = _plot_config(
        "t_auto_rn",
        rn_run,
        rn_tauto_groups,
        deterministic_label="RN",
        title="logistic n5000: RN comparison (T_auto)",
    )
    cn_tauto_plot = _plot_config(
        "t_auto_cn",
        cn_run,
        cn_tauto_groups,
        deterministic_label="CN",
        title="logistic n5000: CN comparison (T_auto)",
    )
    _write(PLOT_ROOT / "logistic_n5000_compare_t_auto_rn.yml", rn_tauto_plot)
    _write(PLOT_ROOT / "logistic_n5000_compare_t_auto_cn.yml", cn_tauto_plot)

    _write(
        PIPELINE_ROOT / "logistic_n5000_compare_t_auto_rn.yml",
        _pipeline_config(
            "logistic_n5000_compare_t_auto_rn",
            rn_tauto_configs,
            "input/plot/logistic_n5000_compare_t_auto_rn.yml",
        ),
    )
    _write(
        PIPELINE_ROOT / "logistic_n5000_compare_t_auto_cn.yml",
        _pipeline_config(
            "logistic_n5000_compare_t_auto_cn",
            cn_tauto_configs,
            "input/plot/logistic_n5000_compare_t_auto_cn.yml",
        ),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZE_ROOT = PROJECT_ROOT / "input" / "optimize" / "real_logistic_compare"
PIPELINE_PATH = PROJECT_ROOT / "input" / "pipeline" / "real_logistic_compare.yml"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot"

DATASETS = (
    ("gisette", "gisette", "data/generated/real_logistic/gisette.npz"),
    ("epsilon", "epsilon_normalized", "data/generated/real_logistic/epsilon_normalized.npz"),
    ("real_sim", "real_sim", "data/generated/real_logistic/real_sim.npz"),
)
SUBSPACE_DIMS = (10, 50, 100)
ARS_CN_T_VALUES = (50, 100)
SEED = 0

TOL = "1.0e-5"
GD_MAX_ITER = 100000
AGD_MAX_ITER = 100000
RS_MAX_ITER = 10000
ARS_CN_MAX_ITER = 1000


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


def _rs_rn_config(dataset_slug: str, source: str, s: int) -> tuple[str, str]:
    run_name = f"rs_rn_real_logistic_{dataset_slug}_s{s}_seed0"
    content = f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(source)}
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
{_line_search_block()}
{_optimize_paths(run_name)}"""
    return run_name, content


def _rs_cn_config(dataset_slug: str, source: str, s: int) -> tuple[str, str]:
    run_name = f"rs_cn_real_logistic_{dataset_slug}_s{s}_seed0"
    content = f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(source)}
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
{_optimize_paths(run_name)}"""
    return run_name, content


def _ars_cn_config(dataset_slug: str, source: str, s: int, t: int) -> tuple[str, str]:
    run_name = f"ars_cn_real_logistic_{dataset_slug}_s{s}_t{t}_seed0"
    content = f"""task: optimize
run_name: {run_name}
seed: {SEED}

{_problem_block(source)}
optimizer:
  type: ars_cn
  max_iter: {ARS_CN_MAX_ITER}
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
    T: {t}
    r: {s}
    seed_offset: {_seed_offset(SEED)}

{_optimize_paths(run_name)}"""
    return run_name, content


def _plot_input(path: str, label: str, color: str, linestyle: str) -> str:
    return f"""  - path: {path}
    label: "{label}"
    color: "{color}"
    linestyle: {linestyle}
"""


def _plot_config(dataset_label: str, dataset_slug: str) -> str:
    body = ""
    body += _plot_input(
        _result_csv(f"gd_real_logistic_{dataset_slug}"),
        "GD",
        "#111111",
        "solid",
    )
    body += _plot_input(
        _result_csv(f"agd_unknown_real_logistic_{dataset_slug}"),
        "AGD-Unknown",
        "#1c7ed6",
        "dashdot",
    )
    for s, color in zip(SUBSPACE_DIMS, ("#f08c00", "#e8590c", "#d9480f")):
        body += _plot_input(
            _result_csv(f"rs_rn_real_logistic_{dataset_slug}_s{s}_seed0"),
            f"RS-RN s={s}",
            color,
            "dashed",
        )
    for s, color in zip(SUBSPACE_DIMS, ("#868e96", "#495057", "#212529")):
        body += _plot_input(
            _result_csv(f"rs_cn_real_logistic_{dataset_slug}_s{s}_seed0"),
            f"RS-CN s={s}",
            color,
            "dashdot",
        )
    ars_colors = {
        (10, 50): "#74b816",
        (10, 100): "#2b8a3e",
        (50, 50): "#15aabf",
        (50, 100): "#1971c2",
        (100, 50): "#9c36b5",
        (100, 100): "#c92a2a",
    }
    for s in SUBSPACE_DIMS:
        for t in ARS_CN_T_VALUES:
            body += _plot_input(
                _result_csv(f"ars_cn_real_logistic_{dataset_slug}_s{s}_t{t}_seed0"),
                f"ARS-CN s=r={s}, T={t}",
                ars_colors[(s, t)],
                "solid" if t == 100 else "dotted",
            )

    return f"""task: plot
plot_name: real_logistic_compare_{dataset_slug}

inputs:
{body}
plot:
  x: cumulative_time
  y: grad_norm
  xscale: linear
  yscale: log
  title: "real logistic: {dataset_label}"
  xlabel: cumulative_time
  ylabel: grad_norm
  grid: true

save:
  path: output/plots/real_logistic_compare_{dataset_slug}.png
  dpi: 180
"""


def _pipeline_config() -> str:
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
        for s in SUBSPACE_DIMS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/rs_rn_s{s}_seed0.yml"""
            )
        for s in SUBSPACE_DIMS:
            steps.append(
                f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/rs_cn_s{s}_seed0.yml"""
            )
        for s in SUBSPACE_DIMS:
            for t in ARS_CN_T_VALUES:
                steps.append(
                    f"""  - command: optimize
    config: input/optimize/real_logistic_compare/{dataset_slug}/ars_cn_s{s}_t{t}_seed0.yml"""
                )
        steps.append(
            f"""  - command: plot
    config: input/plot/real_logistic_compare_{dataset_slug}.yml"""
        )

    return f"""task: pipeline
pipeline_name: real_logistic_compare

steps:
{chr(10).join(steps)}
"""


def main() -> None:
    for dataset_label, dataset_slug, source in DATASETS:
        dataset_dir = OPTIMIZE_ROOT / dataset_slug

        for filename, config_builder in (
            ("gd.yml", _gd_config),
            ("agd_unknown.yml", _agd_config),
        ):
            _run_name, content = config_builder(dataset_slug, source)
            _write(dataset_dir / filename, content)

        for s in SUBSPACE_DIMS:
            _run_name, content = _rs_rn_config(dataset_slug, source, s)
            _write(dataset_dir / f"rs_rn_s{s}_seed0.yml", content)

            _run_name, content = _rs_cn_config(dataset_slug, source, s)
            _write(dataset_dir / f"rs_cn_s{s}_seed0.yml", content)

        for s in SUBSPACE_DIMS:
            for t in ARS_CN_T_VALUES:
                _run_name, content = _ars_cn_config(dataset_slug, source, s, t)
                _write(dataset_dir / f"ars_cn_s{s}_t{t}_seed0.yml", content)

        _write(PLOT_ROOT / f"real_logistic_compare_{dataset_slug}.yml", _plot_config(dataset_label, dataset_slug))

    _write(PIPELINE_PATH, _pipeline_config())


if __name__ == "__main__":
    main()

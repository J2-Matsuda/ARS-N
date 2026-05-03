from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_yaml
from src.utils.paths import resolve_project_path

PROJECT_ROOT = resolve_project_path(".")
GENERATE_ROOT = PROJECT_ROOT / "input" / "generate_data" / "exp"
OPTIMIZE_ROOT = PROJECT_ROOT / "input" / "optimize" / "exp" / "exp_6_madelon_reg0_mfull"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline" / "exp"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot" / "exp"

SEEDS = (0, 1, 2)
RS_SUBSPACE_DIMS = (100, 200)
ARS_SUBSPACE_DIMS = (100, 200)
ARS_R_VALUES = (50, 100, 200)
ARS_T_VALUES = (50, 100)
TOL = "1.0e-5"
RS_MAX_ITER = 1000000
ARS_MAX_ITER = 100000
AUTO_MAX_PARALLEL = 3


def _result_path(run_name: str) -> str:
    return f"output/results/{run_name}/{run_name}.csv"


def _generate_config() -> dict:
    return {
        "task": "generate_data",
        "run_name": "madelon_reg0_mfull",
        "seed": 0,
        "problem": {
            "type": "logistic",
            "source_format": "libsvm",
            "raw_source": "data/raw/libsvm/madelon",
            "download_if_missing": True,
            "download_url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon",
            "n_features": 500,
            "index_base": 1,
            "reg_lambda": 0.0,
        },
        "save": {
            "path": "data/generated/exp/madelon_reg0_mfull.npz",
        },
    }


def _problem_block() -> dict:
    return {
        "problem": {
            "type": "logistic",
            "source": "data/generated/exp/madelon_reg0_mfull.npz",
        },
        "initialization": {
            "type": "zeros",
        },
    }


def _meta_block(run_name: str) -> dict:
    return {
        "log": {
            "enabled": True,
            "csv_path": _result_path(run_name),
            "save_everytime": True,
        },
        "save_meta": {
            "enabled": True,
            "meta_path": f"output/meta/{run_name}.json",
            "resolved_config_path": f"output/meta/{run_name}.resolved.yml",
        },
    }


def _stagnation_block() -> dict:
    return {
        "stop_on_grad_norm_stagnation": True,
        "grad_norm_stagnation_patience": 50,
        "grad_norm_stagnation_rtol": 1.0e-12,
        "grad_norm_stagnation_atol": 1.0e-12,
    }


def _sketch_block() -> dict:
    return {
        "sketch": {
            "mode": "operator",
            "block_size": 512,
            "dtype": "float64",
        }
    }


def _rs_cn_config(s: int, seed: int) -> dict:
    run_name = f"rs_cn_madelon_reg0_mfull_s{s}_seed{seed}"
    optimizer = {
        "type": "rs_cn",
        "max_iter": RS_MAX_ITER,
        "tol": float(TOL),
        **_stagnation_block(),
        "seed": seed,
        "subspace_dim": s,
        "sigma0": 1.0,
        "sigma_min": 1.0e-8,
        "sigma_max": 1.0e8,
        "eta1": 0.05,
        "eta2": 0.9,
        "gamma1": 1.5,
        "gamma2": 2.0,
        "solver": "lanczos",
        "exact_tol": 1.0e-12,
        "krylov_tol": 1.0e-8,
        "solve_each_i_th_krylov_space": 1,
        "keep_Q_matrix_in_memory": False,
        "verbose": True,
        "print_every": 10,
        **_sketch_block(),
    }
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _ars_cn_config(s: int, r_value: int, t_value: int, seed: int) -> dict:
    run_name = f"ars_cn_madelon_reg0_mfull_s{s}_r{r_value}_t{t_value}_seed{seed}"
    optimizer = {
        "type": "ars_cn",
        "max_iter": ARS_MAX_ITER,
        "tol": float(TOL),
        **_stagnation_block(),
        "seed": seed,
        "subspace_dim": s,
        "sigma0": 1.0,
        "sigma_min": 1.0e-8,
        "sigma_max": 1.0e8,
        "eta1": 0.05,
        "eta2": 0.9,
        "gamma1": 1.5,
        "gamma2": 2.0,
        "solver": "lanczos",
        "exact_tol": 1.0e-10,
        "krylov_tol": 1.0e-8,
        "solve_each_i_th_krylov_space": 1,
        "keep_Q_matrix_in_memory": False,
        "verbose": True,
        "print_every": 10,
        **_sketch_block(),
        "rk": {
            "mode": "default",
            "T": t_value,
            "r": r_value,
            "seed_offset": 0,
        },
    }
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _rs_rn_config(s: int, seed: int) -> dict:
    run_name = f"rs_rn_madelon_reg0_mfull_s{s}_seed{seed}"
    optimizer = {
        "type": "rs_rn",
        "max_iter": RS_MAX_ITER,
        "tol": float(TOL),
        **_stagnation_block(),
        "seed": seed,
        "subspace_dim": s,
        "verbose": True,
        "print_every": 20,
        **_sketch_block(),
        "diag_shift": {
            "c1": 2.0,
            "c2": 1.0,
            "gamma": 0.5,
        },
        "line_search": {
            "enabled": True,
            "alpha0": 1.0,
            "c1": 1.0e-4,
            "beta": 0.5,
            "max_iter": 25,
        },
    }
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _group_paths(prefix: str) -> list[str]:
    return [f"output/results/{prefix}_seed{seed}/{prefix}_seed{seed}.csv" for seed in SEEDS]


def _plot_3seed_config(y_key: str, yscale: str, save_name: str) -> dict:
    ylabel = "function_value" if y_key == "f" else "grad_norm"
    ars_inputs = []
    ars_styles = {
        (100, 50, 50): ("#d9480f", "solid"),
        (100, 50, 100): ("#f08c00", "dashed"),
        (100, 100, 50): ("#1971c2", "solid"),
        (100, 100, 100): ("#1c7ed6", "dashed"),
        (100, 200, 50): ("#0b7285", "solid"),
        (100, 200, 100): ("#1098ad", "dashed"),
        (200, 50, 50): ("#5f3dc4", "solid"),
        (200, 50, 100): ("#7950f2", "dashed"),
        (200, 100, 50): ("#2b8a3e", "solid"),
        (200, 100, 100): ("#37b24d", "dashed"),
        (200, 200, 50): ("#c92a2a", "solid"),
        (200, 200, 100): ("#f03e3e", "dashed"),
    }
    for s in ARS_SUBSPACE_DIMS:
        for r_value in ARS_R_VALUES:
            for t_value in ARS_T_VALUES:
                color, linestyle = ars_styles[(s, r_value, t_value)]
                ars_inputs.append(
                    {
                        "paths": _group_paths(f"ars_cn_madelon_reg0_mfull_s{s}_r{r_value}_t{t_value}"),
                        "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                        "label": f"ARS-CN s={s}, r={r_value}, T={t_value}",
                        "color": color,
                        "linestyle": linestyle,
                    }
                )
    return {
        "task": "plot",
        "plot_name": f"exp_6_madelon_reg0_mfull_{ylabel}",
        "inputs": [
            {
                "paths": _group_paths("rs_cn_madelon_reg0_mfull_s100"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "RS-CN s=100",
                "color": "#495057",
                "linestyle": "dashdot",
            },
            {
                "paths": _group_paths("rs_cn_madelon_reg0_mfull_s200"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "RS-CN s=200",
                "color": "#212529",
                "linestyle": "solid",
            },
            *ars_inputs,
            {
                "paths": _group_paths("rs_rn_madelon_reg0_mfull_s100"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "RS-RN s=100",
                "color": "#fd7e14",
                "linestyle": "dotted",
            },
            {
                "paths": _group_paths("rs_rn_madelon_reg0_mfull_s200"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "RS-RN s=200",
                "color": "#e8590c",
                "linestyle": "dotted",
            },
        ],
        "plot": {
            "x": "cumulative_time",
            "y": y_key,
            "xscale": "linear",
            "yscale": yscale,
            "title": "exp_6: madelon reg=0 m=full",
            "xlabel": "cumulative_time",
            "ylabel": ylabel,
            "grid": True,
            "skip_missing": True,
        },
        "save": {
            "path": f"output/plots/exp/{save_name}",
            "dpi": 180,
        },
    }


def _plot_1seed_config(y_key: str, yscale: str, save_name: str) -> dict:
    ylabel = "function_value" if y_key == "f" else "grad_norm"
    return {
        "task": "plot",
        "plot_name": f"pre_exp_6_madelon_reg0_mfull_{ylabel}",
        "inputs": [
            {
                "path": "output/results/rs_cn_madelon_reg0_mfull_s100_seed0/rs_cn_madelon_reg0_mfull_s100_seed0.csv",
                "label": "RS-CN s=100",
                "color": "#495057",
                "linestyle": "dashdot",
            },
            {
                "path": "output/results/rs_cn_madelon_reg0_mfull_s200_seed0/rs_cn_madelon_reg0_mfull_s200_seed0.csv",
                "label": "RS-CN s=200",
                "color": "#212529",
                "linestyle": "solid",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s100_r50_t50_seed0/ars_cn_madelon_reg0_mfull_s100_r50_t50_seed0.csv",
                "label": "ARS-CN s=100, r=50, T=50",
                "color": "#d9480f",
                "linestyle": "solid",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s100_r50_t100_seed0/ars_cn_madelon_reg0_mfull_s100_r50_t100_seed0.csv",
                "label": "ARS-CN s=100, r=50, T=100",
                "color": "#f08c00",
                "linestyle": "dashed",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s100_r100_t50_seed0/ars_cn_madelon_reg0_mfull_s100_r100_t50_seed0.csv",
                "label": "ARS-CN s=100, r=100, T=50",
                "color": "#1971c2",
                "linestyle": "solid",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s100_r100_t100_seed0/ars_cn_madelon_reg0_mfull_s100_r100_t100_seed0.csv",
                "label": "ARS-CN s=100, r=100, T=100",
                "color": "#1c7ed6",
                "linestyle": "dashed",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s100_r200_t50_seed0/ars_cn_madelon_reg0_mfull_s100_r200_t50_seed0.csv",
                "label": "ARS-CN s=100, r=200, T=50",
                "color": "#0b7285",
                "linestyle": "solid",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s100_r200_t100_seed0/ars_cn_madelon_reg0_mfull_s100_r200_t100_seed0.csv",
                "label": "ARS-CN s=100, r=200, T=100",
                "color": "#1098ad",
                "linestyle": "dashed",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s200_r50_t50_seed0/ars_cn_madelon_reg0_mfull_s200_r50_t50_seed0.csv",
                "label": "ARS-CN s=200, r=50, T=50",
                "color": "#5f3dc4",
                "linestyle": "solid",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s200_r50_t100_seed0/ars_cn_madelon_reg0_mfull_s200_r50_t100_seed0.csv",
                "label": "ARS-CN s=200, r=50, T=100",
                "color": "#7950f2",
                "linestyle": "dashed",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s200_r100_t50_seed0/ars_cn_madelon_reg0_mfull_s200_r100_t50_seed0.csv",
                "label": "ARS-CN s=200, r=100, T=50",
                "color": "#2b8a3e",
                "linestyle": "solid",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s200_r100_t100_seed0/ars_cn_madelon_reg0_mfull_s200_r100_t100_seed0.csv",
                "label": "ARS-CN s=200, r=100, T=100",
                "color": "#37b24d",
                "linestyle": "dashed",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s200_r200_t50_seed0/ars_cn_madelon_reg0_mfull_s200_r200_t50_seed0.csv",
                "label": "ARS-CN s=200, r=200, T=50",
                "color": "#c92a2a",
                "linestyle": "solid",
            },
            {
                "path": "output/results/ars_cn_madelon_reg0_mfull_s200_r200_t100_seed0/ars_cn_madelon_reg0_mfull_s200_r200_t100_seed0.csv",
                "label": "ARS-CN s=200, r=200, T=100",
                "color": "#f03e3e",
                "linestyle": "dashed",
            },
            {
                "path": "output/results/rs_rn_madelon_reg0_mfull_s100_seed0/rs_rn_madelon_reg0_mfull_s100_seed0.csv",
                "label": "RS-RN s=100",
                "color": "#fd7e14",
                "linestyle": "dotted",
            },
            {
                "path": "output/results/rs_rn_madelon_reg0_mfull_s200_seed0/rs_rn_madelon_reg0_mfull_s200_seed0.csv",
                "label": "RS-RN s=200",
                "color": "#e8590c",
                "linestyle": "dotted",
            },
        ],
        "plot": {
            "x": "cumulative_time",
            "y": y_key,
            "xscale": "linear",
            "yscale": yscale,
            "title": "pre_exp_6: madelon reg=0 m=full",
            "xlabel": "cumulative_time",
            "ylabel": ylabel,
            "grid": True,
            "skip_missing": True,
        },
        "save": {
            "path": f"output/plots/exp/{save_name}",
            "dpi": 180,
        },
    }


def _parallel_pipeline() -> dict:
    return {
        "task": "pipeline",
        "pipeline_name": "exp_6",
        "steps": [
            {"command": "generate_data", "config": "input/generate_data/exp/madelon_reg0_mfull.yml"},
            *[
                {
                    "command": "optimize_parallel",
                    "max_parallel": AUTO_MAX_PARALLEL,
                    "configs": [f"input/optimize/exp/exp_6_madelon_reg0_mfull/rs_cn_s{s}_seed{seed}.yml" for seed in SEEDS],
                }
                for s in RS_SUBSPACE_DIMS
            ],
            *[
                {
                    "command": "optimize_parallel",
                    "max_parallel": AUTO_MAX_PARALLEL,
                    "configs": [
                        f"input/optimize/exp/exp_6_madelon_reg0_mfull/ars_cn_s{s}_r{r_value}_t{t_value}_seed{seed}.yml"
                        for seed in SEEDS
                    ],
                }
                for s in ARS_SUBSPACE_DIMS
                for r_value in ARS_R_VALUES
                for t_value in ARS_T_VALUES
            ],
            *[
                {
                    "command": "optimize_parallel",
                    "max_parallel": AUTO_MAX_PARALLEL,
                    "configs": [f"input/optimize/exp/exp_6_madelon_reg0_mfull/rs_rn_s{s}_seed{seed}.yml" for seed in SEEDS],
                }
                for s in RS_SUBSPACE_DIMS
            ],
            {"command": "plot", "config": "input/plot/exp/exp_6_madelon_reg0_mfull_grad_norm.yml"},
            {"command": "plot", "config": "input/plot/exp/exp_6_madelon_reg0_mfull_function_value.yml"},
        ],
    }


def _pre_pipeline() -> dict:
    return {
        "task": "pipeline",
        "pipeline_name": "pre_exp_6",
        "steps": [
            {"command": "generate_data", "config": "input/generate_data/exp/madelon_reg0_mfull.yml"},
            *[
                {"command": "optimize", "config": f"input/optimize/exp/exp_6_madelon_reg0_mfull/rs_cn_s{s}_seed0.yml"}
                for s in RS_SUBSPACE_DIMS
            ],
            *[
                {
                    "command": "optimize",
                    "config": f"input/optimize/exp/exp_6_madelon_reg0_mfull/ars_cn_s{s}_r{r_value}_t{t_value}_seed0.yml",
                }
                for s in ARS_SUBSPACE_DIMS
                for r_value in ARS_R_VALUES
                for t_value in ARS_T_VALUES
            ],
            *[
                {"command": "optimize", "config": f"input/optimize/exp/exp_6_madelon_reg0_mfull/rs_rn_s{s}_seed0.yml"}
                for s in RS_SUBSPACE_DIMS
            ],
            {"command": "plot", "config": "input/plot/exp/pre_exp_6_madelon_reg0_mfull_grad_norm.yml"},
            {"command": "plot", "config": "input/plot/exp/pre_exp_6_madelon_reg0_mfull_function_value.yml"},
        ],
    }


def main() -> None:
    GENERATE_ROOT.mkdir(parents=True, exist_ok=True)
    OPTIMIZE_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    save_yaml(_generate_config(), GENERATE_ROOT / "madelon_reg0_mfull.yml")

    for seed in SEEDS:
        for s in RS_SUBSPACE_DIMS:
            save_yaml(_rs_cn_config(s, seed), OPTIMIZE_ROOT / f"rs_cn_s{s}_seed{seed}.yml")
            save_yaml(_rs_rn_config(s, seed), OPTIMIZE_ROOT / f"rs_rn_s{s}_seed{seed}.yml")
        for s in ARS_SUBSPACE_DIMS:
            for r_value in ARS_R_VALUES:
                for t_value in ARS_T_VALUES:
                    save_yaml(
                        _ars_cn_config(s, r_value, t_value, seed),
                        OPTIMIZE_ROOT / f"ars_cn_s{s}_r{r_value}_t{t_value}_seed{seed}.yml",
                    )

    save_yaml(
        _plot_3seed_config("grad_norm", "log", "exp_6_madelon_reg0_mfull_grad_norm.png"),
        PLOT_ROOT / "exp_6_madelon_reg0_mfull_grad_norm.yml",
    )
    save_yaml(
        _plot_3seed_config("f", "linear", "exp_6_madelon_reg0_mfull_function_value.png"),
        PLOT_ROOT / "exp_6_madelon_reg0_mfull_function_value.yml",
    )
    save_yaml(
        _plot_1seed_config("grad_norm", "log", "pre_exp_6_madelon_reg0_mfull_grad_norm.png"),
        PLOT_ROOT / "pre_exp_6_madelon_reg0_mfull_grad_norm.yml",
    )
    save_yaml(
        _plot_1seed_config("f", "linear", "pre_exp_6_madelon_reg0_mfull_function_value.png"),
        PLOT_ROOT / "pre_exp_6_madelon_reg0_mfull_function_value.yml",
    )

    parallel_pipeline = _parallel_pipeline()
    save_yaml(parallel_pipeline, PIPELINE_ROOT / "exp_6.yml")
    save_yaml(parallel_pipeline, PIPELINE_ROOT / "exp_6_madelon_reg0_mfull.yml")

    pre_pipeline = _pre_pipeline()
    save_yaml(pre_pipeline, PIPELINE_ROOT / "pre_exp_6.yml")
    save_yaml(pre_pipeline, PIPELINE_ROOT / "pre_exp_6_madelon_reg0_mfull.yml")


if __name__ == "__main__":
    main()

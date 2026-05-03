from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_yaml


GENERATE_ROOT = PROJECT_ROOT / "input" / "generate_data" / "exp"
OPTIMIZE_PARENT = PROJECT_ROOT / "input" / "optimize" / "exp"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline" / "exp"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot" / "exp"

SEEDS = (0, 1, 2)
RS_CN_SUBSPACE_DIMS = (100, 200, 500)
ARS_CN_CONFIGS = (
    (100, 100, 100),
    (200, 200, 100),
    (500, 500, 100),
    (500, 50, 50),
)
RS_RN_SUBSPACE_DIMS = (100, 200, 500)
TOL = 1.0e-5
RS_MAX_ITER = 1_000_000
ARS_MAX_ITER = 100_000
GD_MAX_ITER = 100_000
AUTO_MAX_PARALLEL = 3
STEP_MAX_TIME = 10_000

VARIANTS = (
    {
        "dataset_key": "flickr_deepwalk_m20000_lam0",
        "experiment_name": "exp_8_flickr_deepwalk_m20000_lam0",
        "pre_experiment_name": "pre_exp_8_flickr_deepwalk_m20000_lam0",
        "reg_lambda": 0.0,
        "title_suffix": "lambda=0",
    },
    {
        "dataset_key": "flickr_deepwalk_m20000_lam1e-10",
        "experiment_name": "exp_8_flickr_deepwalk_m20000_lam1e-10",
        "pre_experiment_name": "pre_exp_8_flickr_deepwalk_m20000_lam1e-10",
        "reg_lambda": 1.0e-10,
        "title_suffix": "lambda=1e-10",
    },
)


def _write_config(path: Path, config: dict) -> None:
    save_yaml(config, path)


def _result_path(run_name: str) -> str:
    return f"output/results/{run_name}/{run_name}.csv"


def _optimize_root(experiment_name: str) -> Path:
    return OPTIMIZE_PARENT / experiment_name


def _data_source(dataset_key: str) -> str:
    return f"data/generated/exp/{dataset_key}.npz"


def _generate_config(dataset_key: str, reg_lambda: float) -> dict:
    return {
        "task": "generate_data",
        "run_name": dataset_key,
        "seed": 0,
        "problem": {
            "type": "multilabel_logistic",
            "source_format": "multilabel_libsvm",
            "raw_source": "data/raw/multilabel/candidates/flickr_deepwalk.svm.bz2",
            "download_if_missing": True,
            "download_if_corrupt": True,
            "download_url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/flickr_deepwalk.svm.bz2",
            "dataset_name": "flickr-deepwalk",
            "n_features": 128,
            "num_labels": 195,
            "index_base": 1,
            "label_index_base": 1,
            "add_bias": True,
            "reg_lambda": reg_lambda,
            "regularize_bias": True,
            "sample_size": 20000,
            "sample_seed": 0,
        },
        "save": {
            "path": _data_source(dataset_key),
        },
    }


def _problem_block(dataset_key: str) -> dict:
    return {
        "problem": {
            "type": "multilabel_logistic",
            "source": _data_source(dataset_key),
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


def _rs_cn_config(dataset_key: str, run_prefix: str, s: int, seed: int) -> dict:
    run_name = f"rs_cn_{run_prefix}_s{s}_seed{seed}"
    optimizer = {
        "type": "rs_cn",
        "max_iter": RS_MAX_ITER,
        "tol": TOL,
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
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(dataset_key), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _ars_cn_config(dataset_key: str, run_prefix: str, s: int, r: int, t: int, seed: int) -> dict:
    run_name = f"ars_cn_{run_prefix}_s{s}_r{r}_t{t}_seed{seed}"
    rk = {
        "mode": "default",
        "T": t,
        "r": r,
        "seed_offset": seed * 1_000_003,
    }
    optimizer = {
        "type": "ars_cn",
        "max_iter": ARS_MAX_ITER,
        "tol": TOL,
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
        "rk": rk,
    }
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(dataset_key), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _gd_config(dataset_key: str, run_prefix: str, seed: int) -> dict:
    run_name = f"gd_{run_prefix}_seed{seed}"
    optimizer = {
        "type": "gd",
        "max_iter": GD_MAX_ITER,
        "tol": TOL,
        "verbose": True,
        "print_every": 1000,
        "line_search": {
            "enabled": True,
            "alpha0": 1.0,
            "c1": 1.0e-4,
            "beta": 0.5,
            "max_iter": 25,
        },
    }
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(dataset_key), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _rs_rn_config(dataset_key: str, run_prefix: str, s: int, seed: int) -> dict:
    run_name = f"rs_rn_{run_prefix}_s{s}_seed{seed}"
    optimizer = {
        "type": "rs_rn",
        "max_iter": RS_MAX_ITER,
        "tol": TOL,
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
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(dataset_key), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _group_paths(run_prefix: str, prefix: str) -> list[str]:
    return [_result_path(f"{prefix}_{run_prefix}_seed{seed}") for seed in SEEDS]


def _plot_inputs(run_prefix: str) -> list[dict]:
    return [
        {
            "paths": _group_paths(run_prefix, f"rs_cn_{run_prefix}_s100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-CN s=100",
            "color": "#495057",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(run_prefix, f"rs_cn_{run_prefix}_s200"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-CN s=200",
            "color": "#343a40",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(run_prefix, f"rs_cn_{run_prefix}_s500"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-CN s=500",
            "color": "#212529",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(run_prefix, f"ars_cn_{run_prefix}_s100_r100_t100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=r=100, T=100",
            "color": "#1971c2",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(run_prefix, f"ars_cn_{run_prefix}_s200_r200_t100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=r=200, T=100",
            "color": "#1864ab",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(run_prefix, f"ars_cn_{run_prefix}_s500_r500_t100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=r=500, T=100",
            "color": "#d9480f",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(run_prefix, f"ars_cn_{run_prefix}_s500_r50_t50"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=500, r=50, T=50",
            "color": "#f76707",
            "linestyle": "dashed",
        },
        {
            "paths": _group_paths(run_prefix, f"gd_{run_prefix}"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "GD",
            "color": "#111111",
            "linestyle": "dotted",
        },
        {
            "paths": _group_paths(run_prefix, f"rs_rn_{run_prefix}_s100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-RN s=100",
            "color": "#2b8a3e",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(run_prefix, f"rs_rn_{run_prefix}_s200"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-RN s=200",
            "color": "#2f9e44",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(run_prefix, f"rs_rn_{run_prefix}_s500"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-RN s=500",
            "color": "#37b24d",
            "linestyle": "solid",
        },
    ]


def _plot_config(experiment_name: str, title_suffix: str, y_key: str, yscale: str, save_name: str) -> dict:
    ylabel = "function_value" if y_key == "f" else "grad_norm"
    run_prefix = experiment_name.removeprefix("exp_8_")
    return {
        "task": "plot",
        "plot_name": f"{experiment_name}_{ylabel}",
        "inputs": _plot_inputs(run_prefix),
        "plot": {
            "x": "cumulative_time",
            "y": y_key,
            "xscale": "linear",
            "yscale": yscale,
            "title": f"exp_8: flickr-deepwalk multilabel logistic m=20000 {title_suffix}",
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


def _pre_plot_config(pre_experiment_name: str, title_suffix: str, y_key: str, yscale: str, save_name: str) -> dict:
    ylabel = "function_value" if y_key == "f" else "grad_norm"
    run_prefix = pre_experiment_name.removeprefix("pre_exp_8_")
    inputs = [
        {"path": _result_path(f"gd_{run_prefix}_seed0"), "label": "GD", "color": "#111111", "linestyle": "solid"},
        {"path": _result_path(f"rs_cn_{run_prefix}_s100_seed0"), "label": "RS-CN s=100", "color": "#495057", "linestyle": "dashdot"},
        {"path": _result_path(f"rs_cn_{run_prefix}_s200_seed0"), "label": "RS-CN s=200", "color": "#343a40", "linestyle": "dashdot"},
        {"path": _result_path(f"rs_cn_{run_prefix}_s500_seed0"), "label": "RS-CN s=500", "color": "#212529", "linestyle": "solid"},
        {"path": _result_path(f"ars_cn_{run_prefix}_s100_r100_t100_seed0"), "label": "ARS-CN s=100, r=100, T=100", "color": "#1971c2", "linestyle": "solid"},
        {"path": _result_path(f"ars_cn_{run_prefix}_s200_r200_t100_seed0"), "label": "ARS-CN s=200, r=200, T=100", "color": "#1864ab", "linestyle": "solid"},
        {"path": _result_path(f"ars_cn_{run_prefix}_s500_r500_t100_seed0"), "label": "ARS-CN s=500, r=500, T=100", "color": "#d9480f", "linestyle": "solid"},
        {"path": _result_path(f"ars_cn_{run_prefix}_s500_r50_t50_seed0"), "label": "ARS-CN s=500, r=50, T=50", "color": "#f76707", "linestyle": "dashed"},
        {"path": _result_path(f"rs_rn_{run_prefix}_s100_seed0"), "label": "RS-RN s=100", "color": "#2b8a3e", "linestyle": "dotted"},
        {"path": _result_path(f"rs_rn_{run_prefix}_s200_seed0"), "label": "RS-RN s=200", "color": "#2f9e44", "linestyle": "dotted"},
        {"path": _result_path(f"rs_rn_{run_prefix}_s500_seed0"), "label": "RS-RN s=500", "color": "#37b24d", "linestyle": "dotted"},
    ]
    return {
        "task": "plot",
        "plot_name": f"{pre_experiment_name}_{ylabel}",
        "inputs": inputs,
        "plot": {
            "x": "cumulative_time",
            "y": y_key,
            "xscale": "linear",
            "yscale": yscale,
            "title": f"pre_exp_8: flickr deepwalk m=20000 {title_suffix}",
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


def _parallel_step(configs: list[str], max_parallel: int = AUTO_MAX_PARALLEL) -> dict:
    return {
        "command": "optimize_parallel",
        "max_parallel": max_parallel,
        "max_time": STEP_MAX_TIME,
        "configs": configs,
    }


def _named_pipeline(dataset_key: str, experiment_name: str) -> dict:
    steps: list[dict] = [
        {
            "command": "generate_data",
            "config": f"input/generate_data/exp/{dataset_key}.yml",
        }
    ]
    for s in RS_CN_SUBSPACE_DIMS:
        steps.append(
            _parallel_step([f"input/optimize/exp/{experiment_name}/rs_cn_s{s}_seed{seed}.yml" for seed in SEEDS])
        )
    for s, r, t in ARS_CN_CONFIGS:
        steps.append(
            _parallel_step([f"input/optimize/exp/{experiment_name}/ars_cn_s{s}_r{r}_t{t}_seed{seed}.yml" for seed in SEEDS])
        )
    steps.append(_parallel_step([f"input/optimize/exp/{experiment_name}/gd_seed{seed}.yml" for seed in SEEDS]))
    for s in RS_RN_SUBSPACE_DIMS:
        steps.append(
            _parallel_step([f"input/optimize/exp/{experiment_name}/rs_rn_s{s}_seed{seed}.yml" for seed in SEEDS])
        )
    steps.extend(
        [
            {"command": "plot", "config": f"input/plot/exp/{experiment_name}_grad_norm.yml"},
            {"command": "plot", "config": f"input/plot/exp/{experiment_name}_function_value.yml"},
        ]
    )
    return {"task": "pipeline", "pipeline_name": experiment_name, "steps": steps}


def _named_pre_pipeline(dataset_key: str, experiment_name: str, pre_experiment_name: str) -> dict:
    steps: list[dict] = [
        {
            "command": "generate_data",
            "config": f"input/generate_data/exp/{dataset_key}.yml",
        }
    ]
    for s in RS_CN_SUBSPACE_DIMS:
        steps.append(
            {
                "command": "optimize",
                "max_time": STEP_MAX_TIME,
                "config": f"input/optimize/exp/{experiment_name}/rs_cn_s{s}_seed0.yml",
            }
        )
    for s, r, t in ARS_CN_CONFIGS:
        steps.append(
            {
                "command": "optimize",
                "max_time": STEP_MAX_TIME,
                "config": f"input/optimize/exp/{experiment_name}/ars_cn_s{s}_r{r}_t{t}_seed0.yml",
            }
        )
    steps.append(
        {
            "command": "optimize",
            "max_time": STEP_MAX_TIME,
            "config": f"input/optimize/exp/{experiment_name}/gd_seed0.yml",
        }
    )
    for s in RS_RN_SUBSPACE_DIMS:
        steps.append(
            {
                "command": "optimize",
                "max_time": STEP_MAX_TIME,
                "config": f"input/optimize/exp/{experiment_name}/rs_rn_s{s}_seed0.yml",
            }
        )
    steps.extend(
        [
            {"command": "plot", "config": f"input/plot/exp/{pre_experiment_name}_grad_norm.yml"},
            {"command": "plot", "config": f"input/plot/exp/{pre_experiment_name}_function_value.yml"},
        ]
    )
    return {"task": "pipeline", "pipeline_name": pre_experiment_name, "steps": steps}


def _append_named_steps(target_steps: list[dict], pipeline: dict) -> None:
    target_steps.extend(pipeline["steps"])


def main() -> None:
    GENERATE_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    exp_8_steps: list[dict] = []
    pre_exp_8_steps: list[dict] = []

    for variant in VARIANTS:
        dataset_key = str(variant["dataset_key"])
        experiment_name = str(variant["experiment_name"])
        pre_experiment_name = str(variant["pre_experiment_name"])
        reg_lambda = float(variant["reg_lambda"])
        title_suffix = str(variant["title_suffix"])
        run_prefix = experiment_name.removeprefix("exp_8_")
        optimize_root = _optimize_root(experiment_name)
        optimize_root.mkdir(parents=True, exist_ok=True)

        _write_config(GENERATE_ROOT / f"{dataset_key}.yml", _generate_config(dataset_key, reg_lambda))

        for seed in SEEDS:
            for s in RS_CN_SUBSPACE_DIMS:
                _write_config(optimize_root / f"rs_cn_s{s}_seed{seed}.yml", _rs_cn_config(dataset_key, run_prefix, s, seed))
            for s, r, t in ARS_CN_CONFIGS:
                _write_config(
                    optimize_root / f"ars_cn_s{s}_r{r}_t{t}_seed{seed}.yml",
                    _ars_cn_config(dataset_key, run_prefix, s, r, t, seed),
                )
            _write_config(optimize_root / f"gd_seed{seed}.yml", _gd_config(dataset_key, run_prefix, seed))
            for s in RS_RN_SUBSPACE_DIMS:
                _write_config(optimize_root / f"rs_rn_s{s}_seed{seed}.yml", _rs_rn_config(dataset_key, run_prefix, s, seed))

        _write_config(
            PLOT_ROOT / f"{experiment_name}_grad_norm.yml",
            _plot_config(experiment_name, title_suffix, "grad_norm", "log", f"{experiment_name}_grad_norm.png"),
        )
        _write_config(
            PLOT_ROOT / f"{experiment_name}_function_value.yml",
            _plot_config(experiment_name, title_suffix, "f", "linear", f"{experiment_name}_function_value.png"),
        )
        _write_config(
            PLOT_ROOT / f"{pre_experiment_name}_grad_norm.yml",
            _pre_plot_config(pre_experiment_name, title_suffix, "grad_norm", "log", f"{pre_experiment_name}_grad_norm.png"),
        )
        _write_config(
            PLOT_ROOT / f"{pre_experiment_name}_function_value.yml",
            _pre_plot_config(pre_experiment_name, title_suffix, "f", "linear", f"{pre_experiment_name}_function_value.png"),
        )

        named_pipeline = _named_pipeline(dataset_key, experiment_name)
        named_pre_pipeline = _named_pre_pipeline(dataset_key, experiment_name, pre_experiment_name)

        _write_config(PIPELINE_ROOT / f"{experiment_name}.yml", named_pipeline)
        _write_config(PIPELINE_ROOT / f"{pre_experiment_name}.yml", named_pre_pipeline)

        _append_named_steps(exp_8_steps, named_pipeline)
        _append_named_steps(pre_exp_8_steps, named_pre_pipeline)

    _write_config(PIPELINE_ROOT / "exp_8.yml", {"task": "pipeline", "pipeline_name": "exp_8", "steps": exp_8_steps})
    _write_config(
        PIPELINE_ROOT / "pre_exp_8.yml",
        {"task": "pipeline", "pipeline_name": "pre_exp_8", "steps": pre_exp_8_steps},
    )


if __name__ == "__main__":
    main()

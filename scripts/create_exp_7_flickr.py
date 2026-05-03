from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_yaml


GENERATE_ROOT = PROJECT_ROOT / "input" / "generate_data" / "exp"
OPTIMIZE_ROOT = PROJECT_ROOT / "input" / "optimize" / "exp" / "exp_7_flickr_deepwalk_m20000_lam1e-3"
PIPELINE_ROOT = PROJECT_ROOT / "input" / "pipeline" / "exp"
PLOT_ROOT = PROJECT_ROOT / "input" / "plot" / "exp"

DATASET_KEY = "flickr_deepwalk_m20000_lam1e-3"
EXPERIMENT_NAME = "exp_7_flickr_deepwalk_m20000_lam1e-3"
DATA_SOURCE = "data/generated/exp/flickr_deepwalk_m20000_lam1e-3.npz"
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


def _write_config(path: Path, config: dict) -> None:
    save_yaml(config, path)


def _result_path(run_name: str) -> str:
    return f"output/results/{run_name}/{run_name}.csv"


def _generate_config() -> dict:
    return {
        "task": "generate_data",
        "run_name": DATASET_KEY,
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
            "reg_lambda": 1.0e-3,
            "regularize_bias": True,
            "sample_size": 20000,
            "sample_seed": 0,
        },
        "save": {
            "path": DATA_SOURCE,
        },
    }


def _problem_block() -> dict:
    return {
        "problem": {
            "type": "multilabel_logistic",
            "source": DATA_SOURCE,
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
    run_name = f"rs_cn_{DATASET_KEY}_s{s}_seed{seed}"
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
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _ars_cn_config(s: int, r: int, t: int, seed: int) -> dict:
    run_name = f"ars_cn_{DATASET_KEY}_s{s}_r{r}_t{t}_seed{seed}"
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
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _gd_config(seed: int) -> dict:
    run_name = f"gd_{DATASET_KEY}_seed{seed}"
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
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _rs_rn_config(s: int, seed: int) -> dict:
    run_name = f"rs_rn_{DATASET_KEY}_s{s}_seed{seed}"
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
    cfg = {"task": "optimize", "run_name": run_name, "seed": seed, **_problem_block(), "optimizer": optimizer}
    cfg.update(_meta_block(run_name))
    return cfg


def _group_paths(prefix: str) -> list[str]:
    return [_result_path(f"{prefix}_seed{seed}") for seed in SEEDS]


def _plot_inputs() -> list[dict]:
    return [
        {
            "paths": _group_paths(f"rs_cn_{DATASET_KEY}_s100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-CN s=100",
            "color": "#495057",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(f"rs_cn_{DATASET_KEY}_s200"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-CN s=200",
            "color": "#343a40",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(f"rs_cn_{DATASET_KEY}_s500"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-CN s=500",
            "color": "#212529",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(f"ars_cn_{DATASET_KEY}_s100_r100_t100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=r=100, T=100",
            "color": "#1971c2",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(f"ars_cn_{DATASET_KEY}_s200_r200_t100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=r=200, T=100",
            "color": "#1864ab",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(f"ars_cn_{DATASET_KEY}_s500_r500_t100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=r=500, T=100",
            "color": "#d9480f",
            "linestyle": "solid",
        },
        {
            "paths": _group_paths(f"ars_cn_{DATASET_KEY}_s500_r50_t50"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "ARS-CN s=500, r=50, T=50",
            "color": "#f76707",
            "linestyle": "dashed",
        },
        {
            "paths": _group_paths(f"gd_{DATASET_KEY}"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "GD",
            "color": "#111111",
            "linestyle": "dotted",
        },
        {
            "paths": _group_paths(f"rs_rn_{DATASET_KEY}_s100"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-RN s=100",
            "color": "#2b8a3e",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(f"rs_rn_{DATASET_KEY}_s200"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-RN s=200",
            "color": "#2f9e44",
            "linestyle": "dashdot",
        },
        {
            "paths": _group_paths(f"rs_rn_{DATASET_KEY}_s500"),
            "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
            "label": "RS-RN s=500",
            "color": "#37b24d",
            "linestyle": "solid",
        },
    ]


def _plot_config(y_key: str, yscale: str, save_name: str) -> dict:
    ylabel = "function_value" if y_key == "f" else "grad_norm"
    return {
        "task": "plot",
        "plot_name": f"{EXPERIMENT_NAME}_{ylabel}",
        "inputs": _plot_inputs(),
        "plot": {
            "x": "cumulative_time",
            "y": y_key,
            "xscale": "linear",
            "yscale": yscale,
            "title": "exp_7: flickr-deepwalk multilabel logistic m=20000 lambda=1e-3",
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


def _pipeline_config() -> dict:
    steps: list[dict] = [
        {
            "command": "generate_data",
            "config": f"input/generate_data/exp/{DATASET_KEY}.yml",
        }
    ]

    for s in RS_CN_SUBSPACE_DIMS:
        steps.append(
            _parallel_step(
                [f"input/optimize/exp/{EXPERIMENT_NAME}/rs_cn_s{s}_seed{seed}.yml" for seed in SEEDS]
            )
        )

    for s, r, t in ARS_CN_CONFIGS:
        steps.append(
            _parallel_step(
                [f"input/optimize/exp/{EXPERIMENT_NAME}/ars_cn_s{s}_r{r}_t{t}_seed{seed}.yml" for seed in SEEDS]
            )
        )

    steps.append(
        _parallel_step([f"input/optimize/exp/{EXPERIMENT_NAME}/gd_seed{seed}.yml" for seed in SEEDS])
    )

    for s in RS_RN_SUBSPACE_DIMS:
        steps.append(
            _parallel_step(
                [f"input/optimize/exp/{EXPERIMENT_NAME}/rs_rn_s{s}_seed{seed}.yml" for seed in SEEDS]
            )
        )

    steps.extend(
        [
            {
                "command": "plot",
                "config": f"input/plot/exp/{EXPERIMENT_NAME}_grad_norm.yml",
            },
            {
                "command": "plot",
                "config": f"input/plot/exp/{EXPERIMENT_NAME}_function_value.yml",
            },
        ]
    )
    return {"task": "pipeline", "pipeline_name": "exp_7", "steps": steps}


def main() -> None:
    GENERATE_ROOT.mkdir(parents=True, exist_ok=True)
    OPTIMIZE_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    _write_config(GENERATE_ROOT / f"{DATASET_KEY}.yml", _generate_config())

    for seed in SEEDS:
        for s in RS_CN_SUBSPACE_DIMS:
            _write_config(OPTIMIZE_ROOT / f"rs_cn_s{s}_seed{seed}.yml", _rs_cn_config(s, seed))
        for s, r, t in ARS_CN_CONFIGS:
            _write_config(
                OPTIMIZE_ROOT / f"ars_cn_s{s}_r{r}_t{t}_seed{seed}.yml",
                _ars_cn_config(s, r, t, seed),
            )
        _write_config(OPTIMIZE_ROOT / f"gd_seed{seed}.yml", _gd_config(seed))
        for s in RS_RN_SUBSPACE_DIMS:
            _write_config(OPTIMIZE_ROOT / f"rs_rn_s{s}_seed{seed}.yml", _rs_rn_config(s, seed))

    _write_config(
        PLOT_ROOT / f"{EXPERIMENT_NAME}_grad_norm.yml",
        _plot_config("grad_norm", "log", f"{EXPERIMENT_NAME}_grad_norm.png"),
    )
    _write_config(
        PLOT_ROOT / f"{EXPERIMENT_NAME}_function_value.yml",
        _plot_config("f", "linear", f"{EXPERIMENT_NAME}_function_value.png"),
    )

    pipeline = _pipeline_config()
    _write_config(PIPELINE_ROOT / "exp_7.yml", pipeline)
    _write_config(PIPELINE_ROOT / f"{EXPERIMENT_NAME}.yml", pipeline)


if __name__ == "__main__":
    main()

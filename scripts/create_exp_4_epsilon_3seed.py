from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml, save_yaml
from src.utils.paths import resolve_project_path

PROJECT_ROOT = resolve_project_path(".")
SOURCE_DIR = PROJECT_ROOT / "input" / "optimize" / "real_noreg_convex_compare" / "epsilon"
TARGET_OPT_DIR = PROJECT_ROOT / "input" / "optimize" / "exp" / "exp_4_epsilon_3seed"
TARGET_PLOT_DIR = PROJECT_ROOT / "input" / "plot" / "exp"
TARGET_PIPELINE_DIR = PROJECT_ROOT / "input" / "pipeline" / "exp"

SEEDS = (0, 1, 2)


def _load_yaml(path: Path) -> dict:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _exp_run_name(base: str, seed: int | None = None) -> str:
    if seed is None:
        return f"{base}_exp_4_epsilon_3seed"
    return f"{base}_exp_4_epsilon_3seed_seed{seed}"


def _rewrite_optimize_config(
    source_name: str,
    target_name: str,
    run_prefix: str,
    seed: int | None,
) -> None:
    source_path = SOURCE_DIR / source_name
    config = _load_yaml(source_path)
    run_name = _exp_run_name(run_prefix, seed)
    config["run_name"] = run_name
    if seed is not None:
        config["seed"] = seed
        optimizer = dict(config.get("optimizer", {}))
        optimizer["seed"] = seed
        config["optimizer"] = optimizer

    log = dict(config["log"])
    log["csv_path"] = f"output/results/{run_name}.csv"
    config["log"] = log

    save_meta = dict(config["save_meta"])
    save_meta["meta_path"] = f"output/meta/{run_name}.json"
    save_meta["resolved_config_path"] = f"output/meta/{run_name}.resolved.yml"
    config["save_meta"] = save_meta

    save_yaml(config, TARGET_OPT_DIR / target_name)


def _group_paths(prefix: str) -> list[str]:
    return [
        f"output/results/{_exp_run_name(prefix, seed)}/{_exp_run_name(prefix, seed)}.csv"
        for seed in SEEDS
    ]


def _plot_config(y_key: str, yscale: str, save_name: str) -> dict:
    ylabel = "function_value" if y_key == "f" else "grad_norm"
    return {
        "task": "plot",
        "plot_name": f"exp_4_epsilon_3seed_{ylabel}",
        "inputs": [
            {
                "path": f"output/results/{_exp_run_name('gd')}/{_exp_run_name('gd')}.csv",
                "label": "GD",
                "color": "#111111",
                "linestyle": "solid",
            },
            {
                "paths": _group_paths("rs_cn_s100"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "RS-CN s=100",
                "color": "#212529",
                "linestyle": "dashdot",
            },
            {
                "paths": _group_paths("rs_cn_s200"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "RS-CN s=200",
                "color": "#495057",
                "linestyle": "dashdot",
            },
            {
                "paths": _group_paths("ars_cn_s100_t100"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "ARS-CN s=r=100, T=100",
                "color": "#1971c2",
                "linestyle": "solid",
            },
            {
                "paths": _group_paths("ars_cn_s200_t100"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "ARS-CN s=r=200, T=100",
                "color": "#0b7285",
                "linestyle": "solid",
            },
            {
                "paths": _group_paths("rs_rn_s100"),
                "aggregate": {"center": "mean", "band": "minmax", "alpha": 0.16},
                "label": "RS-RN s=100",
                "color": "#d9480f",
                "linestyle": "dotted",
            },
            {
                "paths": _group_paths("rs_rn_s200"),
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
            "title": "exp_4: epsilon reg=0 m=10000 (3 seeds)",
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


def _pipeline_config() -> dict:
    return {
        "task": "pipeline",
        "pipeline_name": "exp_4",
        "steps": [
            {
                "command": "generate_data",
                "config": "input/generate_data/real_noreg_convex/epsilon.yml",
            },
            {
                "command": "optimize",
                "config": "input/optimize/exp/exp_4_epsilon_3seed/gd.yml",
                "max_time": 6000,
            },
            {
                "command": "optimize_parallel",
                "max_parallel": 3,
                "max_time": 6000,
                "configs": [
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_cn_s100_seed0.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_cn_s100_seed1.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_cn_s100_seed2.yml",
                ],
            },
            {
                "command": "optimize_parallel",
                "max_parallel": 3,
                "max_time": 6000,
                "configs": [
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_cn_s200_seed0.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_cn_s200_seed1.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_cn_s200_seed2.yml",
                ],
            },
            {
                "command": "optimize_parallel",
                "max_parallel": 3,
                "max_time": 6000,
                "configs": [
                    "input/optimize/exp/exp_4_epsilon_3seed/ars_cn_s100_t100_seed0.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/ars_cn_s100_t100_seed1.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/ars_cn_s100_t100_seed2.yml",
                ],
            },
            {
                "command": "optimize_parallel",
                "max_parallel": 3,
                "max_time": 6000,
                "configs": [
                    "input/optimize/exp/exp_4_epsilon_3seed/ars_cn_s200_t100_seed0.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/ars_cn_s200_t100_seed1.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/ars_cn_s200_t100_seed2.yml",
                ],
            },
            {
                "command": "optimize_parallel",
                "max_parallel": 3,
                "max_time": 6000,
                "configs": [
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_rn_s100_seed0.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_rn_s100_seed1.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_rn_s100_seed2.yml",
                ],
            },
            {
                "command": "optimize_parallel",
                "max_parallel": 3,
                "max_time": 6000,
                "configs": [
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_rn_s200_seed0.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_rn_s200_seed1.yml",
                    "input/optimize/exp/exp_4_epsilon_3seed/rs_rn_s200_seed2.yml",
                ],
            },
            {
                "command": "plot",
                "config": "input/plot/exp/exp_4_epsilon_3seed_grad_norm.yml",
            },
            {
                "command": "plot",
                "config": "input/plot/exp/exp_4_epsilon_3seed_function_value.yml",
            },
        ],
    }


def main() -> None:
    TARGET_OPT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    _rewrite_optimize_config("gd.yml", "gd.yml", "gd", seed=None)
    for seed in SEEDS:
        _rewrite_optimize_config(
            "rs_cn_s100_seed0.yml", f"rs_cn_s100_seed{seed}.yml", "rs_cn_s100", seed=seed
        )
        _rewrite_optimize_config(
            "rs_cn_s200_seed0.yml", f"rs_cn_s200_seed{seed}.yml", "rs_cn_s200", seed=seed
        )
        _rewrite_optimize_config(
            "ars_cn_s100_t100_seed0.yml",
            f"ars_cn_s100_t100_seed{seed}.yml",
            "ars_cn_s100_t100",
            seed=seed,
        )
        _rewrite_optimize_config(
            "ars_cn_s200_t100_seed0.yml",
            f"ars_cn_s200_t100_seed{seed}.yml",
            "ars_cn_s200_t100",
            seed=seed,
        )
        _rewrite_optimize_config(
            "rs_rn_s100_seed0.yml", f"rs_rn_s100_seed{seed}.yml", "rs_rn_s100", seed=seed
        )
        _rewrite_optimize_config(
            "rs_rn_s200_seed0.yml", f"rs_rn_s200_seed{seed}.yml", "rs_rn_s200", seed=seed
        )

    save_yaml(
        _plot_config("grad_norm", "log", "exp_4_epsilon_3seed_grad_norm.png"),
        TARGET_PLOT_DIR / "exp_4_epsilon_3seed_grad_norm.yml",
    )
    save_yaml(
        _plot_config("f", "linear", "exp_4_epsilon_3seed_function_value.png"),
        TARGET_PLOT_DIR / "exp_4_epsilon_3seed_function_value.yml",
    )

    pipeline = _pipeline_config()
    save_yaml(pipeline, TARGET_PIPELINE_DIR / "exp_4.yml")
    save_yaml(pipeline, TARGET_PIPELINE_DIR / "exp_4_epsilon_3seed.yml")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    load_generate_config,
    load_optimize_config,
    load_pipeline_config,
    load_plot_config,
    save_resolved_config,
)
from src.problems.real_classification import clone_problem_data_with_reg_lambda
from src.registry import build_problem, generate_problem_data, get_optimizer
from src.utils.io import load_npz, save_json, save_npz
from src.utils.paths import ensure_standard_directories, resolve_project_path
from src.utils.run_logger import NullRunLogger, RunLogger
from src.utils.seed import set_global_seed
from src.utils.timer import utc_now_iso


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Numerical optimization experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("generate", "generate_data", "optimize", "plot", "pipeline"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, help="Path to YAML config")

    return parser


def _make_initial_point(initialization_config: dict[str, Any], dim: int, seed: int) -> np.ndarray:
    init_type = initialization_config.get("type", "zeros")
    if init_type == "zeros":
        return np.zeros(dim, dtype=float)
    if init_type == "random_normal":
        scale = float(initialization_config.get("scale", 1.0))
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=scale, size=dim)
    raise ValueError(f"Unknown initialization type {init_type!r}")


def _maybe_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_generate(config_path: str) -> Path:
    config = load_generate_config(config_path)
    ensure_standard_directories()
    save_path = resolve_project_path(config["save"]["path"])
    task = str(config["task"])
    if task == "clone_data_with_reg_lambda":
        clone_problem_data_with_reg_lambda(
            source_path=str(config["source"]["path"]),
            save_path=str(save_path),
            reg_lambda=float(config["reg_lambda"]),
        )
        _annotate_generated_npz(
            save_path,
            run_name=str(config["run_name"]),
            task=task,
            expected=config.get("expected"),
        )
        print(f"Cloned data with reg_lambda={float(config['reg_lambda'])}: {save_path}")
        return save_path

    seed = int(config.get("seed", 0))
    set_global_seed(seed)
    generate_problem_data(config["problem"], str(save_path), seed)
    _annotate_generated_npz(
        save_path,
        run_name=str(config["run_name"]),
        task=task,
        expected=config.get("expected"),
    )
    print(f"Generated data: {save_path}")
    return save_path


def _project_relative_path(path: Path) -> str:
    project_root = resolve_project_path(".")
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _annotate_generated_npz(
    path: Path,
    *,
    run_name: str,
    task: str,
    expected: dict[str, Any] | None,
) -> None:
    arrays = load_npz(path)
    raw = np.asarray(arrays.get("generation_config_json", np.asarray("{}")))
    text = str(raw.item()) if raw.shape == () else str(raw.reshape(-1)[0])
    metadata = json.loads(text)
    metadata["run_name"] = run_name
    metadata["task"] = task
    if expected is not None:
        metadata["expected"] = expected
    arrays["generation_config_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    save_npz(path, **arrays)


def _prepare_optimize_output_paths(
    config: dict[str, Any],
    config_path: str,
) -> dict[str, Path]:
    run_name = str(config["run_name"])
    run_dir = resolve_project_path(Path("output") / "results" / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = resolve_project_path(config_path)
    input_config_copy_path = run_dir / source_config_path.name

    csv_name = Path(str(config["log"]["csv_path"])).name
    meta_name = Path(str(config["save_meta"]["meta_path"])).name
    resolved_name = Path(str(config["save_meta"]["resolved_config_path"])).name

    csv_path = run_dir / csv_name
    meta_path = run_dir / meta_name
    resolved_config_path = run_dir / resolved_name

    config["log"]["csv_path"] = _project_relative_path(csv_path)
    config["save_meta"]["meta_path"] = _project_relative_path(meta_path)
    config["save_meta"]["resolved_config_path"] = _project_relative_path(resolved_config_path)

    return {
        "run_dir": run_dir,
        "source_config_path": source_config_path,
        "input_config_copy_path": input_config_copy_path,
        "csv_path": csv_path,
        "meta_path": meta_path,
        "resolved_config_path": resolved_config_path,
    }


def _run_optimize(config_path: str) -> dict[str, Any]:
    config = load_optimize_config(config_path)
    ensure_standard_directories()
    output_paths = _prepare_optimize_output_paths(config, config_path)
    shutil.copy2(output_paths["source_config_path"], output_paths["input_config_copy_path"])
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    built_problem = build_problem(config["problem"])
    x0 = _make_initial_point(config["initialization"], built_problem.dim, seed)
    effective_reg_lambda = float(getattr(built_problem.problem, "reg_lambda", 0.0))
    effective_regularize_bias = bool(getattr(built_problem.problem, "regularize_bias", True))
    source_reg_lambda: float | None = None
    reg_lambda_overridden = False
    if "source" in config["problem"]:
        source_arrays = load_npz(resolve_project_path(config["problem"]["source"]))
        source_reg_lambda = float(np.asarray(source_arrays.get("reg_lambda", np.array(0.0))).item())
        if "reg_lambda" in config["problem"]:
            reg_lambda_overridden = True
            print(
                f"[Optimize:{config['run_name']}] reg_lambda override: "
                f"config.problem.reg_lambda={float(config['problem']['reg_lambda'])} "
                f"replaces source reg_lambda={source_reg_lambda}"
            )
        else:
            print(f"[Optimize:{config['run_name']}] reg_lambda from source npz: {source_reg_lambda}")
    else:
        print(f"[Optimize:{config['run_name']}] reg_lambda from config.problem: {effective_reg_lambda}")
    print(
        f"[Optimize:{config['run_name']}] effective_reg_lambda={effective_reg_lambda}, "
        f"regularize_bias={effective_regularize_bias}"
    )
    problem_obj = built_problem.problem
    dataset_path = config["problem"].get("source")
    problem_type = built_problem.name
    print(f"[Optimize:{config['run_name']}] problem_type={problem_type}")
    if dataset_path is not None:
        print(f"[Optimize:{config['run_name']}] dataset path={dataset_path}")
    if hasattr(problem_obj, "A"):
        print(f"[Optimize:{config['run_name']}] A shape={tuple(int(v) for v in problem_obj.A.shape)}")
    if hasattr(problem_obj, "y"):
        print(f"[Optimize:{config['run_name']}] y shape={tuple(int(v) for v in np.asarray(problem_obj.y).shape)}")
    elif hasattr(problem_obj, "Y"):
        print(f"[Optimize:{config['run_name']}] Y shape={tuple(int(v) for v in problem_obj.Y.shape)}")
    print(f"[Optimize:{config['run_name']}] dim={built_problem.dim}")
    if hasattr(problem_obj, "m"):
        print(f"[Optimize:{config['run_name']}] sample size={int(problem_obj.m)}")
    if hasattr(problem_obj, "num_classes"):
        print(f"[Optimize:{config['run_name']}] num_classes={int(problem_obj.num_classes)}")
    if hasattr(problem_obj, "num_labels"):
        print(f"[Optimize:{config['run_name']}] num_labels={int(problem_obj.num_labels)}")
    optimizer_name = str(config["optimizer"]["type"])
    optimizer = get_optimizer(optimizer_name)
    optimizer_config = dict(config["optimizer"])
    optimizer_config.setdefault("run_name", config["run_name"])
    optimizer_config.setdefault("problem_name", built_problem.name)
    optimizer_config.setdefault("problem_dim", built_problem.dim)

    log_config = config["log"]
    logger: RunLogger | NullRunLogger
    if bool(log_config.get("enabled", False)):
        save_everytime = bool(log_config.get("save_everytime", True))
        logger = RunLogger(
            path=output_paths["csv_path"],
            extra_fields=optimizer.extra_log_fields,
            flush_every=int(log_config.get("flush_every", 1 if save_everytime else 50)),
            save_everytime=save_everytime,
        )
    else:
        logger = NullRunLogger()

    try:
        result = optimizer.run(built_problem.problem, x0, optimizer_config, logger)
    finally:
        logger.close()

    save_meta_config = config["save_meta"]
    if bool(save_meta_config.get("enabled", False)):
        meta_path = output_paths["meta_path"]
        resolved_config_path = output_paths["resolved_config_path"]
        save_resolved_config(config, resolved_config_path)
        save_json(
            {
                "executed_at_utc": utc_now_iso(),
                "run_name": config["run_name"],
                "seed": seed,
                "algorithm": optimizer_name,
                "problem": built_problem.name,
                "effective_reg_lambda": effective_reg_lambda,
                "effective_regularize_bias": effective_regularize_bias,
                "source_reg_lambda": source_reg_lambda,
                "reg_lambda_overridden": reg_lambda_overridden,
                "git_commit_hash": _maybe_git_commit_hash(),
                "run_dir": _project_relative_path(output_paths["run_dir"]),
                "input_config_path": _project_relative_path(output_paths["input_config_copy_path"]),
                "history_path": result.history_path,
                "resolved_config_path": str(resolved_config_path),
                "result": {
                    "f_final": result.f_final,
                    "grad_norm_final": result.grad_norm_final,
                    "n_iter": result.n_iter,
                    "status": result.status,
                },
                "resolved_config": config,
            },
            meta_path,
        )

    summary = {
        "status": result.status,
        "n_iter": result.n_iter,
        "f_final": result.f_final,
        "grad_norm_final": result.grad_norm_final,
        "effective_reg_lambda": effective_reg_lambda,
        "history_path": result.history_path,
        "run_dir": _project_relative_path(output_paths["run_dir"]),
    }
    print(summary)
    return summary


def _run_plot(config_path: str) -> Path:
    from src.plotting.main import plot_from_config

    config = load_plot_config(config_path)
    ensure_standard_directories()
    output_path = plot_from_config(config)
    print(f"Saved plot: {output_path}")
    return output_path


def _run_pipeline(config_path: str) -> list[dict[str, Any]]:
    config = load_pipeline_config(config_path)
    ensure_standard_directories()

    step_runners = {
        "generate": _run_generate,
        "generate_data": _run_generate,
        "optimize": _run_optimize,
        "plot": _run_plot,
    }
    results: list[dict[str, Any]] = []

    for index, step in enumerate(config["steps"], start=1):
        command = str(step["command"])
        step_config_path = str(step["config"])
        print(
            f"[Pipeline:{config['pipeline_name']}] step={index}/{len(config['steps'])} "
            f"command={command} config={step_config_path}"
        )
        result = step_runners[command](step_config_path)
        results.append(
            {
                "step": index,
                "command": command,
                "config": step_config_path,
                "result": str(result),
            }
        )

    return results


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {"generate", "generate_data"}:
        _run_generate(args.config)
        return
    if args.command == "optimize":
        _run_optimize(args.config)
        return
    if args.command == "plot":
        _run_plot(args.config)
        return
    if args.command == "pipeline":
        _run_pipeline(args.config)
        return

    raise ValueError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
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
from src.registry import build_problem, generate_problem_data, get_optimizer
from src.utils.io import save_json
from src.utils.paths import ensure_standard_directories, resolve_project_path
from src.utils.run_logger import NullRunLogger, RunLogger
from src.utils.seed import set_global_seed
from src.utils.timer import utc_now_iso


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Numerical optimization experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("generate", "optimize", "plot", "pipeline"):
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
    seed = int(config.get("seed", 0))
    set_global_seed(seed)
    save_path = resolve_project_path(config["save"]["path"])
    generate_problem_data(config["problem"], str(save_path), seed)
    print(f"Generated data: {save_path}")
    return save_path


def _run_optimize(config_path: str) -> dict[str, Any]:
    config = load_optimize_config(config_path)
    ensure_standard_directories()
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    built_problem = build_problem(config["problem"])
    x0 = _make_initial_point(config["initialization"], built_problem.dim, seed)
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
            path=resolve_project_path(log_config["csv_path"]),
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
        meta_path = resolve_project_path(save_meta_config["meta_path"])
        resolved_config_path = resolve_project_path(save_meta_config["resolved_config_path"])
        save_resolved_config(config, resolved_config_path)
        save_json(
            {
                "executed_at_utc": utc_now_iso(),
                "run_name": config["run_name"],
                "seed": seed,
                "algorithm": optimizer_name,
                "problem": built_problem.name,
                "git_commit_hash": _maybe_git_commit_hash(),
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
        "history_path": result.history_path,
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

    if args.command == "generate":
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

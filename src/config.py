from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import load_yaml, save_yaml
from src.utils.paths import resolve_project_path


def _ensure_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


def _ensure_keys(mapping: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{context} is missing required keys: {joined}")


def _ensure_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean, got {type(value).__name__}")
    return value


def _ensure_positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    return value


def _ensure_nonnegative_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise ValueError(f"{context} must be a non-negative number, got {value!r}")
    return float(value)


def _load_and_check_task(config_path: str | Path, expected_tasks: set[str]) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    config = _ensure_mapping(load_yaml(path), f"Config {path}")
    task = config.get("task")
    if task not in expected_tasks:
        expected = ", ".join(sorted(expected_tasks))
        raise ValueError(f"Config {path} has task={task!r}, expected one of: {expected}")
    return copy.deepcopy(config)


def load_generate_config(config_path: str | Path) -> dict[str, Any]:
    config = _load_and_check_task(config_path, {"generate_data", "clone_data_with_reg_lambda"})
    task = str(config["task"])
    if task == "generate_data":
        _ensure_keys(config, ("run_name", "problem", "save"), "generate_data config")
        _ensure_mapping(config["problem"], "problem")
        save = _ensure_mapping(config["save"], "save")
        _ensure_keys(save, ("path",), "save")
        config.setdefault("seed", 0)
        return config

    _ensure_keys(config, ("run_name", "source", "reg_lambda", "save"), "clone_data_with_reg_lambda config")
    source = _ensure_mapping(config["source"], "source")
    save = _ensure_mapping(config["save"], "save")
    _ensure_keys(source, ("path",), "source")
    _ensure_keys(save, ("path",), "save")
    _ensure_nonnegative_number(config["reg_lambda"], "reg_lambda")
    return config


def load_optimize_config(config_path: str | Path) -> dict[str, Any]:
    config = _load_and_check_task(config_path, {"optimize"})
    _ensure_keys(
        config,
        ("run_name", "problem", "initialization", "optimizer", "log", "save_meta"),
        "optimize config",
    )
    _ensure_mapping(config["problem"], "problem")
    _ensure_mapping(config["initialization"], "initialization")
    optimizer = _ensure_mapping(config["optimizer"], "optimizer")
    log = _ensure_mapping(config["log"], "log")
    save_meta = _ensure_mapping(config["save_meta"], "save_meta")
    _ensure_keys(optimizer, ("type",), "optimizer")
    _ensure_keys(log, ("enabled", "csv_path"), "log")
    _ensure_keys(save_meta, ("enabled", "meta_path", "resolved_config_path"), "save_meta")

    _ensure_bool(log["enabled"], "log.enabled")
    _ensure_bool(save_meta["enabled"], "save_meta.enabled")
    if "save_everytime" in log:
        _ensure_bool(log["save_everytime"], "log.save_everytime")
    if "stop_on_grad_norm_stagnation" in optimizer:
        _ensure_bool(
            optimizer["stop_on_grad_norm_stagnation"],
            "optimizer.stop_on_grad_norm_stagnation",
        )
    if "grad_norm_stagnation_patience" in optimizer:
        _ensure_positive_int(
            optimizer["grad_norm_stagnation_patience"],
            "optimizer.grad_norm_stagnation_patience",
        )
    if "grad_norm_stagnation_rtol" in optimizer:
        _ensure_nonnegative_number(
            optimizer["grad_norm_stagnation_rtol"],
            "optimizer.grad_norm_stagnation_rtol",
        )
    if "grad_norm_stagnation_atol" in optimizer:
        _ensure_nonnegative_number(
            optimizer["grad_norm_stagnation_atol"],
            "optimizer.grad_norm_stagnation_atol",
        )

    optimizer.setdefault("max_iter", 100)
    optimizer.setdefault("tol", 1.0e-8)
    optimizer.setdefault("stop_on_grad_norm_stagnation", False)
    optimizer.setdefault("grad_norm_stagnation_patience", 10)
    optimizer.setdefault("grad_norm_stagnation_rtol", 1.0e-12)
    optimizer.setdefault("grad_norm_stagnation_atol", 1.0e-12)
    line_search = optimizer.setdefault("line_search", {})
    if not isinstance(line_search, dict):
        raise ValueError("optimizer.line_search must be a mapping")
    default_line_search_enabled = str(optimizer.get("type", "")) == "gd"
    line_search.setdefault("enabled", default_line_search_enabled)
    line_search.setdefault("c1", 1.0e-4)
    line_search.setdefault("beta", 0.5)
    line_search.setdefault("max_iter", 25)
    log.setdefault("save_everytime", True)

    config.setdefault("seed", 0)
    return config


def load_plot_config(config_path: str | Path) -> dict[str, Any]:
    config = _load_and_check_task(config_path, {"plot"})
    _ensure_keys(config, ("plot_name", "plot", "save"), "plot config")
    has_inputs = "inputs" in config
    has_panels = "panels" in config
    if not has_inputs and not has_panels:
        raise ValueError("plot config must contain either inputs or panels")
    if has_inputs and (not isinstance(config["inputs"], list) or not config["inputs"]):
        raise ValueError("plot.inputs must be a non-empty list")
    if has_panels and (not isinstance(config["panels"], list) or not config["panels"]):
        raise ValueError("plot.panels must be a non-empty list")
    plot = _ensure_mapping(config["plot"], "plot")
    save = _ensure_mapping(config["save"], "save")
    _ensure_keys(plot, ("x", "y"), "plot")
    _ensure_keys(save, ("path",), "save")
    return config


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    config = _load_and_check_task(config_path, {"pipeline"})
    _ensure_keys(config, ("pipeline_name", "steps"), "pipeline config")
    steps = config["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("pipeline.steps must be a non-empty list")

    allowed_commands = {"generate", "generate_data", "optimize", "plot"}
    for index, step in enumerate(steps, start=1):
        step_context = f"pipeline step {index}"
        step_mapping = _ensure_mapping(step, step_context)
        _ensure_keys(step_mapping, ("command", "config"), step_context)
        command = step_mapping["command"]
        if command not in allowed_commands:
            allowed = ", ".join(sorted(allowed_commands))
            raise ValueError(f"{step_context}.command must be one of: {allowed}")

    return config


def save_resolved_config(config: dict[str, Any], output_path: str | Path) -> None:
    save_yaml(config, resolve_project_path(output_path))

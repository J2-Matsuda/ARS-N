from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Mapping

import numpy as np

from src.algorithms.base import OptimizeResult
from src.problems.base import Problem
from src.problems.logistic import (
    LogisticRegressionProblem,
    build_logistic_problem,
    generate_logistic_from_config,
)
from src.problems.mnist.problem import build_mnist_problem
from src.problems.quadratic import QuadraticProblem, build_quadratic_problem, generate_quadratic_from_config


@dataclass(frozen=True)
class BuiltProblem:
    problem: Problem
    dim: int
    name: str


@dataclass(frozen=True)
class OptimizerSpec:
    run: Callable[[Problem, np.ndarray, Mapping[str, Any], Any], OptimizeResult]
    extra_log_fields: tuple[str, ...]


def _as_built_problem(problem: Problem, dim: int, name: str) -> BuiltProblem:
    return BuiltProblem(problem=problem, dim=dim, name=name)


def _build_quadratic(problem_config: Mapping[str, Any]) -> BuiltProblem:
    problem = build_quadratic_problem(problem_config)
    return _as_built_problem(problem, problem.dim, "quadratic")


def _build_logistic(problem_config: Mapping[str, Any]) -> BuiltProblem:
    problem = build_logistic_problem(problem_config)
    return _as_built_problem(problem, problem.dim, "logistic")


def _build_mnist(problem_config: Mapping[str, Any]) -> BuiltProblem:
    problem, dim = build_mnist_problem(problem_config)
    return _as_built_problem(problem, dim, "mnist")


PROBLEM_BUILDERS: dict[str, Callable[[Mapping[str, Any]], BuiltProblem]] = {
    "quadratic": _build_quadratic,
    "logistic": _build_logistic,
    "mnist": _build_mnist,
}

PROBLEM_GENERATORS: dict[str, Callable[[Mapping[str, Any], str, int], None]] = {
    "quadratic": generate_quadratic_from_config,
    "logistic": generate_logistic_from_config,
}

OPTIMIZER_MODULES: dict[str, str] = {
    "ars_n": "src.algorithms.ars_n.main",
    "ars_cn": "src.algorithms.ars_n.ars_cn",
    "ars_rn": "src.algorithms.ars_n.ars_rn",
    "rn": "src.algorithms.rn.main",
    "cn": "src.algorithms.cn.main",
    "rs_rn": "src.algorithms.rs_rn.main",
    "rs_cn": "src.algorithms.rs_cn.main",
    "newton_cg": "src.algorithms.newton_cg.main",
    "full_newton": "src.algorithms.full_newton.main",
}


def build_problem(problem_config: Mapping[str, Any]) -> BuiltProblem:
    problem_type = str(problem_config.get("type", ""))
    try:
        builder = PROBLEM_BUILDERS[problem_type]
    except KeyError as exc:
        available = ", ".join(sorted(PROBLEM_BUILDERS))
        raise ValueError(f"Unknown problem type {problem_type!r}. Available: {available}") from exc
    return builder(problem_config)


def generate_problem_data(problem_config: Mapping[str, Any], save_path: str, seed: int) -> None:
    problem_type = str(problem_config.get("type", ""))
    try:
        generator = PROBLEM_GENERATORS[problem_type]
    except KeyError as exc:
        available = ", ".join(sorted(PROBLEM_GENERATORS))
        raise ValueError(
            f"Problem type {problem_type!r} does not support generate_data. Available: {available}"
        ) from exc
    generator(problem_config, save_path, seed)


def get_optimizer(optimizer_name: str) -> OptimizerSpec:
    try:
        module_path = OPTIMIZER_MODULES[optimizer_name]
    except KeyError as exc:
        available = ", ".join(sorted(OPTIMIZER_MODULES))
        raise ValueError(f"Unknown optimizer {optimizer_name!r}. Available: {available}") from exc

    module = import_module(module_path)
    run = getattr(module, "run")
    extra_log_fields = tuple(getattr(module, "EXTRA_LOG_FIELDS", ()))
    return OptimizerSpec(run=run, extra_log_fields=extra_log_fields)

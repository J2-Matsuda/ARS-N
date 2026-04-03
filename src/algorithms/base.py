from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from src.problems.base import Problem


@dataclass
class OptimizeResult:
    x_final: np.ndarray
    f_final: float
    grad_norm_final: float
    n_iter: int
    status: str
    history_path: str | None


def evaluate_problem(problem: Problem, x: np.ndarray) -> tuple[float, np.ndarray, float]:
    fx = float(problem.f(x))
    grad = np.asarray(problem.grad(x), dtype=float)
    grad_norm = float(np.linalg.norm(grad))
    return fx, grad, grad_norm


def log_iteration(
    logger: Any,
    iteration: int,
    fx: float,
    grad_norm: float,
    step_norm: float,
    step_size: float,
    cumulative_time: float,
    per_iter_time: float,
    extras: Mapping[str, Any] | None = None,
) -> None:
    row = {
        "iter": iteration,
        "f": fx,
        "grad_norm": grad_norm,
        "step_norm": step_norm,
        "step_size": step_size,
        "cumulative_time": cumulative_time,
        "per_iter_time": per_iter_time,
    }
    if extras:
        row.update(extras)
    logger.log(row)


def armijo_backtracking(
    problem: Problem,
    x: np.ndarray,
    direction: np.ndarray,
    grad: np.ndarray,
    fx: float,
    line_search_config: Mapping[str, Any] | None,
) -> tuple[float, np.ndarray, float, bool]:
    config = dict(line_search_config or {})
    if not bool(config.get("enabled", False)):
        x_next = x + direction
        return 1.0, x_next, float(problem.f(x_next)), True

    c1 = float(config.get("c1", 1.0e-4))
    beta = float(config.get("beta", 0.5))
    max_iter = int(config.get("max_iter", 25))
    directional_derivative = float(np.dot(grad, direction))
    if directional_derivative >= 0.0:
        return 0.0, x.copy(), fx, False

    step_size = 1.0
    for _ in range(max_iter):
        x_next = x + step_size * direction
        f_next = float(problem.f(x_next))
        if f_next <= fx + c1 * step_size * directional_derivative:
            return step_size, x_next, f_next, True
        step_size *= beta

    return 0.0, x.copy(), fx, False


def build_dense_hessian(problem: Problem, x: np.ndarray) -> np.ndarray:
    identity = np.eye(x.size)
    columns = [problem.hvp(x, identity[:, idx]) for idx in range(x.size)]
    return np.column_stack(columns)


def current_time() -> float:
    return perf_counter()


def not_implemented_error(algorithm_name: str, relative_path: str) -> NotImplementedError:
    return NotImplementedError(
        f"{algorithm_name} core update rule is not specified yet. "
        f"Implement the step computation in {relative_path}"
    )

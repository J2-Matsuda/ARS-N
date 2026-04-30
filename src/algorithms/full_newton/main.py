from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

import numpy as np

from src.algorithms.base import (
    GradNormStagnationTracker,
    OptimizeResult,
    armijo_backtracking,
    build_dense_hessian,
    evaluate_problem,
    log_iteration,
    resolve_grad_norm_stagnation_config,
)
from src.problems.base import Problem
from src.utils.timer import Stopwatch

EXTRA_LOG_FIELDS = ("accepted",)


def _dense_newton_direction(problem: Problem, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    hessian = build_dense_hessian(problem, x)
    hessian = 0.5 * (hessian + hessian.T)
    rhs = -grad
    try:
        return np.linalg.solve(hessian, rhs)
    except np.linalg.LinAlgError:
        direction, *_ = np.linalg.lstsq(hessian, rhs, rcond=None)
        return direction


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    max_iter = int(config.get("max_iter", 100))
    tol = float(config.get("tol", 1.0e-8))
    line_search_config = dict(config.get("line_search", {}))
    grad_norm_stagnation = GradNormStagnationTracker(resolve_grad_norm_stagnation_config(config))

    stopwatch = Stopwatch()
    x = np.asarray(x0, dtype=float).copy()
    fx, grad, grad_norm = evaluate_problem(problem, x)
    grad_norm_stagnation.update(grad_norm)
    log_iteration(
        logger,
        iteration=0,
        fx=fx,
        grad_norm=grad_norm,
        step_norm=0.0,
        step_size=0.0,
        cumulative_time=0.0,
        per_iter_time=0.0,
    )

    status = "max_iter"
    n_iter = 0
    if grad_norm <= tol:
        status = "converged"

    for iteration in range(1, max_iter + 1):
        if grad_norm <= tol:
            break

        iter_start = perf_counter()
        direction = _dense_newton_direction(problem, x, grad)
        if float(np.dot(grad, direction)) >= 0.0:
            direction = -grad

        step_size, x_next, f_next, accepted = armijo_backtracking(
            problem=problem,
            x=x,
            direction=direction,
            grad=grad,
            fx=fx,
            line_search_config=line_search_config,
        )
        step = x_next - x
        step_norm = float(np.linalg.norm(step))
        x = x_next
        fx = f_next
        _, grad, grad_norm = evaluate_problem(problem, x)
        per_iter_time = float(perf_counter() - iter_start)
        cumulative_time = float(stopwatch.elapsed())
        n_iter = iteration
        log_iteration(
            logger,
            iteration=iteration,
            fx=fx,
            grad_norm=grad_norm,
            step_norm=step_norm,
            step_size=step_size,
            cumulative_time=cumulative_time,
            per_iter_time=per_iter_time,
            extras={"accepted": int(accepted)},
        )
        if step_size == 0.0:
            status = "line_search_failed"
            break
        if grad_norm <= tol:
            status = "converged"
            break
        if grad_norm_stagnation.update(grad_norm):
            status = "grad_norm_stagnation"
            break

    return OptimizeResult(
        x_final=x,
        f_final=fx,
        grad_norm_final=grad_norm,
        n_iter=n_iter,
        status=status,
        history_path=logger.history_path,
    )

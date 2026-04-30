from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from src.algorithms.base import OptimizeResult, evaluate_problem
from src.problems.base import Problem
from src.utils.timer import Stopwatch

EXTRA_LOG_FIELDS = (
    "accepted",
    "armijo_iters",
    "gtd",
    "hvp_calls_iter",
    "hvp_calls_cum",
)


@dataclass(frozen=True)
class GDResult:
    x_final: np.ndarray
    history: list[dict[str, Any]]
    status: str
    iters: int
    elapsed_sec: float


@dataclass(frozen=True)
class _LineSearchOutcome:
    step_size: float
    x_next: np.ndarray
    f_next: float
    accepted: bool
    armijo_iters: int


def _as_float_vector(vector: np.ndarray | Any) -> np.ndarray:
    return np.asarray(vector, dtype=float).reshape(-1)


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _problem_dim(problem: Problem, x0: np.ndarray) -> int:
    if hasattr(problem, "dim"):
        return int(getattr(problem, "dim"))
    if hasattr(problem, "n"):
        return int(getattr(problem, "n"))
    return int(x0.size)


def _evaluate(problem: Problem, x: np.ndarray) -> tuple[float, np.ndarray, float]:
    fx, grad, grad_norm = evaluate_problem(problem, x)
    return float(fx), _as_float_vector(grad), float(grad_norm)


def _resolve_line_search_config(config: Mapping[str, Any]) -> dict[str, Any]:
    line_search = dict(config.get("line_search", {}))
    legacy_requested = any(key in config for key in ("beta", "tau", "alpha0", "max_ls_iters"))

    if legacy_requested and not bool(line_search.get("enabled", False)):
        line_search["enabled"] = True
    line_search.setdefault("enabled", True)

    if "c1" not in line_search and "beta" in config:
        line_search["c1"] = config["beta"]
    if "beta" not in line_search and "tau" in config:
        line_search["beta"] = config["tau"]
    if "alpha0" not in line_search and "alpha0" in config:
        line_search["alpha0"] = config["alpha0"]
    if "max_iter" not in line_search and "max_ls_iters" in config:
        line_search["max_iter"] = config["max_ls_iters"]

    line_search.setdefault("c1", 1.0e-4)
    line_search.setdefault("beta", 0.5)
    line_search.setdefault("alpha0", 1.0)
    line_search.setdefault("max_iter", 25)

    c1 = float(line_search["c1"])
    shrink = float(line_search["beta"])
    alpha0 = float(line_search["alpha0"])
    max_iter = int(line_search["max_iter"])
    if not (0.0 < c1 < 1.0):
        raise ValueError("optimizer.line_search.c1 must satisfy 0 < c1 < 1")
    if not (0.0 < shrink < 1.0):
        raise ValueError("optimizer.line_search.beta must satisfy 0 < beta < 1")
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        raise ValueError("optimizer.line_search.alpha0 must be positive and finite")
    if max_iter <= 0:
        raise ValueError("optimizer.line_search.max_iter must be positive")
    return line_search


def _armijo_backtracking(
    problem: Problem,
    x: np.ndarray,
    fx: float,
    grad: np.ndarray,
    direction: np.ndarray,
    line_search_config: Mapping[str, Any],
) -> _LineSearchOutcome:
    directional_derivative = float(np.dot(grad, direction))
    if (
        not np.all(np.isfinite(direction))
        or not np.isfinite(directional_derivative)
        or directional_derivative >= 0.0
    ):
        return _LineSearchOutcome(
            step_size=0.0,
            x_next=x.copy(),
            f_next=fx,
            accepted=False,
            armijo_iters=0,
        )

    if not bool(line_search_config.get("enabled", True)):
        x_next = x + direction
        if not np.all(np.isfinite(x_next)):
            return _LineSearchOutcome(0.0, x.copy(), fx, False, 0)
        f_next = float(problem.f(x_next))
        return _LineSearchOutcome(
            step_size=1.0,
            x_next=x_next,
            f_next=f_next,
            accepted=bool(np.isfinite(f_next)),
            armijo_iters=0,
        )

    c1 = float(line_search_config.get("c1", 1.0e-4))
    shrink = float(line_search_config.get("beta", 0.5))
    alpha = float(line_search_config.get("alpha0", 1.0))
    max_iter = int(line_search_config.get("max_iter", 25))

    for armijo_iters in range(max_iter):
        x_next = x + alpha * direction
        if not np.all(np.isfinite(x_next)):
            alpha *= shrink
            continue
        f_next = float(problem.f(x_next))
        if np.isfinite(f_next) and f_next <= fx + c1 * alpha * directional_derivative:
            return _LineSearchOutcome(
                step_size=alpha,
                x_next=x_next,
                f_next=f_next,
                accepted=True,
                armijo_iters=armijo_iters,
            )
        alpha *= shrink

    return _LineSearchOutcome(
        step_size=0.0,
        x_next=x.copy(),
        f_next=fx,
        accepted=False,
        armijo_iters=max_iter,
    )


def _log_row(
    history: list[dict[str, Any]],
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
    history.append(row)
    logger.log(row)


def _print_run_header(config: Mapping[str, Any], dim: int) -> None:
    parts = [
        "[GD]",
        f"run={config.get('run_name', '')}",
        f"dim={dim}",
    ]
    print(" ".join(part for part in parts if not part.endswith("=")), flush=True)


def _print_iter_log(iteration: int, fx: float, grad_norm: float) -> None:
    print(f"[GD] iter={iteration} f={fx:.6e} grad_norm={grad_norm:.3e}", flush=True)


def _run_gd(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> GDResult:
    x = _as_float_vector(x0).copy()
    dim = _problem_dim(problem, x)
    max_iter = int(config.get("max_iter", 100))
    tol = float(config.get("tol", config.get("tol_grad", 1.0e-6)))
    verbose = bool(config.get("verbose", False))
    print_every = max(1, int(config.get("print_every", 10)))
    line_search_config = _resolve_line_search_config(config)

    stopwatch = Stopwatch()
    history: list[dict[str, Any]] = []
    hvp_calls_cum = 0

    fx, grad, grad_norm = _evaluate(problem, x)
    _log_row(
        history=history,
        logger=logger,
        iteration=0,
        fx=fx,
        grad_norm=grad_norm,
        step_norm=0.0,
        step_size=0.0,
        cumulative_time=0.0,
        per_iter_time=0.0,
        extras={
            "hvp_calls_iter": 0,
            "hvp_calls_cum": 0,
        },
    )

    if verbose:
        _print_run_header(config, dim)

    status = "max_iter"
    if grad_norm <= tol:
        status = "converged"

    for iteration in range(1, max_iter + 1):
        if grad_norm <= tol:
            break

        iter_start = perf_counter()
        direction = -grad
        gtd = float(np.dot(grad, direction))
        line_search = _armijo_backtracking(
            problem=problem,
            x=x,
            fx=fx,
            grad=grad,
            direction=direction,
            line_search_config=line_search_config,
        )

        if line_search.accepted:
            step = line_search.x_next - x
            step_norm = _safe_norm(step)
            x = _as_float_vector(line_search.x_next)
            fx = float(line_search.f_next)
            _, grad, grad_norm = _evaluate(problem, x)
        else:
            step_norm = 0.0

        per_iter_time = float(perf_counter() - iter_start)
        cumulative_time = float(stopwatch.elapsed())
        _log_row(
            history=history,
            logger=logger,
            iteration=iteration,
            fx=fx,
            grad_norm=grad_norm,
            step_norm=step_norm,
            step_size=line_search.step_size,
            cumulative_time=cumulative_time,
            per_iter_time=per_iter_time,
            extras={
                "accepted": int(line_search.accepted),
                "armijo_iters": line_search.armijo_iters,
                "gtd": gtd,
                "hvp_calls_iter": 0,
                "hvp_calls_cum": hvp_calls_cum,
            },
        )

        if verbose and iteration % print_every == 0:
            _print_iter_log(iteration, fx, grad_norm)

        if not line_search.accepted:
            status = "line_search_failed"
            break
        if grad_norm <= tol:
            status = "converged"
            break

    return GDResult(
        x_final=x,
        history=history,
        status=status,
        iters=max(0, len(history) - 1),
        elapsed_sec=float(stopwatch.elapsed()),
    )


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    result = _run_gd(problem=problem, x0=x0, config=config, logger=logger)
    final_row = result.history[-1]
    return OptimizeResult(
        x_final=result.x_final,
        f_final=float(final_row["f"]),
        grad_norm_final=float(final_row["grad_norm"]),
        n_iter=result.iters,
        status=result.status,
        history_path=getattr(logger, "history_path", None),
    )

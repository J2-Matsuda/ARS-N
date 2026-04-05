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
    "eta",
    "lambda_min_phpt",
    "lambda_shift",
    "used_grad_fallback",
)


@dataclass(frozen=True)
class RNMResult:
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


@dataclass(frozen=True)
class _RegularizedNewtonInfo:
    eta: float
    lambda_min_phpt: float
    lambda_shift: float


class _CountingProblem:
    def __init__(self, base: Problem) -> None:
        self._base = base
        self.hvp_calls = 0

    def f(self, x: np.ndarray) -> float:
        return float(self._base.f(x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        grad = np.asarray(self._base.grad(x))
        return grad.astype(float, copy=False)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.hvp_calls += 1
        hvp_value = np.asarray(self._base.hvp(x, v))
        return hvp_value.astype(float, copy=False)


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def _problem_dim(problem: Problem, x0: np.ndarray) -> int:
    if hasattr(problem, "dim"):
        return int(getattr(problem, "dim"))
    if hasattr(problem, "n"):
        return int(getattr(problem, "n"))
    return int(x0.size)


def _resolve_line_search_config(config: Mapping[str, Any]) -> dict[str, Any]:
    line_search = dict(config.get("line_search", {}))
    legacy_requested = any(key in config for key in ("beta", "tau", "alpha0", "max_ls_iters"))

    if legacy_requested and not bool(line_search.get("enabled", False)):
        line_search["enabled"] = True
    line_search.setdefault("enabled", False)

    if "c1" not in line_search and "beta" in config:
        line_search["c1"] = config["beta"]
    if "beta" not in line_search and "tau" in config:
        line_search["beta"] = config["tau"]
    if "alpha0" not in line_search and "alpha0" in config:
        line_search["alpha0"] = config["alpha0"]
    if "max_iter" not in line_search and "max_ls_iters" in config:
        line_search["max_iter"] = config["max_ls_iters"]

    # Ueda–Yamashita paper notation:
    #   alpha <-> line_search.c1
    #   beta  <-> line_search.beta
    line_search.setdefault("c1", 1.0e-4)
    line_search.setdefault("beta", 0.5)
    line_search.setdefault("alpha0", 1.0)
    line_search.setdefault("max_iter", 25)
    return line_search


def _build_full_hessian(problem: _CountingProblem, x: np.ndarray, dim: int) -> np.ndarray:
    hessian = np.empty((dim, dim), dtype=float)
    basis = np.zeros((dim, dim), dtype=float)
    np.fill_diagonal(basis, 1.0)
    for column in range(dim):
        hessian[:, column] = problem.hvp(x, basis[:, column])
    return 0.5 * (hessian + hessian.T)


def _solve_regularized_newton(
    hessian: np.ndarray,
    grad: np.ndarray,
    grad_norm: float,
    regularization_config: Mapping[str, Any],
) -> tuple[np.ndarray, _RegularizedNewtonInfo]:
    c1 = float(regularization_config.get("lambda_factor", regularization_config.get("c1", 2.0)))
    c2 = float(regularization_config.get("grad_factor", regularization_config.get("c2", 1.0)))
    delta = float(
        regularization_config.get(
            "grad_exponent",
            regularization_config.get("delta", regularization_config.get("gamma", 1.0)),
        )
    )

    eigvals = np.linalg.eigvalsh(hessian)
    lambda_min = float(eigvals.min()) if eigvals.size else 0.0
    lambda_shift = max(0.0, -lambda_min)
    mu = c1 * lambda_shift + c2 * (grad_norm**delta)

    regularized = hessian + mu * np.eye(hessian.shape[0], dtype=float)
    rhs = -np.asarray(grad, dtype=float).reshape(-1)
    try:
        chol = np.linalg.cholesky(regularized)
        intermediate = np.linalg.solve(chol, rhs)
        step = np.linalg.solve(chol.T, intermediate)
    except np.linalg.LinAlgError:
        try:
            step = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(regularized, rhs, rcond=None)

    return np.asarray(step, dtype=float).reshape(-1), _RegularizedNewtonInfo(
        eta=float(mu),
        lambda_min_phpt=lambda_min,
        lambda_shift=lambda_shift,
    )


def _armijo_backtracking(
    problem: _CountingProblem,
    x: np.ndarray,
    fx: float,
    grad: np.ndarray,
    direction: np.ndarray,
    line_search_config: Mapping[str, Any],
) -> _LineSearchOutcome:
    if not bool(line_search_config.get("enabled", False)):
        x_next = x + direction
        return _LineSearchOutcome(
            step_size=1.0,
            x_next=x_next,
            f_next=float(problem.f(x_next)),
            accepted=True,
            armijo_iters=0,
        )

    c1 = float(line_search_config.get("c1", 1.0e-4))
    shrink = float(line_search_config.get("beta", 0.5))
    alpha = float(line_search_config.get("alpha0", 1.0))
    max_iter = int(line_search_config.get("max_iter", 25))
    directional_derivative = float(np.dot(grad, direction))
    if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
        return _LineSearchOutcome(
            step_size=0.0,
            x_next=x.copy(),
            f_next=fx,
            accepted=False,
            armijo_iters=0,
        )

    for armijo_iters in range(max_iter):
        x_next = x + alpha * direction
        f_next = float(problem.f(x_next))
        if f_next <= fx + c1 * alpha * directional_derivative:
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
        "[RN]",
        f"run={config.get('run_name', '')}",
        f"dim={dim}",
    ]
    print(" ".join(part for part in parts if not part.endswith("=")), flush=True)


def _print_iter_log(iteration: int, fx: float, grad_norm: float) -> None:
    print(f"[RN] iter={iteration} f={fx:.6e} grad_norm={grad_norm:.3e}", flush=True)


def _run_rnm(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> RNMResult:
    x = np.asarray(x0, dtype=float).copy()
    dim = _problem_dim(problem, x)
    max_iter = int(config.get("max_iter", 100))
    tol = float(config.get("tol", config.get("tol_grad", 1.0e-6)))

    verbose = bool(config.get("verbose", False))
    print_every = int(config.get("print_every", 10))
    regularization_config = dict(config.get("diag_shift", config.get("regularization", {})))
    line_search_config = _resolve_line_search_config(config)

    counted_problem = _CountingProblem(problem)
    stopwatch = Stopwatch()
    history: list[dict[str, Any]] = []
    hvp_calls_cum = 0

    fx, grad, grad_norm = evaluate_problem(counted_problem, x)
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
            "accepted": 1,
            "armijo_iters": 0,
            "gtd": float("nan"),
            "hvp_calls_iter": 0,
            "hvp_calls_cum": 0,
            "eta": float("nan"),
            "lambda_min_phpt": float("nan"),
            "lambda_shift": float("nan"),
            "used_grad_fallback": 0,
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
        counted_problem.hvp_calls = 0

        hessian = _build_full_hessian(counted_problem, x, dim)
        direction, reg_info = _solve_regularized_newton(
            hessian=hessian,
            grad=grad,
            grad_norm=grad_norm,
            regularization_config=regularization_config,
        )
        gtd = float(np.dot(grad, direction))
        used_grad_fallback = 0
        if not np.all(np.isfinite(direction)) or not np.isfinite(gtd) or gtd >= 0.0:
            direction = -grad
            gtd = -float(np.dot(grad, grad))
            used_grad_fallback = 1

        line_search = _armijo_backtracking(
            problem=counted_problem,
            x=x,
            fx=fx,
            grad=grad,
            direction=direction,
            line_search_config=line_search_config,
        )

        step = line_search.x_next - x
        step_norm = _safe_norm(step)
        x = line_search.x_next
        fx = line_search.f_next
        _, grad, grad_norm = evaluate_problem(counted_problem, x)

        per_iter_time = float(perf_counter() - iter_start)
        cumulative_time = float(stopwatch.elapsed())
        hvp_calls_cum += counted_problem.hvp_calls
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
                "hvp_calls_iter": counted_problem.hvp_calls,
                "hvp_calls_cum": hvp_calls_cum,
                "eta": reg_info.eta,
                "lambda_min_phpt": reg_info.lambda_min_phpt,
                "lambda_shift": reg_info.lambda_shift,
                "used_grad_fallback": used_grad_fallback,
            },
        )

        if verbose and iteration % max(1, print_every) == 0:
            _print_iter_log(iteration, fx, grad_norm)

        if not line_search.accepted:
            status = "line_search_failed"
            break
        if grad_norm <= tol:
            status = "converged"
            break

    return RNMResult(
        x_final=x,
        history=history,
        status=status,
        iters=max(0, len(history) - 1),
        elapsed_sec=float(stopwatch.elapsed()),
    )


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    result = _run_rnm(problem=problem, x0=x0, config=config, logger=logger)
    final_row = result.history[-1]
    return OptimizeResult(
        x_final=result.x_final,
        f_final=float(final_row["f"]),
        grad_norm_final=float(final_row["grad_norm"]),
        n_iter=result.iters,
        status=result.status,
        history_path=getattr(logger, "history_path", None),
    )

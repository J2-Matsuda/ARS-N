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
    "ls_iters",
    "ls_L",
    "step_inv_L",
    "momentum",
    "restart",
    "f_y",
    "grad_norm_y",
    "hvp_calls_iter",
    "hvp_calls_cum",
)


@dataclass(frozen=True)
class AGDUnknownResult:
    x_final: np.ndarray
    history: list[dict[str, Any]]
    status: str
    iters: int
    elapsed_sec: float


@dataclass(frozen=True)
class _BacktrackingOutcome:
    step_size: float
    L_value: float
    x_next: np.ndarray
    f_next: float
    accepted: bool
    ls_iters: int
    f_y: float
    grad_y: np.ndarray
    grad_norm_y: float


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


def _resolve_backtracking_config(config: Mapping[str, Any]) -> dict[str, Any]:
    backtracking = dict(config.get("backtracking", {}))
    backtracking.setdefault("enabled", True)
    backtracking.setdefault("L0", 1.0)
    backtracking.setdefault("eta", 2.0)
    backtracking.setdefault("max_iter", 50)
    backtracking.setdefault("reuse_previous_L", True)

    L0 = float(backtracking["L0"])
    eta = float(backtracking["eta"])
    max_iter = int(backtracking["max_iter"])
    if not np.isfinite(L0) or L0 <= 0.0:
        raise ValueError("optimizer.backtracking.L0 must be positive and finite")
    if not np.isfinite(eta) or eta <= 1.0:
        raise ValueError("optimizer.backtracking.eta must be finite and greater than 1")
    if max_iter <= 0:
        raise ValueError("optimizer.backtracking.max_iter must be positive")
    return backtracking


def _resolve_restart_config(config: Mapping[str, Any]) -> dict[str, Any]:
    restart = dict(config.get("restart", {}))
    restart.setdefault("enabled", True)
    restart.setdefault("objective_increase", True)
    restart.setdefault("misaligned_momentum", True)
    return restart


def _failed_backtracking(
    y: np.ndarray,
    L_value: float,
    f_y: float,
    grad_y: np.ndarray,
    grad_norm_y: float,
    ls_iters: int,
) -> _BacktrackingOutcome:
    return _BacktrackingOutcome(
        step_size=0.0,
        L_value=L_value,
        x_next=y.copy(),
        f_next=f_y,
        accepted=False,
        ls_iters=ls_iters,
        f_y=f_y,
        grad_y=grad_y,
        grad_norm_y=grad_norm_y,
    )


def _accelerated_backtracking(
    problem: Problem,
    y: np.ndarray,
    L_start: float,
    backtracking_config: Mapping[str, Any],
) -> _BacktrackingOutcome:
    y = _as_float_vector(y)
    if not np.all(np.isfinite(y)):
        nan_grad = np.full_like(y, np.nan, dtype=float)
        return _failed_backtracking(y, float(L_start), np.nan, nan_grad, np.inf, 0)

    f_y, grad_y, grad_norm_y = _evaluate(problem, y)
    L_trial = float(L_start)
    if (
        not np.isfinite(f_y)
        or not np.all(np.isfinite(grad_y))
        or not np.isfinite(grad_norm_y)
        or not np.isfinite(L_trial)
        or L_trial <= 0.0
    ):
        return _failed_backtracking(y, L_trial, f_y, grad_y, grad_norm_y, 0)

    if not bool(backtracking_config.get("enabled", True)):
        step = -grad_y / L_trial
        x_next = y + step
        if not np.all(np.isfinite(x_next)):
            return _failed_backtracking(y, L_trial, f_y, grad_y, grad_norm_y, 0)
        f_next = float(problem.f(x_next))
        return _BacktrackingOutcome(
            step_size=1.0 / L_trial,
            L_value=L_trial,
            x_next=x_next,
            f_next=f_next,
            accepted=bool(np.isfinite(f_next)),
            ls_iters=0,
            f_y=f_y,
            grad_y=grad_y,
            grad_norm_y=grad_norm_y,
        )

    eta = float(backtracking_config.get("eta", 2.0))
    max_iter = int(backtracking_config.get("max_iter", 50))
    for ls_iters in range(max_iter + 1):
        step = -grad_y / L_trial
        x_next = y + step
        if np.all(np.isfinite(x_next)):
            f_next = float(problem.f(x_next))
            model_upper = f_y + float(np.dot(grad_y, step)) + 0.5 * L_trial * float(np.dot(step, step))
            tolerance = 1.0e-12 * max(1.0, abs(f_y), abs(f_next))
            if (
                np.isfinite(f_next)
                and np.isfinite(model_upper)
                and f_next <= model_upper + tolerance
            ):
                return _BacktrackingOutcome(
                    step_size=1.0 / L_trial,
                    L_value=L_trial,
                    x_next=x_next,
                    f_next=f_next,
                    accepted=True,
                    ls_iters=ls_iters,
                    f_y=f_y,
                    grad_y=grad_y,
                    grad_norm_y=grad_norm_y,
                )
        if ls_iters == max_iter:
            break
        L_trial *= eta
        if not np.isfinite(L_trial):
            return _failed_backtracking(y, L_trial, f_y, grad_y, grad_norm_y, ls_iters + 1)

    return _failed_backtracking(y, L_trial, f_y, grad_y, grad_norm_y, max_iter)


def _should_restart(
    restart_config: Mapping[str, Any],
    f_current: float,
    f_candidate: float,
    x_current: np.ndarray,
    x_prev: np.ndarray,
    x_candidate: np.ndarray,
) -> bool:
    if not bool(restart_config.get("enabled", True)):
        return False
    if bool(restart_config.get("objective_increase", True)) and f_candidate > f_current:
        return True
    if bool(restart_config.get("misaligned_momentum", True)):
        displacement_new = x_candidate - x_current
        displacement_old = x_current - x_prev
        alignment = float(np.dot(displacement_new, displacement_old))
        if np.isfinite(alignment) and alignment > 0.0:
            return True
    return False


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
        "[AGD-U]",
        f"run={config.get('run_name', '')}",
        f"dim={dim}",
    ]
    print(" ".join(part for part in parts if not part.endswith("=")), flush=True)


def _print_iter_log(iteration: int, fx: float, grad_norm: float) -> None:
    print(f"[AGD-U] iter={iteration} f={fx:.6e} grad_norm={grad_norm:.3e}", flush=True)


def _run_agd_unknown(
    problem: Problem,
    x0: np.ndarray,
    config: Mapping[str, Any],
    logger: Any,
) -> AGDUnknownResult:
    x = _as_float_vector(x0).copy()
    x_prev = x.copy()
    dim = _problem_dim(problem, x)
    max_iter = int(config.get("max_iter", 100))
    tol = float(config.get("tol", config.get("tol_grad", 1.0e-6)))
    verbose = bool(config.get("verbose", False))
    print_every = max(1, int(config.get("print_every", 10)))
    backtracking_config = _resolve_backtracking_config(config)
    restart_config = _resolve_restart_config(config)

    L0 = float(backtracking_config.get("L0", 1.0))
    L_est = L0
    reuse_previous_L = bool(backtracking_config.get("reuse_previous_L", True))
    t_value = 1.0

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
            "ls_L": L_est,
            "momentum": 0.0,
            "restart": 0,
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
        t_next = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t_value * t_value)))
        momentum = (t_value - 1.0) / t_next
        y = x + momentum * (x - x_prev)
        outcome = _accelerated_backtracking(
            problem=problem,
            y=y,
            L_start=L_est,
            backtracking_config=backtracking_config,
        )
        restart = 0

        if outcome.accepted and _should_restart(
            restart_config=restart_config,
            f_current=fx,
            f_candidate=outcome.f_next,
            x_current=x,
            x_prev=x_prev,
            x_candidate=outcome.x_next,
        ):
            restart = 1
            momentum = 0.0
            outcome = _accelerated_backtracking(
                problem=problem,
                y=x,
                L_start=L_est,
                backtracking_config=backtracking_config,
            )

        if outcome.accepted:
            x_candidate = _as_float_vector(outcome.x_next)
            try:
                f_candidate, grad_candidate, grad_norm_candidate = _evaluate(problem, x_candidate)
            except (FloatingPointError, ValueError):
                f_candidate = np.nan
                grad_candidate = grad
                grad_norm_candidate = np.inf

            if (
                np.all(np.isfinite(x_candidate))
                and np.isfinite(f_candidate)
                and np.all(np.isfinite(grad_candidate))
                and np.isfinite(grad_norm_candidate)
            ):
                step = x_candidate - x
                step_norm = _safe_norm(step)
                x_prev = x
                x = x_candidate
                fx = f_candidate
                grad = grad_candidate
                grad_norm = grad_norm_candidate
                t_value = 1.0 if restart else t_next
                L_est = outcome.L_value if reuse_previous_L else L0
                accepted = True
            else:
                step_norm = 0.0
                accepted = False
        else:
            step_norm = 0.0
            accepted = False

        per_iter_time = float(perf_counter() - iter_start)
        cumulative_time = float(stopwatch.elapsed())
        _log_row(
            history=history,
            logger=logger,
            iteration=iteration,
            fx=fx,
            grad_norm=grad_norm,
            step_norm=step_norm,
            step_size=outcome.step_size if accepted else 0.0,
            cumulative_time=cumulative_time,
            per_iter_time=per_iter_time,
            extras={
                "accepted": int(accepted),
                "ls_iters": outcome.ls_iters,
                "ls_L": outcome.L_value,
                "step_inv_L": outcome.step_size if accepted else 0.0,
                "momentum": momentum,
                "restart": restart,
                "f_y": outcome.f_y,
                "grad_norm_y": outcome.grad_norm_y,
                "hvp_calls_iter": 0,
                "hvp_calls_cum": hvp_calls_cum,
            },
        )

        if verbose and iteration % print_every == 0:
            _print_iter_log(iteration, fx, grad_norm)

        if not accepted:
            status = "line_search_failed"
            break
        if grad_norm <= tol:
            status = "converged"
            break

    return AGDUnknownResult(
        x_final=x,
        history=history,
        status=status,
        iters=max(0, len(history) - 1),
        elapsed_sec=float(stopwatch.elapsed()),
    )


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    result = _run_agd_unknown(problem=problem, x0=x0, config=config, logger=logger)
    final_row = result.history[-1]
    return OptimizeResult(
        x_final=result.x_final,
        f_final=float(final_row["f"]),
        grad_norm_final=float(final_row["grad_norm"]),
        n_iter=result.iters,
        status=result.status,
        history_path=getattr(logger, "history_path", None),
    )

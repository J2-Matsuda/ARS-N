from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from src.algorithms.ars_n.rk import rk_anchor
from src.algorithms.base import (
    GradNormStagnationTracker,
    OptimizeResult,
    evaluate_problem,
    resolve_grad_norm_stagnation_config,
)
from src.problems.base import Problem
from src.utils.sketch import GaussianSketchOperator
from src.utils.timer import Stopwatch

EXTRA_LOG_FIELDS = (
    "f_prev",
    "f_next",
    "grad_norm_prev",
    "grad_norm_next",
    "actual_reduction",
    "accepted",
    "armijo_iters",
    "gtd",
    "used_grad_fallback",
    "eta",
    "lambda_min_phpt",
    "lambda_max_phpt",
    "lambda_shift",
    "cond_phpt_reg",
    "projected_grad_norm",
    "u_norm",
    "y_norm",
    "hvp_calls_iter",
    "hvp_calls_cum",
    "rk_hvp_calls_iter",
    "rk_hvp_calls_cum",
    "subprob_hvp_calls_iter",
    "subprob_hvp_calls_cum",
    "rk_residual_norm_init",
    "rk_residual_norm_final",
    "rk_inner_minres_total_iters",
    "rk_minres_fail",
    "alpha_ws",
    "seed_sketch",
    "rk_seed_base",
    "rk_steps_taken",
    "rk_stop_reason",
    "subproblem_solve_mode",
)


@dataclass(frozen=True)
class ARSRNResult:
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
class _DiagonalShiftInfo:
    eta: float
    lambda_min_phpt: float
    lambda_max_phpt: float
    lambda_shift: float
    cond_phpt_reg: float
    solve_mode: str


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
    return int(np.asarray(x0).size)


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

    line_search.setdefault("c1", 1.0e-4)
    line_search.setdefault("beta", 0.5)
    line_search.setdefault("alpha0", 1.0)
    line_search.setdefault("max_iter", 25)
    return line_search


def _make_gaussian_sketch(
    subspace_dim: int,
    dim: int,
    sketch_config: Mapping[str, Any],
    seed: int,
) -> GaussianSketchOperator:
    if subspace_dim <= 0:
        raise ValueError("subspace_dim must be positive")

    distribution = str(sketch_config.get("distribution", "gaussian"))
    if distribution != "gaussian":
        raise ValueError(f"Unsupported sketch.distribution {distribution!r}. Available: gaussian")
    if bool(sketch_config.get("orthonormalize", False)):
        raise ValueError("ARS-RN requires the raw Gaussian sketch; sketch.orthonormalize is unsupported")

    dtype_name = str(sketch_config.get("dtype", "float64"))
    if dtype_name == "float32":
        dtype = np.float32
    elif dtype_name == "float64":
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported sketch dtype {dtype_name!r}. Available: float32, float64")

    return GaussianSketchOperator(
        shape=(subspace_dim, dim),
        scale=1.0 / np.sqrt(float(subspace_dim)),
        seed=int(seed),
        mode=str(sketch_config.get("mode", "operator")),
        block_size=int(sketch_config.get("block_size", 256)),
        dtype=dtype,
    )


def _build_projected_hessian_from_basis(
    problem: _CountingProblem,
    x: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    basis = np.asarray(basis, dtype=float)
    reduced_dim = int(basis.shape[1])
    hvp_columns = np.empty((basis.shape[0], reduced_dim), dtype=float)
    for column in range(reduced_dim):
        hvp_columns[:, column] = np.asarray(problem.hvp(x, basis[:, column]), dtype=float).reshape(-1)
    projected = basis.T @ hvp_columns
    return 0.5 * (projected + projected.T)


def _solve_diagonal_shift(
    projected_hessian: np.ndarray,
    projected_grad: np.ndarray,
    grad_norm: float,
    regularization_config: Mapping[str, Any],
) -> tuple[np.ndarray, _DiagonalShiftInfo]:
    projected_hessian = np.asarray(projected_hessian, dtype=float)
    projected_grad = np.asarray(projected_grad, dtype=float).reshape(-1)

    lambda_factor = float(regularization_config.get("lambda_factor", 2.0))
    grad_factor = float(regularization_config.get("grad_factor", 1.0))
    grad_exponent = float(regularization_config.get("grad_exponent", 1.0))
    min_eta = float(regularization_config.get("min_eta", 0.0))

    eigvals = np.linalg.eigvalsh(projected_hessian)
    lambda_min = float(eigvals.min()) if eigvals.size else 0.0
    lambda_max = float(eigvals.max()) if eigvals.size else 0.0
    lambda_shift = max(0.0, -lambda_min)
    eta = max(min_eta, lambda_factor * lambda_shift + grad_factor * (grad_norm**grad_exponent))

    regularized = projected_hessian + eta * np.eye(projected_hessian.shape[0], dtype=float)
    try:
        cond_value = float(np.linalg.cond(regularized))
    except np.linalg.LinAlgError:
        cond_value = float("inf")

    rhs = -projected_grad
    solve_mode = "chol"
    try:
        chol = np.linalg.cholesky(regularized)
        intermediate = np.linalg.solve(chol, rhs)
        solution = np.linalg.solve(chol.T, intermediate)
    except np.linalg.LinAlgError:
        try:
            solution = np.linalg.solve(regularized, rhs)
            solve_mode = "solve"
        except np.linalg.LinAlgError:
            solution, *_ = np.linalg.lstsq(regularized, rhs, rcond=None)
            solve_mode = "lstsq"

    solution = np.asarray(solution, dtype=float).reshape(-1)
    if not np.all(np.isfinite(solution)):
        solution = np.zeros_like(rhs)

    return solution, _DiagonalShiftInfo(
        eta=float(eta),
        lambda_min_phpt=lambda_min,
        lambda_max_phpt=lambda_max,
        lambda_shift=float(lambda_shift),
        cond_phpt_reg=cond_value,
        solve_mode=solve_mode,
    )


def _armijo_backtracking(
    problem: _CountingProblem,
    x: np.ndarray,
    fx: float,
    grad: np.ndarray,
    direction: np.ndarray,
    line_search_config: Mapping[str, Any],
) -> _LineSearchOutcome:
    direction = np.asarray(direction, dtype=float).reshape(-1)
    directional_derivative = float(np.dot(grad, direction))
    if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
        return _LineSearchOutcome(
            step_size=0.0,
            x_next=np.asarray(x, dtype=float).copy(),
            f_next=float(fx),
            accepted=False,
            armijo_iters=0,
        )

    if not bool(line_search_config.get("enabled", False)):
        x_next = np.asarray(x, dtype=float) + direction
        f_next = float(problem.f(x_next))
        if not np.isfinite(f_next):
            return _LineSearchOutcome(
                step_size=0.0,
                x_next=np.asarray(x, dtype=float).copy(),
                f_next=float(fx),
                accepted=False,
                armijo_iters=0,
            )
        return _LineSearchOutcome(
            step_size=1.0,
            x_next=x_next,
            f_next=f_next,
            accepted=True,
            armijo_iters=0,
        )

    c1 = float(line_search_config.get("c1", 1.0e-4))
    tau_ls = float(line_search_config.get("beta", 0.5))
    alpha = float(line_search_config.get("alpha0", 1.0))
    max_iter = int(line_search_config.get("max_iter", 25))

    for armijo_iters in range(max_iter):
        x_next = np.asarray(x, dtype=float) + alpha * direction
        f_next = float(problem.f(x_next))
        if np.isfinite(f_next) and f_next <= fx + c1 * alpha * directional_derivative:
            return _LineSearchOutcome(
                step_size=float(alpha),
                x_next=x_next,
                f_next=f_next,
                accepted=True,
                armijo_iters=armijo_iters,
            )
        alpha *= tau_ls

    return _LineSearchOutcome(
        step_size=0.0,
        x_next=np.asarray(x, dtype=float).copy(),
        f_next=float(fx),
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
        "iter": int(iteration),
        "f": float(fx),
        "grad_norm": float(grad_norm),
        "step_norm": float(step_norm),
        "step_size": float(step_size),
        "cumulative_time": float(cumulative_time),
        "per_iter_time": float(per_iter_time),
    }
    if extras:
        row.update(extras)
    history.append(row)
    logger.log(row)


def _print_run_header(config: Mapping[str, Any], dim: int) -> None:
    parts = [
        "[ARS-RN]",
        f"run={config.get('run_name', '')}",
        f"dim={dim}",
        f"subspace_dim={int(config.get('subspace_dim', 0))}",
    ]
    rk_config = dict(config.get("rk", {}))
    if "r" in rk_config:
        parts.append(f"rk_r={int(rk_config['r'])}")
    if "T" in rk_config:
        parts.append(f"rk_T={int(rk_config['T'])}")
    print(" ".join(part for part in parts if not part.endswith("=")), flush=True)


def _print_iter_log(iteration: int, fx: float, grad_norm: float) -> None:
    print(f"[ARS-RN] iter={iteration} f={fx:.6e} grad_norm={grad_norm:.3e}", flush=True)


def _run_ars_rn(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> ARSRNResult:
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    dim = _problem_dim(problem, x)
    max_iter = int(config.get("max_iter", 100))
    tol = float(config.get("tol", 1.0e-6))
    seed = int(config.get("seed", 0))
    verbose = bool(config.get("verbose", False))
    print_every = int(config.get("print_every", 10))
    subspace_dim = int(config.get("subspace_dim", 10))
    if subspace_dim <= 0:
        raise ValueError("subspace_dim must be positive")

    diag_shift_config = dict(config.get("diag_shift", {}))
    line_search_config = _resolve_line_search_config(config)
    sketch_config = dict(config.get("sketch", {}))
    rk_config = dict(config.get("rk", {}))

    counted_problem = _CountingProblem(problem)
    stopwatch = Stopwatch()
    history: list[dict[str, Any]] = []

    fx, grad, grad_norm = evaluate_problem(counted_problem, x)
    if grad_norm > 0.0:
        y0 = np.asarray(rk_config.get("y0", grad / grad_norm), dtype=float).reshape(-1)
        y0_norm = _safe_norm(y0) if np.all(np.isfinite(y0)) and y0.size == dim else float("nan")
    else:
        y0 = None
        y0_norm = float("nan")

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
            "f_prev": fx,
            "f_next": fx,
            "grad_norm_prev": grad_norm,
            "grad_norm_next": grad_norm,
            "actual_reduction": 0.0,
            "accepted": 1,
            "armijo_iters": 0,
            "gtd": float("nan"),
            "used_grad_fallback": 0,
            "eta": float("nan"),
            "lambda_min_phpt": float("nan"),
            "lambda_max_phpt": float("nan"),
            "lambda_shift": float("nan"),
            "cond_phpt_reg": float("nan"),
            "projected_grad_norm": float("nan"),
            "u_norm": float("nan"),
            "y_norm": y0_norm,
            "hvp_calls_iter": 0,
            "hvp_calls_cum": 0,
            "rk_hvp_calls_iter": 0,
            "rk_hvp_calls_cum": 0,
            "subprob_hvp_calls_iter": 0,
            "subprob_hvp_calls_cum": 0,
            "rk_residual_norm_init": float("nan"),
            "rk_residual_norm_final": float("nan"),
            "rk_inner_minres_total_iters": 0,
            "rk_minres_fail": 0,
            "alpha_ws": float("nan"),
            "seed_sketch": "",
            "rk_seed_base": "",
            "rk_steps_taken": 0,
            "rk_stop_reason": "",
            "subproblem_solve_mode": "",
        },
    )

    if grad_norm <= tol:
        return ARSRNResult(
            x_final=x,
            history=history,
            status="converged",
            iters=0,
            elapsed_sec=float(stopwatch.elapsed()),
        )

    if verbose:
        _print_run_header(config, dim)

    sketch_seeds = np.random.SeedSequence(seed).spawn(max_iter)
    grad_norm_stagnation = GradNormStagnationTracker(resolve_grad_norm_stagnation_config(config))
    hvp_calls_cum = 0
    rk_hvp_calls_cum = 0
    subprob_hvp_calls_cum = 0

    x_prev: np.ndarray | None = None
    g_prev: np.ndarray | None = None
    y_prev: np.ndarray | None = None

    status = "max_iter"
    grad_norm_stagnation.update(grad_norm)
    for iteration in range(1, max_iter + 1):
        if grad_norm <= tol:
            status = "converged"
            break

        iter_start = perf_counter()
        x_current = x.copy()
        f_prev = float(fx)
        g_current = np.asarray(grad, dtype=float).reshape(-1)
        grad_norm_prev = float(grad_norm)

        hvp_before_iter = counted_problem.hvp_calls
        hvp_before_rk = counted_problem.hvp_calls
        try:
            y_candidate, rk_info = rk_anchor(
                problem=counted_problem,
                x_k=x_current,
                g_k=g_current,
                k=iteration - 1,
                rk_config=rk_config,
                y0=y0 if iteration == 1 else None,
                x_prev=x_prev,
                g_prev=g_prev,
                y_prev=y_prev,
            )
        except Exception as exc:
            print(f"[ARS-RN][RK] failed: {type(exc).__name__}: {exc}", flush=True)
            rk_info = {
                "alpha_ws": float("nan"),
                "rk_residual_norm_init": float("nan"),
                "rk_residual_norm_final": float("nan"),
                "hvp_calls": counted_problem.hvp_calls - hvp_before_rk,
                "inner_minres_total_iters": 0,
                "minres_fail": 1,
            }
            y_candidate = np.zeros_like(g_current)

        rk_hvp_calls_iter = int(counted_problem.hvp_calls - hvp_before_rk)
        if not np.all(np.isfinite(y_candidate)) or _safe_norm(y_candidate) == 0.0:
            if y_prev is not None and np.all(np.isfinite(y_prev)) and _safe_norm(y_prev) > 0.0:
                y_candidate = y_prev.copy()
            else:
                y_candidate = -g_current

        y_norm = _safe_norm(y_candidate)
        if not np.isfinite(y_norm) or y_norm == 0.0:
            y_candidate = -g_current
            y_norm = _safe_norm(y_candidate)
        q0 = y_candidate / y_norm

        sketch_seed = int(sketch_seeds[iteration - 1].generate_state(1)[0])
        sketch = _make_gaussian_sketch(subspace_dim, dim, sketch_config, sketch_seed)
        sketch_matrix = sketch.dense_matrix()
        Q_k = np.column_stack((q0, sketch_matrix.T))
        projected_grad = np.concatenate(([float(np.dot(q0, g_current))], sketch_matrix @ g_current))
        projected_grad_norm = _safe_norm(projected_grad)

        hvp_before_subprob = counted_problem.hvp_calls
        projected_hessian = _build_projected_hessian_from_basis(counted_problem, x_current, Q_k)
        subprob_hvp_calls_iter = int(counted_problem.hvp_calls - hvp_before_subprob)

        reduced_step, diag_info = _solve_diagonal_shift(
            projected_hessian=projected_hessian,
            projected_grad=projected_grad,
            grad_norm=grad_norm_prev,
            regularization_config=diag_shift_config,
        )
        u_norm = _safe_norm(reduced_step)
        direction = Q_k @ reduced_step

        gtd = float(np.dot(g_current, direction))
        used_grad_fallback = 0
        if (
            not np.all(np.isfinite(direction))
            or not np.isfinite(gtd)
            or gtd >= 0.0
        ):
            direction = -g_current
            gtd = -float(np.dot(g_current, g_current))
            used_grad_fallback = 1

        line_search = _armijo_backtracking(
            problem=counted_problem,
            x=x_current,
            fx=f_prev,
            grad=g_current,
            direction=direction,
            line_search_config=line_search_config,
        )

        x_next = line_search.x_next
        f_next = float(line_search.f_next)
        if line_search.accepted:
            _, grad_next, grad_norm_next = evaluate_problem(counted_problem, x_next)
        else:
            grad_next = g_current
            grad_norm_next = grad_norm_prev

        actual_reduction = float(f_prev - f_next)
        step = x_next - x_current
        step_norm = _safe_norm(step)
        if not line_search.accepted:
            step_norm = 0.0

        hvp_calls_iter = int(counted_problem.hvp_calls - hvp_before_iter)
        hvp_calls_cum += hvp_calls_iter
        rk_hvp_calls_cum += rk_hvp_calls_iter
        subprob_hvp_calls_cum += subprob_hvp_calls_iter

        per_iter_time = float(perf_counter() - iter_start)
        cumulative_time = float(stopwatch.elapsed())
        _log_row(
            history=history,
            logger=logger,
            iteration=iteration,
            fx=f_next,
            grad_norm=grad_norm_next,
            step_norm=step_norm,
            step_size=line_search.step_size,
            cumulative_time=cumulative_time,
            per_iter_time=per_iter_time,
            extras={
                "f_prev": f_prev,
                "f_next": f_next,
                "grad_norm_prev": grad_norm_prev,
                "grad_norm_next": grad_norm_next,
                "actual_reduction": actual_reduction,
                "accepted": int(line_search.accepted),
                "armijo_iters": line_search.armijo_iters,
                "gtd": gtd,
                "used_grad_fallback": used_grad_fallback,
                "eta": diag_info.eta,
                "lambda_min_phpt": diag_info.lambda_min_phpt,
                "lambda_max_phpt": diag_info.lambda_max_phpt,
                "lambda_shift": diag_info.lambda_shift,
                "cond_phpt_reg": diag_info.cond_phpt_reg,
                "projected_grad_norm": projected_grad_norm,
                "u_norm": u_norm,
                "y_norm": y_norm,
                "hvp_calls_iter": hvp_calls_iter,
                "hvp_calls_cum": hvp_calls_cum,
                "rk_hvp_calls_iter": rk_hvp_calls_iter,
                "rk_hvp_calls_cum": rk_hvp_calls_cum,
                "subprob_hvp_calls_iter": subprob_hvp_calls_iter,
                "subprob_hvp_calls_cum": subprob_hvp_calls_cum,
                "rk_residual_norm_init": float(rk_info.get("rk_residual_norm_init", float("nan"))),
                "rk_residual_norm_final": float(rk_info.get("rk_residual_norm_final", float("nan"))),
                "rk_inner_minres_total_iters": int(rk_info.get("inner_minres_total_iters", 0)),
                "rk_minres_fail": int(rk_info.get("minres_fail", 0)),
                "alpha_ws": float(rk_info.get("alpha_ws", float("nan"))),
                "seed_sketch": sketch_seed,
                "rk_seed_base": int(rk_info.get("rk_seed_base", -1)),
                "rk_steps_taken": int(rk_info.get("rk_steps_taken", 0)),
                "rk_stop_reason": str(rk_info.get("rk_stop_reason", "")),
                "subproblem_solve_mode": diag_info.solve_mode,
            },
        )

        if verbose and iteration % max(1, print_every) == 0:
            _print_iter_log(iteration, f_next, grad_norm_next)

        x_prev = x_current
        g_prev = g_current
        y_prev = y_candidate.copy()
        x = x_next
        fx = f_next
        grad = grad_next
        grad_norm = float(grad_norm_next)

        if not line_search.accepted:
            status = "line_search_failed"
            break
        if grad_norm <= tol:
            status = "converged"
            break
        if grad_norm_stagnation.update(grad_norm):
            status = "grad_norm_stagnation"
            break

    return ARSRNResult(
        x_final=x,
        history=history,
        status=status,
        iters=max(0, len(history) - 1),
        elapsed_sec=float(stopwatch.elapsed()),
    )


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    result = _run_ars_rn(problem=problem, x0=x0, config=config, logger=logger)
    final_row = result.history[-1]
    return OptimizeResult(
        x_final=result.x_final,
        f_final=float(final_row["f"]),
        grad_norm_final=float(final_row["grad_norm"]),
        n_iter=result.iters,
        status=result.status,
        history_path=getattr(logger, "history_path", None),
    )

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from src.algorithms.base import (
    GradNormStagnationTracker,
    OptimizeResult,
    evaluate_problem,
    resolve_grad_norm_stagnation_config,
)
from src.problems.base import Problem
from src.utils.timer import Stopwatch

EXTRA_LOG_FIELDS = (
    "variant",
    "accepted",
    "gtd",
    "rho",
    "sigma",
    "model_decrease",
    "actual_decrease",
    "lambda_value",
    "cubic_solver",
    "lambda_min_h",
    "lambda_shift",
    "hvp_calls_iter",
    "hvp_calls_cum",
    "used_grad_fallback",
)


@dataclass(frozen=True)
class CNMResult:
    x_final: np.ndarray
    history: list[dict[str, Any]]
    status: str
    iters: int
    elapsed_sec: float


@dataclass(frozen=True)
class _CubicSolveResult:
    step: np.ndarray
    lambda_value: float
    mode: str


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


def _reshape_to_vector(value: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _problem_dim(problem: Problem, x0: np.ndarray) -> int:
    if hasattr(problem, "dim"):
        return int(getattr(problem, "dim"))
    if hasattr(problem, "n"):
        return int(getattr(problem, "n"))
    return int(np.asarray(x0).size)


def _build_full_hessian(problem: _CountingProblem, x: np.ndarray, dim: int) -> np.ndarray:
    hessian = np.empty((dim, dim), dtype=float)
    identity = np.eye(dim, dtype=float)
    for column in range(dim):
        hessian[:, column] = problem.hvp(x, identity[:, column])
    return 0.5 * (hessian + hessian.T)


def _solve_cauchy_point(
    grad: np.ndarray,
    hessian: np.ndarray,
    sigma: float,
) -> _CubicSolveResult:
    grad = _reshape_to_vector(grad)
    if grad.size == 0:
        return _CubicSolveResult(step=grad, lambda_value=0.0, mode="cauchy_point")

    grad_norm = _safe_norm(grad)
    if grad_norm == 0.0:
        return _CubicSolveResult(step=np.zeros_like(grad), lambda_value=0.0, mode="cauchy_point")

    if not np.isfinite(sigma) or sigma <= 0.0:
        return _CubicSolveResult(step=-grad, lambda_value=0.0, mode="cauchy_point")

    h_grad = np.asarray(hessian, dtype=float) @ grad
    g_h_g = float(np.dot(grad, h_grad))

    a_coeff = sigma * (grad_norm**3)
    b_coeff = g_h_g
    c_coeff = -(grad_norm**2)

    if abs(a_coeff) <= 1.0e-18:
        if abs(b_coeff) <= 1.0e-18:
            step_scale = 0.0
        else:
            step_scale = -c_coeff / b_coeff
    else:
        discriminant = max(b_coeff * b_coeff - 4.0 * a_coeff * c_coeff, 0.0)
        step_scale = (-b_coeff + np.sqrt(discriminant)) / (2.0 * a_coeff)

    step_scale = max(step_scale, 0.0)
    step = -step_scale * grad
    return _CubicSolveResult(
        step=np.asarray(step, dtype=float).reshape(-1),
        lambda_value=float(sigma * _safe_norm(step)),
        mode="cauchy_point",
    )


def _solve_cubic_subproblem_exact(
    grad: np.ndarray,
    hessian: np.ndarray,
    sigma: float,
    exact_tol: float,
    lambda_init: float | None,
) -> tuple[np.ndarray, float]:
    grad = _reshape_to_vector(grad)
    dim = int(grad.size)
    if dim == 0:
        return grad, 0.0

    hessian = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if not np.isfinite(sigma) or sigma <= 0.0:
        try:
            step = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(hessian) @ grad
        return np.asarray(step, dtype=float).reshape(-1), 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    grad_hat = eigenvectors.T @ grad
    d_min = float(eigenvalues.min())
    lambda_psd = max(0.0, -d_min)

    denom_tol = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    grad_tol = 1.0e-12 * max(1.0, _safe_norm(grad_hat))

    def _step_hat_and_norm(lam: float) -> tuple[np.ndarray, float, bool]:
        denom = eigenvalues + lam
        step_hat = np.zeros(dim, dtype=float)
        singular = False
        for index in range(dim):
            if abs(denom[index]) <= denom_tol:
                if abs(grad_hat[index]) > grad_tol:
                    singular = True
                else:
                    step_hat[index] = 0.0
            else:
                step_hat[index] = -grad_hat[index] / denom[index]

        if singular:
            return step_hat, np.inf, True
        return step_hat, _safe_norm(step_hat), False

    lambda_low = lambda_psd
    step_hat_low, norm_low, singular_low = _step_hat_and_norm(lambda_low)
    if singular_low:
        lambda_low += max(1.0e-12, 1.0e-8 * max(1.0, lambda_low))
        step_hat_low, norm_low, singular_low = _step_hat_and_norm(lambda_low)
        if singular_low:
            raise RuntimeError("Exact cubic solver failed near PSD shift")

    phi_low = sigma * norm_low - lambda_low
    if phi_low <= 0.0:
        lambda_hard = lambda_psd
        step_hat0, norm0, _ = _step_hat_and_norm(lambda_hard)
        step0 = eigenvectors @ step_hat0
        if lambda_hard == 0.0:
            return np.asarray(step0, dtype=float).reshape(-1), 0.0

        target = max(lambda_hard / sigma, norm0)
        tau_sq = max(target * target - norm0 * norm0, 0.0)
        eig_tol = 1.0e-12 * max(1.0, abs(d_min))
        min_indices = np.where(np.abs(eigenvalues - d_min) <= eig_tol)[0]
        min_index = int(min_indices[0]) if min_indices.size > 0 else int(np.argmin(eigenvalues))
        step = step0 + np.sqrt(tau_sq) * eigenvectors[:, min_index]
        return np.asarray(step, dtype=float).reshape(-1), float(lambda_hard)

    lambda_high = max(lambda_low, 1.0)
    success_bracket = False
    for _ in range(100):
        _, norm_high, singular_high = _step_hat_and_norm(lambda_high)
        if singular_high:
            lambda_high *= 2.0
            continue
        phi_high = sigma * norm_high - lambda_high
        if phi_high < 0.0:
            success_bracket = True
            break
        lambda_high *= 2.0

    if not success_bracket:
        raise RuntimeError("Exact cubic solver failed to bracket lambda")

    if lambda_init is not None and lambda_low <= lambda_init <= lambda_high:
        lam = float(lambda_init)
    else:
        lam = 0.5 * (lambda_low + lambda_high)

    step_hat = step_hat_low
    for _ in range(100):
        step_hat, norm_step, singular = _step_hat_and_norm(lam)
        if singular:
            lambda_low = max(lambda_low, lam)
            lam = 0.5 * (lambda_low + lambda_high)
            continue

        phi = sigma * norm_step - lam
        if abs(phi) <= exact_tol * max(1.0, lam):
            break

        if phi > 0.0:
            lambda_low = lam
        else:
            lambda_high = lam
        lam = 0.5 * (lambda_low + lambda_high)

    step = eigenvectors @ step_hat
    return np.asarray(step, dtype=float).reshape(-1), float(lam)


def _solve_cubic_step(
    solver: str,
    grad: np.ndarray,
    hessian: np.ndarray,
    sigma: float,
    exact_tol: float,
    lambda_init: float | None,
) -> _CubicSolveResult:
    method = str(solver).lower()
    if method == "cauchy_point":
        return _solve_cauchy_point(grad, hessian, sigma)
    if method == "exact":
        step, lambda_value = _solve_cubic_subproblem_exact(grad, hessian, sigma, exact_tol, lambda_init)
        return _CubicSolveResult(step=step, lambda_value=lambda_value, mode="exact")
    raise ValueError(f"Unknown cubic solver {solver!r}. Available: exact, cauchy_point")


def _resolve_variant(config: Mapping[str, Any]) -> str:
    variant = str(config.get("variant", "arc")).lower()
    if variant not in {"cr", "arc"}:
        raise ValueError(f"Unknown cubic Newton variant {variant!r}. Available: cr, arc")
    return variant


def _resolve_arc_parameters(config: Mapping[str, Any], variant: str) -> tuple[float, float, float, float, float, float, float]:
    sigma_default = float(config.get("sigma", config.get("sigma0", 1.0)))
    sigma = sigma_default
    sigma_min = float(config.get("sigma_min", 1.0e-8))
    sigma_max = float(config.get("sigma_max", 1.0e8))
    eta1 = float(config.get("eta1", 0.1))
    eta2 = float(config.get("eta2", 0.75))
    gamma1 = float(config.get("gamma1", 2.0))
    gamma2 = float(config.get("gamma2", 2.0))

    if variant == "cr":
        sigma_min = sigma
        sigma_max = sigma

    return sigma, sigma_min, sigma_max, eta1, eta2, gamma1, gamma2


def _cubic_model_decrease(
    grad: np.ndarray,
    hessian: np.ndarray,
    step: np.ndarray,
    sigma: float,
) -> tuple[float, float]:
    step = _reshape_to_vector(step)
    gtd = float(np.dot(grad, step))
    h_step = np.asarray(hessian, dtype=float) @ step
    quad_term = 0.5 * float(np.dot(step, h_step))
    cubic_term = (float(sigma) / 3.0) * (_safe_norm(step) ** 3)
    model_decrease = -float(gtd + quad_term + cubic_term)
    return gtd, model_decrease


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


def _print_run_header(config: Mapping[str, Any], dim: int, variant: str, solver: str) -> None:
    parts = [
        "[CN]",
        f"run={config.get('run_name', '')}",
        f"dim={dim}",
        f"variant={variant}",
        f"solver={solver}",
    ]
    print(" ".join(part for part in parts if not part.endswith("=")), flush=True)


def _print_iter_log(iteration: int, fx: float, grad_norm: float, sigma: float) -> None:
    print(f"[CN] iter={iteration} f={fx:.6e} grad_norm={grad_norm:.3e} sigma={sigma:.3e}", flush=True)


def _run_cnm(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> CNMResult:
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    dim = _problem_dim(problem, x)
    max_iter = int(config.get("max_iter", 100))
    tol = float(config.get("tol", 1.0e-6))
    verbose = bool(config.get("verbose", False))
    print_every = int(config.get("print_every", 10))
    variant = _resolve_variant(config)
    solver = str(config.get("solver", "exact")).lower()
    exact_tol = float(config.get("exact_tol", 1.0e-10))
    sigma, sigma_min, sigma_max, eta1, eta2, gamma1, gamma2 = _resolve_arc_parameters(config, variant)
    grad_norm_stagnation = GradNormStagnationTracker(resolve_grad_norm_stagnation_config(config))

    counted_problem = _CountingProblem(problem)
    stopwatch = Stopwatch()
    history: list[dict[str, Any]] = []
    hvp_calls_cum = 0
    lambda_value = 0.0
    successful_flag = False

    fx, grad, grad_norm = evaluate_problem(counted_problem, x)
    grad_norm_stagnation.update(grad_norm)
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
            "variant": variant,
            "accepted": 1,
            "gtd": float("nan"),
            "rho": float("nan"),
            "sigma": sigma,
            "model_decrease": float("nan"),
            "actual_decrease": 0.0,
            "lambda_value": lambda_value,
            "cubic_solver": solver,
            "lambda_min_h": float("nan"),
            "lambda_shift": float("nan"),
            "hvp_calls_iter": 0,
            "hvp_calls_cum": 0,
            "used_grad_fallback": 0,
        },
    )

    if grad_norm <= tol:
        return CNMResult(
            x_final=x,
            history=history,
            status="converged",
            iters=0,
            elapsed_sec=float(stopwatch.elapsed()),
        )

    if verbose:
        _print_run_header(config, dim, variant, solver)

    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        if grad_norm <= tol:
            status = "converged"
            break

        iter_start = perf_counter()
        f_prev = float(fx)
        grad_current = np.asarray(grad, dtype=float).reshape(-1)

        counted_problem.hvp_calls = 0
        hessian = _build_full_hessian(counted_problem, x, dim)
        eigenvalues = np.linalg.eigvalsh(hessian)
        lambda_min_h = float(eigenvalues.min()) if eigenvalues.size else 0.0
        lambda_shift = max(0.0, -lambda_min_h)

        lambda_init = float(lambda_value) if (variant == "arc" and (not successful_flag) and lambda_value > 0.0) else None
        cubic_result = _solve_cubic_step(
            solver=solver,
            grad=grad_current,
            hessian=hessian,
            sigma=sigma,
            exact_tol=exact_tol,
            lambda_init=lambda_init,
        )
        step = np.asarray(cubic_result.step, dtype=float).reshape(-1)
        lambda_value = float(cubic_result.lambda_value)

        gtd, model_decrease = _cubic_model_decrease(grad_current, hessian, step, sigma)
        used_grad_fallback = 0
        if not np.all(np.isfinite(step)) or not np.isfinite(gtd) or gtd >= 0.0:
            step = -grad_current
            lambda_value = 0.0
            gtd, model_decrease = _cubic_model_decrease(grad_current, hessian, step, sigma)
            used_grad_fallback = 1

        step_norm_trial = _safe_norm(step)
        x_trial = x + step
        f_trial = float(counted_problem.f(x_trial))
        actual_decrease = float(f_prev - f_trial)

        if variant == "cr":
            accepted = True
            rho = float("nan")
            x_next = x_trial
            f_next = f_trial
            _, grad_next, grad_norm_next = evaluate_problem(counted_problem, x_next)
            successful_flag = True
        else:
            if np.isfinite(model_decrease) and model_decrease > 0.0 and np.isfinite(actual_decrease):
                rho = float(actual_decrease / model_decrease)
            else:
                rho = float("-inf")

            accepted = bool(rho >= eta1 and actual_decrease >= 0.0)
            if accepted:
                x_next = x_trial
                f_next = f_trial
                _, grad_next, grad_norm_next = evaluate_problem(counted_problem, x_next)
            else:
                x_next = x
                f_next = f_prev
                grad_next = grad_current
                grad_norm_next = float(grad_norm)

            successful_flag = accepted
            if rho >= eta2:
                sigma = max(sigma / gamma2, sigma_min)
            elif rho < eta1 or not np.isfinite(rho):
                sigma = min(gamma1 * sigma, sigma_max)

        hvp_calls_iter = int(counted_problem.hvp_calls)
        hvp_calls_cum += hvp_calls_iter

        per_iter_time = float(perf_counter() - iter_start)
        cumulative_time = float(stopwatch.elapsed())
        _log_row(
            history=history,
            logger=logger,
            iteration=iteration,
            fx=f_next,
            grad_norm=grad_norm_next,
            step_norm=step_norm_trial if accepted or variant == "cr" else 0.0,
            step_size=1.0 if accepted or variant == "cr" else 0.0,
            cumulative_time=cumulative_time,
            per_iter_time=per_iter_time,
            extras={
                "variant": variant,
                "accepted": int(accepted),
                "gtd": gtd,
                "rho": rho,
                "sigma": sigma,
                "model_decrease": model_decrease,
                "actual_decrease": actual_decrease,
                "lambda_value": lambda_value,
                "cubic_solver": cubic_result.mode,
                "lambda_min_h": lambda_min_h,
                "lambda_shift": lambda_shift,
                "hvp_calls_iter": hvp_calls_iter,
                "hvp_calls_cum": hvp_calls_cum,
                "used_grad_fallback": used_grad_fallback,
            },
        )

        if verbose and iteration % max(1, print_every) == 0:
            _print_iter_log(iteration, f_next, grad_norm_next, sigma)

        x = x_next
        fx = f_next
        grad = grad_next
        grad_norm = float(grad_norm_next)
        if grad_norm <= tol:
            status = "converged"
            break
        if grad_norm_stagnation.update(grad_norm):
            status = "grad_norm_stagnation"
            break

    return CNMResult(
        x_final=x,
        history=history,
        status=status,
        iters=max(0, len(history) - 1),
        elapsed_sec=float(stopwatch.elapsed()),
    )


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    result = _run_cnm(problem=problem, x0=x0, config=config, logger=logger)
    final_row = result.history[-1]
    return OptimizeResult(
        x_final=result.x_final,
        f_final=float(final_row["f"]),
        grad_norm_final=float(final_row["grad_norm"]),
        n_iter=result.iters,
        status=result.status,
        history_path=getattr(logger, "history_path", None),
    )

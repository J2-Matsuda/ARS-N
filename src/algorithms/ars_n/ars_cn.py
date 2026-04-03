from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from src.algorithms.ars_n.rk import rk_anchor
from src.algorithms.base import OptimizeResult, evaluate_problem
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
    "gtd",
    "rho",
    "sigma",
    "model_decrease",
    "actual_decrease",
    "lambda_value",
    "cubic_solver",
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
    "alpha_ws",
)


@dataclass(frozen=True)
class ARSCNResult:
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
        raise ValueError("ARS-CN requires the raw Gaussian sketch; sketch.orthonormalize is unsupported")

    dtype_name = str(sketch_config.get("dtype", "float64"))
    if dtype_name not in {"float32", "float64"}:
        raise ValueError(f"Unsupported sketch dtype {dtype_name!r}. Available: float32, float64")

    return GaussianSketchOperator(
        shape=(subspace_dim, dim),
        scale=1.0 / np.sqrt(float(subspace_dim)),
        seed=int(seed),
        mode=str(sketch_config.get("mode", "operator")),
        block_size=int(sketch_config.get("block_size", 256)),
        dtype=np.float32 if dtype_name == "float32" else np.float64,
    )


def _build_tridiagonal(alphas: list[float], betas: list[float]) -> np.ndarray:
    size = len(alphas)
    matrix = np.zeros((size, size), dtype=float)
    for index, alpha in enumerate(alphas):
        matrix[index, index] = alpha
    for index, beta in enumerate(betas):
        matrix[index, index + 1] = beta
        matrix[index + 1, index] = beta
    return matrix


def _explicit_hessian(hv: Callable[[np.ndarray], np.ndarray], dim: int) -> np.ndarray:
    matrix = np.zeros((dim, dim), dtype=float)
    for column in range(dim):
        basis = np.zeros(dim, dtype=float)
        basis[column] = 1.0
        matrix[:, column] = hv(basis)
    return 0.5 * (matrix + matrix.T)


def _resolve_hessian(
    hessian: np.ndarray | Callable[[Any], np.ndarray] | None,
    hv: Callable[[np.ndarray], np.ndarray],
    dim: int,
    w: np.ndarray | None,
) -> np.ndarray:
    if hessian is None:
        matrix = _explicit_hessian(hv, dim)
    elif callable(hessian):
        try:
            matrix = hessian(w)
        except TypeError:
            matrix = hessian(None)
    else:
        matrix = np.asarray(hessian, dtype=float)

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (dim, dim):
        matrix = _explicit_hessian(hv, dim)
    return 0.5 * (matrix + matrix.T)


def _solve_cauchy_point(
    grad: np.ndarray,
    hv: Callable[[np.ndarray], np.ndarray],
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

    h_grad = hv(grad)
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
        discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff
        discriminant = max(discriminant, 0.0)
        step_scale = (-b_coeff + np.sqrt(discriminant)) / (2.0 * a_coeff)

    if step_scale < 0.0:
        step_scale = 0.0

    step = -step_scale * grad
    return _CubicSolveResult(
        step=np.asarray(step, dtype=float),
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
        return np.asarray(step, dtype=float), 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    grad_hat = eigenvectors.T @ grad
    d_min = float(eigenvalues.min())
    lambda_psd = max(0.0, -d_min)

    denom_tol = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    grad_tol = 1.0e-12 * max(1.0, _safe_norm(grad_hat))

    def s_hat_and_norm(lam: float) -> tuple[np.ndarray, float, bool]:
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
    step_hat_low, norm_low, singular_low = s_hat_and_norm(lambda_low)
    if singular_low:
        lambda_low += max(1.0e-12, 1.0e-8 * max(1.0, lambda_low))
        step_hat_low, norm_low, singular_low = s_hat_and_norm(lambda_low)
        if singular_low:
            raise RuntimeError("Exact cubic solver failed near PSD shift")

    phi_low = sigma * norm_low - lambda_low
    if phi_low <= 0.0:
        lambda_hard = lambda_psd
        step_hat0, norm0, _ = s_hat_and_norm(lambda_hard)
        step0 = eigenvectors @ step_hat0
        if lambda_hard == 0.0:
            return np.asarray(step0, dtype=float), 0.0

        target = lambda_hard / sigma
        if target < norm0:
            target = norm0

        tau_sq = max(target * target - norm0 * norm0, 0.0)
        eig_tol = 1.0e-12 * max(1.0, abs(d_min))
        min_indices = np.where(np.abs(eigenvalues - d_min) <= eig_tol)[0]
        min_index = int(min_indices[0]) if min_indices.size > 0 else int(np.argmin(eigenvalues))
        step = step0 + np.sqrt(tau_sq) * eigenvectors[:, min_index]
        return np.asarray(step, dtype=float), lambda_hard

    lambda_high = max(lambda_low, 1.0)
    success_bracket = False
    for _ in range(100):
        _, norm_high, singular_high = s_hat_and_norm(lambda_high)
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
        step_hat, norm_step, singular = s_hat_and_norm(lam)
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
    return np.asarray(step, dtype=float), float(lam)


def _reconstruct_from_lanczos(
    hv: Callable[[np.ndarray], np.ndarray],
    v0: np.ndarray,
    u_hat: np.ndarray,
    betas: list[float],
) -> np.ndarray:
    v0 = _reshape_to_vector(v0)
    norm_v0 = _safe_norm(v0)
    if norm_v0 == 0.0:
        return np.zeros_like(v0)

    q = v0 / norm_v0
    q_prev = np.zeros_like(v0)
    beta_prev = 0.0
    step = np.zeros_like(v0)

    for index in range(u_hat.size):
        step = step + u_hat[index] * q
        z = hv(q) - beta_prev * q_prev
        alpha = float(np.dot(q, z))
        z = z - alpha * q

        if index == u_hat.size - 1:
            break

        if index < len(betas):
            beta = float(betas[index])
        else:
            beta = _safe_norm(z)
        if beta <= 0.0:
            break

        q_prev = q
        q = z / beta
        beta_prev = beta

    return np.asarray(step, dtype=float)


def _solve_arc_subproblem(
    solver: str,
    grad: np.ndarray,
    hv: Callable[[np.ndarray], np.ndarray],
    hessian: np.ndarray | Callable[[Any], np.ndarray] | None,
    sigma: float,
    w: np.ndarray,
    successful_flag: bool,
    lambda_k: float,
    exact_tol: float,
    krylov_tol: float,
    solve_each_i_th_krylov_space: int,
    keep_q_matrix_in_memory: bool,
) -> _CubicSolveResult:
    grad = _reshape_to_vector(grad)
    dim = int(grad.size)
    if dim == 0:
        return _CubicSolveResult(step=grad, lambda_value=0.0, mode=str(solver).lower())

    method = str(solver).lower()
    lambda_init = float(lambda_k) if (not successful_flag and lambda_k > 0.0) else None
    if method == "cauchy_point":
        return _solve_cauchy_point(grad, hv, sigma)

    if method == "exact":
        matrix = _resolve_hessian(hessian, hv, dim, w)
        step, lambda_value = _solve_cubic_subproblem_exact(grad, matrix, sigma, exact_tol, lambda_init)
        return _CubicSolveResult(step=step, lambda_value=lambda_value, mode="exact")

    if method != "lanczos":
        raise ValueError(f"Unknown cubic solver {solver!r}. Available: cauchy_point, exact, lanczos")

    grad_norm = _safe_norm(grad)
    if grad_norm == 0.0:
        return _CubicSolveResult(step=np.zeros(dim, dtype=float), lambda_value=0.0, mode="lanczos")

    max_lanczos_iter = max(1, dim)
    tol = float(krylov_tol) if float(krylov_tol) > 0.0 else 1.0e-12
    solve_frequency = max(1, int(solve_each_i_th_krylov_space))

    alphas: list[float] = []
    betas: list[float] = []
    q_vectors: list[np.ndarray] = []
    q = grad / grad_norm
    q_prev = np.zeros(dim, dtype=float)
    beta_prev = 0.0

    u_hat_best: np.ndarray | None = None
    betas_best: list[float] = []
    lambda_hat = lambda_init

    for iteration in range(1, max_lanczos_iter + 1):
        if keep_q_matrix_in_memory:
            q_vectors.append(q.copy())

        z = hv(q) - beta_prev * q_prev
        alpha = float(np.dot(q, z))
        z = z - alpha * q
        if keep_q_matrix_in_memory:
            for q_i in q_vectors:
                z = z - float(np.dot(q_i, z)) * q_i

        if iteration >= 2:
            betas.append(beta_prev)
        alphas.append(alpha)
        beta = _safe_norm(z)

        should_solve = (
            iteration % solve_frequency == 0
            or iteration == max_lanczos_iter
            or beta <= tol
        )
        if should_solve:
            tridiagonal = _build_tridiagonal(alphas, betas)
            grad_hat = np.zeros(iteration, dtype=float)
            grad_hat[0] = grad_norm
            try:
                u_hat, lambda_hat = _solve_cubic_subproblem_exact(
                    grad_hat,
                    tridiagonal,
                    sigma,
                    exact_tol,
                    lambda_hat,
                )
            except Exception:
                matrix = _resolve_hessian(hessian, hv, dim, w)
                step, lambda_value = _solve_cubic_subproblem_exact(
                    grad,
                    matrix,
                    sigma,
                    exact_tol,
                    lambda_init,
                )
                return _CubicSolveResult(step=step, lambda_value=lambda_value, mode="exact")

            u_hat_best = u_hat
            betas_best = betas.copy()
            if beta <= tol:
                break
            if iteration < max_lanczos_iter:
                if abs(beta * u_hat[-1]) <= tol * max(1.0, _safe_norm(u_hat)):
                    break

        if beta <= tol:
            break

        q_prev = q
        q = z / beta
        beta_prev = beta

    if u_hat_best is None:
        raise RuntimeError("Lanczos failed to produce a cubic subproblem solution")

    if keep_q_matrix_in_memory:
        q_matrix = np.column_stack(q_vectors[: u_hat_best.size])
        step = q_matrix @ u_hat_best
    else:
        step = _reconstruct_from_lanczos(hv, grad, u_hat_best, betas_best)

    return _CubicSolveResult(
        step=np.asarray(step, dtype=float),
        lambda_value=float(sigma * _safe_norm(step)),
        mode="lanczos",
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
        "[ARS-CN]",
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
    print(f"[ARS-CN] iter={iteration} f={fx:.6e} grad_norm={grad_norm:.3e}", flush=True)


def _run_ars_cn(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> ARSCNResult:
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

    rk_config = dict(config.get("rk", {}))
    sketch_config = dict(config.get("sketch", {}))
    sigma = float(config.get("sigma0", 1.0))
    sigma_min = float(config.get("sigma_min", 1.0e-8))
    sigma_max = float(config.get("sigma_max", 1.0e8))
    eta1 = float(config.get("eta1", 0.1))
    eta2 = float(config.get("eta2", 0.75))
    gamma1 = float(config.get("gamma1", 2.0))
    gamma2 = float(config.get("gamma2", 2.0))
    solver = str(config.get("solver", "exact"))
    exact_tol = float(config.get("exact_tol", 1.0e-10))
    krylov_tol = float(config.get("krylov_tol", 1.0e-10))
    solve_each_i_th_krylov_space = int(config.get("solve_each_i_th_krylov_space", 1))
    keep_q_matrix_in_memory = bool(config.get("keep_Q_matrix_in_memory", True))

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

    lambda_value = 0.0
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
            "gtd": float("nan"),
            "rho": float("nan"),
            "sigma": sigma,
            "model_decrease": float("nan"),
            "actual_decrease": 0.0,
            "lambda_value": lambda_value,
            "cubic_solver": solver,
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
            "alpha_ws": float("nan"),
        },
    )

    if grad_norm <= tol:
        return ARSCNResult(
            x_final=x,
            history=history,
            status="converged",
            iters=0,
            elapsed_sec=float(stopwatch.elapsed()),
        )

    if verbose:
        _print_run_header(config, dim)

    sketch_seeds = np.random.SeedSequence(seed).spawn(max_iter)
    hvp_calls_cum = 0
    rk_hvp_calls_cum = 0
    subprob_hvp_calls_cum = 0
    successful_flag = False

    x_prev: np.ndarray | None = None
    g_prev: np.ndarray | None = None
    y_prev: np.ndarray | None = None

    status = "max_iter"
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
        except Exception:
            rk_info = {
                "alpha_ws": float("nan"),
                "rk_residual_norm_init": float("nan"),
                "rk_residual_norm_final": float("nan"),
                "hvp_calls": counted_problem.hvp_calls - hvp_before_rk,
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

        def hv_sub(reduced_vector: np.ndarray) -> np.ndarray:
            lifted = Q_k @ _reshape_to_vector(reduced_vector)
            hvp_value = counted_problem.hvp(x_current, lifted)
            return Q_k.T @ hvp_value

        hvp_before_subprob = counted_problem.hvp_calls
        cubic_result = _solve_arc_subproblem(
            solver=solver,
            grad=projected_grad,
            hv=hv_sub,
            hessian=None,
            sigma=sigma,
            w=np.zeros(Q_k.shape[1], dtype=float),
            successful_flag=successful_flag,
            lambda_k=lambda_value,
            exact_tol=exact_tol,
            krylov_tol=krylov_tol,
            solve_each_i_th_krylov_space=solve_each_i_th_krylov_space,
            keep_q_matrix_in_memory=keep_q_matrix_in_memory,
        )
        reduced_step = np.asarray(cubic_result.step, dtype=float).reshape(-1)
        lambda_value = float(cubic_result.lambda_value)
        direction = Q_k @ reduced_step
        gtd = float(np.dot(projected_grad, reduced_step))
        u_norm = _safe_norm(reduced_step)
        h_u = hv_sub(reduced_step)
        model_decrease = -float(
            gtd + 0.5 * np.dot(reduced_step, h_u) + (sigma / 3.0) * (u_norm**3)
        )
        subprob_hvp_calls_iter = int(counted_problem.hvp_calls - hvp_before_subprob)

        x_trial = x_current + direction
        f_trial = float(counted_problem.f(x_trial))
        actual_decrease = float(f_prev - f_trial)
        if np.isfinite(model_decrease) and model_decrease > 0.0 and np.isfinite(actual_decrease):
            rho = float(actual_decrease / model_decrease)
        else:
            rho = float("-inf")

        accepted = bool(rho >= eta1 and actual_decrease >= 0.0)
        if accepted:
            x_next = x_trial
            f_next = f_trial
            _, grad_next, grad_norm_next = evaluate_problem(counted_problem, x_next)
            step_size = 1.0
            step_norm = _safe_norm(direction)
        else:
            x_next = x_current
            f_next = f_prev
            grad_next = g_current
            grad_norm_next = grad_norm_prev
            step_size = 0.0
            step_norm = 0.0

        successful_flag = accepted
        if rho >= eta2:
            sigma = max(sigma / gamma2, sigma_min)
        elif rho < eta1 or not np.isfinite(rho):
            sigma = min(gamma1 * sigma, sigma_max)

        actual_reduction = float(f_prev - f_next)
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
            step_size=step_size,
            cumulative_time=cumulative_time,
            per_iter_time=per_iter_time,
            extras={
                "f_prev": f_prev,
                "f_next": f_next,
                "grad_norm_prev": grad_norm_prev,
                "grad_norm_next": grad_norm_next,
                "actual_reduction": actual_reduction,
                "accepted": int(accepted),
                "gtd": gtd,
                "rho": rho,
                "sigma": sigma,
                "model_decrease": model_decrease,
                "actual_decrease": actual_decrease,
                "lambda_value": lambda_value,
                "cubic_solver": cubic_result.mode,
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
                "alpha_ws": float(rk_info.get("alpha_ws", float("nan"))),
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
        if grad_norm <= tol:
            status = "converged"
            break

    return ARSCNResult(
        x_final=x,
        history=history,
        status=status,
        iters=max(0, len(history) - 1),
        elapsed_sec=float(stopwatch.elapsed()),
    )


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    result = _run_ars_cn(problem=problem, x0=x0, config=config, logger=logger)
    final_row = result.history[-1]
    return OptimizeResult(
        x_final=result.x_final,
        f_final=float(final_row["f"]),
        grad_norm_final=float(final_row["grad_norm"]),
        n_iter=result.iters,
        status=result.status,
        history_path=getattr(logger, "history_path", None),
    )

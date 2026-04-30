from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

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
    "accepted",
    "gtd",
    "rho",
    "sigma",
    "model_decrease",
    "actual_decrease",
    "hvp_calls_iter",
    "hvp_calls_cum",
    "subspace_dim",
    "lambda_value",
    "seed_sketch",
)


@dataclass(frozen=True)
class RSCNMResult:
    x_final: np.ndarray
    history: list[dict[str, Any]]
    status: str
    iters: int
    elapsed_sec: float


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
    return float(np.linalg.norm(vector))


def _reshape_to_vector(value: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _problem_dim(problem: Problem, x0: np.ndarray) -> int:
    if hasattr(problem, "dim"):
        return int(getattr(problem, "dim"))
    if hasattr(problem, "n"):
        return int(getattr(problem, "n"))
    return int(x0.size)


def _make_gaussian_sketch(
    dim: int,
    subspace_dim: int,
    sketch_config: Mapping[str, Any],
    seed: int,
) -> GaussianSketchOperator:
    distribution = str(sketch_config.get("distribution", "gaussian"))
    if distribution != "gaussian":
        raise ValueError(f"Unsupported sketch.distribution {distribution!r}. Available: gaussian")
    if bool(sketch_config.get("orthonormalize", False)):
        raise ValueError("RS-CN requires the raw Gaussian sketch; sketch.orthonormalize is unsupported")

    dtype_name = str(sketch_config.get("dtype", sketch_config.get("sketch_dtype", "float64")))
    if dtype_name == "float32":
        dtype = np.float32
    elif dtype_name == "float64":
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported sketch dtype {dtype_name!r}. Available: float32, float64")

    return GaussianSketchOperator(
        shape=(subspace_dim, dim),
        scale=1.0 / np.sqrt(float(max(1, subspace_dim))),
        seed=int(seed),
        mode=str(sketch_config.get("mode", "operator")),
        block_size=int(sketch_config.get("block_size", 256)),
        dtype=dtype,
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
) -> tuple[np.ndarray, float]:
    grad = _reshape_to_vector(grad)
    if grad.size == 0:
        return grad, 0.0

    grad_norm = _safe_norm(grad)
    if grad_norm == 0.0:
        return np.zeros_like(grad), 0.0

    if not np.isfinite(sigma) or sigma <= 0.0:
        return -grad, 0.0

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
    return step, float(sigma * _safe_norm(step))


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
            raise RuntimeError("Exact ARC solver failed near PSD shift")

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
        raise RuntimeError("Exact ARC solver failed to bracket lambda")

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
) -> tuple[np.ndarray, float]:
    grad = _reshape_to_vector(grad)
    dim = int(grad.size)
    if dim == 0:
        return grad, 0.0

    method = str(solver).lower()
    lambda_init = float(lambda_k) if (not successful_flag and lambda_k > 0.0) else None
    if method == "cauchy_point":
        return _solve_cauchy_point(grad, hv, sigma)

    if method == "exact":
        matrix = _resolve_hessian(hessian, hv, dim, w)
        return _solve_cubic_subproblem_exact(grad, matrix, sigma, exact_tol, lambda_init)

    if method != "lanczos":
        raise ValueError(f"Unknown ARC solver {solver!r}. Available: cauchy_point, exact, lanczos")

    grad_norm = _safe_norm(grad)
    if grad_norm == 0.0:
        return np.zeros(dim, dtype=float), 0.0

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
                return _solve_cubic_subproblem_exact(grad, matrix, sigma, exact_tol, lambda_init)

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
        raise RuntimeError("Lanczos failed to produce a subproblem solution")

    if keep_q_matrix_in_memory:
        q_matrix = np.column_stack(q_vectors[: u_hat_best.size])
        step = q_matrix @ u_hat_best
    else:
        step = _reconstruct_from_lanczos(hv, grad, u_hat_best, betas_best)
    return np.asarray(step, dtype=float), float(sigma * _safe_norm(step))


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


def _print_run_header(config: Mapping[str, Any], dim: int, subspace_dim: int) -> None:
    parts = [
        "[RS-CN]",
        f"run={config.get('run_name', '')}",
        f"dim={dim}",
        f"subspace_dim={subspace_dim}",
    ]
    print(" ".join(part for part in parts if not part.endswith("=")), flush=True)


def _print_iter_log(iteration: int, fx: float, grad_norm: float) -> None:
    print(f"[RS-CN] iter={iteration} f={fx:.6e} grad_norm={grad_norm:.3e}", flush=True)


def _run_rscn(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> RSCNMResult:
    x = np.asarray(x0, dtype=float).copy()
    dim = _problem_dim(problem, x)
    max_iter = int(config.get("max_iter", 100))
    tol = float(config.get("tol", config.get("tol_grad", 1.0e-6)))
    requested_subspace_dim = int(config.get("subspace_dim", config.get("s", min(20, dim))))
    if requested_subspace_dim <= 0:
        raise ValueError("optimizer.subspace_dim must be positive")
    subspace_dim = min(requested_subspace_dim, dim)

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
    seed = int(config.get("seed", config.get("seed_outer", 0)))
    verbose = bool(config.get("verbose", False))
    print_every = int(config.get("print_every", 10))
    sketch_config = dict(config.get("sketch", {}))
    grad_norm_stagnation = GradNormStagnationTracker(resolve_grad_norm_stagnation_config(config))

    counted_problem = _CountingProblem(problem)
    stopwatch = Stopwatch()
    history: list[dict[str, Any]] = []
    hvp_calls_cum = 0
    lambda_value = 0.0
    successful_flag = False
    child_seeds = np.random.SeedSequence(seed).spawn(max_iter)

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
            "sigma": sigma,
            "hvp_calls_iter": 0,
            "hvp_calls_cum": 0,
            "subspace_dim": subspace_dim,
            "lambda_value": lambda_value,
        },
    )

    if verbose:
        _print_run_header(config, dim, subspace_dim)

    status = "max_iter"
    if grad_norm <= tol:
        status = "converged"

    for iteration in range(1, max_iter + 1):
        if grad_norm <= tol:
            break

        iter_start = perf_counter()
        counted_problem.hvp_calls = 0
        sketch_seed = int(child_seeds[iteration - 1].generate_state(1)[0])
        sketch = _make_gaussian_sketch(
            dim=dim,
            subspace_dim=subspace_dim,
            sketch_config=sketch_config,
            seed=sketch_seed,
        )
        sketch_matrix = sketch.dense_matrix()
        projected_grad = sketch_matrix @ grad

        def hv_sub(subspace_vector: np.ndarray) -> np.ndarray:
            lifted = sketch_matrix.T @ _reshape_to_vector(subspace_vector)
            hvp_value = counted_problem.hvp(x, lifted)
            return sketch_matrix @ hvp_value

        reduced_step, lambda_value = _solve_arc_subproblem(
            solver=solver,
            grad=projected_grad,
            hv=hv_sub,
            hessian=None,
            sigma=sigma,
            w=np.zeros(subspace_dim, dtype=float),
            successful_flag=successful_flag,
            lambda_k=lambda_value,
            exact_tol=exact_tol,
            krylov_tol=krylov_tol,
            solve_each_i_th_krylov_space=solve_each_i_th_krylov_space,
            keep_q_matrix_in_memory=keep_q_matrix_in_memory,
        )

        direction = sketch_matrix.T @ reduced_step
        gtd = float(np.dot(projected_grad, reduced_step))
        step_norm = _safe_norm(direction)
        h_u = hv_sub(reduced_step)
        model_decrease = -float(
            gtd + 0.5 * np.dot(reduced_step, h_u) + (sigma / 3.0) * (_safe_norm(reduced_step) ** 3)
        )

        x_trial = x + direction
        f_trial = float(counted_problem.f(x_trial))
        actual_decrease = float(fx - f_trial)

        if np.isfinite(model_decrease) and model_decrease > 0.0 and np.isfinite(actual_decrease):
            rho = float(actual_decrease / model_decrease)
        else:
            rho = float("-inf")

        accepted = bool(rho >= eta1 and actual_decrease >= 0.0)
        if accepted:
            x_next = x_trial
            fx_next = f_trial
            _, grad_next, grad_norm_next = evaluate_problem(counted_problem, x_next)
            step_size = 1.0
        else:
            x_next = x
            fx_next = fx
            grad_next = grad
            grad_norm_next = grad_norm
            step_norm = 0.0
            step_size = 0.0

        successful_flag = accepted
        if rho >= eta2:
            sigma = max(sigma / gamma2, sigma_min)
        elif rho < eta1 or not np.isfinite(rho):
            sigma = min(gamma1 * sigma, sigma_max)

        hvp_calls_cum += counted_problem.hvp_calls
        per_iter_time = float(perf_counter() - iter_start)
        cumulative_time = float(stopwatch.elapsed())
        _log_row(
            history=history,
            logger=logger,
            iteration=iteration,
            fx=fx_next,
            grad_norm=grad_norm_next,
            step_norm=step_norm,
            step_size=step_size,
            cumulative_time=cumulative_time,
            per_iter_time=per_iter_time,
            extras={
                "accepted": int(accepted),
                "gtd": gtd,
                "rho": rho,
                "sigma": sigma,
                "model_decrease": model_decrease,
                "actual_decrease": actual_decrease,
                "hvp_calls_iter": counted_problem.hvp_calls,
                "hvp_calls_cum": hvp_calls_cum,
                "subspace_dim": subspace_dim,
                "lambda_value": lambda_value,
                "seed_sketch": sketch_seed,
            },
        )

        if verbose and iteration % max(1, print_every) == 0:
            _print_iter_log(iteration, fx_next, grad_norm_next)

        x = x_next
        fx = fx_next
        grad = grad_next
        grad_norm = grad_norm_next
        if grad_norm <= tol:
            status = "converged"
            break
        if grad_norm_stagnation.update(grad_norm):
            status = "grad_norm_stagnation"
            break

    return RSCNMResult(
        x_final=x,
        history=history,
        status=status,
        iters=max(0, len(history) - 1),
        elapsed_sec=float(stopwatch.elapsed()),
    )


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    result = _run_rscn(problem=problem, x0=x0, config=config, logger=logger)
    final_row = result.history[-1]
    return OptimizeResult(
        x_final=result.x_final,
        f_final=float(final_row["f"]),
        grad_norm_final=float(final_row["grad_norm"]),
        n_iter=result.iters,
        status=result.status,
        history_path=getattr(logger, "history_path", None),
    )

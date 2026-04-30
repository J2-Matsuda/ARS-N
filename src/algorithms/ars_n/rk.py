from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from src.problems.base import Problem
from src.utils.sketch import GaussianSketchOperator


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def _load_minres_tools() -> tuple[Any | None, Callable[..., Any] | None]:
    try:
        from scipy.sparse.linalg import LinearOperator, minres
    except ImportError:
        return None, None
    return LinearOperator, minres


def _run_minres(
    minres: Callable[..., Any],
    operator: Any,
    rhs: np.ndarray,
    minres_tol: float,
    minres_maxit: int,
    callback: Callable[[np.ndarray], None],
) -> tuple[np.ndarray, int]:
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    try:
        solution, info = minres(
            operator,
            rhs,
            rtol=minres_tol,
            maxiter=minres_maxit,
            callback=callback,
            show=False,
        )
    except TypeError:
        solution, info = minres(
            operator,
            rhs,
            tol=minres_tol,
            maxiter=minres_maxit,
            callback=callback,
            show=False,
        )

    solution = np.asarray(solution, dtype=float).reshape(-1)
    return solution, int(info)


def _run_cg_fallback(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    tol: float,
    max_iter: int,
    callback: Callable[[np.ndarray], None],
) -> tuple[np.ndarray, int]:
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    rhs_norm = _safe_norm(rhs)
    threshold = float(tol) * max(1.0, rhs_norm)
    if np.sqrt(residual_sq) <= threshold:
        return solution, 0

    for _ in range(max_iter):
        matvec_direction = np.asarray(matvec(direction), dtype=float).reshape(-1)
        denom = float(np.dot(direction, matvec_direction))
        if not np.isfinite(denom) or denom <= 0.0:
            return solution, -1

        alpha = residual_sq / denom
        solution = solution + alpha * direction
        residual = residual - alpha * matvec_direction
        callback(solution)

        next_residual_sq = float(np.dot(residual, residual))
        if not np.isfinite(next_residual_sq):
            return solution, -1
        if np.sqrt(next_residual_sq) <= threshold:
            return solution, 0

        beta = next_residual_sq / residual_sq
        direction = residual + beta * direction
        residual_sq = next_residual_sq

    return solution, max_iter


def _make_rk_gaussian_sketch(
    n: int,
    r: int,
    rk_config: Mapping[str, Any],
    seed: int,
) -> GaussianSketchOperator:
    if n <= 0:
        raise ValueError("n must be positive")
    if r <= 0:
        raise ValueError("r must be positive")

    distribution = str(rk_config.get("distribution", "gaussian"))
    if distribution != "gaussian":
        raise ValueError(f"Unsupported rk.distribution {distribution!r}. Available: gaussian")

    dtype_name = str(rk_config.get("dtype", "float64"))
    if dtype_name == "float32":
        dtype = np.float32
    elif dtype_name == "float64":
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported rk dtype {dtype_name!r}. Available: float32, float64")

    sketch_mode = str(rk_config.get("sketch_mode", rk_config.get("mode", "operator")))
    if sketch_mode in {"default", "T_auto"}:
        sketch_mode = "operator"

    return GaussianSketchOperator(
        shape=(r, n),
        scale=1.0 / np.sqrt(float(r)),
        seed=int(seed),
        mode=sketch_mode,
        block_size=int(rk_config.get("block_size", 256)),
        dtype=dtype,
    )


def rk_anchor(
    problem: Problem,
    x_k: np.ndarray,
    g_k: np.ndarray,
    k: int,
    rk_config: Mapping[str, Any],
    y0: np.ndarray | None = None,
    x_prev: np.ndarray | None = None,
    g_prev: np.ndarray | None = None,
    y_prev: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_k = np.asarray(x_k, dtype=float).reshape(-1)
    g_k = np.asarray(g_k, dtype=float).reshape(-1)

    grad_norm = _safe_norm(g_k)
    if not np.isfinite(grad_norm) or grad_norm <= 0.0:
        raise ValueError("rk_anchor requires a nonzero finite gradient")

    num_inner_steps = int(rk_config.get("T", 0))
    inner_sketch_dim = int(rk_config.get("r", 1))
    rk_mode = str(rk_config.get("mode", "default"))
    if rk_mode in {"operator", "explicit"}:
        rk_mode = "default"
    rk_tol = float(rk_config.get("rk_tol", 0.5))
    seed_offset = int(rk_config.get("seed_offset", 0))
    store_debug_stats = bool(rk_config.get("store_debug_stats", False))
    minres_tol = float(rk_config.get("minres_tol", 1.0e-6))
    minres_maxit = int(rk_config.get("minres_maxit", inner_sketch_dim))
    ridge = float(rk_config.get("ridge", rk_config.get("minres_ridge", 1.0e-8)))
    if num_inner_steps < 0:
        raise ValueError("rk.T must be nonnegative")
    if inner_sketch_dim <= 0:
        raise ValueError("rk.r must be positive")
    if rk_mode not in {"default", "T_auto"}:
        raise ValueError("rk.mode must be one of: default, T_auto")
    if rk_tol <= 0.0 or not np.isfinite(rk_tol):
        raise ValueError("rk.rk_tol must be positive and finite")
    if minres_maxit <= 0:
        raise ValueError("rk.minres_maxit must be positive")
    if minres_tol <= 0.0 or not np.isfinite(minres_tol):
        raise ValueError("rk.minres_tol must be positive and finite")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("rk.ridge must be nonnegative and finite")

    LinearOperator, minres = _load_minres_tools()

    grad_hat_k = g_k / grad_norm
    hvp_calls = 0

    if k == 0:
        if y0 is None:
            raise ValueError("rk_anchor requires a nonzero y0 when k == 0")
        y_t = np.asarray(y0, dtype=float).reshape(-1)
        alpha_ws = 1.0
    else:
        if x_prev is None or g_prev is None or y_prev is None:
            raise ValueError("rk_anchor requires x_prev, g_prev, and y_prev when k > 0")
        x_prev = np.asarray(x_prev, dtype=float).reshape(-1)
        g_prev = np.asarray(g_prev, dtype=float).reshape(-1)
        y_prev = np.asarray(y_prev, dtype=float).reshape(-1)

        grad_prev_norm = _safe_norm(g_prev)
        if not np.isfinite(grad_prev_norm) or grad_prev_norm <= 0.0:
            raise ValueError("rk_anchor requires a nonzero finite previous gradient")

        delta_x = x_k - x_prev
        h_delta = np.asarray(problem.hvp(x_prev, delta_x), dtype=float).reshape(-1)
        hvp_calls += 1

        alpha_ws = 1.0 - float(np.dot(g_prev, h_delta)) / float(np.dot(g_prev, g_prev))
        y_t = alpha_ws * y_prev + delta_x / grad_prev_norm

    y_t = np.asarray(y_t, dtype=float).reshape(-1)
    if not np.all(np.isfinite(y_t)) or _safe_norm(y_t) == 0.0:
        raise ValueError("rk_anchor warm start produced a zero or non-finite vector")

    h_t = np.asarray(problem.hvp(x_k, y_t), dtype=float).reshape(-1)
    hvp_calls += 1
    e_t = h_t - grad_hat_k

    residual_norm_init = _safe_norm(e_t)
    residual_history: list[float] = [residual_norm_init]
    inner_minres_total_iters = 0
    minres_fail = 0
    rk_steps_taken = 0
    rk_stop_reason = "max_steps" if rk_mode == "default" else "max_steps"

    rk_seed_base = seed_offset + int(k)
    rng = np.random.default_rng(rk_seed_base)
    for inner_step in range(1, num_inner_steps + 1):
        sketch_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        R_t = _make_rk_gaussian_sketch(
            n=x_k.size,
            r=inner_sketch_dim,
            rk_config=rk_config,
            seed=sketch_seed,
        )
        rhs = np.asarray(R_t.matvec(e_t), dtype=float).reshape(-1)

        if not np.all(np.isfinite(rhs)):
            u_t = np.zeros(inner_sketch_dim, dtype=float)
            minres_fail = 1
        else:
            def _matvec(u: np.ndarray) -> np.ndarray:
                nonlocal hvp_calls
                u = np.asarray(u, dtype=float).reshape(-1)
                lifted = np.asarray(R_t.rmatvec(u), dtype=float).reshape(-1)
                hv_lifted = np.asarray(problem.hvp(x_k, lifted), dtype=float).reshape(-1)
                hvp_calls += 1
                reduced = np.asarray(R_t.matvec(hv_lifted), dtype=float).reshape(-1)
                if ridge > 0.0:
                    reduced = reduced + ridge * u
                return np.asarray(reduced, dtype=float).reshape(-1)

            callback_iters = 0

            def _callback(_: np.ndarray) -> None:
                nonlocal callback_iters
                callback_iters += 1

            try:
                if LinearOperator is not None and minres is not None:
                    operator = LinearOperator(
                        shape=(inner_sketch_dim, inner_sketch_dim),
                        matvec=_matvec,
                        dtype=float,
                    )
                    u_t, minres_info = _run_minres(
                        minres=minres,
                        operator=operator,
                        rhs=rhs,
                        minres_tol=minres_tol,
                        minres_maxit=minres_maxit,
                        callback=_callback,
                    )
                else:
                    u_t, minres_info = _run_cg_fallback(
                        matvec=_matvec,
                        rhs=rhs,
                        tol=minres_tol,
                        max_iter=minres_maxit,
                        callback=_callback,
                    )
            except Exception:
                u_t = np.zeros(inner_sketch_dim, dtype=float)
                minres_info = -1
                minres_fail = 1

            inner_minres_total_iters += int(callback_iters)
            if minres_info != 0 or not np.all(np.isfinite(u_t)):
                u_t = np.zeros(inner_sketch_dim, dtype=float)
                minres_fail = 1

        correction = np.asarray(R_t.rmatvec(u_t), dtype=float).reshape(-1)
        y_t = y_t - correction
        if not np.all(np.isfinite(y_t)):
            y_t = np.zeros_like(grad_hat_k)
            minres_fail = 1

        h_t = np.asarray(problem.hvp(x_k, y_t), dtype=float).reshape(-1)
        hvp_calls += 1
        e_t = h_t - grad_hat_k
        residual_history.append(_safe_norm(e_t))
        rk_steps_taken = inner_step

        if rk_mode == "T_auto":
            residual_norm_current = residual_history[-1]
            if residual_norm_init == 0.0:
                rk_stop_reason = "rk_tol"
                break
            if residual_norm_current <= rk_tol * residual_norm_init:
                rk_stop_reason = "rk_tol"
                break

    info: dict[str, Any] = {
        "alpha_ws": float(alpha_ws),
        "rk_seed_base": int(rk_seed_base),
        "rk_steps_taken": int(rk_steps_taken),
        "rk_stop_reason": rk_stop_reason,
        "rk_residual_norm_init": float(residual_norm_init),
        "rk_residual_norm_final": float(_safe_norm(e_t)),
        "hvp_calls": int(hvp_calls),
        "inner_minres_total_iters": int(inner_minres_total_iters),
        "minres_fail": int(minres_fail),
    }
    if store_debug_stats:
        info["residual_history"] = residual_history

    return np.asarray(y_t, dtype=float).reshape(-1), info

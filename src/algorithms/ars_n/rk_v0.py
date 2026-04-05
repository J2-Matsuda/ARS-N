from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.problems.base import Problem
from src.utils.sketch import GaussianSketchOperator


@dataclass(frozen=True)
class _SmallSolveResult:
    solution: np.ndarray
    mode: str


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def _solve_small_system(matrix: np.ndarray, rhs: np.ndarray) -> _SmallSolveResult:
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)

    try:
        solution = np.linalg.solve(matrix, rhs)
        mode = "solve"
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        mode = "lstsq"

    solution = np.asarray(solution, dtype=float).reshape(-1)
    if not np.all(np.isfinite(solution)):
        solution = np.zeros_like(rhs)
        mode = "zero"
    return _SmallSolveResult(solution=solution, mode=mode)


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

    return GaussianSketchOperator(
        shape=(r, n),
        scale=1.0 / np.sqrt(float(r)),
        seed=int(seed),
        mode=str(rk_config.get("mode", "operator")),
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
    seed_offset = int(rk_config.get("seed_offset", 0))
    store_debug_stats = bool(rk_config.get("store_debug_stats", False))
    if num_inner_steps < 0:
        raise ValueError("rk.T must be nonnegative")
    if inner_sketch_dim <= 0:
        raise ValueError("rk.r must be positive")

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
    solve_modes: list[str] = []

    rng = np.random.default_rng(seed_offset + int(k))
    for _ in range(num_inner_steps):
        sketch_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        Z_t = _make_rk_gaussian_sketch(
            n=x_k.size,
            r=inner_sketch_dim,
            rk_config=rk_config,
            seed=sketch_seed,
        )
        z_matrix = Z_t.dense_matrix()
        z_columns = z_matrix.T
        W_t = np.empty((x_k.size, inner_sketch_dim), dtype=float)
        for column in range(inner_sketch_dim):
            W_t[:, column] = np.asarray(problem.hvp(x_k, z_columns[:, column]), dtype=float).reshape(-1)
            hvp_calls += 1

        A_t = 0.5 * ((z_matrix @ W_t) + (W_t.T @ z_columns))
        b_t = z_matrix @ e_t
        solve_result = _solve_small_system(A_t, b_t)
        solve_modes.append(solve_result.mode)

        correction = W_t @ solve_result.solution
        y_t = y_t - z_columns @ solve_result.solution
        h_t = h_t - correction
        e_t = e_t - correction
        residual_history.append(_safe_norm(e_t))

    info: dict[str, Any] = {
        "alpha_ws": float(alpha_ws),
        "rk_residual_norm_init": float(residual_norm_init),
        "rk_residual_norm_final": float(_safe_norm(e_t)),
        "hvp_calls": int(hvp_calls),
    }
    if store_debug_stats:
        info["residual_history"] = residual_history
        info["solve_modes"] = solve_modes

    return np.asarray(y_t, dtype=float).reshape(-1), info

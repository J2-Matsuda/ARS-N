from __future__ import annotations

import numpy as np

from src.algorithms.ars_n.rk import rk_anchor


class _IdentityProblem:
    def f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        return 0.5 * float(np.dot(x, x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(-1)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        del x
        return np.asarray(v, dtype=float).reshape(-1)


def test_rk_default_mode_uses_fixed_T() -> None:
    problem = _IdentityProblem()
    x_k = np.array([2.0], dtype=float)
    g_k = np.array([2.0], dtype=float)
    y0 = np.array([3.0], dtype=float)

    y_t, info = rk_anchor(
        problem=problem,
        x_k=x_k,
        g_k=g_k,
        k=0,
        rk_config={
            "mode": "default",
            "T": 3,
            "r": 1,
            "rk_tol": 0.5,
            "ridge": 0.0,
        },
        y0=y0,
    )

    assert np.allclose(y_t, np.array([1.0]))
    assert info["rk_steps_taken"] == 3
    assert info["rk_stop_reason"] == "max_steps"
    assert info["rk_residual_norm_init"] > 0.0
    assert info["rk_residual_norm_final"] == 0.0


def test_rk_t_auto_mode_stops_early_on_rk_tol() -> None:
    problem = _IdentityProblem()
    x_k = np.array([2.0], dtype=float)
    g_k = np.array([2.0], dtype=float)
    y0 = np.array([3.0], dtype=float)

    y_t, info = rk_anchor(
        problem=problem,
        x_k=x_k,
        g_k=g_k,
        k=0,
        rk_config={
            "mode": "T_auto",
            "T": 5,
            "r": 1,
            "rk_tol": 0.5,
            "ridge": 0.0,
        },
        y0=y0,
    )

    assert np.allclose(y_t, np.array([1.0]))
    assert info["rk_steps_taken"] == 1
    assert info["rk_stop_reason"] == "rk_tol"
    assert info["rk_residual_norm_final"] <= 0.5 * info["rk_residual_norm_init"]

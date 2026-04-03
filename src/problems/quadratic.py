from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.utils.io import load_npz, save_npz
from src.utils.paths import resolve_project_path


@dataclass
class QuadraticProblem:
    diag: np.ndarray
    b: np.ndarray
    c: float = 0.0

    @property
    def dim(self) -> int:
        return int(self.diag.shape[0])

    def f(self, x: np.ndarray) -> float:
        return float(0.5 * np.dot(x, self.diag * x) - np.dot(self.b, x) + self.c)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.diag * x - self.b

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        del x
        return self.diag * v


def generate_spectrum(
    dim: int,
    spectrum: str,
    lambda_max: float,
    lambda_min: float,
) -> np.ndarray:
    if dim <= 0:
        raise ValueError("problem.dim must be positive")
    if lambda_max <= 0.0 or lambda_min <= 0.0:
        raise ValueError("lambda_max and lambda_min must be positive")
    if lambda_min > lambda_max:
        raise ValueError("lambda_min must be <= lambda_max")

    if spectrum == "flat":
        return np.full(dim, float(lambda_max))

    if dim == 1:
        return np.array([float(lambda_max)])

    if spectrum == "exponential":
        grid = np.linspace(0.0, 1.0, dim)
        ratio = lambda_min / lambda_max
        return lambda_max * np.power(ratio, grid)

    if spectrum == "polynomial":
        alpha = np.log(lambda_max / lambda_min) / np.log(float(dim))
        diag = lambda_max / np.power(np.arange(1, dim + 1, dtype=float), alpha)
        diag[-1] = lambda_min
        return diag

    raise ValueError("problem.spectrum must be one of: flat, exponential, polynomial")


def generate_quadratic_data(
    dim: int,
    spectrum: str,
    lambda_max: float,
    lambda_min: float,
    b_norm: float,
    seed: int,
    c: float = 0.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    diag = generate_spectrum(dim, spectrum, lambda_max, lambda_min).astype(float)
    b = rng.normal(size=dim)
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        b[0] = float(b_norm)
    else:
        b = b * (float(b_norm) / norm_b)

    return {
        "diag": diag,
        "b": b.astype(float),
        "c": np.array(float(c)),
        "spectrum": np.array(spectrum),
        "lambda_max": np.array(float(lambda_max)),
        "lambda_min": np.array(float(lambda_min)),
    }


def save_quadratic_data(path: str | Path, data: Mapping[str, Any]) -> None:
    save_npz(path, **data)


def load_quadratic_problem(path: str | Path) -> QuadraticProblem:
    arrays = load_npz(path)
    diag = np.asarray(arrays["diag"], dtype=float)
    b = np.asarray(arrays["b"], dtype=float)
    c = float(arrays["c"]) if "c" in arrays else 0.0
    return QuadraticProblem(diag=diag, b=b, c=c)


def build_quadratic_problem(problem_config: Mapping[str, Any]) -> QuadraticProblem:
    if "source" in problem_config:
        return load_quadratic_problem(resolve_project_path(problem_config["source"]))

    data = generate_quadratic_data(
        dim=int(problem_config["dim"]),
        spectrum=str(problem_config.get("spectrum", "exponential")),
        lambda_max=float(problem_config.get("lambda_max", 1.0)),
        lambda_min=float(problem_config.get("lambda_min", 1.0e-6)),
        b_norm=float(problem_config.get("b_norm", 1.0)),
        seed=int(problem_config.get("seed", 0)),
        c=float(problem_config.get("c", 0.0)),
    )
    return QuadraticProblem(diag=data["diag"], b=data["b"], c=float(data["c"]))


def generate_quadratic_from_config(problem_config: Mapping[str, Any], save_path: str, seed: int) -> None:
    data = generate_quadratic_data(
        dim=int(problem_config["dim"]),
        spectrum=str(problem_config.get("spectrum", "exponential")),
        lambda_max=float(problem_config.get("lambda_max", 1.0)),
        lambda_min=float(problem_config.get("lambda_min", 1.0e-6)),
        b_norm=float(problem_config.get("b_norm", 1.0)),
        seed=seed,
        c=float(problem_config.get("c", 0.0)),
    )
    save_quadratic_data(resolve_project_path(save_path), data)

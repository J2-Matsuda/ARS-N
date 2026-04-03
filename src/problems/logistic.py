from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import numpy as np

from src.utils.io import load_npz, save_npz
from src.utils.paths import resolve_project_path

try:
    from scipy.special import expit
    from scipy.sparse import csr_matrix, isspmatrix_csr
except ImportError:  # pragma: no cover - scipy is expected for sparse support
    csr_matrix = None

    def expit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=float), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def isspmatrix_csr(matrix: Any) -> bool:
        del matrix
        return False


MatrixLike = np.ndarray | Any


def _as_float_vector(value: np.ndarray | Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        array = array.astype(float, copy=False)
    return array.reshape(-1)


def _is_csr_matrix(matrix: Any) -> bool:
    """Return True when the matrix is a SciPy CSR matrix."""
    return bool(csr_matrix is not None and isspmatrix_csr(matrix))


def _to_pm1(y: np.ndarray) -> np.ndarray:
    """Convert labels in {0, 1} or {-1, +1} to {-1, +1}."""
    labels = np.asarray(y).reshape(-1)
    unique = np.unique(labels)
    if np.all(np.isin(unique, [0, 1])):
        return np.where(labels > 0, 1.0, -1.0).astype(float, copy=False)
    if np.all(np.isin(unique, [-1, 1])):
        return labels.astype(float, copy=False)
    raise ValueError("Labels must be in {0, 1} or {-1, +1}.")


def generate_logistic_synthetic_data(
    n: int,
    d: int,
    seed: int = 0,
    beta_true: np.ndarray | None = None,
    sparse_beta: bool = False,
    num_nonzero: int | None = None,
    feature_scale: float = 1.0,
    intercept: float = 0.0,
) -> dict[str, np.ndarray]:
    """Generate dense synthetic data for L2-regularized logistic regression."""
    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")
    if feature_scale <= 0.0:
        raise ValueError("feature_scale must be positive")

    rng = np.random.default_rng(seed)
    A = rng.normal(loc=0.0, scale=float(feature_scale), size=(n, d)).astype(float)

    if beta_true is None:
        if sparse_beta:
            nnz = int(num_nonzero if num_nonzero is not None else max(1, min(d, d // 5 or 1)))
            if nnz <= 0 or nnz > d:
                raise ValueError("num_nonzero must satisfy 1 <= num_nonzero <= d")
            beta_true_array = np.zeros(d, dtype=float)
            active = rng.choice(d, size=nnz, replace=False)
            beta_true_array[active] = rng.normal(size=nnz)
        else:
            beta_true_array = rng.normal(size=d).astype(float)
    else:
        beta_true_array = np.asarray(beta_true, dtype=float).reshape(-1)
        if beta_true_array.shape != (d,):
            raise ValueError(f"beta_true must have shape ({d},), got {beta_true_array.shape}")

    logits_true = A @ beta_true_array + float(intercept)
    p_true = expit(logits_true)
    y = rng.binomial(1, p_true, size=n).astype(np.int8)

    return {
        "A": A,
        "y": y,
        "beta_true": beta_true_array.astype(float, copy=False),
        "intercept": np.array(float(intercept), dtype=float),
    }


def save_logistic_dataset(
    path: str | Path,
    data: Mapping[str, Any],
    reg_lambda: float = 0.0,
) -> None:
    """Save a logistic regression dataset to .npz, supporting dense or CSR matrices."""
    A = data["A"]
    arrays: dict[str, Any] = {
        "y": np.asarray(data["y"]).reshape(-1),
        "beta_true": np.asarray(data["beta_true"], dtype=float).reshape(-1),
        "intercept": np.array(float(data.get("intercept", 0.0)), dtype=float),
        "reg_lambda": np.array(float(reg_lambda), dtype=float),
    }

    if _is_csr_matrix(A):
        arrays["A_data"] = np.asarray(A.data, dtype=float)
        arrays["A_indices"] = np.asarray(A.indices, dtype=np.int64)
        arrays["A_indptr"] = np.asarray(A.indptr, dtype=np.int64)
        arrays["A_shape"] = np.asarray(A.shape, dtype=np.int64)
    else:
        arrays["A"] = np.asarray(A, dtype=float)

    save_npz(path, **arrays)


def _load_npz_matrix(npz_data: Mapping[str, Any]) -> MatrixLike:
    """Load a dense matrix or a CSR matrix from .npz contents."""
    if "A" in npz_data:
        return np.asarray(npz_data["A"], dtype=float)
    if "X" in npz_data:
        return np.asarray(npz_data["X"], dtype=float)

    if {"A_data", "A_indices", "A_indptr", "A_shape"}.issubset(npz_data):
        if csr_matrix is None:
            raise ImportError("scipy is required to load CSR matrices from .npz files")
        shape = tuple(int(value) for value in np.asarray(npz_data["A_shape"]).tolist())
        return csr_matrix(
            (
                np.asarray(npz_data["A_data"], dtype=float),
                np.asarray(npz_data["A_indices"], dtype=np.int64),
                np.asarray(npz_data["A_indptr"], dtype=np.int64),
            ),
            shape=shape,
        )

    raise KeyError("Dataset .npz must contain either dense key 'A' or CSR keys 'A_data/A_indices/A_indptr/A_shape'")


@dataclass(init=False)
class LogisticRegressionProblem:
    """Closed-form logistic regression objective with L2 regularization."""

    A: MatrixLike
    y_pm1: np.ndarray
    reg_lambda: float
    _cached_x_ref: np.ndarray | None
    _cached_x_value: np.ndarray | None
    _cached_ax: np.ndarray | None
    _cached_margins: np.ndarray | None
    _cached_probs: np.ndarray | None
    _cached_d: np.ndarray | None

    def __init__(
        self,
        A: MatrixLike | None = None,
        y_pm1: np.ndarray | None = None,
        reg_lambda: float = 0.0,
        *,
        x_matrix: MatrixLike | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        matrix = A if A is not None else x_matrix
        labels = y_pm1 if y_pm1 is not None else y

        if matrix is None:
            raise ValueError("A or x_matrix must be provided")
        if labels is None:
            raise ValueError("y_pm1 or y must be provided")

        if _is_csr_matrix(matrix):
            self.A = matrix.astype(float, copy=False)
        else:
            self.A = np.asarray(matrix, dtype=float)

        self.y_pm1 = _to_pm1(np.asarray(labels)).astype(float, copy=False)
        self.reg_lambda = float(reg_lambda)
        self._cached_x_ref = None
        self._cached_x_value = None
        self._cached_ax = None
        self._cached_margins = None
        self._cached_probs = None
        self._cached_d = None

        if len(self.A.shape) != 2:
            raise ValueError("A must be a 2D matrix")
        if self.A.shape[0] != self.y_pm1.shape[0]:
            raise ValueError("A and y must have the same number of rows")
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be nonnegative")

    @property
    def n(self) -> int:
        """Return the number of parameters."""
        return int(self.A.shape[1])

    @property
    def dim(self) -> int:
        """Alias for compatibility with existing experiment code."""
        return self.n

    @property
    def m(self) -> int:
        """Return the number of samples."""
        return int(self.A.shape[0])

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute A @ x for dense or CSR matrices."""
        vector = _as_float_vector(x)
        return np.asarray(self.A @ vector, dtype=float).reshape(-1)

    def _rmatvec(self, z: np.ndarray) -> np.ndarray:
        """Compute A.T @ z for dense or CSR matrices."""
        vector = _as_float_vector(z)
        return np.asarray(self.A.T @ vector, dtype=float).reshape(-1)

    def _point_cache(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_vec = _as_float_vector(x)
        if self._cached_x_ref is x and self._cached_margins is not None:
            return self._cached_margins, self._cached_probs, self._cached_d
        if (
            self._cached_x_value is not None
            and self._cached_x_value.shape == x_vec.shape
            and np.array_equal(self._cached_x_value, x_vec)
            and self._cached_margins is not None
        ):
            self._cached_x_ref = x if isinstance(x, np.ndarray) else None
            return self._cached_margins, self._cached_probs, self._cached_d

        ax = self._matvec(x_vec)
        margins = -self.y_pm1 * ax
        probs = expit(margins)
        d = probs * (1.0 - probs)

        self._cached_x_ref = x if isinstance(x, np.ndarray) else None
        self._cached_x_value = x_vec.copy()
        self._cached_ax = ax
        self._cached_margins = margins
        self._cached_probs = probs
        self._cached_d = d
        return margins, probs, d

    def f(self, x: np.ndarray) -> float:
        """Evaluate the regularized logistic loss."""
        x = _as_float_vector(x)
        margins, _, _ = self._point_cache(x)
        loss = np.logaddexp(0.0, margins)
        reg = 0.5 * self.reg_lambda * float(np.dot(x, x))
        return float(np.mean(loss) + reg)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the closed-form gradient."""
        x = _as_float_vector(x)
        _, probs, _ = self._point_cache(x)
        weighted = self.y_pm1 * probs
        gradient = -(self._rmatvec(weighted) / self.m)
        gradient += self.reg_lambda * x
        return np.asarray(gradient, dtype=float).reshape(-1)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate the closed-form Hessian-vector product."""
        x = _as_float_vector(x)
        v = _as_float_vector(v)
        _, _, d = self._point_cache(x)
        Av = self._matvec(v)
        hv = self._rmatvec(d * Av) / self.m
        hv += self.reg_lambda * v
        return np.asarray(hv, dtype=float).reshape(-1)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        reg_lambda: float | None = None,
    ) -> "LogisticRegressionProblem":
        """Load a LogisticRegressionProblem from a saved .npz dataset."""
        arrays = load_npz(path)
        A = _load_npz_matrix(arrays)
        if "y" not in arrays:
            raise KeyError("Dataset .npz must contain label key 'y'")
        y = np.asarray(arrays["y"]).reshape(-1)
        loaded_reg_lambda = float(arrays["reg_lambda"]) if "reg_lambda" in arrays else 0.0
        return cls(A=A, y=y, reg_lambda=loaded_reg_lambda if reg_lambda is None else reg_lambda)


def generate_logistic_data(
    num_samples: int,
    dim: int,
    reg_lambda: float,
    noise: float,
    seed: int,
) -> dict[str, Any]:
    """Backward-compatible wrapper around generate_logistic_synthetic_data."""
    data = generate_logistic_synthetic_data(
        n=num_samples,
        d=dim,
        seed=seed,
        feature_scale=1.0,
        intercept=float(noise),
    )
    data["reg_lambda"] = np.array(float(reg_lambda), dtype=float)
    data["X"] = np.asarray(data["A"], dtype=float)
    return data


def save_logistic_data(path: str | Path, data: Mapping[str, Any]) -> None:
    """Backward-compatible alias for save_logistic_dataset."""
    save_logistic_dataset(path, data, reg_lambda=float(data.get("reg_lambda", 0.0)))


def load_logistic_problem(path: str | Path) -> LogisticRegressionProblem:
    """Load a logistic regression problem from disk."""
    return LogisticRegressionProblem.from_npz(path)


def build_logistic_problem(problem_config: Mapping[str, Any]) -> LogisticRegressionProblem:
    """Build a LogisticRegressionProblem from config or a saved dataset."""
    if "source" in problem_config:
        override = problem_config.get("reg_lambda")
        reg_lambda = None if override is None else float(override)
        return LogisticRegressionProblem.from_npz(
            resolve_project_path(problem_config["source"]),
            reg_lambda=reg_lambda,
        )

    beta_true = problem_config.get("beta_true")
    data = generate_logistic_synthetic_data(
        n=int(problem_config.get("n", problem_config.get("num_samples", 1000))),
        d=int(problem_config.get("d", problem_config.get("dim", 20))),
        seed=int(problem_config.get("seed", 0)),
        beta_true=None if beta_true is None else np.asarray(beta_true, dtype=float),
        sparse_beta=bool(problem_config.get("sparse_beta", False)),
        num_nonzero=None if problem_config.get("num_nonzero") is None else int(problem_config["num_nonzero"]),
        feature_scale=float(problem_config.get("feature_scale", 1.0)),
        intercept=float(problem_config.get("intercept", problem_config.get("noise", 0.0))),
    )
    return LogisticRegressionProblem(
        A=data["A"],
        y=data["y"],
        reg_lambda=float(problem_config.get("reg_lambda", 0.0)),
    )


def generate_logistic_from_config(problem_config: Mapping[str, Any], save_path: str, seed: int) -> None:
    """Generate and save a synthetic logistic regression dataset from config."""
    beta_true = problem_config.get("beta_true")
    data = generate_logistic_synthetic_data(
        n=int(problem_config.get("n", problem_config.get("num_samples", 1000))),
        d=int(problem_config.get("d", problem_config.get("dim", 20))),
        seed=seed,
        beta_true=None if beta_true is None else np.asarray(beta_true, dtype=float),
        sparse_beta=bool(problem_config.get("sparse_beta", False)),
        num_nonzero=None if problem_config.get("num_nonzero") is None else int(problem_config["num_nonzero"]),
        feature_scale=float(problem_config.get("feature_scale", 1.0)),
        intercept=float(problem_config.get("intercept", problem_config.get("noise", 0.0))),
    )
    save_logistic_dataset(
        resolve_project_path(save_path),
        data,
        reg_lambda=float(problem_config.get("reg_lambda", 0.0)),
    )


def _main() -> None:
    """Small smoke test for dataset generation and closed-form oracles."""
    n = 1000
    d = 20
    seed = 0
    reg_lambda = 1.0e-2

    data1 = generate_logistic_synthetic_data(
        n=n,
        d=d,
        seed=seed,
        sparse_beta=True,
        num_nonzero=5,
        feature_scale=1.0,
        intercept=0.25,
    )
    data2 = generate_logistic_synthetic_data(
        n=n,
        d=d,
        seed=seed,
        sparse_beta=True,
        num_nonzero=5,
        feature_scale=1.0,
        intercept=0.25,
    )

    same_data = bool(
        np.array_equal(data1["A"], data2["A"])
        and np.array_equal(data1["y"], data2["y"])
        and np.array_equal(data1["beta_true"], data2["beta_true"])
    )

    with TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "logistic_demo.npz"
        save_logistic_dataset(dataset_path, data1, reg_lambda=reg_lambda)
        problem = LogisticRegressionProblem.from_npz(dataset_path)

        x0 = np.zeros(problem.n, dtype=float)
        v = np.linspace(1.0, 2.0, problem.n, dtype=float)
        grad0 = problem.grad(x0)
        hv0 = problem.hvp(x0, v)

        print("A.shape:", problem.A.shape)
        print("y.mean():", float(np.mean(data1["y"])))
        print("beta_true nnz:", int(np.count_nonzero(data1["beta_true"])))
        print("f(x0):", problem.f(x0))
        print("||grad(x0)||:", float(np.linalg.norm(grad0)))
        print("hvp(x0, v).shape:", hv0.shape)
        print("same seed reproducible:", same_data)


if __name__ == "__main__":
    _main()

from __future__ import annotations

import bz2
import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence, TextIO
from urllib.request import urlopen

import numpy as np

from src.utils.io import load_npz, save_npz
from src.utils.paths import resolve_project_path

try:
    from scipy.special import expit
    from scipy.sparse import csr_matrix, hstack as sparse_hstack, isspmatrix_csr
except ImportError:  # pragma: no cover - scipy is expected for sparse support
    csr_matrix = None
    sparse_hstack = None

    def expit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=float), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def isspmatrix_csr(matrix: Any) -> bool:
        del matrix
        return False


MatrixLike = Any
InteractionPair = tuple[int, int]

_ALLOWED_FEATURE_DISTRIBUTIONS = {"gaussian", "student_t"}
_ALLOWED_COVARIANCE_TYPES = {"identity", "toeplitz"}
_ALLOWED_CLASS_BALANCE = {"auto", "target_positive_rate"}
_ALLOWED_RAW_SOURCE_FORMATS = {"libsvm", "svmlight"}


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


def _require_scipy_sparse(feature_name: str) -> None:
    if csr_matrix is None or sparse_hstack is None:
        raise ImportError(f"scipy is required for {feature_name}")


def _open_text_maybe_compressed(path: Path) -> TextIO:
    """Open a plain, .gz, or .bz2 text file."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _normalize_index_base(index_base: Any) -> int | str:
    """Normalize LIBSVM index-base config to 0, 1, or ``"auto"``."""
    if index_base is None:
        return 1
    if isinstance(index_base, str):
        lowered = index_base.lower()
        if lowered == "auto":
            return "auto"
        if lowered in {"0", "zero", "zero_based"}:
            return 0
        if lowered in {"1", "one", "one_based"}:
            return 1
    try:
        parsed = int(index_base)
    except (TypeError, ValueError) as exc:
        raise ValueError("index_base must be 0, 1, or 'auto'") from exc
    if parsed in (0, 1):
        return parsed
    raise ValueError("index_base must be 0, 1, or 'auto'")


def _parse_optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return parsed


def _download_file(url: str, destination: Path) -> None:
    """Download a raw dataset file without adding a new dependency."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as response, temporary_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _is_compressed_path(path: Path) -> bool:
    return path.suffix in {".bz2", ".gz"}


def _redownload_raw_dataset(download_url: str, raw_source_path: Path) -> None:
    if raw_source_path.exists():
        raw_source_path.unlink()
    print(f"Downloading raw dataset: {download_url} -> {raw_source_path}", flush=True)
    _download_file(download_url, raw_source_path)


@dataclass(frozen=True)
class _LogisticGenerationOptions:
    n: int
    d: int
    seed: int
    sparse_beta: bool
    num_nonzero: int | None
    feature_scale: float
    intercept: float
    beta_scale: float
    feature_distribution: str
    t_df: float
    covariance_type: str
    cov_rho: float
    interaction_pairs: tuple[InteractionPair, ...]
    interaction_scale: float
    num_categorical: int
    categorical_cardinalities: tuple[int, ...]
    categorical_effect_scale: float
    class_balance: str
    target_positive_rate: float | None
    label_flip_prob: float
    outlier_fraction: float
    outlier_scale: float
    sparse_X: bool
    x_density: float


def _validate_positive_int(value: int, name: str) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_float(value: Any, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _normalize_interaction_pairs(
    interaction_pairs: list[tuple[int, int]] | None,
    d: int,
) -> tuple[InteractionPair, ...]:
    if interaction_pairs is None:
        return ()
    normalized: list[InteractionPair] = []
    for index, pair in enumerate(interaction_pairs, start=1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"interaction_pairs[{index}] must be a length-2 list or tuple of indices"
            )
        i_raw, j_raw = pair
        i = int(i_raw)
        j = int(j_raw)
        if i == j:
            raise ValueError(f"interaction_pairs[{index}] must use two distinct indices")
        if i < 0 or i >= d or j < 0 or j >= d:
            raise ValueError(
                f"interaction_pairs[{index}] indices must be in [0, {d - 1}]"
            )
        normalized.append((i, j))
    return tuple(normalized)


def _normalize_categorical_cardinalities(
    num_categorical: int,
    categorical_cardinalities: list[int] | None,
) -> tuple[int, ...]:
    if num_categorical < 0:
        raise ValueError("num_categorical must be nonnegative")
    if num_categorical == 0:
        if categorical_cardinalities not in (None, []):
            raise ValueError(
                "categorical_cardinalities must be omitted or empty when num_categorical == 0"
            )
        return ()

    if categorical_cardinalities is None:
        return tuple(3 for _ in range(num_categorical))
    if len(categorical_cardinalities) != num_categorical:
        raise ValueError(
            "categorical_cardinalities must have length equal to num_categorical"
        )

    normalized = tuple(int(cardinality) for cardinality in categorical_cardinalities)
    for index, cardinality in enumerate(normalized, start=1):
        if cardinality < 2:
            raise ValueError(
                f"categorical_cardinalities[{index}] must be at least 2, got {cardinality}"
            )
    return normalized


def _normalize_generation_options(
    *,
    n: int,
    d: int,
    seed: int,
    sparse_beta: bool,
    num_nonzero: int | None,
    feature_scale: float,
    intercept: float,
    beta_scale: float,
    feature_distribution: str,
    t_df: float,
    covariance_type: str,
    cov_rho: float,
    interaction_pairs: list[tuple[int, int]] | None,
    interaction_scale: float,
    num_categorical: int,
    categorical_cardinalities: list[int] | None,
    categorical_effect_scale: float,
    class_balance: str,
    target_positive_rate: float | None,
    label_flip_prob: float,
    outlier_fraction: float,
    outlier_scale: float,
    sparse_X: bool,
    x_density: float,
) -> _LogisticGenerationOptions:
    n = _validate_positive_int(n, "n")
    d = _validate_positive_int(d, "d")
    seed = int(seed)
    feature_scale = _validate_float(feature_scale, "feature_scale")
    if feature_scale <= 0.0:
        raise ValueError("feature_scale must be positive")
    intercept = _validate_float(intercept, "intercept")
    beta_scale = _validate_float(beta_scale, "beta_scale")
    if beta_scale < 0.0:
        raise ValueError("beta_scale must be nonnegative")
    feature_distribution = str(feature_distribution).lower()
    if feature_distribution not in _ALLOWED_FEATURE_DISTRIBUTIONS:
        allowed = ", ".join(sorted(_ALLOWED_FEATURE_DISTRIBUTIONS))
        raise ValueError(
            f"feature_distribution must be one of: {allowed}; got {feature_distribution!r}"
        )
    t_df = _validate_float(t_df, "t_df")
    if t_df <= 0.0:
        raise ValueError("t_df must be positive")
    covariance_type = str(covariance_type).lower()
    if covariance_type not in _ALLOWED_COVARIANCE_TYPES:
        allowed = ", ".join(sorted(_ALLOWED_COVARIANCE_TYPES))
        raise ValueError(f"covariance_type must be one of: {allowed}; got {covariance_type!r}")
    cov_rho = _validate_float(cov_rho, "cov_rho")
    if abs(cov_rho) >= 1.0:
        raise ValueError("cov_rho must satisfy |cov_rho| < 1")
    interaction_scale = _validate_float(interaction_scale, "interaction_scale")
    categorical_effect_scale = _validate_float(
        categorical_effect_scale,
        "categorical_effect_scale",
    )
    class_balance = str(class_balance).lower()
    if class_balance not in _ALLOWED_CLASS_BALANCE:
        allowed = ", ".join(sorted(_ALLOWED_CLASS_BALANCE))
        raise ValueError(f"class_balance must be one of: {allowed}; got {class_balance!r}")
    if class_balance == "target_positive_rate":
        if target_positive_rate is None:
            raise ValueError(
                "target_positive_rate must be provided when class_balance == 'target_positive_rate'"
            )
        target_positive_rate = _validate_float(target_positive_rate, "target_positive_rate")
        if not (0.0 < target_positive_rate < 1.0):
            raise ValueError("target_positive_rate must lie strictly between 0 and 1")
    elif target_positive_rate is not None:
        target_positive_rate = _validate_float(target_positive_rate, "target_positive_rate")
    label_flip_prob = _validate_float(label_flip_prob, "label_flip_prob")
    if not (0.0 <= label_flip_prob <= 1.0):
        raise ValueError("label_flip_prob must satisfy 0 <= label_flip_prob <= 1")
    outlier_fraction = _validate_float(outlier_fraction, "outlier_fraction")
    if not (0.0 <= outlier_fraction <= 1.0):
        raise ValueError("outlier_fraction must satisfy 0 <= outlier_fraction <= 1")
    outlier_scale = _validate_float(outlier_scale, "outlier_scale")
    if outlier_scale <= 0.0:
        raise ValueError("outlier_scale must be positive")
    x_density = _validate_float(x_density, "x_density")
    if not (0.0 < x_density <= 1.0):
        raise ValueError("x_density must satisfy 0 < x_density <= 1")

    if num_nonzero is not None:
        num_nonzero = int(num_nonzero)
        if num_nonzero <= 0 or num_nonzero > d:
            raise ValueError("num_nonzero must satisfy 1 <= num_nonzero <= d")

    interaction_pairs_normalized = _normalize_interaction_pairs(interaction_pairs, d)
    categorical_cardinalities_normalized = _normalize_categorical_cardinalities(
        int(num_categorical),
        categorical_cardinalities,
    )

    if bool(sparse_X) and covariance_type != "identity":
        raise ValueError(
            "sparse_X=True currently supports covariance_type='identity' only"
        )

    return _LogisticGenerationOptions(
        n=n,
        d=d,
        seed=seed,
        sparse_beta=bool(sparse_beta),
        num_nonzero=num_nonzero,
        feature_scale=feature_scale,
        intercept=intercept,
        beta_scale=beta_scale,
        feature_distribution=feature_distribution,
        t_df=t_df,
        covariance_type=covariance_type,
        cov_rho=cov_rho,
        interaction_pairs=interaction_pairs_normalized,
        interaction_scale=interaction_scale,
        num_categorical=int(num_categorical),
        categorical_cardinalities=categorical_cardinalities_normalized,
        categorical_effect_scale=categorical_effect_scale,
        class_balance=class_balance,
        target_positive_rate=target_positive_rate,
        label_flip_prob=label_flip_prob,
        outlier_fraction=outlier_fraction,
        outlier_scale=outlier_scale,
        sparse_X=bool(sparse_X),
        x_density=x_density,
    )


def _sample_feature_values(
    rng: np.random.Generator,
    size: int | tuple[int, ...],
    feature_distribution: str,
    t_df: float,
) -> np.ndarray:
    if feature_distribution == "gaussian":
        return rng.normal(loc=0.0, scale=1.0, size=size).astype(float)
    if feature_distribution == "student_t":
        return rng.standard_t(df=t_df, size=size).astype(float)
    raise ValueError(f"Unsupported feature_distribution {feature_distribution!r}")


def _apply_dense_correlation(
    X: np.ndarray,
    covariance_type: str,
    cov_rho: float,
) -> np.ndarray:
    if covariance_type == "identity" or X.shape[1] == 0:
        return X
    indices = np.arange(X.shape[1])
    covariance = cov_rho ** np.abs(np.subtract.outer(indices, indices))
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(covariance)
        factor = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1.0e-12, None)))
    return np.asarray(X @ factor.T, dtype=float)


def _generate_dense_continuous_features(
    options: _LogisticGenerationOptions,
    rng: np.random.Generator,
) -> np.ndarray:
    X = _sample_feature_values(
        rng,
        size=(options.n, options.d),
        feature_distribution=options.feature_distribution,
        t_df=options.t_df,
    )
    X = _apply_dense_correlation(X, options.covariance_type, options.cov_rho)
    X *= options.feature_scale
    return np.asarray(X, dtype=float)


def _generate_sparse_continuous_features(
    options: _LogisticGenerationOptions,
    rng: np.random.Generator,
) -> Any:
    _require_scipy_sparse("sparse_X=True")
    total_entries = options.n * options.d
    nnz = max(1, int(round(options.x_density * total_entries)))
    flat_indices = rng.choice(total_entries, size=nnz, replace=False)
    row_indices = flat_indices // options.d
    col_indices = flat_indices % options.d
    values = _sample_feature_values(
        rng,
        size=nnz,
        feature_distribution=options.feature_distribution,
        t_df=options.t_df,
    )
    values = np.asarray(values * options.feature_scale, dtype=float)
    return csr_matrix((values, (row_indices, col_indices)), shape=(options.n, options.d))


def _apply_outliers_dense(
    X: np.ndarray,
    outlier_fraction: float,
    outlier_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    num_outliers = int(np.floor(outlier_fraction * X.shape[0]))
    if num_outliers <= 0 or outlier_scale == 1.0:
        return X
    outlier_rows = rng.choice(X.shape[0], size=num_outliers, replace=False)
    X_out = np.asarray(X, dtype=float).copy()
    X_out[outlier_rows] *= outlier_scale
    return X_out


def _apply_outliers_csr(
    X: Any,
    outlier_fraction: float,
    outlier_scale: float,
    rng: np.random.Generator,
) -> Any:
    num_outliers = int(np.floor(outlier_fraction * X.shape[0]))
    if num_outliers <= 0 or outlier_scale == 1.0:
        return X
    X_out = X.copy().tocsr()
    outlier_rows = rng.choice(X.shape[0], size=num_outliers, replace=False)
    for row in outlier_rows:
        start = int(X_out.indptr[row])
        end = int(X_out.indptr[row + 1])
        if start < end:
            X_out.data[start:end] *= outlier_scale
    return X_out


def _generate_continuous_beta(
    d: int,
    rng: np.random.Generator,
    sparse_beta: bool,
    num_nonzero: int | None,
    beta_scale: float,
) -> np.ndarray:
    if sparse_beta:
        nnz = int(num_nonzero if num_nonzero is not None else max(1, min(d, d // 5 or 1)))
        if nnz <= 0 or nnz > d:
            raise ValueError("num_nonzero must satisfy 1 <= num_nonzero <= d")
        beta = np.zeros(d, dtype=float)
        active = rng.choice(d, size=nnz, replace=False)
        beta[active] = rng.normal(size=nnz)
    else:
        beta = rng.normal(size=d).astype(float)
    return np.asarray(beta * beta_scale, dtype=float)


def _generate_categorical_feature_block(
    n: int,
    cardinalities: Sequence[int],
    rng: np.random.Generator,
    sparse_X: bool,
) -> MatrixLike:
    total_width = int(sum(cardinalities))
    if total_width == 0:
        return csr_matrix((n, 0), dtype=float) if sparse_X and csr_matrix is not None else np.zeros((n, 0), dtype=float)

    if sparse_X:
        _require_scipy_sparse("categorical sparse feature generation")
        rows = np.tile(np.arange(n, dtype=np.int64), len(cardinalities))
        cols = np.empty(n * len(cardinalities), dtype=np.int64)
        offset = 0
        cursor = 0
        for cardinality in cardinalities:
            codes = rng.integers(0, cardinality, size=n, endpoint=False)
            cols[cursor : cursor + n] = offset + codes
            cursor += n
            offset += cardinality
        data = np.ones(rows.shape[0], dtype=float)
        return csr_matrix((data, (rows, cols)), shape=(n, total_width))

    block = np.zeros((n, total_width), dtype=float)
    row_indices = np.arange(n)
    offset = 0
    for cardinality in cardinalities:
        codes = rng.integers(0, cardinality, size=n, endpoint=False)
        block[row_indices, offset + codes] = 1.0
        offset += cardinality
    return block


def _append_feature_blocks(
    X_continuous: MatrixLike,
    X_categorical: MatrixLike,
    sparse_X: bool,
) -> MatrixLike:
    if X_categorical.shape[1] == 0:
        return X_continuous
    if sparse_X:
        _require_scipy_sparse("sparse feature concatenation")
        return sparse_hstack([X_continuous, X_categorical], format="csr")
    return np.hstack([
        np.asarray(X_continuous, dtype=float),
        np.asarray(X_categorical, dtype=float),
    ])


def _matvec_matrix(X: MatrixLike, beta: np.ndarray) -> np.ndarray:
    return np.asarray(X @ np.asarray(beta, dtype=float).reshape(-1), dtype=float).reshape(-1)


def _interaction_contribution(
    X_continuous: MatrixLike,
    interaction_pairs: Sequence[InteractionPair],
    interaction_scale: float,
) -> np.ndarray:
    n = int(X_continuous.shape[0])
    if not interaction_pairs or interaction_scale == 0.0:
        return np.zeros(n, dtype=float)

    contribution = np.zeros(n, dtype=float)
    if _is_csr_matrix(X_continuous):
        for i, j in interaction_pairs:
            xi = np.asarray(X_continuous.getcol(i).toarray(), dtype=float).reshape(-1)
            xj = np.asarray(X_continuous.getcol(j).toarray(), dtype=float).reshape(-1)
            contribution += interaction_scale * xi * xj
        return contribution

    X_dense = np.asarray(X_continuous, dtype=float)
    for i, j in interaction_pairs:
        contribution += interaction_scale * X_dense[:, i] * X_dense[:, j]
    return contribution


def _split_or_generate_beta(
    beta_true: np.ndarray | None,
    continuous_dim: int,
    categorical_dim: int,
    rng: np.random.Generator,
    options: _LogisticGenerationOptions,
) -> np.ndarray:
    if categorical_dim < 0:
        raise ValueError("categorical_dim must be nonnegative")
    total_dim = continuous_dim + categorical_dim

    if beta_true is None:
        beta_continuous = _generate_continuous_beta(
            d=continuous_dim,
            rng=rng,
            sparse_beta=options.sparse_beta,
            num_nonzero=options.num_nonzero,
            beta_scale=options.beta_scale,
        )
        if categorical_dim > 0:
            beta_categorical = rng.normal(
                loc=0.0,
                scale=options.categorical_effect_scale,
                size=categorical_dim,
            ).astype(float)
        else:
            beta_categorical = np.zeros(0, dtype=float)
        return np.concatenate([beta_continuous, beta_categorical])

    beta_array = np.asarray(beta_true, dtype=float).reshape(-1)
    if beta_array.shape == (continuous_dim,):
        if categorical_dim == 0:
            return beta_array.astype(float, copy=False)
        beta_categorical = rng.normal(
            loc=0.0,
            scale=options.categorical_effect_scale,
            size=categorical_dim,
        ).astype(float)
        return np.concatenate([beta_array.astype(float, copy=False), beta_categorical])

    if beta_array.shape == (total_dim,):
        return beta_array.astype(float, copy=False)

    raise ValueError(
        "beta_true must have shape "
        f"({continuous_dim},) or ({total_dim},), got {beta_array.shape}"
    )


def _find_bias_shift_for_target_positive_rate(
    raw_logits: np.ndarray,
    target_positive_rate: float,
    *,
    tol: float = 1.0e-10,
    max_iter: int = 200,
) -> float:
    logits = np.asarray(raw_logits, dtype=float).reshape(-1)
    low = -80.0
    high = 80.0

    def mean_prob(shift: float) -> float:
        return float(np.mean(expit(logits + shift)))

    low_value = mean_prob(low)
    high_value = mean_prob(high)
    while low_value > target_positive_rate:
        low *= 2.0
        low_value = mean_prob(low)
    while high_value < target_positive_rate:
        high *= 2.0
        high_value = mean_prob(high)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_value = mean_prob(mid)
        if abs(mid_value - target_positive_rate) <= tol:
            return float(mid)
        if mid_value < target_positive_rate:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def _apply_label_noise(
    y: np.ndarray,
    label_flip_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int8).reshape(-1).copy()
    if label_flip_prob == 0.0:
        return labels
    flips = rng.random(labels.shape[0]) < label_flip_prob
    labels[flips] = np.int8(1) - labels[flips]
    return labels


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _generation_config_dict(
    options: _LogisticGenerationOptions,
    beta_true_supplied: bool,
) -> dict[str, Any]:
    return {
        "n": options.n,
        "d": options.d,
        "seed": options.seed,
        "sparse_beta": options.sparse_beta,
        "num_nonzero": options.num_nonzero,
        "feature_scale": options.feature_scale,
        "intercept": options.intercept,
        "beta_scale": options.beta_scale,
        "feature_distribution": options.feature_distribution,
        "t_df": options.t_df,
        "covariance_type": options.covariance_type,
        "cov_rho": options.cov_rho,
        "interaction_pairs": [list(pair) for pair in options.interaction_pairs],
        "interaction_scale": options.interaction_scale,
        "num_categorical": options.num_categorical,
        "categorical_cardinalities": list(options.categorical_cardinalities),
        "categorical_effect_scale": options.categorical_effect_scale,
        "class_balance": options.class_balance,
        "target_positive_rate": options.target_positive_rate,
        "label_flip_prob": options.label_flip_prob,
        "outlier_fraction": options.outlier_fraction,
        "outlier_scale": options.outlier_scale,
        "sparse_X": options.sparse_X,
        "x_density": options.x_density,
        "beta_true_supplied": beta_true_supplied,
    }


def _matrix_equal(left: MatrixLike, right: MatrixLike) -> bool:
    if _is_csr_matrix(left) and _is_csr_matrix(right):
        return bool(
            left.shape == right.shape
            and np.array_equal(left.data, right.data)
            and np.array_equal(left.indices, right.indices)
            and np.array_equal(left.indptr, right.indptr)
        )
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def load_libsvm_logistic_dataset(
    path: str | Path,
    *,
    n_features: int | None = None,
    index_base: int | str = 1,
    sample_size: int | None = None,
    sample_seed: int = 0,
    max_rows: int | None = None,
    add_bias: bool = False,
) -> dict[str, Any]:
    """Load a LIBSVM/SVMLight classification dataset as a CSR logistic dataset.

    The loader expects labels in ``{-1, +1}`` or ``{0, 1}`` and stores a CSR
    feature matrix compatible with :class:`LogisticRegressionProblem`. The raw
    file may be plain text, ``.gz``, or ``.bz2``. Feature indices are interpreted
    as one-based by default, matching standard LIBSVM files; set
    ``index_base: 0`` or ``index_base: auto`` in YAML when needed.
    """
    from src.problems.real_classification import _maybe_add_bias_column, load_libsvm_classification_dataset

    dataset = load_libsvm_classification_dataset(
        path,
        n_features=n_features,
        index_base=index_base,
        sample_size=sample_size,
        sample_seed=sample_seed,
        max_rows=max_rows,
        label_mode="binary",
    )
    matrix = _maybe_add_bias_column(dataset["A"], add_bias=add_bias)
    y_pm1 = _to_pm1(np.asarray(dataset["y"], dtype=float))
    generation_config = dict(dataset["generation_config"])
    generation_config.update(
        {
            "n": int(matrix.shape[0]),
            "d": int(matrix.shape[1]),
            "nnz": int(matrix.nnz) if _is_csr_matrix(matrix) else int(np.count_nonzero(matrix)),
            "add_bias": bool(add_bias),
            "n_features_with_bias": int(matrix.shape[1]),
        }
    )

    return {
        "A": matrix,
        "y": y_pm1,
        "beta_true": np.zeros(matrix.shape[1], dtype=float),
        "intercept": np.array(0.0, dtype=float),
        "positive_rate": np.array(float(np.mean(y_pm1 > 0.0)), dtype=float),
        "generation_config": generation_config,
    }


def generate_logistic_synthetic_data(
    n: int,
    d: int,
    seed: int = 0,
    beta_true: np.ndarray | None = None,
    sparse_beta: bool = False,
    num_nonzero: int | None = None,
    feature_scale: float = 1.0,
    intercept: float = 0.0,
    beta_scale: float = 1.0,
    feature_distribution: str = "gaussian",
    t_df: float = 3.0,
    covariance_type: str = "identity",
    cov_rho: float = 0.0,
    interaction_pairs: list[tuple[int, int]] | None = None,
    interaction_scale: float = 0.0,
    num_categorical: int = 0,
    categorical_cardinalities: list[int] | None = None,
    categorical_effect_scale: float = 1.0,
    class_balance: str = "auto",
    target_positive_rate: float | None = None,
    label_flip_prob: float = 0.0,
    outlier_fraction: float = 0.0,
    outlier_scale: float = 10.0,
    sparse_X: bool = False,
    x_density: float = 0.01,
) -> dict[str, Any]:
    """Generate synthetic logistic regression data with optional benchmark complexity.

    When all new arguments are omitted, the generator preserves the previous behavior:
    dense Gaussian features with independent coordinates, a linear logit model,
    and Bernoulli labels.

    Notes:
    - ``d`` refers to the number of continuous features before optional one-hot
      categorical columns are appended.
    - If ``beta_true`` is provided and categorical variables are requested, it may
      have shape ``(d,)`` for the continuous coefficients only or the final feature
      dimension after one-hot expansion.
    - If ``beta_true`` is generated internally, ``beta_scale`` is applied after
      sparse/dense coefficient generation. User-supplied ``beta_true`` is left as-is.
    """
    options = _normalize_generation_options(
        n=n,
        d=d,
        seed=seed,
        sparse_beta=sparse_beta,
        num_nonzero=num_nonzero,
        feature_scale=feature_scale,
        intercept=intercept,
        beta_scale=beta_scale,
        feature_distribution=feature_distribution,
        t_df=t_df,
        covariance_type=covariance_type,
        cov_rho=cov_rho,
        interaction_pairs=interaction_pairs,
        interaction_scale=interaction_scale,
        num_categorical=num_categorical,
        categorical_cardinalities=categorical_cardinalities,
        categorical_effect_scale=categorical_effect_scale,
        class_balance=class_balance,
        target_positive_rate=target_positive_rate,
        label_flip_prob=label_flip_prob,
        outlier_fraction=outlier_fraction,
        outlier_scale=outlier_scale,
        sparse_X=sparse_X,
        x_density=x_density,
    )

    rng = np.random.default_rng(options.seed)
    if options.sparse_X:
        X_continuous = _generate_sparse_continuous_features(options, rng)
        X_continuous = _apply_outliers_csr(
            X_continuous,
            options.outlier_fraction,
            options.outlier_scale,
            rng,
        )
    else:
        X_continuous = _generate_dense_continuous_features(options, rng)
        X_continuous = _apply_outliers_dense(
            X_continuous,
            options.outlier_fraction,
            options.outlier_scale,
            rng,
        )

    X_categorical = _generate_categorical_feature_block(
        options.n,
        options.categorical_cardinalities,
        rng,
        sparse_X=options.sparse_X,
    )
    categorical_dim = int(X_categorical.shape[1])
    beta_true_array = _split_or_generate_beta(
        beta_true=beta_true,
        continuous_dim=options.d,
        categorical_dim=categorical_dim,
        rng=rng,
        options=options,
    )

    A = _append_feature_blocks(X_continuous, X_categorical, sparse_X=options.sparse_X)
    linear_contribution = _matvec_matrix(A, beta_true_array)
    interaction_contribution = _interaction_contribution(
        X_continuous,
        options.interaction_pairs,
        options.interaction_scale,
    )
    raw_logits = linear_contribution + interaction_contribution + options.intercept

    intercept_shift = 0.0
    if options.class_balance == "target_positive_rate":
        intercept_shift = _find_bias_shift_for_target_positive_rate(
            raw_logits,
            float(options.target_positive_rate),
        )
    final_intercept = float(options.intercept + intercept_shift)
    logits_true = raw_logits + intercept_shift
    p_true = expit(logits_true)
    y = rng.binomial(1, p_true, size=options.n).astype(np.int8)
    y = _apply_label_noise(y, options.label_flip_prob, rng)

    positive_rate = float(np.mean(y))
    generation_config = _generation_config_dict(options, beta_true_supplied=beta_true is not None)

    return {
        "A": A,
        "y": y,
        "beta_true": beta_true_array.astype(float, copy=False),
        "intercept": np.array(final_intercept, dtype=float),
        "positive_rate": np.array(positive_rate, dtype=float),
        "generation_config": generation_config,
    }


def save_logistic_dataset(
    path: str | Path,
    data: Mapping[str, Any],
    reg_lambda: float = 0.0,
    generation_config: Mapping[str, Any] | None = None,
    regularize_bias: bool = True,
) -> None:
    """Save a logistic regression dataset to .npz, supporting dense or CSR matrices.

    The saved archive remains backward-compatible with the current loader and also
    stores lightweight metadata fields for experiment bookkeeping.
    """
    A = data["A"]
    y = np.asarray(data["y"]).reshape(-1)
    metadata_config = generation_config if generation_config is not None else data.get("generation_config")
    positive_rate = float(data.get("positive_rate", float(np.mean(y))))
    generation_config_json = json.dumps(_json_ready(metadata_config or {}), sort_keys=True)

    arrays: dict[str, Any] = {
        "y": y,
        "beta_true": np.asarray(data["beta_true"], dtype=float).reshape(-1),
        "intercept": np.array(float(data.get("intercept", 0.0)), dtype=float),
        "reg_lambda": np.array(float(reg_lambda), dtype=float),
        "regularize_bias": np.array(bool(regularize_bias)),
        "problem_type": np.asarray("logistic"),
        "positive_rate": np.array(positive_rate, dtype=float),
        "generation_config_json": np.asarray(generation_config_json),
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

    if {"X_data", "X_indices", "X_indptr", "X_shape"}.issubset(npz_data):
        if csr_matrix is None:
            raise ImportError("scipy is required to load CSR matrices from .npz files")
        shape = tuple(int(value) for value in np.asarray(npz_data["X_shape"]).tolist())
        return csr_matrix(
            (
                np.asarray(npz_data["X_data"], dtype=float),
                np.asarray(npz_data["X_indices"], dtype=np.int64),
                np.asarray(npz_data["X_indptr"], dtype=np.int64),
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
    regularize_bias: bool
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
        regularize_bias: bool = True,
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
        self.regularize_bias = bool(regularize_bias)
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
        if self.regularize_bias:
            reg_vector = x
        else:
            reg_vector = x.copy()
            reg_vector[-1] = 0.0
        reg = 0.5 * self.reg_lambda * float(np.dot(reg_vector, reg_vector))
        return float(np.mean(loss) + reg)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the closed-form gradient."""
        x = _as_float_vector(x)
        _, probs, _ = self._point_cache(x)
        weighted = self.y_pm1 * probs
        gradient = -(self._rmatvec(weighted) / self.m)
        reg_vector = x if self.regularize_bias else np.concatenate([x[:-1], np.zeros(1, dtype=float)])
        gradient += self.reg_lambda * reg_vector
        return np.asarray(gradient, dtype=float).reshape(-1)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate the closed-form Hessian-vector product."""
        x = _as_float_vector(x)
        v = _as_float_vector(v)
        _, _, d = self._point_cache(x)
        Av = self._matvec(v)
        hv = self._rmatvec(d * Av) / self.m
        reg_vector = v if self.regularize_bias else np.concatenate([v[:-1], np.zeros(1, dtype=float)])
        hv += self.reg_lambda * reg_vector
        return np.asarray(hv, dtype=float).reshape(-1)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        reg_lambda: float | None = None,
        regularize_bias: bool | None = None,
    ) -> "LogisticRegressionProblem":
        """Load a LogisticRegressionProblem from a saved .npz dataset."""
        arrays = load_npz(path)
        A = _load_npz_matrix(arrays)
        if "y" not in arrays:
            raise KeyError("Dataset .npz must contain label key 'y'")
        y = np.asarray(arrays["y"]).reshape(-1)
        loaded_reg_lambda = float(arrays["reg_lambda"]) if "reg_lambda" in arrays else 0.0
        loaded_regularize_bias = bool(np.asarray(arrays.get("regularize_bias", np.array(True))).item())
        return cls(
            A=A,
            y=y,
            reg_lambda=loaded_reg_lambda if reg_lambda is None else reg_lambda,
            regularize_bias=loaded_regularize_bias if regularize_bias is None else bool(regularize_bias),
        )


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
    data["X"] = np.asarray(data["A"].toarray() if _is_csr_matrix(data["A"]) else data["A"], dtype=float)
    return data


def save_logistic_data(path: str | Path, data: Mapping[str, Any]) -> None:
    """Backward-compatible alias for save_logistic_dataset."""
    save_logistic_dataset(
        path,
        data,
        reg_lambda=float(data.get("reg_lambda", 0.0)),
        regularize_bias=bool(data.get("regularize_bias", True)),
    )


def load_logistic_problem(path: str | Path) -> LogisticRegressionProblem:
    """Load a logistic regression problem from disk."""
    return LogisticRegressionProblem.from_npz(path)


def _logistic_generation_kwargs_from_config(
    problem_config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    beta_true = problem_config.get("beta_true")
    interaction_pairs = problem_config.get("interaction_pairs")
    categorical_cardinalities = problem_config.get("categorical_cardinalities")

    return {
        "n": int(problem_config.get("n", problem_config.get("num_samples", 1000))),
        "d": int(problem_config.get("d", problem_config.get("dim", 20))),
        "seed": int(seed),
        "beta_true": None if beta_true is None else np.asarray(beta_true, dtype=float),
        "sparse_beta": bool(problem_config.get("sparse_beta", False)),
        "num_nonzero": (
            None if problem_config.get("num_nonzero") is None else int(problem_config["num_nonzero"])
        ),
        "feature_scale": float(problem_config.get("feature_scale", 1.0)),
        "intercept": float(problem_config.get("intercept", problem_config.get("noise", 0.0))),
        "beta_scale": float(problem_config.get("beta_scale", 1.0)),
        "feature_distribution": str(problem_config.get("feature_distribution", "gaussian")),
        "t_df": float(problem_config.get("t_df", 3.0)),
        "covariance_type": str(problem_config.get("covariance_type", "identity")),
        "cov_rho": float(problem_config.get("cov_rho", 0.0)),
        "interaction_pairs": None if interaction_pairs is None else list(interaction_pairs),
        "interaction_scale": float(problem_config.get("interaction_scale", 0.0)),
        "num_categorical": int(problem_config.get("num_categorical", 0)),
        "categorical_cardinalities": (
            None if categorical_cardinalities is None else list(categorical_cardinalities)
        ),
        "categorical_effect_scale": float(problem_config.get("categorical_effect_scale", 1.0)),
        "class_balance": str(problem_config.get("class_balance", "auto")),
        "target_positive_rate": problem_config.get("target_positive_rate"),
        "label_flip_prob": float(problem_config.get("label_flip_prob", 0.0)),
        "outlier_fraction": float(problem_config.get("outlier_fraction", 0.0)),
        "outlier_scale": float(problem_config.get("outlier_scale", 10.0)),
        "sparse_X": bool(problem_config.get("sparse_X", False)),
        "x_density": float(problem_config.get("x_density", 0.01)),
    }


def build_logistic_problem(problem_config: Mapping[str, Any]) -> LogisticRegressionProblem:
    """Build a LogisticRegressionProblem from config or a saved dataset."""
    if "source" in problem_config:
        override = problem_config.get("reg_lambda")
        reg_lambda = None if override is None else float(override)
        return LogisticRegressionProblem.from_npz(
            resolve_project_path(problem_config["source"]),
            reg_lambda=reg_lambda,
            regularize_bias=problem_config.get("regularize_bias"),
        )

    data = generate_logistic_synthetic_data(
        **_logistic_generation_kwargs_from_config(
            problem_config,
            seed=int(problem_config.get("seed", 0)),
        )
    )
    return LogisticRegressionProblem(
        A=data["A"],
        y=data["y"],
        reg_lambda=float(problem_config.get("reg_lambda", 0.0)),
        regularize_bias=bool(problem_config.get("regularize_bias", True)),
    )


def generate_logistic_from_config(problem_config: Mapping[str, Any], save_path: str, seed: int) -> None:
    """Generate and save a synthetic logistic regression dataset from config."""
    source_format = str(problem_config.get("source_format", "")).lower()
    if "raw_source" in problem_config or source_format in _ALLOWED_RAW_SOURCE_FORMATS:
        if source_format and source_format not in _ALLOWED_RAW_SOURCE_FORMATS:
            allowed = ", ".join(sorted(_ALLOWED_RAW_SOURCE_FORMATS))
            raise ValueError(f"source_format must be one of: {allowed}; got {source_format!r}")
        if "raw_source" not in problem_config:
            raise ValueError("problem.raw_source is required for LIBSVM/SVMLight generation")
        raw_source_path = resolve_project_path(problem_config["raw_source"])
        download_url = problem_config.get("download_url")
        download_if_missing = bool(problem_config.get("download_if_missing", bool(download_url)))
        if not raw_source_path.exists() and download_if_missing:
            if not download_url:
                raise ValueError(
                    "problem.download_url is required when download_if_missing is true "
                    "and raw_source does not exist"
                )
            _redownload_raw_dataset(str(download_url), raw_source_path)
        n_features = problem_config.get("n_features", problem_config.get("d"))
        try:
            data = load_libsvm_logistic_dataset(
                problem_config["raw_source"],
                n_features=None if n_features is None else int(n_features),
                index_base=problem_config.get("index_base", 1),
                sample_size=problem_config.get("sample_size"),
                sample_seed=int(problem_config.get("sample_seed", 0)),
                max_rows=problem_config.get("max_rows"),
                add_bias=bool(problem_config.get("add_bias", True)),
            )
        except (EOFError, OSError) as exc:
            download_if_corrupt = bool(problem_config.get("download_if_corrupt", bool(download_url)))
            if not (download_url and download_if_corrupt and _is_compressed_path(raw_source_path)):
                raise
            print(
                f"Raw compressed dataset appears incomplete or corrupted: {raw_source_path}. "
                "Re-downloading once.",
                flush=True,
            )
            _redownload_raw_dataset(str(download_url), raw_source_path)
            try:
                data = load_libsvm_logistic_dataset(
                    problem_config["raw_source"],
                    n_features=None if n_features is None else int(n_features),
                    index_base=problem_config.get("index_base", 1),
                    sample_size=problem_config.get("sample_size"),
                    sample_seed=int(problem_config.get("sample_seed", 0)),
                    max_rows=problem_config.get("max_rows"),
                    add_bias=bool(problem_config.get("add_bias", True)),
                )
            except (EOFError, OSError) as retry_exc:
                raise RuntimeError(
                    f"Failed to read {raw_source_path} after re-downloading. "
                    "The download may be incomplete; remove the raw file and try again, "
                    "or download it manually."
                ) from retry_exc
        save_logistic_dataset(
            resolve_project_path(save_path),
            data,
            reg_lambda=float(problem_config.get("reg_lambda", 1.0e-3)),
            generation_config=data.get("generation_config"),
            regularize_bias=bool(problem_config.get("regularize_bias", True)),
        )
        return

    generation_kwargs = _logistic_generation_kwargs_from_config(problem_config, seed=seed)
    data = generate_logistic_synthetic_data(**generation_kwargs)
    save_logistic_dataset(
        resolve_project_path(save_path),
        data,
        reg_lambda=float(problem_config.get("reg_lambda", 0.0)),
        generation_config=data.get("generation_config"),
        regularize_bias=bool(problem_config.get("regularize_bias", True)),
    )


def _main() -> None:
    """Small smoke test for dataset generation and closed-form oracles."""
    n = 1000
    d = 20
    seed = 0
    reg_lambda = 1.0e-2

    smoke_kwargs = {
        "n": n,
        "d": d,
        "seed": seed,
        "sparse_beta": True,
        "num_nonzero": 5,
        "feature_scale": 1.0,
        "intercept": 0.25,
        "beta_scale": 0.5,
        "feature_distribution": "student_t",
        "t_df": 3.0,
        "covariance_type": "toeplitz",
        "cov_rho": 0.5,
        "interaction_pairs": [(0, 1), (2, 3)],
        "interaction_scale": 0.3,
        "num_categorical": 2,
        "categorical_cardinalities": [3, 4],
        "categorical_effect_scale": 0.5,
        "class_balance": "target_positive_rate",
        "target_positive_rate": 0.2,
        "label_flip_prob": 0.01,
        "outlier_fraction": 0.01,
        "outlier_scale": 10.0,
        "sparse_X": False,
    }
    data1 = generate_logistic_synthetic_data(**smoke_kwargs)
    data2 = generate_logistic_synthetic_data(**smoke_kwargs)

    same_data = bool(
        _matrix_equal(data1["A"], data2["A"])
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
        print("positive rate:", float(np.mean(data1["y"])))
        print("final dimension:", int(problem.n))
        print("sparse_X used:", bool(_is_csr_matrix(data1["A"])))
        print("beta_true nnz:", int(np.count_nonzero(data1["beta_true"])))
        print("f(x0):", problem.f(x0))
        print("||grad(x0)||:", float(np.linalg.norm(grad0)))
        print("hvp(x0, v).shape:", hv0.shape)
        print("same seed reproducible:", same_data)


if __name__ == "__main__":
    _main()


# Example benchmark configs
#
# 1. baseline linear logistic
# problem:
#   type: logistic
#   n: 5000
#   d: 1000
#   reg_lambda: 1.0e-3
#   seed: 0
#   feature_scale: 1.0
#   sparse_beta: false
#
# 2. non-separable / overlapping
# problem:
#   type: logistic
#   n: 5000
#   d: 1000
#   seed: 0
#   beta_scale: 0.1
#   feature_scale: 2.0
#   intercept: 0.0
#
# 3. interaction misspecification
# problem:
#   type: logistic
#   n: 5000
#   d: 1000
#   seed: 0
#   interaction_pairs:
#     - [0, 1]
#     - [2, 3]
#   interaction_scale: 1.0
#
# 4. imbalanced
# problem:
#   type: logistic
#   n: 5000
#   d: 1000
#   seed: 0
#   class_balance: target_positive_rate
#   target_positive_rate: 0.05
#
# 5. p >> n sparse beta
# problem:
#   type: logistic
#   n: 200
#   d: 5000
#   seed: 0
#   sparse_beta: true
#   num_nonzero: 20
#   beta_scale: 0.2
#
# 6. correlated ill-conditioned
# problem:
#   type: logistic
#   n: 5000
#   d: 1000
#   seed: 0
#   covariance_type: toeplitz
#   cov_rho: 0.95
#
# 7. heavy-tailed with outliers
# problem:
#   type: logistic
#   n: 5000
#   d: 1000
#   seed: 0
#   feature_distribution: student_t
#   t_df: 3.0
#   outlier_fraction: 0.01
#   outlier_scale: 20.0
#
# 8. categorical mixed features
# problem:
#   type: logistic
#   n: 5000
#   d: 1000
#   seed: 0
#   num_categorical: 2
#   categorical_cardinalities: [3, 4]
#   categorical_effect_scale: 0.5

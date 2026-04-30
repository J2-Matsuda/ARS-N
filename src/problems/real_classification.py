from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.io import load_npz, save_npz
from src.utils.paths import resolve_project_path

from src.problems.logistic import (
    MatrixLike,
    _as_float_vector,
    _is_compressed_path,
    _is_csr_matrix,
    _json_ready,
    _load_npz_matrix,
    _normalize_index_base,
    _open_text_maybe_compressed,
    _parse_optional_positive_int,
    _redownload_raw_dataset,
    _require_scipy_sparse,
    _to_pm1,
    csr_matrix,
    expit,
    isspmatrix_csr,
    sparse_hstack,
)

try:
    from scipy.special import logsumexp
except ImportError:  # pragma: no cover
    def logsumexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        max_values = np.max(array, axis=axis, keepdims=True)
        stabilized = np.exp(array - max_values)
        summed = np.sum(stabilized, axis=axis, keepdims=True)
        result = np.log(summed) + max_values
        if keepdims:
            return result
        if axis is None:
            return np.asarray(result).reshape(())
        return np.squeeze(result, axis=axis)

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None


def _as_int_vector(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Labels must be finite")
    rounded = np.rint(array)
    if not np.allclose(array, rounded):
        raise ValueError("Labels must be integer-valued")
    return rounded.astype(np.int64, copy=False)


def _read_scalar_string(npz_data: Mapping[str, Any], key: str, default: str) -> str:
    if key not in npz_data:
        return default
    value = np.asarray(npz_data[key])
    if value.shape == ():
        return str(value.item())
    return str(value.reshape(-1)[0])


def _read_scalar_bool(npz_data: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in npz_data:
        return default
    value = np.asarray(npz_data[key])
    if value.shape == ():
        return bool(value.item())
    return bool(value.reshape(-1)[0])


def _load_generation_config_from_npz(npz_data: Mapping[str, Any]) -> dict[str, Any]:
    raw = np.asarray(npz_data.get("generation_config_json", np.asarray("{}")))
    text = str(raw.item()) if raw.shape == () else str(raw.reshape(-1)[0])
    return json.loads(text)


def _subset_rows(matrix: MatrixLike, row_indices: np.ndarray) -> MatrixLike:
    if _is_csr_matrix(matrix):
        return matrix[row_indices]
    return np.asarray(matrix)[row_indices]


def _sample_rows(
    A: MatrixLike,
    primary_target: Any,
    *,
    sample_size: int | None,
    sample_seed: int,
    secondary_target: Any | None = None,
) -> tuple[MatrixLike, Any, Any | None]:
    if sample_size is None:
        return A, primary_target, secondary_target

    m = int(A.shape[0])
    if sample_size > m:
        raise ValueError(f"sample_size={sample_size} exceeds number of rows m={m}")

    rng = np.random.default_rng(int(sample_seed))
    row_indices = rng.choice(m, size=sample_size, replace=False)
    A_sampled = _subset_rows(A, row_indices)
    primary_sampled = _subset_rows(primary_target, row_indices)
    secondary_sampled = None if secondary_target is None else _subset_rows(secondary_target, row_indices)
    return A_sampled, primary_sampled, secondary_sampled


def _infer_multiclass_labels(
    raw_labels: np.ndarray,
    num_classes: int | None,
) -> tuple[np.ndarray, dict[str, int]]:
    integer_labels = _as_int_vector(raw_labels)
    unique = np.unique(integer_labels)
    if unique.size == 0:
        raise ValueError("Classification dataset has no labels")

    mapping: dict[int, int]
    if num_classes is not None and np.array_equal(unique, np.arange(num_classes, dtype=np.int64)):
        mapping = {int(label): int(label) for label in unique}
    elif num_classes is not None and np.array_equal(
        unique, np.arange(1, num_classes + 1, dtype=np.int64)
    ):
        mapping = {int(label): int(label - 1) for label in unique}
    else:
        mapping = {int(label): index for index, label in enumerate(unique.tolist())}
        if num_classes is not None and len(mapping) > num_classes:
            raise ValueError(
                f"Observed {len(mapping)} classes, which exceeds num_classes={num_classes}"
            )

    mapped = np.asarray([mapping[int(label)] for label in integer_labels], dtype=np.int64)
    return mapped, {str(key): int(value) for key, value in mapping.items()}


def _maybe_add_bias_column(A: MatrixLike, add_bias: bool) -> MatrixLike:
    if not add_bias:
        return A
    if _is_csr_matrix(A):
        _require_scipy_sparse("add_bias=True")
        ones = csr_matrix(np.ones((A.shape[0], 1), dtype=float))
        return sparse_hstack([A, ones], format="csr")
    dense = np.asarray(A, dtype=float)
    ones = np.ones((dense.shape[0], 1), dtype=float)
    return np.hstack([dense, ones])


def _multilabel_csr_from_rows(
    label_rows: Sequence[Sequence[int]],
    *,
    num_labels: int,
) -> Any:
    _require_scipy_sparse("multilabel CSR target construction")
    rows: list[int] = []
    cols: list[int] = []
    for row_index, label_indices in enumerate(label_rows):
        for label_index in label_indices:
            rows.append(int(row_index))
            cols.append(int(label_index))
    data = np.ones(len(rows), dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(len(label_rows), int(num_labels)), dtype=float)


def _load_npz_target_matrix(npz_data: Mapping[str, Any], prefix: str) -> MatrixLike:
    if prefix in npz_data:
        return np.asarray(npz_data[prefix])

    data_key = f"{prefix}_data"
    indices_key = f"{prefix}_indices"
    indptr_key = f"{prefix}_indptr"
    shape_key = f"{prefix}_shape"
    if {data_key, indices_key, indptr_key, shape_key}.issubset(npz_data):
        if csr_matrix is None:
            raise ImportError(f"scipy is required to load CSR target matrix {prefix!r}")
        shape = tuple(int(value) for value in np.asarray(npz_data[shape_key]).tolist())
        return csr_matrix(
            (
                np.asarray(npz_data[data_key], dtype=float),
                np.asarray(npz_data[indices_key], dtype=np.int64),
                np.asarray(npz_data[indptr_key], dtype=np.int64),
            ),
            shape=shape,
        )

    raise KeyError(
        f"Dataset .npz must contain either dense key {prefix!r} or CSR keys "
        f"{data_key}/{indices_key}/{indptr_key}/{shape_key}"
    )


def _dense_target_matrix(matrix: MatrixLike) -> np.ndarray:
    if _is_csr_matrix(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _require_text_dataset_dependencies(feature_name: str) -> None:
    missing: list[str] = []
    if load_dataset is None:
        missing.append("datasets")
    if TfidfVectorizer is None:
        missing.append("scikit-learn")
    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            f"{feature_name} requires optional dependencies: {joined}. "
            "Install them with: pip install -e .[lexglue]"
        )


def _regularization_mask(num_rows: int, width: int, regularize_bias: bool) -> np.ndarray:
    mask = np.ones((num_rows, width), dtype=float)
    if not regularize_bias:
        mask[-1, :] = 0.0
    return mask


def _regularized_frobenius_term(W: np.ndarray, reg_lambda: float, regularize_bias: bool) -> float:
    if reg_lambda == 0.0:
        return 0.0
    mask = _regularization_mask(W.shape[0], W.shape[1], regularize_bias)
    return 0.5 * reg_lambda * float(np.sum(mask * W * W))


def _dataset_metadata(
    *,
    dataset_name: str,
    source_format: str,
    raw_source: str,
    download_url: str | None,
    original_num_samples: int,
    sample_size: int | None,
    sample_seed: int,
    n_features_without_bias: int,
    n_features_with_bias: int,
    reg_lambda: float,
    add_bias: bool,
    regularize_bias: bool,
    num_classes: int | None = None,
    num_labels: int | None = None,
) -> dict[str, Any]:
    dim = n_features_with_bias
    if num_classes is not None:
        dim *= int(num_classes)
    if num_labels is not None:
        dim *= int(num_labels)
    metadata = {
        "dataset_name": dataset_name,
        "source_format": source_format,
        "raw_source": raw_source,
        "download_url": download_url,
        "original_num_samples": int(original_num_samples),
        "sample_size": None if sample_size is None else int(sample_size),
        "sample_seed": int(sample_seed),
        "n_features_without_bias": int(n_features_without_bias),
        "n_features_with_bias": int(n_features_with_bias),
        "dim": int(dim),
        "reg_lambda": float(reg_lambda),
        "add_bias": bool(add_bias),
        "regularize_bias": bool(regularize_bias),
    }
    if num_classes is not None:
        metadata["num_classes"] = int(num_classes)
    if num_labels is not None:
        metadata["num_labels"] = int(num_labels)
    return metadata


def load_libsvm_classification_dataset(
    path: str | Path,
    *,
    n_features: int | None = None,
    index_base: int | str = 1,
    sample_size: int | None = None,
    sample_seed: int = 0,
    max_rows: int | None = None,
    label_mode: str = "auto",
    num_classes: int | None = None,
) -> dict[str, Any]:
    _require_scipy_sparse("LIBSVM/SVMLight classification dataset loading")
    source_path = resolve_project_path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Raw LIBSVM/SVMLight dataset not found: {source_path}")

    n_features = _parse_optional_positive_int(n_features, "n_features")
    sample_size = _parse_optional_positive_int(sample_size, "sample_size")
    max_rows = _parse_optional_positive_int(max_rows, "max_rows")
    normalized_index_base = _normalize_index_base(index_base)

    raw_labels: list[float] = []
    data_values: list[float] = []
    column_indices: list[int] = []
    indptr: list[int] = [0]
    min_raw_index: int | None = None

    with _open_text_maybe_compressed(source_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if sample_size is None and max_rows is not None and len(raw_labels) >= max_rows:
                break

            content = raw_line.split("#", 1)[0].strip()
            if not content:
                continue

            fields = content.split()
            try:
                raw_labels.append(float(fields[0]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label at {source_path}:{line_number}: {fields[0]!r}"
                ) from exc

            for token in fields[1:]:
                if ":" not in token:
                    raise ValueError(
                        f"Invalid LIBSVM token at {source_path}:{line_number}: {token!r}"
                    )
                index_text, value_text = token.split(":", 1)
                if index_text == "qid":
                    continue
                try:
                    raw_index = int(index_text)
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid feature token at {source_path}:{line_number}: {token!r}"
                    ) from exc
                if value == 0.0:
                    continue

                min_raw_index = raw_index if min_raw_index is None else min(min_raw_index, raw_index)
                if normalized_index_base == "auto":
                    column = raw_index
                else:
                    column = raw_index - normalized_index_base
                    if column < 0:
                        raise ValueError(
                            f"Feature index {raw_index} at {source_path}:{line_number} "
                            f"is invalid for index_base={normalized_index_base}"
                        )
                    if n_features is not None and column >= n_features:
                        raise ValueError(
                            f"Feature index {raw_index} at {source_path}:{line_number} "
                            f"exceeds n_features={n_features}"
                        )

                column_indices.append(column)
                data_values.append(value)

            indptr.append(len(column_indices))

    if not raw_labels:
        raise ValueError(f"Raw LIBSVM/SVMLight dataset is empty: {source_path}")

    if normalized_index_base == "auto":
        inferred_base = 0 if min_raw_index == 0 else 1
        column_indices = [index - inferred_base for index in column_indices]
        normalized_index_base = inferred_base

    if not column_indices:
        raise ValueError(f"Raw LIBSVM/SVMLight dataset has no nonzero features: {source_path}")

    inferred_n_features = int(max(column_indices) + 1)
    if n_features is None:
        n_features = inferred_n_features
    elif inferred_n_features > n_features:
        raise ValueError(
            f"Dataset uses feature column {inferred_n_features - 1}, which exceeds n_features={n_features}"
        )

    A = csr_matrix(
        (
            np.asarray(data_values, dtype=float),
            np.asarray(column_indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(raw_labels), n_features),
    )
    A.sum_duplicates()
    A.sort_indices()

    label_mode_normalized = str(label_mode).lower()
    if label_mode_normalized == "binary":
        labels = _to_pm1(np.asarray(raw_labels, dtype=float))
        label_mapping = {str(value): int(mapped) for value, mapped in zip(sorted(set(raw_labels)), sorted(set(labels)))}
    else:
        labels, label_mapping = _infer_multiclass_labels(
            np.asarray(raw_labels, dtype=float),
            num_classes=num_classes,
        )

    original_num_samples = int(A.shape[0])
    A, labels, _ = _sample_rows(
        A,
        labels,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )

    generation_config = {
        "source_format": "libsvm",
        "raw_source": str(path),
        "original_num_samples": original_num_samples,
        "sample_size": None if sample_size is None else int(sample_size),
        "sample_seed": int(sample_seed),
        "n_features_without_bias": int(n_features),
        "index_base": normalized_index_base,
        "max_rows": max_rows,
        "label_mapping": label_mapping,
    }

    return {
        "A": A,
        "y": labels,
        "label_mapping": label_mapping,
        "generation_config": generation_config,
    }


def load_multilabel_dataset(
    path: str | Path,
    *,
    source_format: str,
    n_features: int | None = None,
    num_labels: int,
    index_base: int | str = 1,
    label_index_base: int | str = 1,
    sample_size: int | None = None,
    sample_seed: int = 0,
    max_rows: int | None = None,
) -> dict[str, Any]:
    source_format_normalized = str(source_format).lower()
    sample_size = _parse_optional_positive_int(sample_size, "sample_size")
    max_rows = _parse_optional_positive_int(max_rows, "max_rows")
    num_labels = int(num_labels)
    if num_labels <= 0:
        raise ValueError("num_labels must be positive")

    if source_format_normalized == "lexglue_unfair_tos_tfidf":
        _require_text_dataset_dependencies("LexGLUE UNFAIR-ToS TF-IDF loading")
        cache_dir = resolve_project_path(path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = "coastalcph/lex_glue"
        dataset_config = "unfair_tos"
        split_name = "train"
        text_key = "text"

        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split_name,
            cache_dir=str(cache_dir),
        )

        texts: list[str] = []
        label_rows: list[list[int]] = []
        for row in dataset:
            texts.append(str(row[text_key]))
            if "labels" in row and row["labels"] is not None:
                labels = sorted({int(value) for value in row["labels"]})
            elif "label" in row and row["label"] is not None:
                scalar = int(row["label"])
                if scalar == num_labels:
                    labels = []
                elif 0 <= scalar < num_labels:
                    labels = [scalar]
                else:
                    raise ValueError(
                        f"Unexpected scalar label {scalar} for num_labels={num_labels} "
                        "in LexGLUE UNFAIR-ToS"
                    )
            else:
                labels = []

            for label_index in labels:
                if label_index < 0 or label_index >= num_labels:
                    raise ValueError(
                        f"Label index {label_index} is invalid for num_labels={num_labels}"
                    )
            label_rows.append(labels)

        if not texts:
            raise ValueError("LexGLUE UNFAIR-ToS split is empty")

        original_num_samples = len(texts)
        if sample_size is not None:
            if sample_size > original_num_samples:
                raise ValueError(
                    f"sample_size={sample_size} exceeds number of rows m={original_num_samples}"
                )
            rng = np.random.default_rng(int(sample_seed))
            row_indices = rng.choice(original_num_samples, size=sample_size, replace=False)
            texts = [texts[int(index)] for index in row_indices]
            label_rows = [label_rows[int(index)] for index in row_indices]

        vectorizer = TfidfVectorizer(dtype=np.float64)
        A = vectorizer.fit_transform(texts).tocsr()
        Y = _multilabel_csr_from_rows(label_rows, num_labels=num_labels)
        generation_config = {
            "source_format": source_format_normalized,
            "raw_source": str(path),
            "hf_dataset_name": dataset_name,
            "hf_dataset_config": dataset_config,
            "hf_split": split_name,
            "text_key": text_key,
            "original_num_samples": int(original_num_samples),
            "sample_size": None if sample_size is None else int(sample_size),
            "sample_seed": int(sample_seed),
            "n_features_without_bias": int(A.shape[1]),
            "num_labels": int(num_labels),
        }
        return {"A": A, "Y": Y, "generation_config": generation_config}

    if source_format_normalized == "npz":
        arrays = load_npz(resolve_project_path(path))
        A = _load_npz_matrix(arrays)
        if "Y" in arrays or "Y_data" in arrays:
            Y = _load_npz_target_matrix(arrays, "Y")
        elif "y_multilabel" in arrays:
            Y = np.asarray(arrays["y_multilabel"], dtype=float)
        else:
            raise KeyError("Multilabel .npz must contain Y or y_multilabel")
        if n_features is not None and int(A.shape[1]) != int(n_features):
            raise ValueError(
                f"Feature dimension mismatch: npz has {int(A.shape[1])}, expected n_features={int(n_features)}"
            )
        if int(Y.shape[1]) != num_labels:
            raise ValueError(
                f"Label dimension mismatch: npz has {int(Y.shape[1])}, expected num_labels={num_labels}"
            )
        original_num_samples = int(A.shape[0])
        A, Y, _ = _sample_rows(A, Y, sample_size=sample_size, sample_seed=sample_seed)
        generation_config = {
            "source_format": source_format_normalized,
            "raw_source": str(path),
            "original_num_samples": original_num_samples,
            "sample_size": None if sample_size is None else int(sample_size),
            "sample_seed": int(sample_seed),
            "n_features_without_bias": int(A.shape[1]),
            "num_labels": num_labels,
        }
        return {"A": A, "Y": Y, "generation_config": generation_config}

    if source_format_normalized != "multilabel_libsvm":
        raise ValueError(
            "source_format must be 'multilabel_libsvm', 'npz', or 'lexglue_unfair_tos_tfidf'"
        )

    _require_scipy_sparse("multilabel LIBSVM dataset loading")
    source_path = resolve_project_path(path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw multilabel dataset not found: {source_path}. "
            "Place the file there, or set problem.download_if_missing: true "
            "with a valid problem.download_url."
        )

    n_features = _parse_optional_positive_int(n_features, "n_features")
    normalized_feature_index_base = _normalize_index_base(index_base)
    normalized_label_index_base = _normalize_index_base(label_index_base)
    if normalized_label_index_base == "auto":
        normalized_label_index_base = 1

    label_rows: list[list[int]] = []
    data_values: list[float] = []
    column_indices: list[int] = []
    indptr: list[int] = [0]
    min_raw_feature_index: int | None = None

    with _open_text_maybe_compressed(source_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if sample_size is None and max_rows is not None and len(label_rows) >= max_rows:
                break

            content = raw_line.split("#", 1)[0].strip()
            if not content:
                continue

            fields = content.split()
            label_token = ""
            feature_tokens = fields
            if fields and ":" not in fields[0]:
                label_token = fields[0]
                feature_tokens = fields[1:]

            label_indices: list[int] = []
            if label_token:
                for raw_label in label_token.split(","):
                    if not raw_label:
                        continue
                    parsed_label = int(raw_label)
                    label_index = parsed_label - int(normalized_label_index_base)
                    if label_index < 0 or label_index >= num_labels:
                        raise ValueError(
                            f"Label index {parsed_label} at {source_path}:{line_number} "
                            f"is invalid for num_labels={num_labels}"
                        )
                    label_indices.append(label_index)
            label_rows.append(label_indices)

            for token in feature_tokens:
                if ":" not in token:
                    raise ValueError(
                        f"Invalid multilabel token at {source_path}:{line_number}: {token!r}"
                    )
                index_text, value_text = token.split(":", 1)
                raw_index = int(index_text)
                value = float(value_text)
                if value == 0.0:
                    continue
                min_raw_feature_index = (
                    raw_index if min_raw_feature_index is None else min(min_raw_feature_index, raw_index)
                )
                if normalized_feature_index_base == "auto":
                    column = raw_index
                else:
                    column = raw_index - int(normalized_feature_index_base)
                    if column < 0:
                        raise ValueError(
                            f"Feature index {raw_index} at {source_path}:{line_number} "
                            f"is invalid for index_base={normalized_feature_index_base}"
                        )
                    if n_features is not None and column >= n_features:
                        raise ValueError(
                            f"Feature index {raw_index} at {source_path}:{line_number} exceeds n_features={n_features}"
                        )
                column_indices.append(column)
                data_values.append(value)

            indptr.append(len(column_indices))

    if not label_rows:
        raise ValueError(f"Raw multilabel dataset is empty: {source_path}")

    if normalized_feature_index_base == "auto":
        inferred_base = 0 if min_raw_feature_index == 0 else 1
        column_indices = [index - inferred_base for index in column_indices]
        normalized_feature_index_base = inferred_base

    inferred_n_features = int(max(column_indices) + 1) if column_indices else 0
    if n_features is None:
        n_features = inferred_n_features
    elif inferred_n_features > n_features:
        raise ValueError(
            f"Dataset uses feature column {inferred_n_features - 1}, which exceeds n_features={n_features}"
        )

    A = csr_matrix(
        (
            np.asarray(data_values, dtype=float),
            np.asarray(column_indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(label_rows), n_features),
    )
    A.sum_duplicates()
    A.sort_indices()

    Y = _multilabel_csr_from_rows(label_rows, num_labels=num_labels)
    original_num_samples = int(A.shape[0])
    A, Y, _ = _sample_rows(A, Y, sample_size=sample_size, sample_seed=sample_seed)
    generation_config = {
        "source_format": source_format_normalized,
        "raw_source": str(path),
        "original_num_samples": original_num_samples,
        "sample_size": None if sample_size is None else int(sample_size),
        "sample_seed": int(sample_seed),
        "n_features_without_bias": int(n_features),
        "num_labels": int(num_labels),
        "index_base": normalized_feature_index_base,
        "label_index_base": normalized_label_index_base,
        "max_rows": max_rows,
    }
    return {"A": A, "Y": Y, "generation_config": generation_config}


def save_softmax_dataset(
    path: str | Path,
    data: Mapping[str, Any],
    *,
    reg_lambda: float = 0.0,
    generation_config: Mapping[str, Any] | None = None,
    regularize_bias: bool = True,
) -> None:
    A = data["A"]
    y = _as_int_vector(data["y"])
    arrays: dict[str, Any] = {
        "y": y,
        "num_classes": np.array(int(data["num_classes"]), dtype=np.int64),
        "reg_lambda": np.array(float(reg_lambda), dtype=float),
        "regularize_bias": np.array(bool(regularize_bias)),
        "problem_type": np.asarray("softmax"),
        "generation_config_json": np.asarray(
            json.dumps(_json_ready(generation_config or data.get("generation_config", {})), sort_keys=True)
        ),
    }
    if _is_csr_matrix(A):
        arrays["A_data"] = np.asarray(A.data, dtype=float)
        arrays["A_indices"] = np.asarray(A.indices, dtype=np.int64)
        arrays["A_indptr"] = np.asarray(A.indptr, dtype=np.int64)
        arrays["A_shape"] = np.asarray(A.shape, dtype=np.int64)
    else:
        arrays["A"] = np.asarray(A, dtype=float)
    save_npz(path, **arrays)


def save_multilabel_logistic_dataset(
    path: str | Path,
    data: Mapping[str, Any],
    *,
    reg_lambda: float = 0.0,
    generation_config: Mapping[str, Any] | None = None,
    regularize_bias: bool = True,
) -> None:
    A = data["A"]
    Y = data["Y"]
    arrays: dict[str, Any] = {
        "num_labels": np.array(int(data["num_labels"]), dtype=np.int64),
        "reg_lambda": np.array(float(reg_lambda), dtype=float),
        "regularize_bias": np.array(bool(regularize_bias)),
        "problem_type": np.asarray("multilabel_logistic"),
        "generation_config_json": np.asarray(
            json.dumps(_json_ready(generation_config or data.get("generation_config", {})), sort_keys=True)
        ),
    }
    if _is_csr_matrix(A):
        arrays["A_data"] = np.asarray(A.data, dtype=float)
        arrays["A_indices"] = np.asarray(A.indices, dtype=np.int64)
        arrays["A_indptr"] = np.asarray(A.indptr, dtype=np.int64)
        arrays["A_shape"] = np.asarray(A.shape, dtype=np.int64)
    else:
        arrays["A"] = np.asarray(A, dtype=float)

    if _is_csr_matrix(Y):
        arrays["Y_data"] = np.asarray(Y.data, dtype=float)
        arrays["Y_indices"] = np.asarray(Y.indices, dtype=np.int64)
        arrays["Y_indptr"] = np.asarray(Y.indptr, dtype=np.int64)
        arrays["Y_shape"] = np.asarray(Y.shape, dtype=np.int64)
    else:
        arrays["Y"] = np.asarray(Y, dtype=float)

    save_npz(path, **arrays)


def save_mlp_multilabel_logistic_dataset(
    path: str | Path,
    data: Mapping[str, Any],
    *,
    reg_lambda: float = 0.0,
    generation_config: Mapping[str, Any] | None = None,
    regularize_bias: bool = True,
    hidden_width: int,
    activation: str,
    init_scale: float,
    loss_average: str,
) -> None:
    A = data["A"]
    Y = data["Y"]
    arrays: dict[str, Any] = {
        "num_labels": np.array(int(data["num_labels"]), dtype=np.int64),
        "hidden_width": np.array(int(hidden_width), dtype=np.int64),
        "reg_lambda": np.array(float(reg_lambda), dtype=float),
        "regularize_bias": np.array(bool(regularize_bias)),
        "activation": np.asarray(str(activation)),
        "init_scale": np.array(float(init_scale), dtype=float),
        "loss_average": np.asarray(str(loss_average)),
        "problem_type": np.asarray("mlp_multilabel_logistic"),
        "generation_config_json": np.asarray(
            json.dumps(_json_ready(generation_config or data.get("generation_config", {})), sort_keys=True)
        ),
    }
    if _is_csr_matrix(A):
        arrays["A_data"] = np.asarray(A.data, dtype=float)
        arrays["A_indices"] = np.asarray(A.indices, dtype=np.int64)
        arrays["A_indptr"] = np.asarray(A.indptr, dtype=np.int64)
        arrays["A_shape"] = np.asarray(A.shape, dtype=np.int64)
    else:
        arrays["A"] = np.asarray(A, dtype=float)

    if _is_csr_matrix(Y):
        arrays["Y_data"] = np.asarray(Y.data, dtype=float)
        arrays["Y_indices"] = np.asarray(Y.indices, dtype=np.int64)
        arrays["Y_indptr"] = np.asarray(Y.indptr, dtype=np.int64)
        arrays["Y_shape"] = np.asarray(Y.shape, dtype=np.int64)
    else:
        arrays["Y"] = np.asarray(Y, dtype=float)

    save_npz(path, **arrays)


@dataclass(init=False)
class SoftmaxRegressionProblem:
    A: MatrixLike
    y: np.ndarray
    num_classes: int
    reg_lambda: float
    regularize_bias: bool
    _cached_x_ref: np.ndarray | None
    _cached_x_value: np.ndarray | None
    _cached_logits: np.ndarray | None
    _cached_probs: np.ndarray | None

    def __init__(
        self,
        A: MatrixLike,
        y: np.ndarray,
        num_classes: int,
        reg_lambda: float = 0.0,
        *,
        regularize_bias: bool = True,
    ) -> None:
        self.A = A.astype(float, copy=False) if _is_csr_matrix(A) else np.asarray(A, dtype=float)
        self.y = _as_int_vector(y)
        self.num_classes = int(num_classes)
        self.reg_lambda = float(reg_lambda)
        self.regularize_bias = bool(regularize_bias)
        self._cached_x_ref = None
        self._cached_x_value = None
        self._cached_logits = None
        self._cached_probs = None

        if self.A.ndim != 2:
            raise ValueError("A must be a 2D matrix")
        if self.A.shape[0] != self.y.shape[0]:
            raise ValueError("A and y must have the same number of rows")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if np.min(self.y) < 0 or np.max(self.y) >= self.num_classes:
            raise ValueError("y must contain labels in {0, ..., K-1}")

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def p(self) -> int:
        return int(self.A.shape[1])

    @property
    def dim(self) -> int:
        return int(self.p * self.num_classes)

    @property
    def n(self) -> int:
        return self.dim

    def _matmat(self, W: np.ndarray) -> np.ndarray:
        return np.asarray(self.A @ W, dtype=float)

    def _rmatmat(self, M: np.ndarray) -> np.ndarray:
        return np.asarray(self.A.T @ M, dtype=float)

    def _reshape_weights(self, x: np.ndarray) -> np.ndarray:
        x_vec = _as_float_vector(x)
        if x_vec.size != self.dim:
            raise ValueError(f"x must have shape ({self.dim},)")
        return x_vec.reshape(self.p, self.num_classes)

    def _point_cache(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_vec = _as_float_vector(x)
        if self._cached_x_ref is x and self._cached_logits is not None:
            return self._cached_logits, self._cached_probs
        if (
            self._cached_x_value is not None
            and self._cached_x_value.shape == x_vec.shape
            and np.array_equal(self._cached_x_value, x_vec)
            and self._cached_logits is not None
        ):
            self._cached_x_ref = x if isinstance(x, np.ndarray) else None
            return self._cached_logits, self._cached_probs

        W = x_vec.reshape(self.p, self.num_classes)
        logits = self._matmat(W)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        self._cached_x_ref = x if isinstance(x, np.ndarray) else None
        self._cached_x_value = x_vec.copy()
        self._cached_logits = logits
        self._cached_probs = probs
        return logits, probs

    def f(self, x: np.ndarray) -> float:
        logits, _ = self._point_cache(x)
        W = self._reshape_weights(x)
        losses = logsumexp(logits, axis=1) - logits[np.arange(self.m), self.y]
        reg = _regularized_frobenius_term(W, self.reg_lambda, self.regularize_bias)
        return float(np.mean(losses) + reg)

    def grad(self, x: np.ndarray) -> np.ndarray:
        _, probs = self._point_cache(x)
        W = self._reshape_weights(x)
        residual = probs.copy()
        residual[np.arange(self.m), self.y] -= 1.0
        grad_matrix = self._rmatmat(residual) / self.m
        if self.reg_lambda != 0.0:
            grad_matrix += self.reg_lambda * _regularization_mask(
                self.p, self.num_classes, self.regularize_bias
            ) * W
        return np.asarray(grad_matrix, dtype=float).reshape(-1)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        _, probs = self._point_cache(x)
        V = _as_float_vector(v).reshape(self.p, self.num_classes)
        S = self._matmat(V)
        dot = np.sum(probs * S, axis=1, keepdims=True)
        R = probs * (S - dot)
        hv_matrix = self._rmatmat(R) / self.m
        if self.reg_lambda != 0.0:
            hv_matrix += self.reg_lambda * _regularization_mask(
                self.p, self.num_classes, self.regularize_bias
            ) * V
        return np.asarray(hv_matrix, dtype=float).reshape(-1)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        reg_lambda: float | None = None,
        regularize_bias: bool | None = None,
    ) -> "SoftmaxRegressionProblem":
        arrays = load_npz(path)
        A = _load_npz_matrix(arrays)
        if "y" not in arrays:
            raise KeyError("Dataset .npz must contain label key 'y'")
        if "num_classes" not in arrays:
            raise KeyError("Dataset .npz must contain 'num_classes'")
        loaded_reg_lambda = float(arrays["reg_lambda"]) if "reg_lambda" in arrays else 0.0
        loaded_regularize_bias = _read_scalar_bool(arrays, "regularize_bias", True)
        return cls(
            A=A,
            y=np.asarray(arrays["y"]).reshape(-1),
            num_classes=int(np.asarray(arrays["num_classes"]).item()),
            reg_lambda=loaded_reg_lambda if reg_lambda is None else float(reg_lambda),
            regularize_bias=loaded_regularize_bias if regularize_bias is None else bool(regularize_bias),
        )


@dataclass(init=False)
class MultiLabelLogisticProblem:
    A: MatrixLike
    Y: MatrixLike
    num_labels: int
    reg_lambda: float
    regularize_bias: bool
    _Y_dense: np.ndarray
    _cached_x_ref: np.ndarray | None
    _cached_x_value: np.ndarray | None
    _cached_logits: np.ndarray | None
    _cached_probs: np.ndarray | None

    def __init__(
        self,
        A: MatrixLike,
        Y: MatrixLike,
        num_labels: int,
        reg_lambda: float = 0.0,
        *,
        regularize_bias: bool = True,
    ) -> None:
        self.A = A.astype(float, copy=False) if _is_csr_matrix(A) else np.asarray(A, dtype=float)
        self.Y = Y.astype(float, copy=False) if _is_csr_matrix(Y) else np.asarray(Y, dtype=float)
        self.num_labels = int(num_labels)
        self.reg_lambda = float(reg_lambda)
        self.regularize_bias = bool(regularize_bias)
        self._Y_dense = _dense_target_matrix(self.Y)
        if _is_csr_matrix(Y):
            approx_mb = float(self._Y_dense.nbytes) / (1024.0 * 1024.0)
            print(
                f"[MultiLabelLogisticProblem] cached dense Y with shape={self._Y_dense.shape}, "
                f"approx_mb={approx_mb:.2f}"
            )
        self._cached_x_ref = None
        self._cached_x_value = None
        self._cached_logits = None
        self._cached_probs = None

        if self.A.ndim != 2:
            raise ValueError("A must be a 2D matrix")
        if self._Y_dense.ndim != 2:
            raise ValueError("Y must be a 2D matrix")
        if self.A.shape[0] != self._Y_dense.shape[0]:
            raise ValueError("A and Y must have the same number of rows")
        if self._Y_dense.shape[1] != self.num_labels:
            raise ValueError("Y must have num_labels columns")

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def p(self) -> int:
        return int(self.A.shape[1])

    @property
    def dim(self) -> int:
        return int(self.p * self.num_labels)

    @property
    def n(self) -> int:
        return self.dim

    def _reshape_weights(self, x: np.ndarray) -> np.ndarray:
        x_vec = _as_float_vector(x)
        if x_vec.size != self.dim:
            raise ValueError(f"x must have shape ({self.dim},)")
        return x_vec.reshape(self.p, self.num_labels)

    def _matmat(self, W: np.ndarray) -> np.ndarray:
        return np.asarray(self.A @ W, dtype=float)

    def _rmatmat(self, M: np.ndarray) -> np.ndarray:
        return np.asarray(self.A.T @ M, dtype=float)

    def _point_cache(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_vec = _as_float_vector(x)
        if self._cached_x_ref is x and self._cached_logits is not None:
            return self._cached_logits, self._cached_probs
        if (
            self._cached_x_value is not None
            and self._cached_x_value.shape == x_vec.shape
            and np.array_equal(self._cached_x_value, x_vec)
            and self._cached_logits is not None
        ):
            self._cached_x_ref = x if isinstance(x, np.ndarray) else None
            return self._cached_logits, self._cached_probs

        W = x_vec.reshape(self.p, self.num_labels)
        logits = self._matmat(W)
        probs = expit(logits)
        self._cached_x_ref = x if isinstance(x, np.ndarray) else None
        self._cached_x_value = x_vec.copy()
        self._cached_logits = logits
        self._cached_probs = probs
        return logits, probs

    def f(self, x: np.ndarray) -> float:
        logits, _ = self._point_cache(x)
        W = self._reshape_weights(x)
        losses = np.logaddexp(0.0, logits) - self._Y_dense * logits
        reg = _regularized_frobenius_term(W, self.reg_lambda, self.regularize_bias)
        return float(np.mean(losses) + reg)

    def grad(self, x: np.ndarray) -> np.ndarray:
        _, probs = self._point_cache(x)
        W = self._reshape_weights(x)
        residual = probs - self._Y_dense
        grad_matrix = self._rmatmat(residual) / (self.m * self.num_labels)
        if self.reg_lambda != 0.0:
            grad_matrix += self.reg_lambda * _regularization_mask(
                self.p, self.num_labels, self.regularize_bias
            ) * W
        return np.asarray(grad_matrix, dtype=float).reshape(-1)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        _, probs = self._point_cache(x)
        V = _as_float_vector(v).reshape(self.p, self.num_labels)
        S = self._matmat(V)
        D = probs * (1.0 - probs)
        hv_matrix = self._rmatmat(D * S) / (self.m * self.num_labels)
        if self.reg_lambda != 0.0:
            hv_matrix += self.reg_lambda * _regularization_mask(
                self.p, self.num_labels, self.regularize_bias
            ) * V
        return np.asarray(hv_matrix, dtype=float).reshape(-1)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        reg_lambda: float | None = None,
        regularize_bias: bool | None = None,
    ) -> "MultiLabelLogisticProblem":
        arrays = load_npz(path)
        A = _load_npz_matrix(arrays)
        Y = _load_npz_target_matrix(arrays, "Y") if ("Y" in arrays or "Y_data" in arrays) else _load_npz_target_matrix(arrays, "y_multilabel")
        if "num_labels" not in arrays:
            raise KeyError("Dataset .npz must contain 'num_labels'")
        loaded_reg_lambda = float(arrays["reg_lambda"]) if "reg_lambda" in arrays else 0.0
        loaded_regularize_bias = _read_scalar_bool(arrays, "regularize_bias", True)
        return cls(
            A=A,
            Y=Y,
            num_labels=int(np.asarray(arrays["num_labels"]).item()),
            reg_lambda=loaded_reg_lambda if reg_lambda is None else float(reg_lambda),
            regularize_bias=loaded_regularize_bias if regularize_bias is None else bool(regularize_bias),
        )


@dataclass(init=False)
class MLPMultiLabelLogisticProblem:
    A: MatrixLike
    Y: MatrixLike
    num_labels: int
    hidden_width: int
    activation: str
    reg_lambda: float
    init_scale: float
    regularize_bias: bool
    loss_average: str
    _Y_dense: np.ndarray
    _cached_x_ref: np.ndarray | None
    _cached_x_value: np.ndarray | None
    _cached_hidden_linear: np.ndarray | None
    _cached_hidden: np.ndarray | None
    _cached_logits: np.ndarray | None
    _cached_probs: np.ndarray | None

    def __init__(
        self,
        A: MatrixLike,
        Y: MatrixLike,
        num_labels: int,
        hidden_width: int,
        *,
        activation: str = "tanh",
        reg_lambda: float = 0.0,
        init_scale: float = 1.0e-2,
        regularize_bias: bool = True,
        loss_average: str = "sample_label",
    ) -> None:
        self.A = A.astype(float, copy=False) if _is_csr_matrix(A) else np.asarray(A, dtype=float)
        self.Y = Y.astype(float, copy=False) if _is_csr_matrix(Y) else np.asarray(Y, dtype=float)
        self.num_labels = int(num_labels)
        self.hidden_width = int(hidden_width)
        self.activation = str(activation).lower()
        self.reg_lambda = float(reg_lambda)
        self.init_scale = float(init_scale)
        self.regularize_bias = bool(regularize_bias)
        self.loss_average = str(loss_average).lower()
        self._Y_dense = _dense_target_matrix(self.Y)
        if _is_csr_matrix(Y):
            approx_mb = float(self._Y_dense.nbytes) / (1024.0 * 1024.0)
            print(
                f"[MLPMultiLabelLogisticProblem] cached dense Y with shape={self._Y_dense.shape}, "
                f"approx_mb={approx_mb:.2f}"
            )
        self._cached_x_ref = None
        self._cached_x_value = None
        self._cached_hidden_linear = None
        self._cached_hidden = None
        self._cached_logits = None
        self._cached_probs = None

        if self.A.ndim != 2:
            raise ValueError("A must be a 2D matrix")
        if self._Y_dense.ndim != 2:
            raise ValueError("Y must be a 2D matrix")
        if self.A.shape[0] != self._Y_dense.shape[0]:
            raise ValueError("A and Y must have the same number of rows")
        if self._Y_dense.shape[1] != self.num_labels:
            raise ValueError("Y must have num_labels columns")
        if self.hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if self.activation != "tanh":
            raise ValueError("activation must be 'tanh'")
        if self.loss_average != "sample_label":
            raise ValueError("loss_average must be 'sample_label'")

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def p(self) -> int:
        return int(self.A.shape[1])

    @property
    def dim(self) -> int:
        return int(self.p * self.hidden_width + self.hidden_width * self.num_labels + self.num_labels)

    @property
    def n(self) -> int:
        return self.dim

    @property
    def scale(self) -> float:
        return 1.0 / float(self.m * self.num_labels)

    def _unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_vec = _as_float_vector(x)
        if x_vec.size != self.dim:
            raise ValueError(f"x must have shape ({self.dim},)")
        w1_size = self.p * self.hidden_width
        b_size = self.hidden_width * self.num_labels
        W1 = x_vec[:w1_size].reshape(self.p, self.hidden_width)
        B = x_vec[w1_size : w1_size + b_size].reshape(self.hidden_width, self.num_labels)
        c = x_vec[w1_size + b_size :]
        return W1, B, c

    def _pack(self, W1: np.ndarray, B: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(W1, dtype=float).reshape(-1),
                np.asarray(B, dtype=float).reshape(-1),
                np.asarray(c, dtype=float).reshape(-1),
            ]
        )

    def _regularization_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        W1_mask = np.ones((self.p, self.hidden_width), dtype=float)
        if not self.regularize_bias:
            W1_mask[-1, :] = 0.0
            c_mask = np.zeros(self.num_labels, dtype=float)
        else:
            c_mask = np.ones(self.num_labels, dtype=float)
        B_mask = np.ones((self.hidden_width, self.num_labels), dtype=float)
        return W1_mask, B_mask, c_mask

    def _point_cache(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_vec = _as_float_vector(x)
        if self._cached_x_ref is x and self._cached_hidden_linear is not None:
            return self._cached_hidden_linear, self._cached_hidden, self._cached_logits, self._cached_probs
        if (
            self._cached_x_value is not None
            and self._cached_x_value.shape == x_vec.shape
            and np.array_equal(self._cached_x_value, x_vec)
            and self._cached_hidden_linear is not None
        ):
            self._cached_x_ref = x if isinstance(x, np.ndarray) else None
            return self._cached_hidden_linear, self._cached_hidden, self._cached_logits, self._cached_probs

        W1, B, c = self._unpack(x_vec)
        hidden_linear = np.asarray(self.A @ W1, dtype=float)
        hidden = np.tanh(hidden_linear)
        logits = hidden @ B + c.reshape(1, self.num_labels)
        probs = expit(logits)

        self._cached_x_ref = x if isinstance(x, np.ndarray) else None
        self._cached_x_value = x_vec.copy()
        self._cached_hidden_linear = hidden_linear
        self._cached_hidden = hidden
        self._cached_logits = logits
        self._cached_probs = probs
        return hidden_linear, hidden, logits, probs

    def f(self, x: np.ndarray) -> float:
        _, _, logits, _ = self._point_cache(x)
        W1, B, c = self._unpack(x)
        losses = np.logaddexp(0.0, logits) - self._Y_dense * logits
        W1_mask, B_mask, c_mask = self._regularization_masks()
        reg = 0.0
        if self.reg_lambda != 0.0:
            reg = 0.5 * self.reg_lambda * float(
                np.sum(W1_mask * W1 * W1)
                + np.sum(B_mask * B * B)
                + np.sum(c_mask * c * c)
            )
        return float(np.mean(losses) + reg)

    def grad(self, x: np.ndarray) -> np.ndarray:
        _, hidden, _, probs = self._point_cache(x)
        W1, B, c = self._unpack(x)
        G = (probs - self._Y_dense) * self.scale
        grad_B = hidden.T @ G
        grad_c = np.sum(G, axis=0)
        hidden_residual = (G @ B.T) * (1.0 - hidden * hidden)
        grad_W1 = np.asarray(self.A.T @ hidden_residual, dtype=float)
        if self.reg_lambda != 0.0:
            W1_mask, B_mask, c_mask = self._regularization_masks()
            grad_W1 += self.reg_lambda * W1_mask * W1
            grad_B += self.reg_lambda * B_mask * B
            grad_c += self.reg_lambda * c_mask * c
        return self._pack(grad_W1, grad_B, grad_c)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        _, hidden, _, probs = self._point_cache(x)
        _, B, _ = self._unpack(x)
        V1, VB, vc = self._unpack(v)

        base_G = (probs - self._Y_dense) * self.scale
        activation_prime = 1.0 - hidden * hidden
        d_hidden_linear = np.asarray(self.A @ V1, dtype=float)
        d_hidden = activation_prime * d_hidden_linear
        d_logits = d_hidden @ B + hidden @ VB + vc.reshape(1, self.num_labels)
        d_probs = (probs * (1.0 - probs) * d_logits) * self.scale

        hv_B = d_hidden.T @ base_G + hidden.T @ d_probs
        hv_c = np.sum(d_probs, axis=0)

        base_hidden_term = base_G @ B.T
        d_hidden_term = d_probs @ B.T + base_G @ VB.T
        second_factor = -2.0 * hidden * d_hidden
        d_hidden_residual = d_hidden_term * activation_prime + base_hidden_term * second_factor
        hv_W1 = np.asarray(self.A.T @ d_hidden_residual, dtype=float)

        if self.reg_lambda != 0.0:
            W1_mask, B_mask, c_mask = self._regularization_masks()
            hv_W1 += self.reg_lambda * W1_mask * V1
            hv_B += self.reg_lambda * B_mask * VB
            hv_c += self.reg_lambda * c_mask * vc

        return self._pack(hv_W1, hv_B, hv_c)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        reg_lambda: float | None = None,
        regularize_bias: bool | None = None,
        hidden_width: int | None = None,
        activation: str | None = None,
        init_scale: float | None = None,
        loss_average: str | None = None,
    ) -> "MLPMultiLabelLogisticProblem":
        arrays = load_npz(path)
        A = _load_npz_matrix(arrays)
        Y = _load_npz_target_matrix(arrays, "Y")
        if "num_labels" not in arrays:
            raise KeyError("Dataset .npz must contain 'num_labels'")
        if "hidden_width" not in arrays:
            raise KeyError("Dataset .npz must contain 'hidden_width'")
        loaded_reg_lambda = float(arrays["reg_lambda"]) if "reg_lambda" in arrays else 0.0
        loaded_regularize_bias = _read_scalar_bool(arrays, "regularize_bias", True)
        loaded_activation = _read_scalar_string(arrays, "activation", "tanh")
        loaded_loss_average = _read_scalar_string(arrays, "loss_average", "sample_label")
        loaded_init_scale = float(np.asarray(arrays.get("init_scale", np.array(1.0e-2))).item())
        return cls(
            A=A,
            Y=Y,
            num_labels=int(np.asarray(arrays["num_labels"]).item()),
            hidden_width=int(np.asarray(arrays["hidden_width"]).item())
            if hidden_width is None
            else int(hidden_width),
            activation=loaded_activation if activation is None else str(activation),
            reg_lambda=loaded_reg_lambda if reg_lambda is None else float(reg_lambda),
            init_scale=loaded_init_scale if init_scale is None else float(init_scale),
            regularize_bias=loaded_regularize_bias if regularize_bias is None else bool(regularize_bias),
            loss_average=loaded_loss_average if loss_average is None else str(loss_average),
        )


def infer_problem_type_from_npz(path: str | Path) -> str:
    arrays = load_npz(resolve_project_path(path))
    return _read_scalar_string(arrays, "problem_type", "logistic")


def clone_problem_data_with_reg_lambda(
    *,
    source_path: str | Path,
    save_path: str | Path,
    reg_lambda: float,
) -> None:
    resolved_source = resolve_project_path(source_path)
    arrays = load_npz(resolved_source)
    problem_type = infer_problem_type_from_npz(resolved_source)
    metadata = _load_generation_config_from_npz(arrays)
    metadata["reg_lambda"] = float(reg_lambda)
    metadata["cloned_from"] = str(source_path)
    metadata["clone_reg_lambda_only"] = True

    cloned = dict(arrays)
    cloned["problem_type"] = np.asarray(problem_type)
    cloned["reg_lambda"] = np.asarray(float(reg_lambda), dtype=float)
    cloned["generation_config_json"] = np.asarray(
        json.dumps(_json_ready(metadata), sort_keys=True)
    )
    save_npz(resolve_project_path(save_path), **cloned)


def load_problem_from_npz(
    path: str | Path,
    *,
    problem_type: str | None = None,
    reg_lambda: float | None = None,
    regularize_bias: bool | None = None,
) -> Any:
    resolved_problem_type = problem_type or infer_problem_type_from_npz(path)
    if resolved_problem_type == "logistic":
        from src.problems.logistic import LogisticRegressionProblem

        return LogisticRegressionProblem.from_npz(
            path,
            reg_lambda=reg_lambda,
            regularize_bias=regularize_bias,
        )
    if resolved_problem_type == "softmax":
        return SoftmaxRegressionProblem.from_npz(
            path,
            reg_lambda=reg_lambda,
            regularize_bias=regularize_bias,
        )
    if resolved_problem_type == "multilabel_logistic":
        return MultiLabelLogisticProblem.from_npz(
            path,
            reg_lambda=reg_lambda,
            regularize_bias=regularize_bias,
        )
    if resolved_problem_type == "mlp_multilabel_logistic":
        return MLPMultiLabelLogisticProblem.from_npz(
            path,
            reg_lambda=reg_lambda,
            regularize_bias=regularize_bias,
        )
    raise ValueError(f"Unsupported problem_type {resolved_problem_type!r}")


def build_softmax_problem(problem_config: Mapping[str, Any]) -> SoftmaxRegressionProblem:
    if "source" not in problem_config:
        raise ValueError("softmax problems currently require problem.source")
    return SoftmaxRegressionProblem.from_npz(
        resolve_project_path(problem_config["source"]),
        reg_lambda=problem_config.get("reg_lambda"),
        regularize_bias=problem_config.get("regularize_bias"),
    )


def build_multilabel_logistic_problem(
    problem_config: Mapping[str, Any]
) -> MultiLabelLogisticProblem:
    if "source" not in problem_config:
        raise ValueError("multilabel_logistic problems currently require problem.source")
    return MultiLabelLogisticProblem.from_npz(
        resolve_project_path(problem_config["source"]),
        reg_lambda=problem_config.get("reg_lambda"),
        regularize_bias=problem_config.get("regularize_bias"),
    )


def build_mlp_multilabel_logistic_problem(
    problem_config: Mapping[str, Any]
) -> MLPMultiLabelLogisticProblem:
    if "source" not in problem_config:
        raise ValueError("mlp_multilabel_logistic problems currently require problem.source")
    return MLPMultiLabelLogisticProblem.from_npz(
        resolve_project_path(problem_config["source"]),
        reg_lambda=problem_config.get("reg_lambda"),
        regularize_bias=problem_config.get("regularize_bias"),
        hidden_width=problem_config.get("hidden_width"),
        activation=problem_config.get("activation"),
        init_scale=problem_config.get("init_scale"),
        loss_average=problem_config.get("loss_average"),
    )


def generate_softmax_from_config(problem_config: Mapping[str, Any], save_path: str, seed: int) -> None:
    del seed
    source_format = str(problem_config.get("source_format", "libsvm")).lower()
    if source_format != "libsvm":
        raise ValueError("softmax generate_data currently supports source_format='libsvm' only")
    if "raw_source" not in problem_config:
        raise ValueError("problem.raw_source is required for softmax generate_data")

    raw_source_path = resolve_project_path(problem_config["raw_source"])
    download_url = problem_config.get("download_url")
    download_if_missing = bool(problem_config.get("download_if_missing", bool(download_url)))
    if not raw_source_path.exists() and download_if_missing:
        if not download_url:
            raise ValueError("problem.download_url is required when download_if_missing is true")
        _redownload_raw_dataset(str(download_url), raw_source_path)

    n_features = problem_config.get("n_features", problem_config.get("d"))
    num_classes = int(problem_config["num_classes"])
    sample_size = problem_config.get("sample_size")
    sample_seed = int(problem_config.get("sample_seed", 0))
    add_bias = bool(problem_config.get("add_bias", True))
    reg_lambda = float(problem_config.get("reg_lambda", 1.0e-3))
    regularize_bias = bool(problem_config.get("regularize_bias", True))

    try:
        data = load_libsvm_classification_dataset(
            problem_config["raw_source"],
            n_features=None if n_features is None else int(n_features),
            index_base=problem_config.get("index_base", 1),
            sample_size=sample_size,
            sample_seed=sample_seed,
            max_rows=problem_config.get("max_rows"),
            label_mode="multiclass",
            num_classes=num_classes,
        )
    except (EOFError, OSError) as exc:
        download_if_corrupt = bool(problem_config.get("download_if_corrupt", bool(download_url)))
        if not (download_url and download_if_corrupt and _is_compressed_path(raw_source_path)):
            raise
        print(
            f"Raw compressed dataset appears incomplete or corrupted: {raw_source_path}. Re-downloading once.",
            flush=True,
        )
        _redownload_raw_dataset(str(download_url), raw_source_path)
        data = load_libsvm_classification_dataset(
            problem_config["raw_source"],
            n_features=None if n_features is None else int(n_features),
            index_base=problem_config.get("index_base", 1),
            sample_size=sample_size,
            sample_seed=sample_seed,
            max_rows=problem_config.get("max_rows"),
            label_mode="multiclass",
            num_classes=num_classes,
        )

    A = _maybe_add_bias_column(data["A"], add_bias)
    metadata = dict(data["generation_config"])
    metadata.update(
        _dataset_metadata(
            dataset_name=str(problem_config.get("dataset_name", Path(str(problem_config["raw_source"])).stem)),
            source_format=source_format,
            raw_source=str(problem_config["raw_source"]),
            download_url=None if download_url is None else str(download_url),
            original_num_samples=int(data["generation_config"]["original_num_samples"]),
            sample_size=None if sample_size is None else int(sample_size),
            sample_seed=sample_seed,
            n_features_without_bias=int(n_features if n_features is not None else data["A"].shape[1]),
            n_features_with_bias=int(A.shape[1]),
            num_classes=num_classes,
            reg_lambda=reg_lambda,
            add_bias=add_bias,
            regularize_bias=regularize_bias,
        )
    )
    save_softmax_dataset(
        resolve_project_path(save_path),
        {"A": A, "y": data["y"], "num_classes": num_classes, "generation_config": metadata},
        reg_lambda=reg_lambda,
        generation_config=metadata,
        regularize_bias=regularize_bias,
    )


def generate_multilabel_logistic_from_config(
    problem_config: Mapping[str, Any],
    save_path: str,
    seed: int,
) -> None:
    del seed
    if "raw_source" not in problem_config:
        raise ValueError("problem.raw_source is required for multilabel_logistic generate_data")

    source_format = str(problem_config.get("source_format", "multilabel_libsvm")).lower()
    raw_source_path = resolve_project_path(problem_config["raw_source"])
    download_url = problem_config.get("download_url")
    download_if_missing = bool(problem_config.get("download_if_missing", bool(download_url)))
    if not raw_source_path.exists() and download_if_missing:
        if not download_url:
            raise ValueError("problem.download_url is required when download_if_missing is true")
        _redownload_raw_dataset(str(download_url), raw_source_path)

    n_features = problem_config.get("n_features", problem_config.get("d"))
    num_labels = int(problem_config["num_labels"])
    sample_size = problem_config.get("sample_size")
    sample_seed = int(problem_config.get("sample_seed", 0))
    add_bias = bool(problem_config.get("add_bias", True))
    reg_lambda = float(problem_config.get("reg_lambda", 1.0e-3))
    regularize_bias = bool(problem_config.get("regularize_bias", True))

    try:
        data = load_multilabel_dataset(
            problem_config["raw_source"],
            source_format=source_format,
            n_features=None if n_features is None else int(n_features),
            num_labels=num_labels,
            index_base=problem_config.get("index_base", 1),
            label_index_base=problem_config.get("label_index_base", 1),
            sample_size=sample_size,
            sample_seed=sample_seed,
            max_rows=problem_config.get("max_rows"),
        )
    except (EOFError, OSError) as exc:
        download_if_corrupt = bool(problem_config.get("download_if_corrupt", bool(download_url)))
        if not (download_url and download_if_corrupt and _is_compressed_path(raw_source_path)):
            raise
        print(
            f"Raw compressed dataset appears incomplete or corrupted: {raw_source_path}. Re-downloading once.",
            flush=True,
        )
        _redownload_raw_dataset(str(download_url), raw_source_path)
        data = load_multilabel_dataset(
            problem_config["raw_source"],
            source_format=source_format,
            n_features=None if n_features is None else int(n_features),
            num_labels=num_labels,
            index_base=problem_config.get("index_base", 1),
            label_index_base=problem_config.get("label_index_base", 1),
            sample_size=sample_size,
            sample_seed=sample_seed,
            max_rows=problem_config.get("max_rows"),
        )

    A = _maybe_add_bias_column(data["A"], add_bias)
    metadata = dict(data["generation_config"])
    metadata.update(
        _dataset_metadata(
            dataset_name=str(problem_config.get("dataset_name", Path(str(problem_config["raw_source"])).stem)),
            source_format=source_format,
            raw_source=str(problem_config["raw_source"]),
            download_url=None if download_url is None else str(download_url),
            original_num_samples=int(data["generation_config"]["original_num_samples"]),
            sample_size=None if sample_size is None else int(sample_size),
            sample_seed=sample_seed,
            n_features_without_bias=int(n_features if n_features is not None else data["A"].shape[1]),
            n_features_with_bias=int(A.shape[1]),
            num_labels=num_labels,
            reg_lambda=reg_lambda,
            add_bias=add_bias,
            regularize_bias=regularize_bias,
        )
    )
    save_multilabel_logistic_dataset(
        resolve_project_path(save_path),
        {"A": A, "Y": data["Y"], "num_labels": num_labels, "generation_config": metadata},
        reg_lambda=reg_lambda,
        generation_config=metadata,
        regularize_bias=regularize_bias,
    )


def generate_mlp_multilabel_logistic_from_config(
    problem_config: Mapping[str, Any],
    save_path: str,
    seed: int,
) -> None:
    del seed
    if "raw_source" not in problem_config:
        raise ValueError("problem.raw_source is required for mlp_multilabel_logistic generate_data")

    source_format = str(problem_config.get("source_format", "multilabel_libsvm")).lower()
    raw_source_path = resolve_project_path(problem_config["raw_source"])
    download_url = problem_config.get("download_url")
    download_if_missing = bool(problem_config.get("download_if_missing", bool(download_url)))
    if not raw_source_path.exists() and download_if_missing:
        if not download_url:
            raise ValueError("problem.download_url is required when download_if_missing is true")
        _redownload_raw_dataset(str(download_url), raw_source_path)

    n_features = problem_config.get("n_features", problem_config.get("d"))
    num_labels = int(problem_config["num_labels"])
    sample_size = problem_config.get("sample_size")
    sample_seed = int(problem_config.get("sample_seed", 0))
    add_bias = bool(problem_config.get("add_bias", True))
    hidden_width = int(problem_config.get("hidden_width", 1))
    activation = str(problem_config.get("activation", "tanh")).lower()
    reg_lambda = float(problem_config.get("reg_lambda", 1.0e-3))
    init_scale = float(problem_config.get("init_scale", 1.0e-2))
    regularize_bias = bool(problem_config.get("regularize_bias", True))
    loss_average = str(problem_config.get("loss_average", "sample_label")).lower()

    try:
        data = load_multilabel_dataset(
            problem_config["raw_source"],
            source_format=source_format,
            n_features=None if n_features is None else int(n_features),
            num_labels=num_labels,
            index_base=problem_config.get("index_base", 1),
            label_index_base=problem_config.get("label_index_base", 1),
            sample_size=sample_size,
            sample_seed=sample_seed,
            max_rows=problem_config.get("max_rows"),
        )
    except (EOFError, OSError):
        download_if_corrupt = bool(problem_config.get("download_if_corrupt", bool(download_url)))
        if not (download_url and download_if_corrupt and _is_compressed_path(raw_source_path)):
            raise
        print(
            f"Raw compressed dataset appears incomplete or corrupted: {raw_source_path}. Re-downloading once.",
            flush=True,
        )
        _redownload_raw_dataset(str(download_url), raw_source_path)
        data = load_multilabel_dataset(
            problem_config["raw_source"],
            source_format=source_format,
            n_features=None if n_features is None else int(n_features),
            num_labels=num_labels,
            index_base=problem_config.get("index_base", 1),
            label_index_base=problem_config.get("label_index_base", 1),
            sample_size=sample_size,
            sample_seed=sample_seed,
            max_rows=problem_config.get("max_rows"),
        )

    A = _maybe_add_bias_column(data["A"], add_bias)
    metadata = dict(data["generation_config"])
    metadata.update(
        _dataset_metadata(
            dataset_name=str(problem_config.get("dataset_name", Path(str(problem_config["raw_source"])).stem)),
            source_format=source_format,
            raw_source=str(problem_config["raw_source"]),
            download_url=None if download_url is None else str(download_url),
            original_num_samples=int(data["generation_config"]["original_num_samples"]),
            sample_size=None if sample_size is None else int(sample_size),
            sample_seed=sample_seed,
            n_features_without_bias=int(n_features if n_features is not None else data["A"].shape[1]),
            n_features_with_bias=int(A.shape[1]),
            num_labels=num_labels,
            reg_lambda=reg_lambda,
            add_bias=add_bias,
            regularize_bias=regularize_bias,
        )
    )
    metadata["hidden_width"] = hidden_width
    metadata["activation"] = activation
    metadata["init_scale"] = init_scale
    metadata["loss_average"] = loss_average
    metadata["dim"] = int(A.shape[1] * hidden_width + hidden_width * num_labels + num_labels)

    problem = MLPMultiLabelLogisticProblem(
        A=A,
        Y=data["Y"],
        num_labels=num_labels,
        hidden_width=hidden_width,
        activation=activation,
        reg_lambda=reg_lambda,
        init_scale=init_scale,
        regularize_bias=regularize_bias,
        loss_average=loss_average,
    )
    if int(problem.dim) == 44200:
        raise AssertionError("BUG: unfair-tos shared-MLP benchmark produced dim=44200 instead of 6307")
    expected_dim = int(A.shape[1] * hidden_width + hidden_width * num_labels + num_labels)
    if int(problem.dim) != expected_dim:
        raise AssertionError(f"MLP dimension mismatch: got {problem.dim}, expected {expected_dim}")

    save_mlp_multilabel_logistic_dataset(
        resolve_project_path(save_path),
        {
            "A": A,
            "Y": data["Y"],
            "num_labels": num_labels,
            "generation_config": metadata,
        },
        reg_lambda=reg_lambda,
        generation_config=metadata,
        regularize_bias=regularize_bias,
        hidden_width=hidden_width,
        activation=activation,
        init_scale=init_scale,
        loss_average=loss_average,
    )

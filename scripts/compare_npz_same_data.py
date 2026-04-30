from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.problems.real_classification import infer_problem_type_from_npz
from src.utils.io import load_npz
from src.utils.paths import resolve_project_path


def _shape_from_npz(npz_data: dict[str, object], key: str) -> tuple[int, ...]:
    shape_key = f"{key}_shape"
    if shape_key in npz_data:
        return tuple(int(value) for value in np.asarray(npz_data[shape_key]).tolist())
    return tuple(np.asarray(npz_data[key]).shape)


def _compare_matrix_storage(left: dict[str, object], right: dict[str, object], prefix: str) -> list[str]:
    errors: list[str] = []
    dense_key = prefix
    csr_keys = [f"{prefix}_data", f"{prefix}_indices", f"{prefix}_indptr", f"{prefix}_shape"]
    left_is_dense = dense_key in left
    right_is_dense = dense_key in right
    left_is_csr = all(key in left for key in csr_keys)
    right_is_csr = all(key in right for key in csr_keys)

    if (left_is_dense != right_is_dense) or (left_is_csr != right_is_csr):
        return [f"{prefix} storage format differs"]

    if left_is_dense:
        if not np.array_equal(np.asarray(left[dense_key]), np.asarray(right[dense_key])):
            errors.append(f"{prefix} differs")
        return errors

    for key in csr_keys:
        if not np.array_equal(np.asarray(left[key]), np.asarray(right[key])):
            errors.append(f"{key} differs")
    return errors


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("Usage: python scripts/compare_npz_same_data.py <path1.npz> <path2.npz>")

    path1 = resolve_project_path(args[0])
    path2 = resolve_project_path(args[1])
    npz1 = load_npz(path1)
    npz2 = load_npz(path2)

    errors: list[str] = []
    problem_type_1 = infer_problem_type_from_npz(path1)
    problem_type_2 = infer_problem_type_from_npz(path2)
    if problem_type_1 != problem_type_2:
        errors.append(f"problem_type differs: {problem_type_1} vs {problem_type_2}")

    a_shape_1 = _shape_from_npz(npz1, "A")
    a_shape_2 = _shape_from_npz(npz2, "A")
    if a_shape_1 != a_shape_2:
        errors.append(f"A shape differs: {a_shape_1} vs {a_shape_2}")
    errors.extend(_compare_matrix_storage(npz1, npz2, "A"))

    if problem_type_1 in {"logistic", "softmax"} and problem_type_2 in {"logistic", "softmax"}:
        y1 = np.asarray(npz1["y"])
        y2 = np.asarray(npz2["y"])
        if y1.shape != y2.shape:
            errors.append(f"y shape differs: {tuple(y1.shape)} vs {tuple(y2.shape)}")
        elif not np.array_equal(y1, y2):
            errors.append("y differs")
    elif problem_type_1 == "multilabel_logistic" and problem_type_2 == "multilabel_logistic":
        y_shape_1 = _shape_from_npz(npz1, "Y")
        y_shape_2 = _shape_from_npz(npz2, "Y")
        if y_shape_1 != y_shape_2:
            errors.append(f"Y shape differs: {y_shape_1} vs {y_shape_2}")
        errors.extend(_compare_matrix_storage(npz1, npz2, "Y"))

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print(f"problem_type: {problem_type_1}")
    print(f"A shape: {a_shape_1}")
    if problem_type_1 == "multilabel_logistic":
        print(f"Y shape: {_shape_from_npz(npz1, 'Y')}")
    else:
        print(f"y shape: {tuple(np.asarray(npz1['y']).shape)}")
    print("same data: OK")


if __name__ == "__main__":
    main()

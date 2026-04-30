from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.problems.real_classification import infer_problem_type_from_npz
from src.utils.io import load_npz
from src.utils.paths import resolve_project_path


def _load_generation_config(npz_data: dict[str, object]) -> dict[str, object]:
    raw = np.asarray(npz_data.get("generation_config_json", np.asarray("{}")))
    text = str(raw.item()) if raw.shape == () else str(raw.reshape(-1)[0])
    return json.loads(text)


def _shape(npz_data: dict[str, object], key: str) -> tuple[int, ...]:
    shape_key = f"{key}_shape"
    if shape_key in npz_data:
        return tuple(int(value) for value in np.asarray(npz_data[shape_key]).tolist())
    return tuple(np.asarray(npz_data[key]).shape)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("Usage: python scripts/compare_npz_problem_metadata.py <path1.npz> <path2.npz>")

    path1 = resolve_project_path(args[0])
    path2 = resolve_project_path(args[1])
    npz1 = load_npz(path1)
    npz2 = load_npz(path2)
    meta1 = _load_generation_config(npz1)
    meta2 = _load_generation_config(npz2)

    errors: list[str] = []
    problem_type_1 = infer_problem_type_from_npz(path1)
    problem_type_2 = infer_problem_type_from_npz(path2)
    if problem_type_1 != problem_type_2:
        errors.append(f"problem_type differs: {problem_type_1} vs {problem_type_2}")

    a_shape_1 = _shape(npz1, "A")
    a_shape_2 = _shape(npz2, "A")
    if a_shape_1[1] != a_shape_2[1]:
        errors.append(f"A column count differs: {a_shape_1[1]} vs {a_shape_2[1]}")

    n_features_1 = int(meta1.get("n_features_with_bias", a_shape_1[1]))
    n_features_2 = int(meta2.get("n_features_with_bias", a_shape_2[1]))
    if n_features_1 != n_features_2:
        errors.append(f"n_features_with_bias differs: {n_features_1} vs {n_features_2}")

    dim1 = int(meta1.get("dim", a_shape_1[1]))
    dim2 = int(meta2.get("dim", a_shape_2[1]))
    if dim1 != dim2:
        errors.append(f"dim differs: {dim1} vs {dim2}")

    reg1 = float(np.asarray(npz1["reg_lambda"]).item())
    reg2 = float(np.asarray(npz2["reg_lambda"]).item())
    if not np.isclose(reg1, reg2):
        errors.append(f"reg_lambda differs: {reg1} vs {reg2}")

    if problem_type_1 == "softmax" and problem_type_2 == "softmax":
        num_classes_1 = int(np.asarray(npz1["num_classes"]).item())
        num_classes_2 = int(np.asarray(npz2["num_classes"]).item())
        if num_classes_1 != num_classes_2:
            errors.append(f"num_classes differs: {num_classes_1} vs {num_classes_2}")
    elif problem_type_1 == "multilabel_logistic" and problem_type_2 == "multilabel_logistic":
        num_labels_1 = int(np.asarray(npz1["num_labels"]).item())
        num_labels_2 = int(np.asarray(npz2["num_labels"]).item())
        if num_labels_1 != num_labels_2:
            errors.append(f"num_labels differs: {num_labels_1} vs {num_labels_2}")
        y_shape_1 = _shape(npz1, "Y")
        y_shape_2 = _shape(npz2, "Y")
        if y_shape_1[1] != y_shape_2[1]:
            errors.append(f"Y column count differs: {y_shape_1[1]} vs {y_shape_2[1]}")

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    sample_size_1 = int(meta1.get("sample_size", a_shape_1[0]))
    sample_size_2 = int(meta2.get("sample_size", a_shape_2[0]))
    print(f"problem_type: {problem_type_1}")
    print(f"n_features_with_bias: {n_features_1}")
    if problem_type_1 == "softmax":
        print(f"num_classes: {int(np.asarray(npz1['num_classes']).item())}")
    elif problem_type_1 == "multilabel_logistic":
        print(f"num_labels: {int(np.asarray(npz1['num_labels']).item())}")
    print(f"dim: {dim1}")
    print(f"reg_lambda: {reg1}")
    if sample_size_1 != sample_size_2:
        print(f"sample_size differs: {sample_size_1} vs {sample_size_2}")
    else:
        print(f"sample_size: {sample_size_1}")
    print("metadata compatibility: OK")


if __name__ == "__main__":
    main()

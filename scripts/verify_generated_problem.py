from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.problems.real_classification import infer_problem_type_from_npz, load_problem_from_npz
from src.utils.io import load_npz
from src.utils.paths import resolve_project_path


def _load_generation_config(npz_data: dict[str, object]) -> dict[str, object]:
    raw = np.asarray(npz_data.get("generation_config_json", np.asarray("{}")))
    text = str(raw.item()) if raw.shape == () else str(raw.reshape(-1)[0])
    return json.loads(text)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python scripts/verify_generated_problem.py <path-to-npz>")

    dataset_path = resolve_project_path(args[0])
    npz_data = load_npz(dataset_path)
    metadata = _load_generation_config(npz_data)
    problem_type = infer_problem_type_from_npz(dataset_path)
    problem = load_problem_from_npz(dataset_path, problem_type=problem_type)

    x0 = np.zeros(problem.dim, dtype=float)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(problem.dim)
    f0 = float(problem.f(x0))
    grad0 = np.asarray(problem.grad(x0), dtype=float).reshape(-1)
    hv0 = np.asarray(problem.hvp(x0, v), dtype=float).reshape(-1)

    if not np.isfinite(f0):
        raise RuntimeError("f(x0) is not finite")
    if grad0.shape != (problem.dim,):
        raise RuntimeError("grad(x0) has unexpected shape")
    if hv0.shape != (problem.dim,):
        raise RuntimeError("hvp(x0, v) has unexpected shape")
    if not np.all(np.isfinite(grad0)):
        raise RuntimeError("grad(x0) contains NaN/Inf")
    if not np.all(np.isfinite(hv0)):
        raise RuntimeError("hvp(x0, v) contains NaN/Inf")

    A_shape = tuple(int(value) for value in np.asarray(npz_data["A_shape"]).tolist()) if "A_shape" in npz_data else tuple(np.asarray(npz_data["A"]).shape)
    m = int(A_shape[0])
    dim = int(problem.dim)
    expected_dim = metadata.get("dim")
    expected_sample_size = metadata.get("sample_size", m)
    if expected_dim is not None and dim != int(expected_dim):
        raise RuntimeError(f"dim mismatch: got {dim}, expected {expected_dim}")
    if dim == 44200:
        raise RuntimeError("BUG: loaded unfair-tos MLP benchmark has dim=44200, expected shared-MLP dim=6307")
    if expected_sample_size is not None and m != int(expected_sample_size):
        raise RuntimeError(f"sample size mismatch: got {m}, expected {expected_sample_size}")

    expected = metadata.get("expected", {})
    if expected:
        expected_dataset = str(expected.get("dataset", "")).lower()
        expected_model = str(expected.get("model", "")).lower()
        expected_reg_lambda = expected.get("reg_lambda")
        expected_augmented_dim = expected.get("augmented_dim")
        expected_optimization_dim = expected.get("optimization_dim")
        expected_num_classes = expected.get("num_classes")
        expected_num_labels = expected.get("num_labels")
        expected_hidden_width = expected.get("hidden_width")

        if expected_reg_lambda is not None:
            reg_lambda = float(np.asarray(npz_data["reg_lambda"]).item())
            if not np.isclose(reg_lambda, float(expected_reg_lambda)):
                raise RuntimeError(f"reg_lambda mismatch: got {reg_lambda}, expected {expected_reg_lambda}")
        if expected_augmented_dim is not None and int(A_shape[1]) != int(expected_augmented_dim):
            raise RuntimeError(f"A column mismatch: got {A_shape[1]}, expected {expected_augmented_dim}")
        if expected_optimization_dim is not None and dim != int(expected_optimization_dim):
            raise RuntimeError(f"optimization dim mismatch: got {dim}, expected {expected_optimization_dim}")
        if expected_num_classes is not None:
            num_classes = int(np.asarray(npz_data["num_classes"]).item())
            if num_classes != int(expected_num_classes):
                raise RuntimeError(f"num_classes mismatch: got {num_classes}, expected {expected_num_classes}")
            if problem_type != "softmax":
                raise RuntimeError(f"problem_type mismatch for {expected_dataset}: expected softmax, got {problem_type}")
        if expected_num_labels is not None:
            num_labels = int(np.asarray(npz_data["num_labels"]).item())
            if num_labels != int(expected_num_labels):
                raise RuntimeError(f"num_labels mismatch: got {num_labels}, expected {expected_num_labels}")
            if problem_type not in {"multilabel_logistic", "mlp_multilabel_logistic"}:
                raise RuntimeError(
                    f"problem_type mismatch for {expected_dataset}: expected multilabel_logistic or mlp_multilabel_logistic, got {problem_type}"
                )
        if expected_hidden_width is not None:
            hidden_width = int(np.asarray(npz_data["hidden_width"]).item())
            if hidden_width != int(expected_hidden_width):
                raise RuntimeError(
                    f"hidden_width mismatch: got {hidden_width}, expected {expected_hidden_width}"
                )
            if problem_type != "mlp_multilabel_logistic":
                raise RuntimeError(
                    f"problem_type mismatch for {expected_dataset}: expected mlp_multilabel_logistic, got {problem_type}"
                )
        if expected_model == "l2-logistic" and problem_type != "logistic":
            raise RuntimeError(f"problem_type mismatch: expected logistic, got {problem_type}")

    print(f"problem_type: {problem_type}")
    print(f"A shape: {A_shape}")
    if problem_type == "softmax":
        print(f"y shape: {tuple(np.asarray(npz_data['y']).shape)}")
        print(f"num_classes: {int(np.asarray(npz_data['num_classes']).item())}")
    elif problem_type == "multilabel_logistic":
        if "Y_shape" in npz_data:
            y_shape = tuple(int(value) for value in np.asarray(npz_data["Y_shape"]).tolist())
        else:
            y_shape = tuple(np.asarray(npz_data["Y"]).shape)
        print(f"Y shape: {y_shape}")
        print(f"num_labels: {int(np.asarray(npz_data['num_labels']).item())}")
    elif problem_type == "mlp_multilabel_logistic":
        if "Y_shape" in npz_data:
            y_shape = tuple(int(value) for value in np.asarray(npz_data["Y_shape"]).tolist())
        else:
            y_shape = tuple(np.asarray(npz_data["Y"]).shape)
        print(f"Y shape: {y_shape}")
        print(f"num_labels: {int(np.asarray(npz_data['num_labels']).item())}")
        print(f"hidden_width: {int(np.asarray(npz_data['hidden_width']).item())}")
    else:
        print(f"y shape: {tuple(np.asarray(npz_data['y']).shape)}")
    print(f"reg_lambda: {float(np.asarray(npz_data['reg_lambda']).item())}")
    print(f"dim: {dim}")
    print(f"sample size m: {m}")
    print(f"f(x0): {f0}")
    print(f"||grad(x0)||: {float(np.linalg.norm(grad0))}")
    print(f"hvp(x0, v).shape: {hv0.shape}")
    print(f"finite check: {bool(np.isfinite(f0) and np.all(np.isfinite(grad0)) and np.all(np.isfinite(hv0)))}")
    regularize_bias = bool(np.asarray(npz_data.get('regularize_bias', np.asarray(True))).item())
    if problem_type == "mlp_multilabel_logistic":
        print("strong convexity: this benchmark is intentionally nonconvex, so the convex strong-convexity statement does not apply.")
    elif float(np.asarray(npz_data["reg_lambda"]).item()) > 0.0 and regularize_bias:
        print(
            "strong convexity: Since reg_lambda > 0 and the loss is convex, "
            "the objective is reg_lambda-strongly convex when regularize_bias=true."
        )
    else:
        print("strong convexity: bias is unregularized or reg_lambda == 0, so the default statement does not apply.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from src.problems.logistic import LogisticRegressionProblem, load_libsvm_logistic_dataset, save_logistic_dataset
from src.problems.real_classification import (
    MultiLabelLogisticProblem,
    SoftmaxRegressionProblem,
    clone_problem_data_with_reg_lambda,
    infer_problem_type_from_npz,
    load_libsvm_classification_dataset,
    load_multilabel_dataset,
    save_multilabel_logistic_dataset,
    save_softmax_dataset,
)
from src.utils.io import load_npz


def _small_sparse_A() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [1.5, 0.5, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=float,
        )
    )


def _finite_difference_hvp(problem: object, x: np.ndarray, v: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    grad_plus = np.asarray(problem.grad(x + eps * v), dtype=float)
    grad_base = np.asarray(problem.grad(x), dtype=float)
    return (grad_plus - grad_base) / eps


def test_softmax_problem_shapes_and_hvp() -> None:
    A = _small_sparse_A()
    y = np.array([0, 1, 2, 1], dtype=int)
    problem = SoftmaxRegressionProblem(A=A, y=y, num_classes=3, reg_lambda=1.0e-3)
    x = np.linspace(-0.2, 0.3, problem.dim)
    v = np.linspace(0.5, -0.4, problem.dim)

    assert np.isfinite(problem.f(x))
    assert problem.grad(x).shape == (problem.dim,)
    assert problem.hvp(x, v).shape == (problem.dim,)

    fd = _finite_difference_hvp(problem, x, v)
    hv = problem.hvp(x, v)
    assert np.allclose(fd, hv, atol=5.0e-5, rtol=5.0e-4)


def test_multilabel_problem_shapes_and_hvp() -> None:
    A = _small_sparse_A()
    Y = csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )
    )
    problem = MultiLabelLogisticProblem(A=A, Y=Y, num_labels=2, reg_lambda=1.0e-3)
    x = np.linspace(-0.3, 0.25, problem.dim)
    v = np.linspace(0.1, 0.8, problem.dim)

    assert np.isfinite(problem.f(x))
    assert problem.grad(x).shape == (problem.dim,)
    assert problem.hvp(x, v).shape == (problem.dim,)

    fd = _finite_difference_hvp(problem, x, v)
    hv = problem.hvp(x, v)
    assert np.allclose(fd, hv, atol=5.0e-5, rtol=5.0e-4)


def test_save_load_roundtrip_for_all_problem_types(tmp_path: Path) -> None:
    A = _small_sparse_A()

    logistic_path = tmp_path / "epsilon_like.npz"
    logistic_data = {
        "A": A,
        "y": np.array([1.0, -1.0, 1.0, -1.0]),
        "beta_true": np.zeros(A.shape[1], dtype=float),
        "intercept": np.array(0.0, dtype=float),
        "positive_rate": np.array(0.5, dtype=float),
    }
    save_logistic_dataset(logistic_path, logistic_data, reg_lambda=1.0e-3, regularize_bias=False)
    logistic_problem = LogisticRegressionProblem.from_npz(logistic_path)
    x_log = np.linspace(-0.1, 0.2, logistic_problem.dim)
    v_log = np.linspace(0.3, -0.2, logistic_problem.dim)
    assert infer_problem_type_from_npz(logistic_path) == "logistic"
    assert np.isfinite(logistic_problem.f(x_log))
    assert logistic_problem.grad(x_log).shape == (logistic_problem.dim,)
    assert logistic_problem.hvp(x_log, v_log).shape == (logistic_problem.dim,)

    softmax_path = tmp_path / "mnist_like.npz"
    save_softmax_dataset(
        softmax_path,
        {"A": A, "y": np.array([0, 1, 2, 1]), "num_classes": 3},
        reg_lambda=1.0e-3,
        regularize_bias=False,
    )
    softmax_problem = SoftmaxRegressionProblem.from_npz(softmax_path)
    x_soft = np.linspace(-0.2, 0.25, softmax_problem.dim)
    v_soft = np.linspace(0.4, -0.1, softmax_problem.dim)
    assert infer_problem_type_from_npz(softmax_path) == "softmax"
    assert np.isfinite(softmax_problem.f(x_soft))
    assert softmax_problem.grad(x_soft).shape == (softmax_problem.dim,)
    assert softmax_problem.hvp(x_soft, v_soft).shape == (softmax_problem.dim,)

    multilabel_path = tmp_path / "blogcatalog_like.npz"
    Y = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=float))
    save_multilabel_logistic_dataset(
        multilabel_path,
        {"A": A, "Y": Y, "num_labels": 2},
        reg_lambda=1.0e-3,
        regularize_bias=False,
    )
    multilabel_problem = MultiLabelLogisticProblem.from_npz(multilabel_path)
    x_multi = np.linspace(-0.15, 0.35, multilabel_problem.dim)
    v_multi = np.linspace(0.2, 0.7, multilabel_problem.dim)
    assert infer_problem_type_from_npz(multilabel_path) == "multilabel_logistic"
    assert np.isfinite(multilabel_problem.f(x_multi))
    assert multilabel_problem.grad(x_multi).shape == (multilabel_problem.dim,)
    assert multilabel_problem.hvp(x_multi, v_multi).shape == (multilabel_problem.dim,)


def test_sample_size_and_seed_are_reproducible(tmp_path: Path) -> None:
    raw_path = tmp_path / "toy.libsvm"
    raw_path.write_text(
        "\n".join(
            [
                "1 1:1.0 2:0.0",
                "-1 1:0.0 2:1.0",
                "1 1:2.0 2:0.5",
                "-1 1:0.1 2:0.3",
                "1 1:1.4 2:0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_a = load_libsvm_classification_dataset(
        raw_path,
        n_features=2,
        sample_size=3,
        sample_seed=7,
        max_rows=2,
        label_mode="binary",
    )
    run_b = load_libsvm_classification_dataset(
        raw_path,
        n_features=2,
        sample_size=3,
        sample_seed=7,
        max_rows=2,
        label_mode="binary",
    )
    run_c = load_libsvm_classification_dataset(
        raw_path,
        n_features=2,
        sample_size=3,
        sample_seed=8,
        max_rows=2,
        label_mode="binary",
    )

    assert (run_a["A"] != run_b["A"]).nnz == 0
    assert np.array_equal(run_a["y"], run_b["y"])
    assert not np.array_equal(run_a["y"], run_c["y"]) or (run_a["A"] != run_c["A"]).nnz != 0


def test_expected_dimensions_match_saved_metadata(tmp_path: Path) -> None:
    A_binary = csr_matrix(np.ones((6, 2001), dtype=float))
    binary_path = tmp_path / "epsilon_l2.npz"
    save_logistic_dataset(
        binary_path,
        {
            "A": A_binary,
            "y": np.array([1, -1, 1, -1, 1, -1], dtype=float),
            "beta_true": np.zeros(2001, dtype=float),
            "intercept": np.array(0.0, dtype=float),
            "positive_rate": np.array(0.5, dtype=float),
            "generation_config": {"dim": 2001, "sample_size": 6},
        },
        reg_lambda=1.0e-3,
    )
    assert LogisticRegressionProblem.from_npz(binary_path).dim == 2001

    A_soft = csr_matrix(np.ones((5, 781), dtype=float))
    softmax_path = tmp_path / "mnist_l2.npz"
    save_softmax_dataset(
        softmax_path,
        {
            "A": A_soft,
            "y": np.array([0, 1, 2, 3, 4], dtype=int),
            "num_classes": 10,
            "generation_config": {"dim": 7810, "sample_size": 5},
        },
        reg_lambda=1.0e-3,
    )
    assert SoftmaxRegressionProblem.from_npz(softmax_path).dim == 7810

    A_multi = csr_matrix(np.ones((7, 129), dtype=float))
    Y_multi = csr_matrix(np.zeros((7, 39), dtype=float))
    multilabel_path = tmp_path / "blogcatalog_l2.npz"
    save_multilabel_logistic_dataset(
        multilabel_path,
        {
            "A": A_multi,
            "Y": Y_multi,
            "num_labels": 39,
            "generation_config": {"dim": 5031, "sample_size": 7},
        },
        reg_lambda=1.0e-3,
    )
    assert MultiLabelLogisticProblem.from_npz(multilabel_path).dim == 5031


def test_logistic_loader_add_bias_and_sampling() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "binary.libsvm"
        raw_path.write_text(
            "\n".join(
                [
                    "1 1:1.0 2:2.0",
                    "0 1:0.5 2:1.0",
                    "1 1:3.0 2:0.0",
                    "0 1:0.0 2:4.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = load_libsvm_logistic_dataset(
            raw_path,
            n_features=2,
            sample_size=3,
            sample_seed=0,
            add_bias=True,
        )
        assert dataset["A"].shape == (3, 3)
        assert dataset["beta_true"].shape == (3,)


def test_clone_problem_data_with_reg_lambda_preserves_data(tmp_path: Path) -> None:
    base_path = tmp_path / "epsilon_base.npz"
    clone_path = tmp_path / "epsilon_clone.npz"
    A = csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float))
    save_logistic_dataset(
        base_path,
        {
            "A": A,
            "y": np.array([1.0, -1.0]),
            "beta_true": np.zeros(A.shape[1], dtype=float),
            "intercept": np.array(0.0, dtype=float),
            "positive_rate": np.array(0.5, dtype=float),
            "generation_config": {"dataset_name": "epsilon", "reg_lambda": 1.0e-3},
        },
        reg_lambda=1.0e-3,
        regularize_bias=True,
    )

    clone_problem_data_with_reg_lambda(
        source_path=str(base_path),
        save_path=str(clone_path),
        reg_lambda=1.0e-1,
    )

    base_npz = load_npz(base_path)
    clone_npz = load_npz(clone_path)
    assert infer_problem_type_from_npz(clone_path) == "logistic"
    assert float(np.asarray(clone_npz["reg_lambda"]).item()) == 1.0e-1
    assert np.array_equal(np.asarray(base_npz["y"]), np.asarray(clone_npz["y"]))
    assert np.array_equal(np.asarray(base_npz["A_data"]), np.asarray(clone_npz["A_data"]))
    assert np.array_equal(np.asarray(base_npz["A_indices"]), np.asarray(clone_npz["A_indices"]))
    assert np.array_equal(np.asarray(base_npz["A_indptr"]), np.asarray(clone_npz["A_indptr"]))
    assert np.array_equal(np.asarray(base_npz["A_shape"]), np.asarray(clone_npz["A_shape"]))


def test_load_multilabel_dataset_from_npz_with_x_keys(tmp_path: Path) -> None:
    npz_path = tmp_path / "mediamill_like.npz"
    X = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=float))
    Y = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float))
    np.savez(
        npz_path,
        X_data=np.asarray(X.data, dtype=float),
        X_indices=np.asarray(X.indices, dtype=np.int64),
        X_indptr=np.asarray(X.indptr, dtype=np.int64),
        X_shape=np.asarray(X.shape, dtype=np.int64),
        Y_data=np.asarray(Y.data, dtype=float),
        Y_indices=np.asarray(Y.indices, dtype=np.int64),
        Y_indptr=np.asarray(Y.indptr, dtype=np.int64),
        Y_shape=np.asarray(Y.shape, dtype=np.int64),
    )

    loaded = load_multilabel_dataset(
        npz_path,
        source_format="npz",
        n_features=2,
        num_labels=2,
        sample_size=2,
        sample_seed=0,
    )
    assert loaded["A"].shape == (2, 2)
    assert loaded["Y"].shape == (2, 2)

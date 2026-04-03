from __future__ import annotations

from typing import Mapping

from src.problems.logistic import LogisticRegressionProblem
from src.problems.mnist.dataset import load_mnist_binary
from src.utils.paths import resolve_project_path


def build_mnist_problem(problem_config: Mapping[str, object]) -> tuple[LogisticRegressionProblem, int]:
    data_root = resolve_project_path(str(problem_config.get("data_root", "data/generated/mnist")))
    digits = tuple(problem_config.get("digits", (0, 1)))
    train = bool(problem_config.get("train", True))
    limit = problem_config.get("limit")
    reg_lambda = float(problem_config.get("reg_lambda", 1.0e-2))

    x_matrix, y = load_mnist_binary(
        root=data_root,
        digits=digits,
        train=train,
        limit=None if limit is None else int(limit),
    )
    problem = LogisticRegressionProblem(x_matrix=x_matrix, y=y, reg_lambda=reg_lambda)
    return problem, x_matrix.shape[1]

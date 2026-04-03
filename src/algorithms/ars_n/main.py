from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.algorithms.base import OptimizeResult, not_implemented_error
from src.problems.base import Problem


def run(problem: Problem, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> OptimizeResult:
    del problem, x0, config, logger
    raise not_implemented_error("ARS-N", "src/algorithms/ars_n/main.py")

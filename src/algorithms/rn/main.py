from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.algorithms.base import not_implemented_error


def run(problem: Any, x0: np.ndarray, config: Mapping[str, Any], logger: Any) -> Any:
    del problem, x0, config, logger
    raise not_implemented_error("RN", "src/algorithms/rn/main.py")

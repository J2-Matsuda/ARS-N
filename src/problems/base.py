from typing import Protocol

import numpy as np


class Problem(Protocol):
    def f(self, x: np.ndarray) -> float:
        ...

    def grad(self, x: np.ndarray) -> np.ndarray:
        ...

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        ...

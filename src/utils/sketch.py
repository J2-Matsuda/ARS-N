from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianSketchOperator:
    shape: tuple[int, int]
    scale: float
    seed: int
    mode: str = "operator"
    block_size: int = 256
    dtype: np.dtype | type = float
    _mat: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows, cols = self.shape
        if rows <= 0 or cols <= 0:
            raise ValueError("GaussianSketchOperator shape must be positive.")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("GaussianSketchOperator scale must be positive and finite.")
        if self.mode not in {"operator", "explicit"}:
            raise ValueError(f"Unknown sketch mode: {self.mode}")
        self.block_size = max(1, int(self.block_size))
        self.dtype = np.dtype(self.dtype)

        if self.mode == "explicit":
            self._mat = self._materialize_matrix()

    def dense_matrix(self) -> np.ndarray:
        if self._mat is None:
            self._mat = self._materialize_matrix()
        return self._mat

    def _materialize_matrix(self) -> np.ndarray:
        rows, cols = self.shape
        matrix = np.empty((rows, cols), dtype=self.dtype)
        rng = np.random.default_rng(self.seed)
        row = 0
        while row < rows:
            block_rows = min(self.block_size, rows - row)
            block = rng.standard_normal(size=(block_rows, cols)).astype(self.dtype, copy=False)
            matrix[row : row + block_rows] = (self.scale * block).astype(self.dtype, copy=False)
            row += block_rows
        return matrix

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(-1)
        rows, cols = self.shape
        if vector.shape != (cols,):
            raise ValueError(f"matvec expects v shape ({cols},), got {vector.shape}")
        return np.asarray(self.dense_matrix() @ vector, dtype=float).reshape(-1)

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(-1)
        rows, cols = self.shape
        if vector.shape != (rows,):
            raise ValueError(f"rmatvec expects u shape ({rows},), got {vector.shape}")
        return np.asarray(self.dense_matrix().T @ vector, dtype=float).reshape(-1)

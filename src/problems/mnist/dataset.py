from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_mnist_binary(
    root: str | Path,
    digits: Sequence[int] = (0, 1),
    train: bool = True,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise ImportError(
            "MNIST support requires torchvision. Install the optional dependency set: pip install -e .[mnist]"
        ) from exc

    dataset = MNIST(root=str(root), train=train, download=True)
    images = dataset.data.numpy().reshape(len(dataset), -1).astype(np.float64) / 255.0
    targets = dataset.targets.numpy()

    digits = tuple(digits)
    if len(digits) != 2:
        raise ValueError("MNIST helper currently supports binary classification with exactly two digits")

    mask = np.isin(targets, digits)
    filtered_images = images[mask]
    filtered_targets = targets[mask]
    labels = (filtered_targets == digits[1]).astype(np.float64)

    if limit is not None:
        filtered_images = filtered_images[:limit]
        labels = labels[:limit]

    return filtered_images, labels

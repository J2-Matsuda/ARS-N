from __future__ import annotations

import csv
from pathlib import Path

from src.plotting.main import plot_from_config


def _write_history(path: Path, grad_norms: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iter",
                "f",
                "grad_norm",
                "step_norm",
                "step_size",
                "cumulative_time",
                "per_iter_time",
            ],
        )
        writer.writeheader()
        for iteration, grad_norm in enumerate(grad_norms):
            writer.writerow(
                {
                    "iter": iteration,
                    "f": grad_norm,
                    "grad_norm": grad_norm,
                    "step_norm": 0.0,
                    "step_size": 1.0,
                    "cumulative_time": float(iteration),
                    "per_iter_time": 1.0,
                }
            )


def test_plot_from_config_accepts_grouped_paths(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a.csv"
    run_b = tmp_path / "run_b.csv"
    plot_path = tmp_path / "aggregate.png"
    _write_history(run_a, [3.0, 2.0, 1.0])
    _write_history(run_b, [4.0, 2.5, 1.5])

    output_path = plot_from_config(
        {
            "plot_name": "aggregate_smoke",
            "inputs": [
                {
                    "paths": [str(run_a), str(run_b)],
                    "label": "mean with band",
                    "aggregate": {
                        "center": "mean",
                        "band": "minmax",
                    },
                }
            ],
            "plot": {
                "x": "iter",
                "y": "grad_norm",
                "yscale": "log",
                "grid": True,
            },
            "save": {"path": str(plot_path)},
        }
    )

    assert output_path == plot_path
    assert plot_path.exists()

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


def test_plot_from_config_accepts_panels(tmp_path: Path) -> None:
    gd_a = tmp_path / "gd_a.csv"
    gd_b = tmp_path / "gd_b.csv"
    rs_a_seed0 = tmp_path / "rs_a_seed0.csv"
    rs_a_seed1 = tmp_path / "rs_a_seed1.csv"
    rs_b_seed0 = tmp_path / "rs_b_seed0.csv"
    rs_b_seed1 = tmp_path / "rs_b_seed1.csv"
    plot_path = tmp_path / "panels.png"

    _write_history(gd_a, [3.0, 2.0, 1.0])
    _write_history(gd_b, [2.5, 2.0, 1.5])
    _write_history(rs_a_seed0, [4.0, 2.5, 1.5])
    _write_history(rs_a_seed1, [3.5, 2.4, 1.4])
    _write_history(rs_b_seed0, [5.0, 3.0, 2.0])
    _write_history(rs_b_seed1, [4.5, 2.8, 1.9])

    output_path = plot_from_config(
        {
            "plot_name": "panel_smoke",
            "panels": [
                {
                    "title": "gisette",
                    "inputs": [
                        {"path": str(gd_a), "label": "GD"},
                        {
                            "paths": [str(rs_a_seed0), str(rs_a_seed1)],
                            "label": "RS",
                            "aggregate": {"center": "mean", "band": "minmax"},
                        },
                    ],
                },
                {
                    "title": "epsilon",
                    "inputs": [
                        {"path": str(gd_b), "label": "GD"},
                        {
                            "paths": [str(rs_b_seed0), str(rs_b_seed1)],
                            "label": "RS",
                            "aggregate": {"center": "mean", "band": "minmax"},
                        },
                    ],
                },
            ],
            "plot": {
                "x": "iter",
                "y": "grad_norm",
                "yscale": "log",
                "grid": True,
                "shared_legend": True,
                "layout": {"ncols": 2, "sharey": True},
            },
            "save": {"path": str(plot_path)},
        }
    )

    assert output_path == plot_path
    assert plot_path.exists()

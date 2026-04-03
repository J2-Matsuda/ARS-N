from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from src.cli import main


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_smoke_quadratic_pipeline(tmp_path: Path) -> None:
    quadratic_path = tmp_path / "quadratic_problem.npz"
    newton_cg_csv = tmp_path / "newton_cg.csv"
    full_newton_csv = tmp_path / "full_newton.csv"
    newton_cg_meta = tmp_path / "newton_cg_meta.json"
    full_newton_meta = tmp_path / "full_newton_meta.json"
    newton_cg_resolved = tmp_path / "newton_cg_resolved.yml"
    full_newton_resolved = tmp_path / "full_newton_resolved.yml"
    plot_path = tmp_path / "comparison.png"

    generate_config = {
        "task": "generate_data",
        "run_name": "smoke_quadratic",
        "seed": 7,
        "problem": {
            "type": "quadratic",
            "dim": 8,
            "spectrum": "exponential",
            "lambda_max": 1.0,
            "lambda_min": 1.0e-3,
            "b_norm": 1.0,
        },
        "save": {"path": str(quadratic_path)},
    }
    newton_cg_config = {
        "task": "optimize",
        "run_name": "smoke_newton_cg",
        "seed": 7,
        "problem": {"type": "quadratic", "source": str(quadratic_path)},
        "initialization": {"type": "zeros"},
        "optimizer": {
            "type": "newton_cg",
            "max_iter": 20,
            "tol": 1.0e-10,
            "cg_max_iter": 16,
            "cg_tol": 1.0e-8,
            "line_search": {"enabled": True, "c1": 1.0e-4, "beta": 0.5, "max_iter": 25},
        },
        "log": {"enabled": True, "csv_path": str(newton_cg_csv)},
        "save_meta": {
            "enabled": True,
            "meta_path": str(newton_cg_meta),
            "resolved_config_path": str(newton_cg_resolved),
        },
    }
    full_newton_config = {
        "task": "optimize",
        "run_name": "smoke_full_newton",
        "seed": 7,
        "problem": {"type": "quadratic", "source": str(quadratic_path)},
        "initialization": {"type": "zeros"},
        "optimizer": {
            "type": "full_newton",
            "max_iter": 8,
            "tol": 1.0e-10,
            "line_search": {"enabled": True, "c1": 1.0e-4, "beta": 0.5, "max_iter": 25},
        },
        "log": {"enabled": True, "csv_path": str(full_newton_csv)},
        "save_meta": {
            "enabled": True,
            "meta_path": str(full_newton_meta),
            "resolved_config_path": str(full_newton_resolved),
        },
    }
    plot_config = {
        "task": "plot",
        "plot_name": "smoke_comparison",
        "inputs": [
            {"path": str(newton_cg_csv), "label": "Newton-CG"},
            {"path": str(full_newton_csv), "label": "Full Newton"},
        ],
        "plot": {
            "x": "cumulative_time",
            "y": "grad_norm",
            "xscale": "linear",
            "yscale": "log",
            "title": "Smoke Test",
            "xlabel": "cumulative_time",
            "ylabel": "grad_norm",
            "grid": True,
        },
        "save": {"path": str(plot_path)},
    }

    generate_yaml = tmp_path / "generate.yml"
    newton_cg_yaml = tmp_path / "newton_cg.yml"
    full_newton_yaml = tmp_path / "full_newton.yml"
    plot_yaml = tmp_path / "plot.yml"
    _write_yaml(generate_yaml, generate_config)
    _write_yaml(newton_cg_yaml, newton_cg_config)
    _write_yaml(full_newton_yaml, full_newton_config)
    _write_yaml(plot_yaml, plot_config)

    main(["generate", "--config", str(generate_yaml)])
    main(["optimize", "--config", str(newton_cg_yaml)])
    main(["optimize", "--config", str(full_newton_yaml)])
    main(["plot", "--config", str(plot_yaml)])

    assert quadratic_path.exists()
    assert newton_cg_csv.exists()
    assert full_newton_csv.exists()
    assert plot_path.exists()

    with newton_cg_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames[:7] == [
            "iter",
            "f",
            "grad_norm",
            "step_norm",
            "step_size",
            "cumulative_time",
            "per_iter_time",
        ]
    assert rows[0]["iter"] == "0"
    assert float(rows[-1]["grad_norm"]) <= float(rows[0]["grad_norm"])

    with newton_cg_meta.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    assert meta["algorithm"] == "newton_cg"
    assert meta["resolved_config"]["run_name"] == "smoke_newton_cg"

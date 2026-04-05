from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from src.cli import main
from src.config import load_pipeline_config
from src.config import load_optimize_config
from src.utils.run_logger import RunLogger


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sample_log_row() -> dict[str, float]:
    return {
        "iter": 0,
        "f": 1.0,
        "grad_norm": 0.5,
        "step_norm": 0.0,
        "step_size": 0.0,
        "cumulative_time": 0.0,
        "per_iter_time": 0.0,
    }


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


def test_load_optimize_config_defaults_save_everytime_to_true(tmp_path: Path) -> None:
    config_path = tmp_path / "optimize.yml"
    _write_yaml(
        config_path,
        {
            "task": "optimize",
            "run_name": "config_default",
            "problem": {"type": "quadratic", "source": str(tmp_path / "problem.npz")},
            "initialization": {"type": "zeros"},
            "optimizer": {"type": "full_newton"},
            "log": {"enabled": True, "csv_path": str(tmp_path / "history.csv")},
            "save_meta": {
                "enabled": False,
                "meta_path": str(tmp_path / "meta.json"),
                "resolved_config_path": str(tmp_path / "resolved.yml"),
            },
        },
    )

    loaded = load_optimize_config(config_path)

    assert loaded["log"]["save_everytime"] is True


def test_pipeline_command_runs_steps_in_order(tmp_path: Path) -> None:
    quadratic_path = tmp_path / "quadratic_problem.npz"
    newton_cg_csv = tmp_path / "newton_cg.csv"
    newton_cg_meta = tmp_path / "newton_cg_meta.json"
    newton_cg_resolved = tmp_path / "newton_cg_resolved.yml"
    plot_path = tmp_path / "comparison.png"

    generate_config = {
        "task": "generate_data",
        "run_name": "pipeline_generate",
        "seed": 11,
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
    optimize_config = {
        "task": "optimize",
        "run_name": "pipeline_optimize",
        "seed": 11,
        "problem": {"type": "quadratic", "source": str(quadratic_path)},
        "initialization": {"type": "zeros"},
        "optimizer": {
            "type": "newton_cg",
            "max_iter": 5,
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
    plot_config = {
        "task": "plot",
        "plot_name": "pipeline_plot",
        "inputs": [{"path": str(newton_cg_csv), "label": "Newton-CG"}],
        "plot": {
            "x": "cumulative_time",
            "y": "grad_norm",
            "xscale": "linear",
            "yscale": "log",
            "title": "Pipeline Test",
            "xlabel": "cumulative_time",
            "ylabel": "grad_norm",
            "grid": True,
        },
        "save": {"path": str(plot_path)},
    }
    pipeline_config = {
        "task": "pipeline",
        "pipeline_name": "smoke_pipeline",
        "steps": [
            {"command": "generate", "config": str(tmp_path / "generate.yml")},
            {"command": "optimize", "config": str(tmp_path / "optimize.yml")},
            {"command": "plot", "config": str(tmp_path / "plot.yml")},
        ],
    }

    generate_yaml = tmp_path / "generate.yml"
    optimize_yaml = tmp_path / "optimize.yml"
    plot_yaml = tmp_path / "plot.yml"
    pipeline_yaml = tmp_path / "pipeline.yml"
    _write_yaml(generate_yaml, generate_config)
    _write_yaml(optimize_yaml, optimize_config)
    _write_yaml(plot_yaml, plot_config)
    _write_yaml(pipeline_yaml, pipeline_config)

    loaded = load_pipeline_config(pipeline_yaml)
    assert loaded["steps"][1]["command"] == "optimize"

    main(["pipeline", "--config", str(pipeline_yaml)])

    assert quadratic_path.exists()
    assert newton_cg_csv.exists()
    assert newton_cg_meta.exists()
    assert plot_path.exists()


def test_rn_quadratic_smoke(tmp_path: Path) -> None:
    quadratic_path = tmp_path / "quadratic_problem.npz"
    rn_csv = tmp_path / "rn.csv"
    rn_meta = tmp_path / "rn_meta.json"
    rn_resolved = tmp_path / "rn_resolved.yml"

    generate_config = {
        "task": "generate_data",
        "run_name": "rn_generate",
        "seed": 5,
        "problem": {
            "type": "quadratic",
            "dim": 6,
            "spectrum": "exponential",
            "lambda_max": 1.0,
            "lambda_min": 1.0e-3,
            "b_norm": 1.0,
        },
        "save": {"path": str(quadratic_path)},
    }
    rn_config = {
        "task": "optimize",
        "run_name": "rn_quadratic",
        "seed": 5,
        "problem": {"type": "quadratic", "source": str(quadratic_path)},
        "initialization": {"type": "zeros"},
        "optimizer": {
            "type": "rn",
            "max_iter": 10,
            "tol": 1.0e-10,
            "diag_shift": {"c1": 2.0, "c2": 1.0, "gamma": 1.0},
            "line_search": {"enabled": True, "c1": 1.0e-4, "beta": 0.5, "max_iter": 25},
        },
        "log": {"enabled": True, "csv_path": str(rn_csv)},
        "save_meta": {
            "enabled": True,
            "meta_path": str(rn_meta),
            "resolved_config_path": str(rn_resolved),
        },
    }

    generate_yaml = tmp_path / "generate.yml"
    rn_yaml = tmp_path / "rn.yml"
    _write_yaml(generate_yaml, generate_config)
    _write_yaml(rn_yaml, rn_config)

    main(["generate", "--config", str(generate_yaml)])
    main(["optimize", "--config", str(rn_yaml)])

    assert quadratic_path.exists()
    assert rn_csv.exists()
    assert rn_meta.exists()

    with rn_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert "eta" in (reader.fieldnames or [])
    assert "lambda_min_phpt" in (reader.fieldnames or [])
    assert "lambda_shift" in (reader.fieldnames or [])
    assert rows[0]["iter"] == "0"
    assert float(rows[-1]["grad_norm"]) <= float(rows[0]["grad_norm"])


def test_run_logger_save_everytime_controls_write_timing(tmp_path: Path) -> None:
    immediate_path = tmp_path / "immediate.csv"
    delayed_path = tmp_path / "delayed.csv"
    row = _sample_log_row()

    immediate_logger = RunLogger(immediate_path, save_everytime=True, flush_every=1)
    immediate_logger.log(row)
    immediate_contents = immediate_path.read_text(encoding="utf-8")
    immediate_logger.close()

    delayed_logger = RunLogger(delayed_path, save_everytime=False)
    delayed_logger.log(row)
    assert not delayed_path.exists()
    delayed_logger.close()
    delayed_contents = delayed_path.read_text(encoding="utf-8")

    assert "iter,f,grad_norm,step_norm,step_size,cumulative_time,per_iter_time" in immediate_contents
    assert "0,1.0,0.5,0.0,0.0,0.0,0.0" in immediate_contents
    assert "iter,f,grad_norm,step_norm,step_size,cumulative_time,per_iter_time" in delayed_contents
    assert "0,1.0,0.5,0.0,0.0,0.0,0.0" in delayed_contents

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping

from src.utils.io import ensure_parent_dir
from src.utils.paths import OUTPUT_DIR, resolve_project_path

_MPL_CONFIG_DIR = OUTPUT_DIR / "meta" / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _load_xy_series(path: str | Path, x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for key in (x_key, y_key):
            if key not in fieldnames:
                raise ValueError(f"CSV {path} does not contain column {key!r}")

        x_values: list[float] = []
        y_values: list[float] = []
        for row in reader:
            x_raw = row.get(x_key)
            y_raw = row.get(y_key)
            if x_raw in ("", None) or y_raw in ("", None):
                continue
            try:
                x_value = float(x_raw)
                y_value = float(y_raw)
            except (TypeError, ValueError):
                continue
            if not (x_value == x_value and y_value == y_value):
                continue
            x_values.append(x_value)
            y_values.append(y_value)

    if not x_values:
        raise ValueError(f"CSV {path} has no valid data for columns {x_key!r}, {y_key!r}")
    return x_values, y_values


def _resolve_linestyle(value: Any) -> str:
    linestyle = str(value).strip().lower()
    linestyle_map = {
        "solid": "-",
        "-": "-",
        "dashed": "--",
        "--": "--",
        "dotted": ":",
        ":": ":",
        "dashdot": "-.",
        "-.": "-.",
    }
    try:
        return linestyle_map[linestyle]
    except KeyError as exc:
        raise ValueError(
            "Unsupported linestyle "
            f"{value!r}. Available: solid, dashed, dotted, dashdot, -, --, :, -."
        ) from exc


def plot_from_config(config: Mapping[str, Any]) -> Path:
    plot_config = dict(config["plot"])
    save_path = resolve_project_path(config["save"]["path"])
    dpi = int(config["save"].get("dpi", 150))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x_key = str(plot_config["x"])
    y_key = str(plot_config["y"])

    for item in config["inputs"]:
        csv_path = resolve_project_path(item["path"])
        label = str(item.get("label", csv_path.stem))
        x_values, y_values = _load_xy_series(csv_path, x_key=x_key, y_key=y_key)
        plot_kwargs: dict[str, Any] = {"label": label, "linewidth": 2.0}
        if "color" in item:
            plot_kwargs["color"] = str(item["color"])
        if "linestyle" in item:
            plot_kwargs["linestyle"] = _resolve_linestyle(item["linestyle"])
        ax.plot(x_values, y_values, **plot_kwargs)

    ax.set_xscale(str(plot_config.get("xscale", "linear")))
    ax.set_yscale(str(plot_config.get("yscale", "linear")))
    ax.set_title(str(plot_config.get("title", "")))
    ax.set_xlabel(str(plot_config.get("xlabel", x_key)))
    ax.set_ylabel(str(plot_config.get("ylabel", y_key)))
    if bool(plot_config.get("grid", False)):
        ax.grid(True, alpha=0.3)
    if len(config["inputs"]) > 1:
        ax.legend()

    fig.tight_layout()
    ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path

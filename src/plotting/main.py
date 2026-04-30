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
import numpy as np


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


def _as_sorted_arrays(
    x_values: list[float],
    y_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    if x_array.size != y_array.size:
        raise ValueError("x and y series must have the same length")

    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = y_array[order]
    unique_x, unique_indices = np.unique(x_array, return_index=True)
    return unique_x, y_array[unique_indices]


def _common_grid(
    series: list[tuple[np.ndarray, np.ndarray]],
    aggregate_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    first_x = series[0][0]
    if all(x.shape == first_x.shape and np.allclose(x, first_x) for x, _ in series):
        return first_x, np.vstack([y for _, y in series])

    lower = max(float(x[0]) for x, _ in series)
    upper = min(float(x[-1]) for x, _ in series)
    if lower > upper:
        raise ValueError("Input series do not have an overlapping x range")

    if lower == upper:
        x_grid = np.asarray([lower], dtype=float)
    else:
        grid_mode = str(aggregate_config.get("grid", "first")).lower()
        if grid_mode == "linspace":
            default_points = min(x.size for x, _ in series)
            n_points = max(2, int(aggregate_config.get("num_points", default_points)))
            x_grid = np.linspace(lower, upper, n_points)
        elif grid_mode == "union":
            x_grid = np.unique(
                np.concatenate([x[(x >= lower) & (x <= upper)] for x, _ in series])
            )
        else:
            candidate = first_x[(first_x >= lower) & (first_x <= upper)]
            if candidate.size >= 2:
                x_grid = candidate
            else:
                n_points = max(2, min(x.size for x, _ in series))
                x_grid = np.linspace(lower, upper, n_points)

    y_stack = np.vstack([np.interp(x_grid, x, y) for x, y in series])
    return x_grid, y_stack


def _aggregate_xy_series(
    paths: list[str | Path],
    x_key: str,
    y_key: str,
    aggregate_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for path in paths:
        x_values, y_values = _load_xy_series(path, x_key=x_key, y_key=y_key)
        series.append(_as_sorted_arrays(x_values, y_values))

    if not series:
        raise ValueError("Grouped plot input must contain at least one path")

    x_grid, y_stack = _common_grid(series, aggregate_config)
    center = str(aggregate_config.get("center", "mean")).lower()
    if center == "median":
        y_center = np.median(y_stack, axis=0)
    elif center == "mean":
        y_center = np.mean(y_stack, axis=0)
    else:
        raise ValueError(f"Unsupported aggregate.center {center!r}. Available: mean, median")

    band = str(aggregate_config.get("band", "minmax")).lower()
    if band in {"none", "false", "off"} or len(series) == 1:
        return x_grid, y_center, None, None
    if band in {"minmax", "range"}:
        y_lower = np.min(y_stack, axis=0)
        y_upper = np.max(y_stack, axis=0)
    elif band == "std":
        y_std = np.std(y_stack, axis=0, ddof=1 if len(series) > 1 else 0)
        y_lower = y_center - y_std
        y_upper = y_center + y_std
    elif band == "sem":
        y_std = np.std(y_stack, axis=0, ddof=1 if len(series) > 1 else 0)
        y_sem = y_std / np.sqrt(float(len(series)))
        y_lower = y_center - y_sem
        y_upper = y_center + y_sem
    else:
        raise ValueError(
            f"Unsupported aggregate.band {band!r}. Available: minmax, std, sem, none"
        )

    return x_grid, y_center, y_lower, y_upper


def _coerce_path_list(value: Any, context: str) -> list[str | Path]:
    if isinstance(value, (str, Path)):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty path or list of paths")
    return [str(item) for item in value]


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


def _resolve_axis_limits(value: Any, context: str) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context} must be a length-2 list or tuple")

    limits: list[float | None] = []
    for item in value:
        if item is None:
            limits.append(None)
            continue
        try:
            numeric_value = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} entries must be numbers or null") from exc
        limits.append(numeric_value)

    lower, upper = limits
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"{context} must satisfy min < max")
    return lower, upper


def _plot_inputs_on_axis(
    ax: Any,
    inputs: list[Mapping[str, Any]],
    plot_config: Mapping[str, Any],
) -> int:
    x_key = str(plot_config["x"])
    y_key = str(plot_config["y"])
    plotted_count = 0
    skip_missing = bool(plot_config.get("skip_missing", True))

    for item in inputs:
        if "paths" in item:
            raw_paths = _coerce_path_list(item["paths"], "plot input paths")
            csv_paths_all = [resolve_project_path(path) for path in raw_paths]
            csv_paths = [path for path in csv_paths_all if path.exists()]
            missing_paths = [path for path in csv_paths_all if not path.exists()]
            for missing_path in missing_paths:
                print(f"[plot] skipping missing CSV: {missing_path}")
            if not csv_paths:
                if skip_missing:
                    continue
                raise FileNotFoundError(
                    f"All grouped plot inputs are missing for {raw_paths!r}"
                )
            label = str(item.get("label", Path(str(csv_paths[0])).stem))
            aggregate_config = dict(item.get("aggregate", {}))
            x_values, y_values, y_lower, y_upper = _aggregate_xy_series(
                paths=csv_paths,
                x_key=x_key,
                y_key=y_key,
                aggregate_config=aggregate_config,
            )
        elif "path" in item:
            csv_path = resolve_project_path(item["path"])
            if not csv_path.exists():
                print(f"[plot] skipping missing CSV: {csv_path}")
                if skip_missing:
                    continue
                raise FileNotFoundError(csv_path)
            label = str(item.get("label", csv_path.stem))
            x_values, y_values = _load_xy_series(csv_path, x_key=x_key, y_key=y_key)
            y_lower = None
            y_upper = None
            aggregate_config = {}
        else:
            raise ValueError("Each plot input must contain either 'path' or 'paths'")

        plot_kwargs: dict[str, Any] = {"label": label, "linewidth": 2.0}
        if "color" in item:
            plot_kwargs["color"] = str(item["color"])
        if "linestyle" in item:
            plot_kwargs["linestyle"] = _resolve_linestyle(item["linestyle"])
        (line,) = ax.plot(x_values, y_values, **plot_kwargs)
        plotted_count += 1
        if y_lower is not None and y_upper is not None:
            band_alpha = float(aggregate_config.get("alpha", item.get("band_alpha", 0.18)))
            ax.fill_between(
                x_values,
                y_lower,
                y_upper,
                color=line.get_color(),
                alpha=band_alpha,
                linewidth=0.0,
                label="_nolegend_",
            )

    ax.set_xscale(str(plot_config.get("xscale", "linear")))
    ax.set_yscale(str(plot_config.get("yscale", "linear")))
    ax.set_title(str(plot_config.get("title", "")))
    ax.set_xlabel(str(plot_config.get("xlabel", x_key)))
    ax.set_ylabel(str(plot_config.get("ylabel", y_key)))
    if "x_limit" in plot_config:
        ax.set_xlim(*_resolve_axis_limits(plot_config["x_limit"], "plot.x_limit"))
    if "y_limit" in plot_config:
        ax.set_ylim(*_resolve_axis_limits(plot_config["y_limit"], "plot.y_limit"))
    if bool(plot_config.get("grid", False)):
        ax.grid(True, alpha=0.3)
    return plotted_count


def _merge_plot_config(
    base_plot_config: Mapping[str, Any],
    panel_plot_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base_plot_config)
    if panel_plot_config:
        merged.update(panel_plot_config)
    return merged


def plot_from_config(config: Mapping[str, Any]) -> Path:
    plot_config = dict(config["plot"])
    save_path = resolve_project_path(config["save"]["path"])
    dpi = int(config["save"].get("dpi", 150))

    if "panels" in config:
        panels = list(config["panels"])
        n_panels = len(panels)
        layout = dict(plot_config.get("layout", {}))
        ncols = max(1, int(layout.get("ncols", n_panels)))
        nrows = int(np.ceil(n_panels / ncols))
        figsize = (
            float(layout.get("width", 6.0 * ncols)),
            float(layout.get("height", 4.2 * nrows)),
        )
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=False,
            sharex=bool(layout.get("sharex", False)),
            sharey=bool(layout.get("sharey", False)),
        )
        axes_flat = list(axes.flat)

        for axis, panel in zip(axes_flat, panels):
            panel_mapping = dict(panel)
            panel_inputs = panel_mapping.get("inputs")
            if not isinstance(panel_inputs, list) or not panel_inputs:
                raise ValueError("Each plot panel must contain a non-empty inputs list")
            panel_plot = panel_mapping.get("plot")
            if panel_plot is not None and not isinstance(panel_plot, Mapping):
                raise ValueError("panel.plot must be a mapping when provided")
            merged_plot = _merge_plot_config(plot_config, panel_plot)
            if "title" in panel_mapping:
                merged_plot["title"] = panel_mapping["title"]
            if "xlabel" in panel_mapping:
                merged_plot["xlabel"] = panel_mapping["xlabel"]
            if "ylabel" in panel_mapping:
                merged_plot["ylabel"] = panel_mapping["ylabel"]
            plotted_count = _plot_inputs_on_axis(axis, panel_inputs, merged_plot)
            if (
                not bool(plot_config.get("shared_legend", False))
                and plotted_count > 1
            ):
                axis.legend()

        for axis in axes_flat[n_panels:]:
            axis.set_visible(False)

        if "figure_title" in plot_config:
            fig.suptitle(str(plot_config["figure_title"]))
        if bool(plot_config.get("shared_legend", False)):
            legend_handles: list[Any] = []
            legend_labels: list[str] = []
            seen_labels: set[str] = set()
            for axis in axes_flat[:n_panels]:
                handles, labels = axis.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label == "_nolegend_" or label in seen_labels:
                        continue
                    seen_labels.add(label)
                    legend_handles.append(handle)
                    legend_labels.append(label)
            if legend_handles:
                location = str(plot_config.get("shared_legend_location", "lower center"))
                ncol = int(plot_config.get("shared_legend_ncol", max(1, len(legend_labels))))
                fig.legend(legend_handles, legend_labels, loc=location, ncol=ncol)
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.96] if "figure_title" in plot_config else None)
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        plotted_count = _plot_inputs_on_axis(ax, list(config["inputs"]), plot_config)
        if plotted_count > 1:
            ax.legend()
        fig.tight_layout()

    ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path

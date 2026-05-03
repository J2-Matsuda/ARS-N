from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


RESULT_ROOT = Path("output/results/candidate_multilabel")
THRESHOLDS = (1.0e-4, 1.0e-5)


def _time_to_threshold(csv_path: Path, threshold: float) -> float | None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                grad_norm = float(row["grad_norm"])
                cumulative_time = float(row["cumulative_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(grad_norm) and grad_norm <= threshold:
                return cumulative_time
    return None


def _load_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for meta_path in RESULT_ROOT.rglob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        algorithm = str(meta.get("algorithm", ""))
        if algorithm not in {"ars_cn", "rs_cn"}:
            continue
        config = meta.get("resolved_config", {})
        if not isinstance(config, dict):
            continue
        problem = config.get("problem", {})
        optimizer = config.get("optimizer", {})
        if not isinstance(problem, dict) or not isinstance(optimizer, dict):
            continue
        source = str(problem.get("source", ""))
        slug = Path(source).stem
        csv_path = Path(str(meta.get("history_path", "")))
        if not csv_path.exists():
            candidates = list(meta_path.parent.glob("*.csv"))
            if not candidates:
                continue
            csv_path = candidates[0]

        record: dict[str, object] = {
            "slug": slug,
            "algorithm": algorithm,
            "run_name": meta.get("run_name"),
            "seed": meta.get("seed"),
            "subspace_dim": optimizer.get("subspace_dim"),
        }
        rk = optimizer.get("rk", {})
        if isinstance(rk, dict):
            record["T"] = rk.get("T")
            record["r"] = rk.get("r")
        for threshold in THRESHOLDS:
            record[f"t_grad_{threshold:g}"] = _time_to_threshold(csv_path, threshold)
        runs.append(record)
    return runs


def main() -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in _load_runs():
        grouped[str(run["slug"])].append(run)

    if not grouped:
        print(f"No candidate results found under {RESULT_ROOT}")
        return

    for threshold in THRESHOLDS:
        key = f"t_grad_{threshold:g}"
        print(f"\n=== grad_norm <= {threshold:g} ===")
        for slug in sorted(grouped):
            ars = [run for run in grouped[slug] if run["algorithm"] == "ars_cn" and run.get(key) is not None]
            rs = [run for run in grouped[slug] if run["algorithm"] == "rs_cn" and run.get(key) is not None]
            if not ars or not rs:
                continue
            best_ars = min(ars, key=lambda run: float(run[key]))
            best_rs = min(rs, key=lambda run: float(run[key]))
            best_rs_time = float(best_rs[key])
            win_count = sum(1 for run in ars if float(run[key]) < best_rs_time)
            ratio = best_rs_time / float(best_ars[key])
            print(
                f"{slug:42s} ratio={ratio:6.2f} "
                f"ARS<{win_count:02d}/{len(ars):02d} "
                f"best_ars={float(best_ars[key]):9.1f}s {best_ars['run_name']} "
                f"best_rs={best_rs_time:9.1f}s {best_rs['run_name']}"
            )


if __name__ == "__main__":
    main()

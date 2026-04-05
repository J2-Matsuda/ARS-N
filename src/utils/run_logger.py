from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.io import ensure_parent_dir

REQUIRED_COLUMNS = (
    "iter",
    "f",
    "grad_norm",
    "step_norm",
    "step_size",
    "cumulative_time",
    "per_iter_time",
)


class RunLogger:
    def __init__(self, path: str | Path, extra_fields: Sequence[str] = (), flush_every: int = 1, save_everytime: bool = True) -> None:
        self.path = Path(path)
        ensure_parent_dir(self.path)
        deduplicated = [field for field in extra_fields if field not in REQUIRED_COLUMNS]
        self.fieldnames = [*REQUIRED_COLUMNS, *deduplicated]
        self.save_everytime = save_everytime
        if self.save_everytime:
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._file.flush()
            self._flush_every = max(1, int(flush_every))
            self._rows_since_flush = 0
        else:
            self._history = []
        self.history_path = str(self.path)
        self.enabled = True

    def log(self, row: Mapping[str, Any]) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Run logger row is missing required columns: {joined}")

        unexpected = set(row) - set(self.fieldnames)
        if unexpected:
            joined = ", ".join(sorted(unexpected))
            raise ValueError(f"Run logger row contains unknown columns: {joined}")

        if self.save_everytime:
            serialized = {field: row.get(field, "") for field in self.fieldnames}
            self._writer.writerow(serialized)
            self._rows_since_flush += 1
            if self._rows_since_flush >= self._flush_every:
                self._file.flush()
                self._rows_since_flush = 0
        else:
            self._history.append(dict(row))

    def close(self) -> None:
        if self.save_everytime:
            self._file.flush()
            self._file.close()
        else:
            # Write all history at once
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                for row in self._history:
                    serialized = {field: row.get(field, "") for field in self.fieldnames}
                    writer.writerow(serialized)


class NullRunLogger:
    enabled = False
    history_path = None

    def log(self, row: Mapping[str, Any]) -> None:
        del row

    def close(self) -> None:
        return None

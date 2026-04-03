from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter


@dataclass
class Stopwatch:
    start_time: float = field(default_factory=perf_counter)

    def elapsed(self) -> float:
        return perf_counter() - self.start_time


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

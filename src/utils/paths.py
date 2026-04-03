from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "input"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

STANDARD_DIRECTORIES = (
    INPUT_DIR / "generate_data",
    INPUT_DIR / "optimize",
    INPUT_DIR / "plot",
    DATA_DIR / "generated",
    OUTPUT_DIR / "results",
    OUTPUT_DIR / "plots",
    OUTPUT_DIR / "meta",
)


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_standard_directories() -> None:
    for directory in STANDARD_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

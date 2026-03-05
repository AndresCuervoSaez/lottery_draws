from __future__ import annotations

from pathlib import Path


def ensure_project_dirs(base_dir: Path) -> None:
    for rel in ["data/raw", "data/processed", "outputs"]:
        (base_dir / rel).mkdir(parents=True, exist_ok=True)


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace(" ", "_")
    )

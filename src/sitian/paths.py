"""Resolve dataset case locations independently of worker working directories."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_case_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    legacy_root = os.environ.get("SITIAN_LEGACY_PROJECT_ROOT")
    if path.is_absolute() and legacy_root:
        try:
            relative = path.relative_to(Path(legacy_root).expanduser())
        except ValueError:
            pass
        else:
            path = PROJECT_ROOT / relative
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not (path / "case.json").is_file():
        raise FileNotFoundError(
            f"case.json missing in {path}; restore the case snapshot or set "
            "SITIAN_LEGACY_PROJECT_ROOT to the old project root for migrated Parquet"
        )
    return path

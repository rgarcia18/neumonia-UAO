from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory.

    The root is detected by walking up from the current file location until a
    `pyproject.toml` file is found. This makes resource paths (e.g. `model/`)
    robust regardless of the current working directory.

    Returns:
        Absolute path to the repository root. If not found, falls back to the
        current working directory.
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()

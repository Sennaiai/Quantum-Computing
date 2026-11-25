"""Validate generated-output locations before any filesystem mutation."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_DIRECTORIES = ("src", "python_classes", "qbe_modules", "pipeline", "tests",
                         ".agents", ".codex", ".aws")


def validate_output_dir(output_dir, *, project_root=PROJECT_ROOT) -> Path:
    """Return a resolved artifact directory without creating it.

    Project ancestors cannot receive mirrored notebook paths. Source trees,
    tests, and Git metadata cannot receive generated artifacts. Resolution also
    prevents an output symlink from bypassing these checks.
    """
    if isinstance(output_dir, str) and not output_dir.strip():
        raise ValueError("Output directory must not be empty")
    try:
        output = Path(output_dir).expanduser().resolve()
        root = Path(project_root).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("Output directory must be a valid filesystem path") from exc
    if output == root or output in root.parents:
        raise ValueError("Output directory must not be the project root or an ancestor")
    if ".git" in output.parts:
        raise ValueError("Output directory must be outside Git metadata")
    for relative in PROTECTED_DIRECTORIES:
        protected = (root / relative).resolve()
        if output == protected or protected in output.parents:
            raise ValueError("Output artifacts must be outside source and test directories")
    if output.exists() and not output.is_dir():
        raise ValueError("Output path must be a directory")
    return output

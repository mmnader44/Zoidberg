from pathlib import Path


# Project root is the parent of `src` (this file lives in `src/utils/`)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def relative_path(path: Path) -> Path:
    """
    Return the path relative to the project root.

    If the path is not inside the project (cannot be relativized),
    the original path is returned.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def format_file_size(path: Path) -> str:
    """
    Human-readable file size using decimal units (o, Ko, Mo, Go).
    """
    size = path.stat().st_size

    if size < 1024:
        return f"{size} o"
    if size < 1024**2:
        return f"{size / 1024:.1f} Ko"
    if size < 1024**3:
        return f"{size / 1024**2:.1f} Mo"
    return f"{size / 1024**3:.1f} Go"


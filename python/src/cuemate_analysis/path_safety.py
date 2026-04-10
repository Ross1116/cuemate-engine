from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_DEFAULT_POSIX_SERVICE_ROOTS = ("/workspace", "/host")


def resolve_allowed_roots(
    raw_roots: str | None,
    *,
    default_posix_roots: Iterable[str] = _DEFAULT_POSIX_SERVICE_ROOTS,
) -> list[Path]:
    """Return canonical allowed roots for service-facing file validation.

    On POSIX service runtimes, default to `/workspace` and `/host` when no
    explicit override is provided. On Windows, default to no root restriction
    so local host-side development still works.
    """
    if raw_roots:
        parts = [item.strip() for item in raw_roots.split(os.pathsep) if item.strip()]
    elif os.name != "nt":
        parts = [str(item).strip() for item in default_posix_roots if str(item).strip()]
    else:
        parts = []
    return [Path(item).expanduser().resolve() for item in parts]


def resolve_existing_file_path(
    raw_path: str | Path,
    label: str,
    *,
    allowed_roots: Iterable[Path] | None = None,
) -> Path:
    roots = list(allowed_roots or [])
    resolved = _resolve_path(raw_path, label, roots)
    _ensure_allowed_root(resolved, label, roots)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a readable file: {resolved}")
    return resolved


def resolve_existing_directory_path(
    raw_path: str | Path,
    label: str,
    *,
    allowed_roots: Iterable[Path] | None = None,
) -> Path:
    roots = list(allowed_roots or [])
    resolved = _resolve_path(raw_path, label, roots)
    _ensure_allowed_root(resolved, label, roots)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a readable directory: {resolved}")
    return resolved


# Backward-compatible alias for service modules that still use the shorter name.
resolve_existing_dir_path = resolve_existing_directory_path


def _resolve_path(raw_path: str | Path, label: str, allowed_roots: Iterable[Path]) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise ValueError(f"{label} is required.")
    candidate = Path(text).expanduser()
    roots = list(allowed_roots)
    if candidate.is_absolute() or not roots:
        return candidate.resolve()
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.is_relative_to(root):
            return resolved
    allowed = ", ".join(root.as_posix() for root in roots)
    raise ValueError(f"{label} must stay within allowed roots: {allowed}")


def _ensure_allowed_root(
    resolved_path: Path,
    label: str,
    allowed_roots: Iterable[Path] | None,
) -> None:
    roots = list(allowed_roots or [])
    if not roots:
        return
    if any(resolved_path.is_relative_to(root) for root in roots):
        return
    allowed = ", ".join(root.as_posix() for root in roots)
    raise ValueError(f"{label} must stay within allowed roots: {allowed}")

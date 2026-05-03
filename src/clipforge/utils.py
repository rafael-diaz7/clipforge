"""Small shared helpers for local paths, filenames, and timestamps."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(root: Path, *parts: str | Path) -> Path:
    """Resolve a path under the project root."""

    return root.joinpath(*parts).resolve()


def ensure_project_subdir(root: Path, *parts: str | Path) -> Path:
    """Create and return a project-local subdirectory."""

    return ensure_directory(resolve_project_path(root, *parts))


def safe_filename(value: str, *, fallback: str = "clip") -> str:
    """Return a filesystem-safe filename stem or filename."""

    cleaned = _SAFE_FILENAME_PATTERN.sub("_", value.strip()).strip("._-")
    return cleaned or fallback


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for metadata."""

    return datetime.now(UTC).isoformat()


def clip_slug_from_url(url: str) -> str:
    """Extract a stable slug from a Twitch clip URL, falling back safely."""

    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]

    if "clip" in path_parts:
        clip_index = path_parts.index("clip")
        if clip_index + 1 < len(path_parts):
            return safe_filename(path_parts[clip_index + 1])

    if path_parts:
        return safe_filename(path_parts[-1])

    return safe_filename(parsed.netloc, fallback="clip")

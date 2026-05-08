"""Small shared helpers for local paths, filenames, and timestamps."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
HTTP_URL_SCHEMES = frozenset({"http", "https"})


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


def is_http_url(value: str) -> bool:
    """Return whether a value is an absolute HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in HTTP_URL_SCHEMES and bool(parsed.netloc)


def normalized_host(netloc: str) -> str:
    """Normalize a URL netloc for host comparisons."""
    host = netloc.lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    return host


def redact_secrets(text: str, *, secrets: tuple[str, ...]) -> str:
    """Replace secret values in a text fragment with a stable marker."""
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def response_text_excerpt(
    text: str,
    *,
    secrets: tuple[str, ...] = (),
    limit: int = 240,
    empty_fallback: str | None = None,
) -> str:
    """Return a compact, redacted response body excerpt for error messages."""
    cleaned = " ".join(redact_secrets(text, secrets=secrets).split())
    if not cleaned:
        return empty_fallback or ""
    if len(cleaned) > limit:
        return f"{cleaned[:limit].rstrip()}..."
    return cleaned


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


def twitch_clip_slug_from_url(url: str) -> str:
    """Extract a Twitch clip slug from a supported Twitch clip URL."""
    parsed = urlparse(url)
    host = normalized_host(parsed.netloc)

    if parsed.scheme not in HTTP_URL_SCHEMES:
        raise ValueError("Twitch clip URL must use http or https.")

    path_parts = [part for part in parsed.path.split("/") if part]
    if host == "clips.twitch.tv" and path_parts:
        return safe_filename(path_parts[0])

    if host == "twitch.tv" and "clip" in path_parts:
        clip_index = path_parts.index("clip")
        if clip_index + 1 < len(path_parts):
            return safe_filename(path_parts[clip_index + 1])

    raise ValueError(
        "Unsupported Twitch clip URL. Expected clips.twitch.tv/<slug> "
        "or twitch.tv/<channel>/clip/<slug>."
    )

"""Shared artifact path builders."""

from __future__ import annotations

from pathlib import Path

from clipforge.core.config import ClipforgeConfig
from clipforge.utils.paths import safe_filename


MAX_TITLE_PATH_PART_LENGTH = 72
MAX_READY_EXPORT_TITLE_LENGTH = 80


def sanitize_path_part(
    value: str | None,
    *,
    fallback: str = "untitled",
    max_length: int | None = MAX_TITLE_PATH_PART_LENGTH,
) -> str:
    """Return a deterministic filesystem-safe path component."""

    cleaned = safe_filename(value or "", fallback=fallback)
    if max_length is None or len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip("._-") or fallback


def clip_folder_name(title: str | None, clip_id: str) -> str:
    """Return the human-facing export folder name for one clip."""

    safe_title = sanitize_path_part(title, fallback="untitled")
    safe_clip_id = sanitize_path_part(clip_id, fallback="clip", max_length=None)
    return f"{safe_title}__{safe_clip_id}"


def download_dir(config: ClipforgeConfig, *, clip_id: str, engine: str) -> Path:
    """Return the backend-scoped source download directory for one clip."""

    return config.downloads_dir / _artifact_id(clip_id) / _engine(engine)


def backend_download_dir(downloads_dir: Path, *, clip_id: str, backend: str) -> Path:
    """Return the backend-scoped source download directory for one clip."""

    return downloads_dir / _artifact_id(clip_id) / _engine(backend)


def render_path(
    config: ClipforgeConfig,
    *,
    streamer: str | None,
    clip_id: str,
    engine: str,
    layout: str,
) -> Path:
    """Return the render candidate path for one layout."""

    return (
        config.renders_dir
        / sanitize_path_part(streamer, fallback="unknown_streamer")
        / _artifact_id(clip_id)
        / _engine(engine)
        / f"{_layout(layout)}.{config.output_format}"
    )


def render_preview_path(
    config: ClipforgeConfig,
    *,
    streamer: str | None,
    clip_id: str,
    engine: str,
    layout: str,
    width: int,
    height: int,
) -> Path:
    """Return the review-resolution render candidate path for one layout."""

    return (
        render_path(
            config,
            streamer=streamer,
            clip_id=clip_id,
            engine=engine,
            layout=layout,
        ).parent
        / f"preview_{width}x{height}"
        / f"{_layout(layout)}.{config.output_format}"
    )


def export_path(
    config: ClipforgeConfig,
    *,
    streamer: str,
    title: str | None,
    clip_id: str,
    layout: str,
) -> Path:
    """Return the final human-facing export path for one selected layout."""

    return (
        config.exports_dir
        / sanitize_path_part(streamer, fallback="unknown_streamer")
        / clip_folder_name(title, clip_id)
        / f"{_layout(layout)}.{config.output_format}"
    )


def ready_export_path(
    config: ClipforgeConfig,
    *,
    streamer: str,
    title: str | None = None,
    clip_id: str,
    layout: str,
) -> Path:
    """Return the phone-review ready export path for one selected candidate."""

    return (
        config.exports_dir
        / "ready"
        / sanitize_path_part(streamer, fallback="unknown_streamer")
        / _artifact_id(clip_id)
        / ready_export_filename(
            title=title,
            clip_id=clip_id,
            layout=layout,
            extension=config.output_format,
        )
    )


def ready_export_filename(
    *,
    title: str | None,
    clip_id: str,
    layout: str,
    extension: str,
) -> str:
    """Return the phone-download filename for one selected render candidate."""

    safe_title = sanitize_path_part(
        title,
        fallback="clip",
        max_length=MAX_READY_EXPORT_TITLE_LENGTH,
    )
    stem = f"{safe_title}-{_artifact_id(clip_id)}"
    return f"{stem}.{_extension(extension)}"


def _artifact_id(value: str) -> str:
    return sanitize_path_part(value, fallback="clip", max_length=None)


def _engine(value: str) -> str:
    return sanitize_path_part(value, fallback="unknown_engine", max_length=None)


def _layout(value: str) -> str:
    return sanitize_path_part(value, fallback="layout", max_length=None)


def _extension(value: str) -> str:
    return sanitize_path_part(value.lstrip("."), fallback="mp4", max_length=None)

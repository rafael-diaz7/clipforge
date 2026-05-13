"""Reusable artifact readiness and reuse helpers for pipeline stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from clipforge.core.config import ClipforgeConfig
from clipforge.media.analyze import FRAME_SAMPLE_EXTENSION, sample_frames, sample_timestamps
from clipforge.media.captions import (
    caption_metadata_path as deterministic_caption_metadata_path,
    generate_caption_metadata,
)
from clipforge.media.download import DownloadResult, download_twitch_clip
from clipforge.media.layouts import (
    GENERATED_LAYOUT_NAMES,
    generate_detected_layout_candidates,
    load_layout,
)
from clipforge.media.overlay import analyze_overlay
from clipforge.storage.paths import backend_download_dir
from clipforge.utils.paths import clip_analysis_dir, safe_filename


LOGGER = logging.getLogger("clipforge.pipeline.artifact_reuse")


def ensure_source_video(
    twitch_clip_url: str,
    *,
    clip_id: str,
    clip_state,
    prefer_existing: bool,
    config: ClipforgeConfig,
    on_media_url_resolved: Callable[[str], None] | None = None,
) -> tuple[DownloadResult, bool]:
    if prefer_existing:
        existing = existing_download_result(
            clip_id=clip_id,
            clip_state=clip_state,
            config=config,
        )
        if existing is not None:
            LOGGER.info(
                "Reusing existing source video for clip %s from %s.",
                clip_id,
                existing.source_path,
            )
            return existing, True

    result = download_twitch_clip(
        twitch_clip_url,
        clip_id=clip_id,
        config=config,
        on_media_url_resolved=on_media_url_resolved,
    )
    return result, False


def existing_download_result(
    *,
    clip_id: str,
    clip_state,
    config: ClipforgeConfig,
) -> DownloadResult | None:
    state_download_path = getattr(clip_state, "download_path", None)
    if state_download_path:
        source_path = Path(state_download_path)
        if source_path.is_file():
            return DownloadResult(
                source_path=source_path,
                backend=backend_from_source_path(source_path, clip_id=clip_id, config=config),
            )

    state_metadata_path = getattr(clip_state, "metadata_path", None)
    if state_metadata_path:
        metadata_result = download_result_from_metadata(Path(state_metadata_path))
        if metadata_result is not None:
            return metadata_result

    configured_backend = config.require_downloader_backend()
    configured_dir = backend_download_dir(
        config.downloads_dir,
        clip_id=clip_id,
        backend=configured_backend,
    )
    source_path = first_video_file(configured_dir)
    if source_path is not None:
        return DownloadResult(source_path=source_path, backend=configured_backend)

    clip_download_dir = config.downloads_dir / safe_filename(clip_id)
    if not clip_download_dir.is_dir():
        return None
    for backend_dir in sorted(path for path in clip_download_dir.iterdir() if path.is_dir()):
        source_path = first_video_file(backend_dir)
        if source_path is not None:
            return DownloadResult(source_path=source_path, backend=backend_dir.name)
    return None


def download_result_from_metadata(metadata_path: Path) -> DownloadResult | None:
    payload = read_json_object_if_available(metadata_path)
    if payload is None:
        return None
    source_value = payload.get("source_path")
    if not isinstance(source_value, str):
        return None
    source_path = Path(source_value)
    if not source_path.is_file():
        return None
    backend = payload.get("downloader_backend")
    return DownloadResult(
        source_path=source_path,
        backend=backend if isinstance(backend, str) and backend else source_path.parent.name,
        media_url=payload.get("download_media_url")
        if isinstance(payload.get("download_media_url"), str)
        else None,
    )


def first_video_file(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            return path
    return None


def backend_from_source_path(
    source_path: Path,
    *,
    clip_id: str,
    config: ClipforgeConfig,
) -> str:
    safe_clip_id = safe_filename(clip_id)
    try:
        if source_path.parent.parent.name == safe_clip_id:
            return source_path.parent.name
    except IndexError:
        pass
    return config.require_downloader_backend()


def ensure_caption_metadata(
    source_path: Path,
    *,
    clip_id: str,
    force_captions: bool,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    existing_caption_metadata_path = deterministic_caption_metadata_path(
        clip_id,
        config=config,
    )
    if existing_caption_metadata_path.exists() and not force_captions:
        LOGGER.info(
            "Reusing existing caption metadata for clip %s from %s.",
            clip_id,
            existing_caption_metadata_path,
        )
        return existing_caption_metadata_path, True

    if force_captions and existing_caption_metadata_path.exists():
        LOGGER.info("Regenerating captions for clip %s from %s.", clip_id, source_path)
    else:
        LOGGER.info("Generating captions for clip %s from %s.", clip_id, source_path)
    return generate_caption_metadata(source_path, clip_id=clip_id, config=config), False


def require_existing_caption_metadata(
    clip_id: str,
    *,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    existing_caption_metadata_path = deterministic_caption_metadata_path(
        clip_id,
        config=config,
    )
    if existing_caption_metadata_path.exists():
        LOGGER.info(
            "Reusing existing caption metadata for rerender of clip %s from %s.",
            clip_id,
            existing_caption_metadata_path,
        )
        return existing_caption_metadata_path, True

    raise RuntimeError(
        "Captions are missing and rerender mode does not regenerate transcriptions. "
        "Run with --generate-captions first."
    )


def ensure_sampled_frames(
    source_path: Path,
    *,
    clip_id: str,
    duration_seconds: float | None,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    metadata_path = frames_metadata_path(clip_id, config=config)
    if not force and frames_artifacts_ready(
        metadata_path,
        expected_count=len(sample_timestamps()),
        expected_suffix=FRAME_SAMPLE_EXTENSION,
    ):
        LOGGER.info("Reusing sampled frames for clip %s from %s.", clip_id, metadata_path)
        return metadata_path, True

    LOGGER.info("Sampling analysis frames for clip %s from %s.", clip_id, source_path)
    return (
        sample_frames(
            source_path,
            clip_id=clip_id,
            analysis_dir=config.analysis_dir,
            duration_seconds=duration_seconds,
        ),
        False,
    )


def ensure_overlay_analysis(
    *,
    clip_id: str,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    path = overlay_path(clip_id, config=config)
    if not force and json_object_file_ready(path):
        LOGGER.info("Reusing overlay analysis for clip %s from %s.", clip_id, path)
        return path, True

    LOGGER.info("Analyzing overlay for clip %s.", clip_id)
    return analyze_overlay(clip_id=clip_id, analysis_dir=config.analysis_dir), False


def ensure_layout_analysis(
    *,
    clip_id: str,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[tuple[Path, ...], bool]:
    existing_layout_paths = generated_layout_paths(clip_id, config=config)
    if not force and layout_artifacts_ready(existing_layout_paths):
        LOGGER.info("Reusing generated layouts for clip %s.", clip_id)
        return existing_layout_paths, True

    LOGGER.info("Generating detected layouts for clip %s.", clip_id)
    generated_paths = generate_detected_layout_candidates(
        clip_id=clip_id,
        analysis_dir=config.analysis_dir,
        example_layouts_dir=config.example_layouts_dir,
    )
    return generated_paths, False


def frames_metadata_path(clip_id: str, *, config: ClipforgeConfig) -> Path:
    return clip_analysis_path(clip_id, config=config) / "frames.json"


def overlay_path(clip_id: str, *, config: ClipforgeConfig) -> Path:
    return clip_analysis_path(clip_id, config=config) / "overlay.json"


def generated_layout_paths(clip_id: str, *, config: ClipforgeConfig) -> tuple[Path, ...]:
    layouts_dir = clip_analysis_path(clip_id, config=config) / "layouts"
    return tuple(layouts_dir / f"{layout_name}.json" for layout_name in GENERATED_LAYOUT_NAMES)


def clip_analysis_path(clip_id: str, *, config: ClipforgeConfig) -> Path:
    return clip_analysis_dir(config.analysis_dir, clip_id)


def frames_artifacts_ready(
    metadata_path: Path,
    *,
    expected_count: int,
    expected_suffix: str,
) -> bool:
    payload = read_json_object_if_available(metadata_path)
    if payload is None:
        return False
    frame_paths = payload.get("frame_paths")
    if not isinstance(frame_paths, list) or not frame_paths:
        return False
    if len(frame_paths) != expected_count:
        return False
    sampled_timestamps = payload.get("sampled_timestamps")
    if not isinstance(sampled_timestamps, list) or len(sampled_timestamps) != expected_count:
        return False
    return all(
        isinstance(value, str)
        and Path(value).suffix.lower() == expected_suffix
        and Path(value).is_file()
        for value in frame_paths
    )


def json_object_file_ready(path: Path) -> bool:
    return read_json_object_if_available(path) is not None


def layout_artifacts_ready(paths: tuple[Path, ...]) -> bool:
    if not paths:
        return False
    for path in paths:
        if not path.is_file():
            return False
        try:
            load_layout(path)
        except Exception:
            return False
    return True


def read_json_object_if_available(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload

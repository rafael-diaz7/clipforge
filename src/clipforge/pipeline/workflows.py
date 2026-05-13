"""Reusable pipeline workflows."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.utils.paths import (
    clip_analysis_dir,
    ensure_directory,
    safe_filename,
    twitch_clip_slug_from_url,
)
from clipforge.integrations.clipr import CliprClient
from clipforge.media.download import (
    DownloadResult,
    backend_download_dir,
    download_clip,
    download_twitch_clip,
)
from clipforge.media.captions import (
    CaptionMetadata,
    caption_metadata_path as deterministic_caption_metadata_path,
    generate_caption_metadata,
    load_caption_metadata,
)
from clipforge.media.analyze import FRAME_SAMPLE_EXTENSION, sample_frames, sample_timestamps
from clipforge.media.layouts import (
    DEFAULT_LAYOUT_NAMES,
    GENERATED_LAYOUT_NAMES,
    Layout,
    OutputSize,
    generate_detected_layout_candidates,
    load_example_layouts,
    load_example_layout,
    load_layout,
    parse_layout,
)
from clipforge.media.overlay import analyze_overlay
from clipforge.media.render import (
    CaptionStyle,
    CaptionVerticalSafeArea,
    Watermark,
    load_streamer_watermark,
    render_layout,
)
from clipforge.media.render_settings import (
    DEFAULT_FFMPEG_RENDER_SETTINGS,
    FFmpegRenderSettings,
)
from clipforge.pipeline.artifacts import write_metadata
from clipforge.pipeline.state_sync import record_rendered_clip
from clipforge.storage.state import get_clip


LOGGER = logging.getLogger("clipforge.pipeline.workflows")

GENERATED_LAYOUT_REPLACEMENTS = {
    "facecam_focus": "detected_streamer_focus",
    "hybrid": "detected_hybrid",
    "hybrid_full_game_bottom": "detected_hybrid_full_game_bottom",
}


class ClipProcessingError(RuntimeError):
    """Raised when a named processing stage fails."""


def resolve_download_url(twitch_clip_url: str, *, config: ClipforgeConfig | None = None) -> str:
    """Resolve a Twitch clip URL to a direct downloadable media URL."""

    config = config or load_config()
    LOGGER.info("Resolving Twitch clip URL with Clipr.")
    return CliprClient.from_config(config).get_download_url(twitch_clip_url)


def download_media_url(
    media_url: str,
    *,
    clip_id: str | None = None,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Download a direct media URL into the configured downloads directory."""

    config = config or load_config()
    LOGGER.info("Downloading clip media to %s.", config.downloads_dir)
    return download_clip(
        media_url,
        downloads_dir=config.downloads_dir,
        filename_stem=clip_id,
    )


def render_candidate(
    source_path: Path,
    *,
    layout_ref: str,
    clip_id: str | None = None,
    caption_metadata_path: Path | None = None,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Render one local source clip with one example layout or layout file."""

    config = config or load_config()
    layout = _load_layout_ref(layout_ref, config=config)
    preview_output_size = _review_output_size_for_layout(layout, config=config)
    preview_layout = _layout_with_output_size(layout, preview_output_size)
    output_path = _render_output_path(
        source_path,
        layout,
        clip_id=clip_id,
        output_size=preview_output_size,
        config=config,
    )
    caption_metadata = _load_optional_caption_metadata(caption_metadata_path)
    caption_style = _caption_style_from_config(config)
    LOGGER.info(
        "Rendering layout %s preview at %sx%s to %s.",
        layout.name,
        preview_output_size.width,
        preview_output_size.height,
        output_path,
    )
    return _render_layout_with_optional_captions(
        source_path,
        output_path,
        preview_layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=None,
        review=True,
        style_scale=_output_scale(layout.output, preview_output_size),
        config=config,
    )


def render_all_candidates(
    source_path: Path,
    *,
    clip_id: str | None = None,
    caption_metadata_path: Path | None = None,
    use_generated_layouts: bool = True,
    config: ClipforgeConfig | None = None,
) -> tuple[Path, ...]:
    """Render the default candidate layouts for one source clip."""

    config = config or load_config()
    layouts = _candidate_layouts(
        clip_id=clip_id,
        use_generated_layouts=use_generated_layouts,
        config=config,
    )
    caption_metadata = _load_optional_caption_metadata(caption_metadata_path)
    caption_style = _caption_style_from_config(config)
    return tuple(
        _render_candidate_layout(
            source_path,
            layout,
            clip_id=clip_id,
            caption_metadata=caption_metadata,
            caption_style=caption_style,
            config=config,
        )
        for layout in layouts
    )


def process_clip(
    twitch_clip_url: str,
    *,
    generate_captions: bool | None = None,
    force_captions: bool = False,
    force: bool = False,
    rerender: bool = False,
    channel: str | None = None,
    use_generated_layouts: bool = True,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Run the full MVP pipeline and return the metadata path."""

    config = config or load_config()
    clip_id = twitch_clip_slug_from_url(twitch_clip_url)
    clip_state = get_clip(clip_id, db_path=config.state_db_path)
    state_channel = clip_state.streamer_login if clip_state is not None else None
    channel = channel or state_channel
    force_visuals = force or rerender
    LOGGER.info("Starting clip pipeline for clip %s.", clip_id)
    caption_metadata_path = None
    reused_caption_metadata = False
    if rerender:
        caption_metadata_path, reused_caption_metadata = _run_stage(
            "captions",
            lambda: _require_existing_caption_metadata(clip_id, config=config),
        )

    download_result, reused_source = _run_stage(
        "download",
        lambda: _ensure_source_video(
            twitch_clip_url,
            clip_id=clip_id,
            clip_state=clip_state,
            prefer_existing=rerender,
            config=config,
        ),
    )
    source_path = download_result.source_path
    if not rerender and _should_generate_captions(generate_captions, config=config):
        caption_metadata_path, reused_caption_metadata = _run_stage(
            "captions",
            lambda: _ensure_caption_metadata(
                source_path,
                clip_id=clip_id,
                force_captions=force_captions,
                config=config,
            )
        )
    caption_metadata = _load_optional_caption_metadata(caption_metadata_path)
    caption_style = _caption_style_from_config(config)

    frames_metadata_path: Path | None = None
    reused_frames = False
    overlay_path: Path | None = None
    reused_overlay = False
    generated_layout_paths: tuple[Path, ...] = ()
    reused_layouts = False
    if use_generated_layouts:
        frames_metadata_path, reused_frames = _run_stage(
            "frames",
            lambda: _ensure_sampled_frames(
                source_path,
                clip_id=clip_id,
                duration_seconds=getattr(clip_state, "duration_seconds", None),
                force=force_visuals,
                config=config,
            ),
        )
        overlay_path, reused_overlay = _run_stage(
            "overlay",
            lambda: _ensure_overlay_analysis(
                clip_id=clip_id,
                force=force_visuals,
                config=config,
            ),
        )
        generated_layout_paths, reused_layouts = _run_stage(
            "layouts",
            lambda: _ensure_layout_analysis(
                clip_id=clip_id,
                force=force_visuals,
                config=config,
            ),
        )

    layouts = _run_stage(
        "layouts",
        lambda: _candidate_layouts(
            clip_id=clip_id,
            use_generated_layouts=use_generated_layouts,
            config=config,
        ),
    )

    outputs = []
    for layout in layouts:
        output_path, _reused_render = _run_stage(
            "renders",
            lambda layout=layout: _ensure_rendered_candidate_layout(
                source_path,
                layout,
                clip_id=clip_id,
                backend=download_result.backend,
                channel=channel,
                caption_metadata=caption_metadata,
                caption_style=caption_style,
                force=force_visuals,
                config=config,
            ),
        )
        preview_output_size = _review_output_size_for_layout(layout, config=config)
        outputs.append(
            {
                "layout": layout.name,
                "path": str(output_path),
                "resolution": _output_size_payload(preview_output_size),
                "render_profile": "review",
                "render_settings": _render_settings_payload(
                    config.render_settings_for(review=True)
                ),
            }
        )

    metadata_path = _run_stage(
        "metadata",
        lambda: write_metadata(
            clip_id=clip_id,
            twitch_clip_url=twitch_clip_url,
            download_result=download_result,
            source_path=source_path,
            layouts=layouts,
            outputs=outputs,
            config=config,
            caption_metadata_path=caption_metadata_path,
        ),
    )
    _run_stage(
        "metadata",
        lambda: record_rendered_clip(
            clip_id=clip_id,
            twitch_clip_url=twitch_clip_url,
            render_dir=Path(outputs[0]["path"]).parent,
            metadata_path=metadata_path,
            config=config,
        ),
    )

    if rerender:
        print("rerender: regenerating visual artifacts; preserving source and captions")
    source_prefix = "source: reusing existing" if reused_source else "source:"
    print(f"{source_prefix} {source_path}")
    if caption_metadata_path is not None:
        if reused_caption_metadata:
            print(f"captions: reusing existing {caption_metadata_path}")
        else:
            print(f"captions: {caption_metadata_path}")
    if frames_metadata_path is not None:
        prefix = "frames: reusing existing" if reused_frames else "frames:"
        print(f"{prefix} {frames_metadata_path}")
    if overlay_path is not None:
        prefix = "overlay: reusing existing" if reused_overlay else "overlay:"
        print(f"{prefix} {overlay_path}")
    for layout_path in generated_layout_paths:
        prefix = "layout: reusing existing" if reused_layouts else "layout:"
        print(f"{prefix} {layout_path}")
    for output in outputs:
        print(f"{output['layout']}: {output['path']}")
    print(f"metadata: {metadata_path}")
    return metadata_path


def _run_stage(stage_name: str, callback):
    try:
        return callback()
    except ClipProcessingError:
        raise
    except Exception as exc:
        raise ClipProcessingError(f"{stage_name} stage failed: {exc}") from exc


def _ensure_source_video(
    twitch_clip_url: str,
    *,
    clip_id: str,
    clip_state,
    prefer_existing: bool,
    config: ClipforgeConfig,
) -> tuple[DownloadResult, bool]:
    if prefer_existing:
        existing = _existing_download_result(
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
        on_media_url_resolved=lambda media_url: print(f"download_url: {media_url}"),
    )
    return result, False


def _existing_download_result(
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
                backend=_backend_from_source_path(source_path, clip_id=clip_id, config=config),
            )

    state_metadata_path = getattr(clip_state, "metadata_path", None)
    if state_metadata_path:
        metadata_result = _download_result_from_metadata(Path(state_metadata_path))
        if metadata_result is not None:
            return metadata_result

    configured_backend = config.require_downloader_backend()
    configured_dir = backend_download_dir(
        config.downloads_dir,
        clip_id=clip_id,
        backend=configured_backend,
    )
    source_path = _first_video_file(configured_dir)
    if source_path is not None:
        return DownloadResult(source_path=source_path, backend=configured_backend)

    clip_download_dir = config.downloads_dir / safe_filename(clip_id)
    if not clip_download_dir.is_dir():
        return None
    for backend_dir in sorted(path for path in clip_download_dir.iterdir() if path.is_dir()):
        source_path = _first_video_file(backend_dir)
        if source_path is not None:
            return DownloadResult(source_path=source_path, backend=backend_dir.name)
    return None


def _download_result_from_metadata(metadata_path: Path) -> DownloadResult | None:
    payload = _read_json_object_if_available(metadata_path)
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


def _first_video_file(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            return path
    return None


def _backend_from_source_path(
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


def _ensure_caption_metadata(
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


def _require_existing_caption_metadata(
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

    raise ClipProcessingError(
        "Captions are missing and rerender mode does not regenerate transcriptions. "
        "Run with --generate-captions first."
    )


def _ensure_sampled_frames(
    source_path: Path,
    *,
    clip_id: str,
    duration_seconds: float | None,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    metadata_path = _frames_metadata_path(clip_id, config=config)
    if not force and _frames_artifacts_ready(
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


def _ensure_overlay_analysis(
    *,
    clip_id: str,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    overlay_path = _overlay_path(clip_id, config=config)
    if not force and _json_object_file_ready(overlay_path):
        LOGGER.info("Reusing overlay analysis for clip %s from %s.", clip_id, overlay_path)
        return overlay_path, True

    LOGGER.info("Analyzing overlay for clip %s.", clip_id)
    return analyze_overlay(clip_id=clip_id, analysis_dir=config.analysis_dir), False


def _ensure_layout_analysis(
    *,
    clip_id: str,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[tuple[Path, ...], bool]:
    existing_layout_paths = _generated_layout_paths(clip_id, config=config)
    if not force and _layout_artifacts_ready(existing_layout_paths):
        LOGGER.info("Reusing generated layouts for clip %s.", clip_id)
        return existing_layout_paths, True

    LOGGER.info("Generating detected layouts for clip %s.", clip_id)
    generated_paths = generate_detected_layout_candidates(
        clip_id=clip_id,
        analysis_dir=config.analysis_dir,
        example_layouts_dir=config.example_layouts_dir,
    )
    return generated_paths, False


def _ensure_rendered_candidate_layout(
    source_path: Path,
    layout: Layout,
    *,
    clip_id: str | None,
    backend: str | None,
    channel: str | None,
    caption_metadata: CaptionMetadata | None,
    caption_style: CaptionStyle,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    preview_output_size = _review_output_size_for_layout(layout, config=config)
    preview_layout = _layout_with_output_size(layout, preview_output_size)
    output_path = _render_output_path(
        source_path,
        layout,
        clip_id=clip_id,
        backend=backend,
        channel=channel,
        output_size=preview_output_size,
        config=config,
    )
    if output_path.is_file() and not force:
        LOGGER.info("Reusing rendered layout %s from %s.", layout.name, output_path)
        return output_path, True

    LOGGER.info(
        "Rendering layout %s preview at %sx%s to %s.",
        layout.name,
        preview_output_size.width,
        preview_output_size.height,
        output_path,
    )
    return (
        _render_layout_with_optional_captions(
            source_path,
            output_path,
            preview_layout,
            caption_metadata=caption_metadata,
            caption_style=caption_style,
            channel=channel,
            review=True,
            style_scale=_output_scale(layout.output, preview_output_size),
            config=config,
        ),
        False,
    )


def _frames_metadata_path(clip_id: str, *, config: ClipforgeConfig) -> Path:
    return _clip_analysis_dir(clip_id, config=config) / "frames.json"


def _overlay_path(clip_id: str, *, config: ClipforgeConfig) -> Path:
    return _clip_analysis_dir(clip_id, config=config) / "overlay.json"


def _generated_layout_paths(clip_id: str, *, config: ClipforgeConfig) -> tuple[Path, ...]:
    layouts_dir = _clip_analysis_dir(clip_id, config=config) / "layouts"
    return tuple(layouts_dir / f"{layout_name}.json" for layout_name in GENERATED_LAYOUT_NAMES)


def _clip_analysis_dir(clip_id: str, *, config: ClipforgeConfig) -> Path:
    return clip_analysis_dir(config.analysis_dir, clip_id)


def _frames_artifacts_ready(
    metadata_path: Path,
    *,
    expected_count: int,
    expected_suffix: str,
) -> bool:
    payload = _read_json_object_if_available(metadata_path)
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


def _json_object_file_ready(path: Path) -> bool:
    return _read_json_object_if_available(path) is not None


def _layout_artifacts_ready(paths: tuple[Path, ...]) -> bool:
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


def _read_json_object_if_available(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _should_generate_captions(
    generate_captions: bool | None,
    *,
    config: ClipforgeConfig,
) -> bool:
    if generate_captions is not None:
        return generate_captions
    return config.generate_captions


def _load_layout_ref(layout_ref: str, *, config: ClipforgeConfig) -> Layout:
    path = Path(layout_ref)
    if path.suffix.lower() == ".json" or path.exists():
        LOGGER.info("Loading layout from %s.", path)
        return load_layout(path)

    layout_path = config.example_layouts_dir / f"{layout_ref}.json"
    LOGGER.info("Loading example layout %s from %s.", layout_ref, layout_path)
    return load_layout(layout_path)


def _candidate_layouts(
    *,
    clip_id: str | None,
    use_generated_layouts: bool,
    config: ClipforgeConfig,
) -> tuple[Layout, ...]:
    if not use_generated_layouts or clip_id is None:
        return load_example_layouts(
            DEFAULT_LAYOUT_NAMES,
            layouts_dir=config.example_layouts_dir,
        )

    generated_layouts_dir = _clip_analysis_dir(clip_id, config=config) / "layouts"
    layouts: list[Layout] = []
    for layout_name in DEFAULT_LAYOUT_NAMES:
        generated_name = GENERATED_LAYOUT_REPLACEMENTS.get(layout_name)
        generated_path = (
            generated_layouts_dir / f"{generated_name}.json"
            if generated_name is not None
            else None
        )
        if generated_path is not None and generated_path.is_file():
            LOGGER.info(
                "Using generated layout %s instead of static %s.",
                generated_path,
                layout_name,
            )
            layouts.append(load_layout(generated_path))
        else:
            layouts.append(
                load_example_layout(layout_name, layouts_dir=config.example_layouts_dir)
            )
    return tuple(layouts)


def _render_output_path(
    source_path: Path,
    layout: Layout,
    *,
    clip_id: str | None,
    backend: str | None = None,
    channel: str | None = None,
    output_size: OutputSize | None = None,
    config: ClipforgeConfig,
) -> Path:
    stem = clip_id or source_path.stem
    if backend is not None:
        output_dir = config.renders_dir
        if channel:
            output_dir = output_dir / safe_filename(channel)
        output_dir = ensure_directory(
            output_dir / safe_filename(stem) / safe_filename(backend)
        )
        if output_size is not None and output_size != layout.output:
            output_dir = ensure_directory(
                output_dir / f"preview_{output_size.width}x{output_size.height}"
            )
        return output_dir / f"{layout.name}.{config.output_format}"

    output_dir = ensure_directory(config.renders_dir)
    if output_size is not None and output_size != layout.output:
        return (
            output_dir
            / f"{stem}_{layout.name}_{output_size.width}x{output_size.height}."
            f"{config.output_format}"
        )
    return output_dir / f"{stem}_{layout.name}.{config.output_format}"


def _render_candidate_layout(
    source_path: Path,
    layout: Layout,
    *,
    clip_id: str | None,
    backend: str | None = None,
    channel: str | None = None,
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = CaptionStyle(),
    config: ClipforgeConfig,
) -> Path:
    preview_output_size = _review_output_size_for_layout(layout, config=config)
    preview_layout = _layout_with_output_size(layout, preview_output_size)
    output_path = _render_output_path(
        source_path,
        layout,
        clip_id=clip_id,
        backend=backend,
        channel=channel,
        output_size=preview_output_size,
        config=config,
    )
    LOGGER.info(
        "Rendering layout %s preview at %sx%s to %s.",
        layout.name,
        preview_output_size.width,
        preview_output_size.height,
        output_path,
    )
    return _render_layout_with_optional_captions(
        source_path,
        output_path,
        preview_layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=channel,
        review=True,
        style_scale=_output_scale(layout.output, preview_output_size),
        config=config,
    )


def _load_optional_caption_metadata(path: Path | None) -> CaptionMetadata | None:
    if path is None:
        return None
    return load_caption_metadata(path)


def _render_layout_with_optional_captions(
    source_path: Path,
    output_path: Path,
    layout: Layout,
    *,
    caption_metadata: CaptionMetadata | None,
    caption_style: CaptionStyle,
    channel: str | None,
    review: bool,
    style_scale: float = 1.0,
    config: ClipforgeConfig,
) -> Path:
    watermark = _scale_watermark(
        load_streamer_watermark(channel, base_dir=config.project_root),
        style_scale,
    )
    caption_style = _scale_caption_style(caption_style, style_scale)
    render_kwargs = {}
    if watermark is not None:
        render_kwargs["watermark"] = watermark
    render_settings = config.render_settings_for(review=review)
    if render_settings != DEFAULT_FFMPEG_RENDER_SETTINGS:
        render_kwargs["render_settings"] = render_settings

    if caption_metadata is None:
        return render_layout(source_path, output_path, layout, **render_kwargs)
    if caption_style == CaptionStyle():
        return render_layout(
            source_path,
            output_path,
            layout,
            caption_metadata=caption_metadata,
            caption_renderer_backend=config.require_caption_renderer_backend(),
            ass_temp_dir=config.ass_temp_dir,
            **render_kwargs,
        )
    return render_layout(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        caption_renderer_backend=config.require_caption_renderer_backend(),
        ass_temp_dir=config.ass_temp_dir,
        **render_kwargs,
    )


def render_selected_layout_from_metadata(
    metadata_path: Path,
    *,
    selected_layout: str,
    output_path: Path,
    channel: str | None,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Render one selected layout from pipeline metadata with final settings."""

    config = config or load_config()
    payload = _read_pipeline_metadata(metadata_path)
    source_path = _metadata_source_path(payload, metadata_path=metadata_path)
    layout = _metadata_layout(payload, selected_layout=selected_layout)
    caption_metadata = _load_optional_caption_metadata(
        _metadata_optional_path(payload, "caption_metadata_path")
    )
    caption_style = _caption_style_from_config(config)
    ensure_directory(output_path.parent)
    LOGGER.info(
        "Rendering selected layout %s at final size %sx%s to %s.",
        layout.name,
        layout.output.width,
        layout.output.height,
        output_path,
    )
    return _render_layout_with_optional_captions(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=channel,
        review=False,
        config=config,
    )


def _caption_style_from_config(config: ClipforgeConfig) -> CaptionStyle:
    if (
        config.caption_font_file is None
        and config.caption_font_fallbacks == CaptionStyle().font_fallbacks
    ):
        return CaptionStyle()
    return CaptionStyle(
        font_file=config.caption_font_file,
        font_fallbacks=config.caption_font_fallbacks,
    )


def _review_output_size_for_layout(
    layout: Layout,
    *,
    config: ClipforgeConfig,
) -> OutputSize:
    width, height = config.review_resolution_for(
        width=layout.output.width,
        height=layout.output.height,
    )
    return OutputSize(width=width, height=height)


def _layout_with_output_size(layout: Layout, output_size: OutputSize) -> Layout:
    if layout.output == output_size:
        return layout
    return replace(layout, output=output_size)


def _output_scale(normal_output_size: OutputSize, output_size: OutputSize) -> float:
    if normal_output_size.width <= 0:
        return 1.0
    return output_size.width / normal_output_size.width


def _scale_caption_style(caption_style: CaptionStyle, scale: float) -> CaptionStyle:
    if scale == 1.0:
        return caption_style
    vertical_safe_area = caption_style.vertical_safe_area
    if vertical_safe_area is not None:
        vertical_safe_area = CaptionVerticalSafeArea(
            top=_scale_int(vertical_safe_area.top, scale),
            bottom=_scale_int(vertical_safe_area.bottom, scale),
            center=vertical_safe_area.center,
        )
    return replace(
        caption_style,
        font_size=_scale_int(caption_style.font_size, scale),
        box_border_width=_scale_int(caption_style.box_border_width, scale),
        line_spacing=_scale_int(caption_style.line_spacing, scale),
        outline_width=_scale_int(caption_style.outline_width, scale),
        outline_thickness=_scale_optional_int(caption_style.outline_thickness, scale),
        shadow_offset=_scale_int(caption_style.shadow_offset, scale),
        shadow_strength=_scale_optional_int(caption_style.shadow_strength, scale),
        safe_margin_x=_scale_int(caption_style.safe_margin_x, scale),
        safe_margin_bottom=_scale_int(caption_style.safe_margin_bottom, scale),
        vertical_safe_area=vertical_safe_area,
    )


def _scale_watermark(watermark: Watermark | None, scale: float) -> Watermark | None:
    if watermark is None or scale == 1.0:
        return watermark
    return replace(
        watermark,
        native_width=_scale_int(watermark.native_width, scale),
        native_height=_scale_int(watermark.native_height, scale),
        margin=_scale_int(watermark.margin, scale),
    )


def _scale_optional_int(value: int | None, scale: float) -> int | None:
    if value is None:
        return None
    return _scale_int(value, scale)


def _scale_int(value: int, scale: float) -> int:
    if value == 0:
        return 0
    return max(1, round(value * scale))


def _output_size_payload(output_size: OutputSize) -> dict[str, int]:
    return {"width": output_size.width, "height": output_size.height}


def _render_settings_payload(
    render_settings: FFmpegRenderSettings,
) -> dict[str, object]:
    return asdict(render_settings.normalized())


def _read_pipeline_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClipProcessingError(
            f"Could not read pipeline metadata {metadata_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ClipProcessingError(f"Pipeline metadata is not valid JSON: {metadata_path}") from exc
    if not isinstance(payload, dict):
        raise ClipProcessingError(f"Pipeline metadata must be a JSON object: {metadata_path}")
    return payload


def _metadata_source_path(payload: dict[str, Any], *, metadata_path: Path) -> Path:
    value = payload.get("source_path")
    if not isinstance(value, str) or not value:
        raise ClipProcessingError(f"Pipeline metadata is missing source_path: {metadata_path}")
    return Path(value)


def _metadata_layout(payload: dict[str, Any], *, selected_layout: str) -> Layout:
    layouts = payload.get("layouts")
    if not isinstance(layouts, list):
        raise ClipProcessingError("Pipeline metadata is missing layouts.")
    for layout_payload in layouts:
        if not isinstance(layout_payload, dict):
            continue
        if layout_payload.get("name") == selected_layout:
            return parse_layout(layout_payload)
    raise ClipProcessingError(
        f"Pipeline metadata does not contain selected layout: {selected_layout}."
    )


def _metadata_optional_path(payload: dict[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        return None
    return Path(value)

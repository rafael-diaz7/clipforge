"""Reusable pipeline workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.utils.paths import (
    clip_analysis_dir,
    ensure_directory,
    safe_filename,
    twitch_clip_slug_from_url,
)
from clipforge.integrations.clipr import CliprClient
from clipforge.media.download import download_clip, download_twitch_clip
from clipforge.media.captions import (
    CaptionMetadata,
    caption_metadata_path as deterministic_caption_metadata_path,
    generate_caption_metadata,
    load_caption_metadata,
)
from clipforge.media.analyze import sample_frames
from clipforge.media.layouts import (
    DEFAULT_LAYOUT_NAMES,
    GENERATED_LAYOUT_NAMES,
    Layout,
    generate_detected_layout_candidates,
    load_example_layouts,
    load_example_layout,
    load_layout,
)
from clipforge.media.overlay import analyze_overlay
from clipforge.media.render import load_streamer_watermark, render_layout
from clipforge.media.render import CaptionStyle
from clipforge.pipeline.artifacts import write_metadata
from clipforge.pipeline.state_sync import record_rendered_clip
from clipforge.storage.state import get_clip


LOGGER = logging.getLogger("clipforge.pipeline.workflows")

GENERATED_LAYOUT_REPLACEMENTS = {
    "facecam_focus": "detected_streamer_focus",
    "hybrid": "detected_hybrid",
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
    output_path = _render_output_path(source_path, layout, clip_id=clip_id, config=config)
    caption_metadata = _load_optional_caption_metadata(caption_metadata_path)
    caption_style = _caption_style_from_config(config)
    LOGGER.info("Rendering layout %s to %s.", layout.name, output_path)
    return _render_layout_with_optional_captions(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=None,
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
    """Render the default MVP candidate layouts for one source clip."""

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
    use_generated_layouts: bool = True,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Run the full MVP pipeline and return the metadata path."""

    config = config or load_config()
    clip_id = twitch_clip_slug_from_url(twitch_clip_url)
    clip_state = get_clip(clip_id, db_path=config.state_db_path)
    channel = clip_state.streamer_login if clip_state is not None else None
    LOGGER.info("Starting clip pipeline for clip %s.", clip_id)
    download_result = _run_stage(
        "download",
        lambda: download_twitch_clip(
            twitch_clip_url,
            clip_id=clip_id,
            config=config,
            on_media_url_resolved=lambda media_url: print(f"download_url: {media_url}"),
        ),
    )
    source_path = download_result.source_path
    caption_metadata_path = None
    reused_caption_metadata = False
    if _should_generate_captions(generate_captions, config=config):
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
                force=force,
                config=config,
            ),
        )
        overlay_path, reused_overlay = _run_stage(
            "overlay",
            lambda: _ensure_overlay_analysis(
                clip_id=clip_id,
                force=force,
                config=config,
            ),
        )
        generated_layout_paths, reused_layouts = _run_stage(
            "layouts",
            lambda: _ensure_layout_analysis(
                clip_id=clip_id,
                force=force,
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
                force=force,
                config=config,
            ),
        )
        outputs.append({"layout": layout.name, "path": str(output_path)})

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

    print(f"source: {source_path}")
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


def _ensure_sampled_frames(
    source_path: Path,
    *,
    clip_id: str,
    force: bool,
    config: ClipforgeConfig,
) -> tuple[Path, bool]:
    metadata_path = _frames_metadata_path(clip_id, config=config)
    if not force and _frames_artifacts_ready(metadata_path):
        LOGGER.info("Reusing sampled frames for clip %s from %s.", clip_id, metadata_path)
        return metadata_path, True

    LOGGER.info("Sampling analysis frames for clip %s from %s.", clip_id, source_path)
    return (
        sample_frames(source_path, clip_id=clip_id, analysis_dir=config.analysis_dir),
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
    output_path = _render_output_path(
        source_path,
        layout,
        clip_id=clip_id,
        backend=backend,
        channel=channel,
        config=config,
    )
    if output_path.is_file() and not force:
        LOGGER.info("Reusing rendered layout %s from %s.", layout.name, output_path)
        return output_path, True

    LOGGER.info("Rendering layout %s to %s.", layout.name, output_path)
    return (
        _render_layout_with_optional_captions(
            source_path,
            output_path,
            layout,
            caption_metadata=caption_metadata,
            caption_style=caption_style,
            channel=channel,
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


def _frames_artifacts_ready(metadata_path: Path) -> bool:
    payload = _read_json_object_if_available(metadata_path)
    if payload is None:
        return False
    frame_paths = payload.get("frame_paths")
    if not isinstance(frame_paths, list) or not frame_paths:
        return False
    return all(isinstance(value, str) and Path(value).is_file() for value in frame_paths)


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
        return output_dir / f"{layout.name}.{config.output_format}"

    output_dir = ensure_directory(config.renders_dir)
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
    output_path = _render_output_path(
        source_path,
        layout,
        clip_id=clip_id,
        backend=backend,
        channel=channel,
        config=config,
    )
    LOGGER.info("Rendering layout %s to %s.", layout.name, output_path)
    return _render_layout_with_optional_captions(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=channel,
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
    config: ClipforgeConfig,
) -> Path:
    watermark = load_streamer_watermark(channel, base_dir=config.project_root)
    render_kwargs = {}
    if watermark is not None:
        render_kwargs["watermark"] = watermark

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

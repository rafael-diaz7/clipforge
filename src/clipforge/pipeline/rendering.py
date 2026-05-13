"""Render candidate and final output orchestration helpers."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.media.captions import CaptionMetadata, load_caption_metadata
from clipforge.media.caption_rendering import CaptionStyle, CaptionVerticalSafeArea
from clipforge.media.layouts import (
    DEFAULT_LAYOUT_NAMES,
    Layout,
    OutputSize,
    load_example_layout,
    load_example_layouts,
    load_layout,
)
from clipforge.media.render import Watermark, load_streamer_watermark, render_layout
from clipforge.media.render_settings import (
    DEFAULT_FFMPEG_RENDER_SETTINGS,
    FFmpegRenderSettings,
)
from clipforge.pipeline.artifact_reuse import clip_analysis_path
from clipforge.pipeline.metadata import (
    PipelineMetadataError,
    metadata_layout,
    metadata_optional_path,
    metadata_source_path,
    read_pipeline_metadata,
)
from clipforge.storage.paths import render_path, render_preview_path
from clipforge.utils.paths import ensure_directory


LOGGER = logging.getLogger("clipforge.pipeline.rendering")

GENERATED_LAYOUT_REPLACEMENTS = {
    "facecam_focus": "detected_streamer_focus",
    "hybrid": "detected_hybrid",
    "hybrid_full_game_bottom": "detected_hybrid_full_game_bottom",
}


class RenderSelectionError(RuntimeError):
    """Raised when a selected render cannot be resolved from metadata."""


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
    layout = load_layout_ref(layout_ref, config=config)
    preview_output_size = review_output_size_for_layout(layout, config=config)
    preview_layout = layout_with_output_size(layout, preview_output_size)
    output_path = render_output_path(
        source_path,
        layout,
        clip_id=clip_id,
        output_size=preview_output_size,
        config=config,
    )
    caption_metadata = load_optional_caption_metadata(caption_metadata_path)
    caption_style = caption_style_from_config(config)
    LOGGER.info(
        "Rendering layout %s preview at %sx%s to %s.",
        layout.name,
        preview_output_size.width,
        preview_output_size.height,
        output_path,
    )
    return render_layout_with_optional_captions(
        source_path,
        output_path,
        preview_layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=None,
        review=True,
        style_scale=output_scale(layout.output, preview_output_size),
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
    layouts = candidate_layouts(
        clip_id=clip_id,
        use_generated_layouts=use_generated_layouts,
        config=config,
    )
    caption_metadata = load_optional_caption_metadata(caption_metadata_path)
    caption_style = caption_style_from_config(config)
    return tuple(
        render_candidate_layout(
            source_path,
            layout,
            clip_id=clip_id,
            caption_metadata=caption_metadata,
            caption_style=caption_style,
            config=config,
        )
        for layout in layouts
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
    try:
        payload = read_pipeline_metadata(metadata_path)
        source_path = metadata_source_path(payload, metadata_path=metadata_path)
        layout = metadata_layout(payload, selected_layout=selected_layout)
    except PipelineMetadataError as exc:
        raise RenderSelectionError(str(exc)) from exc
    caption_metadata = load_optional_caption_metadata(
        metadata_optional_path(payload, "caption_metadata_path")
    )
    caption_style = caption_style_from_config(config)
    ensure_directory(output_path.parent)
    LOGGER.info(
        "Rendering selected layout %s at final size %sx%s to %s.",
        layout.name,
        layout.output.width,
        layout.output.height,
        output_path,
    )
    return render_layout_with_optional_captions(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=channel,
        review=False,
        config=config,
    )


def ensure_rendered_candidate_layout(
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
    preview_output_size = review_output_size_for_layout(layout, config=config)
    preview_layout = layout_with_output_size(layout, preview_output_size)
    output_path = render_output_path(
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
        render_layout_with_optional_captions(
            source_path,
            output_path,
            preview_layout,
            caption_metadata=caption_metadata,
            caption_style=caption_style,
            channel=channel,
            review=True,
            style_scale=output_scale(layout.output, preview_output_size),
            config=config,
        ),
        False,
    )


def load_layout_ref(layout_ref: str, *, config: ClipforgeConfig) -> Layout:
    path = Path(layout_ref)
    if path.suffix.lower() == ".json" or path.exists():
        LOGGER.info("Loading layout from %s.", path)
        return load_layout(path)

    layout_path = config.example_layouts_dir / f"{layout_ref}.json"
    LOGGER.info("Loading example layout %s from %s.", layout_ref, layout_path)
    return load_layout(layout_path)


def candidate_layouts(
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

    generated_layouts_dir = clip_analysis_path(clip_id, config=config) / "layouts"
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


def render_output_path(
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
        if output_size is not None and output_size != layout.output:
            output_path = render_preview_path(
                config,
                streamer=channel,
                clip_id=stem,
                engine=backend,
                layout=layout.name,
                width=output_size.width,
                height=output_size.height,
            )
            ensure_directory(output_path.parent)
            return output_path
        output_path = render_path(
            config,
            streamer=channel,
            clip_id=stem,
            engine=backend,
            layout=layout.name,
        )
        ensure_directory(output_path.parent)
        return output_path

    output_dir = ensure_directory(config.renders_dir)
    if output_size is not None and output_size != layout.output:
        return (
            output_dir
            / f"{stem}_{layout.name}_{output_size.width}x{output_size.height}."
            f"{config.output_format}"
        )
    return output_dir / f"{stem}_{layout.name}.{config.output_format}"


def render_candidate_layout(
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
    preview_output_size = review_output_size_for_layout(layout, config=config)
    preview_layout = layout_with_output_size(layout, preview_output_size)
    output_path = render_output_path(
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
    return render_layout_with_optional_captions(
        source_path,
        output_path,
        preview_layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        channel=channel,
        review=True,
        style_scale=output_scale(layout.output, preview_output_size),
        config=config,
    )


def load_optional_caption_metadata(path: Path | None) -> CaptionMetadata | None:
    if path is None:
        return None
    return load_caption_metadata(path)


def render_layout_with_optional_captions(
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
    watermark = scale_watermark(
        load_streamer_watermark(channel, base_dir=config.project_root),
        style_scale,
    )
    caption_style = scale_caption_style(caption_style, style_scale)
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


def caption_style_from_config(config: ClipforgeConfig) -> CaptionStyle:
    if (
        config.caption_font_file is None
        and config.caption_font_fallbacks == CaptionStyle().font_fallbacks
    ):
        return CaptionStyle()
    return CaptionStyle(
        font_file=config.caption_font_file,
        font_fallbacks=config.caption_font_fallbacks,
    )


def review_output_size_for_layout(
    layout: Layout,
    *,
    config: ClipforgeConfig,
) -> OutputSize:
    width, height = config.review_resolution_for(
        width=layout.output.width,
        height=layout.output.height,
    )
    return OutputSize(width=width, height=height)


def layout_with_output_size(layout: Layout, output_size: OutputSize) -> Layout:
    if layout.output == output_size:
        return layout
    return replace(layout, output=output_size)


def output_scale(normal_output_size: OutputSize, output_size: OutputSize) -> float:
    if normal_output_size.width <= 0:
        return 1.0
    return output_size.width / normal_output_size.width


def scale_caption_style(caption_style: CaptionStyle, scale: float) -> CaptionStyle:
    if scale == 1.0:
        return caption_style
    vertical_safe_area = caption_style.vertical_safe_area
    if vertical_safe_area is not None:
        vertical_safe_area = CaptionVerticalSafeArea(
            top=scale_int(vertical_safe_area.top, scale),
            bottom=scale_int(vertical_safe_area.bottom, scale),
            center=vertical_safe_area.center,
        )
    return replace(
        caption_style,
        font_size=scale_int(caption_style.font_size, scale),
        box_border_width=scale_int(caption_style.box_border_width, scale),
        line_spacing=scale_int(caption_style.line_spacing, scale),
        outline_width=scale_int(caption_style.outline_width, scale),
        outline_thickness=scale_optional_int(caption_style.outline_thickness, scale),
        shadow_offset=scale_int(caption_style.shadow_offset, scale),
        shadow_strength=scale_optional_int(caption_style.shadow_strength, scale),
        safe_margin_x=scale_int(caption_style.safe_margin_x, scale),
        safe_margin_bottom=scale_int(caption_style.safe_margin_bottom, scale),
        vertical_safe_area=vertical_safe_area,
    )


def scale_watermark(watermark: Watermark | None, scale: float) -> Watermark | None:
    if watermark is None or scale == 1.0:
        return watermark
    return replace(
        watermark,
        native_width=scale_int(watermark.native_width, scale),
        native_height=scale_int(watermark.native_height, scale),
        margin=scale_int(watermark.margin, scale),
    )


def scale_optional_int(value: int | None, scale: float) -> int | None:
    if value is None:
        return None
    return scale_int(value, scale)


def scale_int(value: int, scale: float) -> int:
    if value == 0:
        return 0
    return max(1, round(value * scale))


def output_size_payload(output_size: OutputSize) -> dict[str, int]:
    return {"width": output_size.width, "height": output_size.height}


def render_settings_payload(
    render_settings: FFmpegRenderSettings,
) -> dict[str, object]:
    return asdict(render_settings.normalized())

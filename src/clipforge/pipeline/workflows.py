"""Reusable pipeline workflows."""

from __future__ import annotations

import logging
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.utils import ensure_directory, safe_filename, twitch_clip_slug_from_url
from clipforge.integrations.clipr import CliprClient
from clipforge.media.download import download_clip, download_twitch_clip
from clipforge.media.captions import CaptionMetadata, generate_caption_metadata, load_caption_metadata
from clipforge.media.layouts import (
    DEFAULT_LAYOUT_NAMES,
    Layout,
    load_example_layouts,
    load_layout,
)
from clipforge.media.render import render_layout
from clipforge.media.render import CaptionStyle
from clipforge.pipeline.artifacts import write_metadata
from clipforge.pipeline.state_sync import record_rendered_clip
from clipforge.storage.state import get_clip


LOGGER = logging.getLogger("clipforge.pipeline.workflows")


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
        config=config,
    )


def render_all_candidates(
    source_path: Path,
    *,
    clip_id: str | None = None,
    caption_metadata_path: Path | None = None,
    config: ClipforgeConfig | None = None,
) -> tuple[Path, ...]:
    """Render the default MVP candidate layouts for one source clip."""

    config = config or load_config()
    layouts = load_example_layouts(DEFAULT_LAYOUT_NAMES, layouts_dir=config.example_layouts_dir)
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
    config: ClipforgeConfig | None = None,
) -> Path:
    """Run the full MVP pipeline and return the metadata path."""

    config = config or load_config()
    clip_id = twitch_clip_slug_from_url(twitch_clip_url)
    clip_state = get_clip(clip_id, db_path=config.state_db_path)
    channel = clip_state.streamer_login if clip_state is not None else None
    LOGGER.info("Starting clip pipeline for clip %s.", clip_id)
    download_result = download_twitch_clip(
        twitch_clip_url,
        clip_id=clip_id,
        config=config,
        on_media_url_resolved=lambda media_url: print(f"download_url: {media_url}"),
    )
    source_path = download_result.source_path
    caption_metadata_path = None
    if _should_generate_captions(generate_captions, config=config):
        LOGGER.info("Generating captions for clip %s from %s.", clip_id, source_path)
        caption_metadata_path = generate_caption_metadata(
            source_path,
            clip_id=clip_id,
            config=config,
        )
    caption_metadata = _load_optional_caption_metadata(caption_metadata_path)
    caption_style = _caption_style_from_config(config)

    layouts = load_example_layouts(DEFAULT_LAYOUT_NAMES, layouts_dir=config.example_layouts_dir)

    outputs = []
    for layout in layouts:
        output_path = _render_candidate_layout(
            source_path,
            layout,
            clip_id=clip_id,
            backend=download_result.backend,
            channel=channel,
            caption_metadata=caption_metadata,
            caption_style=caption_style,
            config=config,
        )
        outputs.append({"layout": layout.name, "path": str(output_path)})

    metadata_path = write_metadata(
        clip_id=clip_id,
        twitch_clip_url=twitch_clip_url,
        download_result=download_result,
        source_path=source_path,
        layouts=layouts,
        outputs=outputs,
        config=config,
        caption_metadata_path=caption_metadata_path,
    )
    record_rendered_clip(
        clip_id=clip_id,
        twitch_clip_url=twitch_clip_url,
        render_dir=Path(outputs[0]["path"]).parent,
        metadata_path=metadata_path,
        config=config,
    )

    print(f"source: {source_path}")
    if caption_metadata_path is not None:
        print(f"captions: {caption_metadata_path}")
    for output in outputs:
        print(f"{output['layout']}: {output['path']}")
    print(f"metadata: {metadata_path}")
    return metadata_path


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
    config: ClipforgeConfig,
) -> Path:
    if caption_metadata is None:
        return render_layout(source_path, output_path, layout)
    if caption_style == CaptionStyle():
        return render_layout(
            source_path,
            output_path,
            layout,
            caption_metadata=caption_metadata,
            caption_renderer_backend=config.require_caption_renderer_backend(),
            ass_temp_dir=config.ass_temp_dir,
        )
    return render_layout(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        caption_renderer_backend=config.require_caption_renderer_backend(),
        ass_temp_dir=config.ass_temp_dir,
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

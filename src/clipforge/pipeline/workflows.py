"""Public pipeline workflow facade."""

from __future__ import annotations

import logging
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.integrations.clipr import CliprClient
from clipforge.media.download import download_clip
from clipforge.pipeline.clip_processing import (
    ClipProcessingError,
    ClipProcessingResult,
    process_clip_stages,
)
from clipforge.pipeline.rendering import (
    RenderSelectionError,
    render_all_candidates,
    render_candidate,
    render_selected_layout_from_metadata as _render_selected_layout_from_metadata,
)


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
    result = process_clip_stages(
        twitch_clip_url,
        generate_captions=generate_captions,
        force_captions=force_captions,
        force=force,
        rerender=rerender,
        channel=channel,
        use_generated_layouts=use_generated_layouts,
        config=config,
        on_media_url_resolved=lambda media_url: LOGGER.info(
            "Resolved download URL: %s.",
            media_url,
        ),
    )
    _log_process_clip_result(result)
    _print_clip_summary(result)
    return result.metadata_path


def render_selected_layout_from_metadata(
    metadata_path: Path,
    *,
    selected_layout: str,
    output_path: Path,
    channel: str | None,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Render one selected layout from pipeline metadata with final settings."""

    try:
        return _render_selected_layout_from_metadata(
            metadata_path,
            selected_layout=selected_layout,
            output_path=output_path,
            channel=channel,
            config=config,
        )
    except RenderSelectionError as exc:
        raise ClipProcessingError(str(exc)) from exc


def _log_process_clip_result(result: ClipProcessingResult) -> None:
    if result.rerender:
        LOGGER.info("Rerendering visual artifacts while preserving source and captions.")

    source_prefix = "source: reusing existing" if result.reused_source else "source:"
    LOGGER.info("%s %s", source_prefix, result.source_path)

    if result.caption_metadata_path is not None:
        if result.reused_caption_metadata:
            LOGGER.info(
                "captions: reusing existing %s",
                result.caption_metadata_path,
            )
        else:
            LOGGER.info("captions: %s", result.caption_metadata_path)

    if result.frames_metadata_path is not None:
        prefix = "frames: reusing existing" if result.reused_frames else "frames:"
        LOGGER.info("%s %s", prefix, result.frames_metadata_path)

    if result.overlay_path is not None:
        prefix = "overlay: reusing existing" if result.reused_overlay else "overlay:"
        LOGGER.info("%s %s", prefix, result.overlay_path)

    for layout_path in result.generated_layout_paths:
        prefix = "layout: reusing existing" if result.reused_layouts else "layout:"
        LOGGER.info("%s %s", prefix, layout_path)

    for output in result.outputs:
        LOGGER.info("%s: %s", output["layout"], output["path"])
    LOGGER.info("metadata: %s", result.metadata_path)


def _print_clip_summary(result: ClipProcessingResult) -> None:
    print(f"Slug: {result.clip_id}")
    if result.title:
        print(f"Title: {result.title}")
    if result.streamer_login:
        print(f"Streamer: {result.streamer_login}")
    if result.view_count is not None:
        print(f"Views: {result.view_count}")
    if result.created_at:
        print(f"Created: {result.created_at}")
    if result.duration_seconds is not None:
        print(f"Duration: {result.duration_seconds:g}s")

"""Stage orchestration for processing one Twitch clip."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.artifact_reuse import (
    ensure_caption_metadata,
    ensure_layout_analysis,
    ensure_overlay_analysis,
    ensure_sampled_frames,
    ensure_source_video,
    require_existing_caption_metadata,
)
from clipforge.pipeline.artifacts import write_metadata
from clipforge.pipeline.rendering import (
    candidate_layouts,
    caption_style_from_config,
    ensure_rendered_candidate_layout,
    load_optional_caption_metadata,
    output_size_payload,
    render_settings_payload,
    review_output_size_for_layout,
)
from clipforge.pipeline.state_sync import record_rendered_clip
from clipforge.storage.state import get_clip
from clipforge.utils.paths import twitch_clip_slug_from_url


LOGGER = logging.getLogger("clipforge.pipeline.clip_processing")


class ClipProcessingError(RuntimeError):
    """Raised when a named processing stage fails."""


@dataclass(frozen=True)
class ClipProcessingResult:
    clip_id: str
    title: str | None
    streamer_login: str | None
    view_count: int | None
    created_at: str | None
    duration_seconds: float | None
    metadata_path: Path
    source_path: Path
    reused_source: bool
    caption_metadata_path: Path | None
    reused_caption_metadata: bool
    frames_metadata_path: Path | None
    reused_frames: bool
    overlay_path: Path | None
    reused_overlay: bool
    generated_layout_paths: tuple[Path, ...]
    reused_layouts: bool
    outputs: tuple[dict[str, object], ...]
    rerender: bool


def process_clip_stages(
    twitch_clip_url: str,
    *,
    generate_captions: bool | None,
    force_captions: bool,
    force: bool,
    rerender: bool,
    channel: str | None,
    use_generated_layouts: bool,
    config: ClipforgeConfig,
    on_media_url_resolved: Callable[[str], None] | None = None,
) -> ClipProcessingResult:
    """Run reusable processing stages and return the artifacts produced."""

    clip_id = twitch_clip_slug_from_url(twitch_clip_url)
    clip_state = get_clip(clip_id, db_path=config.state_db_path)
    state_channel = clip_state.streamer_login if clip_state is not None else None
    channel = channel or state_channel
    force_visuals = force or rerender
    LOGGER.info("Starting clip pipeline for clip %s.", clip_id)

    caption_metadata_path = None
    reused_caption_metadata = False
    if rerender:
        caption_metadata_path, reused_caption_metadata = run_stage(
            "captions",
            lambda: require_existing_caption_metadata(clip_id, config=config),
        )

    download_result, reused_source = run_stage(
        "download",
        lambda: ensure_source_video(
            twitch_clip_url,
            clip_id=clip_id,
            clip_state=clip_state,
            prefer_existing=rerender,
            config=config,
            on_media_url_resolved=on_media_url_resolved,
        ),
    )
    source_path = download_result.source_path
    if not rerender and should_generate_captions(generate_captions, config=config):
        caption_metadata_path, reused_caption_metadata = run_stage(
            "captions",
            lambda: ensure_caption_metadata(
                source_path,
                clip_id=clip_id,
                force_captions=force_captions,
                config=config,
            ),
        )
    caption_metadata = load_optional_caption_metadata(caption_metadata_path)
    caption_style = caption_style_from_config(config)

    frames_metadata_path: Path | None = None
    reused_frames = False
    overlay_path: Path | None = None
    reused_overlay = False
    generated_layout_paths: tuple[Path, ...] = ()
    reused_layouts = False
    if use_generated_layouts:
        frames_metadata_path, reused_frames = run_stage(
            "frames",
            lambda: ensure_sampled_frames(
                source_path,
                clip_id=clip_id,
                duration_seconds=getattr(clip_state, "duration_seconds", None),
                force=force_visuals,
                config=config,
            ),
        )
        overlay_path, reused_overlay = run_stage(
            "overlay",
            lambda: ensure_overlay_analysis(
                clip_id=clip_id,
                force=force_visuals,
                config=config,
            ),
        )
        generated_layout_paths, reused_layouts = run_stage(
            "layouts",
            lambda: ensure_layout_analysis(
                clip_id=clip_id,
                force=force_visuals,
                config=config,
            ),
        )

    layouts = run_stage(
        "layouts",
        lambda: candidate_layouts(
            clip_id=clip_id,
            use_generated_layouts=use_generated_layouts,
            config=config,
        ),
    )

    outputs = []
    for layout in layouts:
        output_path, _reused_render = run_stage(
            "renders",
            lambda layout=layout: ensure_rendered_candidate_layout(
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
        preview_output_size = review_output_size_for_layout(layout, config=config)
        outputs.append(
            {
                "layout": layout.name,
                "path": str(output_path),
                "resolution": output_size_payload(preview_output_size),
                "render_profile": "review",
                "render_settings": render_settings_payload(
                    config.render_settings_for(review=True)
                ),
            }
        )

    metadata_path = run_stage(
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
    run_stage(
        "metadata",
        lambda: record_rendered_clip(
            clip_id=clip_id,
            twitch_clip_url=twitch_clip_url,
            render_dir=Path(outputs[0]["path"]).parent,
            metadata_path=metadata_path,
            config=config,
        ),
    )

    return ClipProcessingResult(
        clip_id=clip_id,
        title=getattr(clip_state, "title", None),
        streamer_login=channel,
        view_count=getattr(clip_state, "view_count", None),
        created_at=getattr(clip_state, "created_at", None),
        duration_seconds=getattr(clip_state, "duration_seconds", None),
        metadata_path=metadata_path,
        source_path=source_path,
        reused_source=reused_source,
        caption_metadata_path=caption_metadata_path,
        reused_caption_metadata=reused_caption_metadata,
        frames_metadata_path=frames_metadata_path,
        reused_frames=reused_frames,
        overlay_path=overlay_path,
        reused_overlay=reused_overlay,
        generated_layout_paths=generated_layout_paths,
        reused_layouts=reused_layouts,
        outputs=tuple(outputs),
        rerender=rerender,
    )


def run_stage(stage_name: str, callback):
    try:
        return callback()
    except ClipProcessingError:
        raise
    except Exception as exc:
        raise ClipProcessingError(f"{stage_name} stage failed: {exc}") from exc


def should_generate_captions(
    generate_captions: bool | None,
    *,
    config: ClipforgeConfig,
) -> bool:
    if generate_captions is not None:
        return generate_captions
    return config.generate_captions

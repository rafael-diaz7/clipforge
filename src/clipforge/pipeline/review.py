"""Manual streamer review workflow."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.integrations.twitch import (
    list_channel_clips,
    twitch_channel_login_from_input,
)
from clipforge.pipeline.state_sync import record_discovered_clips
from clipforge.pipeline.workflows import process_clip
from clipforge.storage.state import (
    REVIEW_EXCLUDED_STATUSES,
    ClipState,
    get_clip,
    get_review_eligible_clips,
    mark_clip_exported,
    mark_clip_failed,
)
from clipforge.utils.paths import ensure_directory, safe_filename


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


class ClipReviewError(RuntimeError):
    """Raised when manual clip review cannot complete."""


@dataclass(frozen=True)
class RenderOption:
    layout: str
    path: Path


def review_streamer_clips(
    *,
    streamer: str,
    count: int = 3,
    force: bool = False,
    rerender: bool = False,
    generate_captions: bool | None = None,
    force_captions: bool = False,
    clip_ids: Sequence[str] = (),
    started_at: str | None = None,
    ended_at: str | None = None,
    discovery_limit: int | None = None,
    use_generated_layouts: bool = True,
    config: ClipforgeConfig | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> tuple[Path, ...]:
    """Discover, rank, process, and manually export final renders for a streamer."""

    if count < 1:
        raise ClipReviewError("--count must be at least 1.")

    config = config or load_config()
    streamer_login = twitch_channel_login_from_input(streamer)
    discovered = list_channel_clips(
        streamer,
        limit=discovery_limit or max(count, 10),
        started_at=started_at,
        ended_at=ended_at,
        config=config,
    )
    record_discovered_clips(clips=discovered, channel=streamer, config=config)

    selected_clips = _selected_review_clips(
        clip_ids=clip_ids,
        count=count,
        streamer_login=streamer_login,
        config=config,
    )
    if not selected_clips:
        raise ClipReviewError(
            f"No review-eligible clips found for streamer: {streamer_login}."
        )

    exported_paths: list[Path] = []
    for clip in selected_clips:
        output_fn(_format_clip_header(clip))
        process_kwargs = {"config": config, "use_generated_layouts": use_generated_layouts}
        if rerender:
            process_kwargs["rerender"] = True
        if generate_captions is not None:
            process_kwargs["generate_captions"] = generate_captions
        if force_captions:
            process_kwargs["force_captions"] = True

        try:
            metadata_path = process_clip(clip.url, **process_kwargs)
        except Exception as exc:
            mark_clip_failed(
                clip.clip_id,
                error_message=str(exc),
                db_path=config.state_db_path,
            )
            raise
        render_options = _render_options_from_metadata(metadata_path)
        selected = _prompt_for_render_selection(
            clip,
            render_options,
            input_fn=input_fn,
            output_fn=output_fn,
        )
        if selected is None:
            output_fn(f"skipped: {clip.clip_id}")
            continue

        export_path = _copy_selected_render(
            clip=clip,
            streamer_login=streamer_login,
            selected=selected,
            force=force,
            config=config,
        )
        mark_clip_exported(
            clip.clip_id,
            selected_render_layout=selected.layout,
            selected_render_path=selected.path,
            export_path=export_path,
            db_path=config.state_db_path,
        )
        exported_paths.append(export_path)
        output_fn(f"exported: {export_path}")

    return tuple(exported_paths)


def _selected_review_clips(
    *,
    clip_ids: Sequence[str],
    count: int,
    streamer_login: str,
    config: ClipforgeConfig,
) -> tuple[ClipState, ...]:
    if not clip_ids:
        return get_review_eligible_clips(
            db_path=config.state_db_path,
            streamer_login=streamer_login,
            limit=count,
        )

    clips: list[ClipState] = []
    for clip_id in clip_ids:
        clip = get_clip(clip_id, db_path=config.state_db_path)
        if clip is None:
            raise ClipReviewError(f"Clip not found after discovery: {clip_id}.")
        _ensure_manual_clip_is_eligible(clip, streamer_login=streamer_login)
        clips.append(clip)
    return tuple(clips)


def _ensure_manual_clip_is_eligible(clip: ClipState, *, streamer_login: str) -> None:
    if clip.status in REVIEW_EXCLUDED_STATUSES:
        raise ClipReviewError(
            f"Clip is not review-eligible: {clip.clip_id} ({clip.status})."
        )
    if (
        clip.streamer_login is not None
        and clip.streamer_login.lower() != streamer_login.lower()
    ):
        raise ClipReviewError(
            f"Clip {clip.clip_id} belongs to streamer {clip.streamer_login}, "
            f"not {streamer_login}."
        )


def _render_options_from_metadata(metadata_path: Path) -> tuple[RenderOption, ...]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClipReviewError(
            f"Could not read pipeline metadata {metadata_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ClipReviewError(f"Pipeline metadata is not valid JSON: {metadata_path}") from exc

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ClipReviewError(
            f"Pipeline metadata has no render outputs: {metadata_path}."
        )

    options: list[RenderOption] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        layout = output.get("layout")
        path = output.get("path")
        if isinstance(layout, str) and isinstance(path, str):
            options.append(RenderOption(layout=layout, path=Path(path)))

    if not options:
        raise ClipReviewError(
            f"Pipeline metadata has no usable render outputs: {metadata_path}."
        )
    return tuple(options)


def _prompt_for_render_selection(
    clip: ClipState,
    options: Sequence[RenderOption],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> RenderOption | None:
    output_fn("render options:")
    for index, option in enumerate(options, start=1):
        output_fn(f"  {index}. {option.layout}: {option.path}")

    prompt = f"Select render for {clip.clip_id} [1-{len(options)} or s to skip]: "
    while True:
        value = input_fn(prompt).strip().lower()
        if value in {"s", "skip"}:
            return None
        try:
            selected_index = int(value)
        except ValueError:
            output_fn(f"Invalid selection. Enter 1-{len(options)} or s to skip.")
            continue
        if 1 <= selected_index <= len(options):
            return options[selected_index - 1]
        output_fn(f"Invalid selection. Enter 1-{len(options)} or s to skip.")


def _copy_selected_render(
    *,
    clip: ClipState,
    streamer_login: str,
    selected: RenderOption,
    force: bool,
    config: ClipforgeConfig,
) -> Path:
    source_path = selected.path
    export_path = (
        config.exports_dir
        / "ready"
        / safe_filename(streamer_login)
        / safe_filename(clip.clip_id)
        / f"{safe_filename(selected.layout)}.{config.output_format}"
    )
    if export_path.exists() and not force:
        raise ClipReviewError(f"Export already exists: {export_path}. Re-run with --force.")

    ensure_directory(export_path.parent)
    try:
        shutil.copy2(source_path, export_path)
    except OSError as exc:
        raise ClipReviewError(
            f"Could not export selected render to {export_path}: {exc}"
        ) from exc
    return export_path


def _format_clip_header(clip: ClipState) -> str:
    score = "" if clip.rank_score is None else f"{clip.rank_score:g}"
    views = "" if clip.view_count is None else str(clip.view_count)
    duration = "" if clip.duration_seconds is None else f"{clip.duration_seconds:g}s"
    parts = [
        f"clip: {clip.clip_id}",
        f"streamer: {clip.streamer_login or ''}",
        f"score: {score}",
        f"views: {views}",
        f"duration: {duration}",
        f"created: {clip.created_at or ''}",
    ]
    title = clip.title or ""
    if title:
        parts.append(f"title: {title}")
    return "\n".join(parts)

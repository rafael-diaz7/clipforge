"""Manual streamer review workflow."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.integrations.twitch import (
    list_channel_clips,
    twitch_channel_login_from_input,
)
from clipforge.pipeline.exports import export_review_selection
from clipforge.pipeline.metadata import RenderCandidate, render_candidates_from_metadata
from clipforge.pipeline.state_sync import record_discovered_clips
from clipforge.pipeline.workflows import process_clip, render_selected_layout_from_metadata
from clipforge.storage.state import (
    REVIEW_EXCLUDED_STATUSES,
    ClipState,
    get_clip,
    get_persisted_clips,
    get_review_eligible_clips,
    get_unprocessed_clips,
    mark_clip_failed,
    mark_clip_needs_rerender,
    mark_clip_skipped,
)


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


class ClipReviewError(RuntimeError):
    """Raised when manual clip review cannot complete."""


class ReviewAction(Enum):
    SKIP = "skip"
    RERENDER = "rerender"


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
        force=force,
        rerender=rerender,
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
        process_kwargs = {
            "channel": streamer_login,
            "config": config,
            "use_generated_layouts": use_generated_layouts,
        }
        if rerender:
            process_kwargs["rerender"] = True
        if generate_captions is not None:
            process_kwargs["generate_captions"] = generate_captions
        if force_captions:
            process_kwargs["force_captions"] = True

        metadata_path = _existing_render_metadata_path(clip)
        if metadata_path is None:
            try:
                metadata_path = process_clip(clip.url, **process_kwargs)
            except Exception as exc:
                mark_clip_failed(
                    clip.clip_id,
                    error_message=str(exc),
                    db_path=config.state_db_path,
                )
                raise
        try:
            render_options = render_candidates_from_metadata(metadata_path)
        except Exception as exc:
            mark_clip_failed(
                clip.clip_id,
                error_message=str(exc),
                db_path=config.state_db_path,
            )
            raise ClipReviewError(str(exc)) from exc
        choice = _prompt_for_render_selection(
            clip,
            render_options,
            input_fn=input_fn,
            output_fn=output_fn,
        )
        if choice is ReviewAction.SKIP:
            mark_clip_skipped(
                clip.clip_id,
                skip_reason="review skipped after candidates generated",
                db_path=config.state_db_path,
            )
            output_fn(
                f"skipped: {clip.clip_id} "
                "(will not be picked again by normal review)"
            )
            continue
        if choice is ReviewAction.RERENDER:
            mark_clip_needs_rerender(
                clip.clip_id,
                skip_reason="review requested rerender after candidates generated",
                db_path=config.state_db_path,
            )
            output_fn(
                f"rerender requested: {clip.clip_id} "
                "(will be picked up by processing or review --rerender)"
            )
            continue

        try:
            selected_export = export_review_selection(
                clip=clip,
                streamer_login=streamer_login,
                metadata_path=metadata_path,
                selected=choice,
                force=force,
                config=config,
                render_selected=render_selected_layout_from_metadata,
            )
        except Exception as exc:
            mark_clip_failed(
                clip.clip_id,
                error_message=str(exc),
                db_path=config.state_db_path,
            )
            raise ClipReviewError(str(exc)) from exc
        exported_paths.append(selected_export.export_path)
        output_fn(f"exported: {selected_export.export_path}")

    return tuple(exported_paths)


def _selected_review_clips(
    *,
    clip_ids: Sequence[str],
    count: int,
    force: bool,
    rerender: bool,
    streamer_login: str,
    config: ClipforgeConfig,
) -> tuple[ClipState, ...]:
    if not clip_ids:
        eligible = list(
            get_review_eligible_clips(
                db_path=config.state_db_path,
                streamer_login=streamer_login,
                limit=count,
                include_needs_rerender=rerender,
            )
        )
        if len(eligible) >= count:
            return tuple(eligible)

        selected_ids = {clip.clip_id for clip in eligible}
        candidates = get_unprocessed_clips(
            db_path=config.state_db_path,
            streamer_login=streamer_login,
        )
        for clip in candidates:
            if clip.clip_id in selected_ids:
                continue
            if clip.status == "needs_rerender" and not rerender:
                continue
            eligible.append(clip)
            selected_ids.add(clip.clip_id)
            if len(eligible) >= count:
                break
        if force and len(eligible) < count:
            for clip in get_persisted_clips(
                db_path=config.state_db_path,
                streamer_login=streamer_login,
            ):
                if clip.clip_id in selected_ids:
                    continue
                _ensure_manual_clip_is_eligible(
                    clip,
                    streamer_login=streamer_login,
                    force=True,
                    allow_needs_rerender=rerender,
                )
                eligible.append(clip)
                selected_ids.add(clip.clip_id)
                if len(eligible) >= count:
                    break
        return tuple(eligible)

    clips: list[ClipState] = []
    for clip_id in clip_ids:
        clip = get_clip(clip_id, db_path=config.state_db_path)
        if clip is None:
            raise ClipReviewError(f"Clip not found after discovery: {clip_id}.")
        _ensure_manual_clip_is_eligible(
            clip,
            streamer_login=streamer_login,
            force=force,
            allow_needs_rerender=rerender,
        )
        clips.append(clip)
    return tuple(clips)


def _ensure_manual_clip_is_eligible(
    clip: ClipState,
    *,
    streamer_login: str,
    force: bool,
    allow_needs_rerender: bool,
) -> None:
    if clip.status == "needs_rerender" and not allow_needs_rerender:
        raise ClipReviewError(
            f"Clip needs rerender before review: {clip.clip_id}. "
            "Re-run with --rerender."
        )
    if (
        clip.status in REVIEW_EXCLUDED_STATUSES
        and not (clip.status == "needs_rerender" and allow_needs_rerender)
        and not force
    ):
        raise ClipReviewError(
            f"Clip is not review-eligible: {clip.clip_id} ({clip.status}). "
            "Re-run with --force to review it anyway."
        )
    if (
        clip.streamer_login is not None
        and clip.streamer_login.lower() != streamer_login.lower()
    ):
        raise ClipReviewError(
            f"Clip {clip.clip_id} belongs to streamer {clip.streamer_login}, "
            f"not {streamer_login}."
        )


def _existing_render_metadata_path(clip: ClipState) -> Path | None:
    if clip.status != "rendered" or clip.metadata_path is None:
        return None
    return Path(clip.metadata_path)


def _prompt_for_render_selection(
    clip: ClipState,
    options: Sequence[RenderCandidate],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> RenderCandidate | ReviewAction:
    output_fn("render options:")
    for index, option in enumerate(options, start=1):
        output_fn(f"  {index}. {option.layout}: {option.path}")

    prompt = (
        f"Select render for {clip.clip_id} "
        f"[1-{len(options)}, s to skip, or r to rerender later]: "
    )
    while True:
        value = input_fn(prompt).strip().lower()
        if value in {"s", "skip"}:
            return ReviewAction.SKIP
        if value in {"r", "rerender"}:
            return ReviewAction.RERENDER
        try:
            selected_index = int(value)
        except ValueError:
            output_fn(
                f"Invalid selection. Enter 1-{len(options)}, s to skip, or r to rerender."
            )
            continue
        if 1 <= selected_index <= len(options):
            return options[selected_index - 1]
        output_fn(f"Invalid selection. Enter 1-{len(options)}, s to skip, or r to rerender.")


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

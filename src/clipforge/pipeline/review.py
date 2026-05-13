"""Manual streamer review workflow."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.integrations.twitch import (
    list_channel_clips,
    twitch_channel_login_from_input,
)
from clipforge.pipeline.state_sync import record_discovered_clips
from clipforge.media.render_settings import FFmpegRenderSettings
from clipforge.pipeline.workflows import process_clip, render_selected_layout_from_metadata
from clipforge.storage.paths import export_path as selected_export_path
from clipforge.storage.state import (
    REVIEW_EXCLUDED_STATUSES,
    ClipState,
    get_clip,
    get_persisted_clips,
    get_review_eligible_clips,
    get_unprocessed_clips,
    mark_clip_exported,
    mark_clip_failed,
    mark_clip_needs_rerender,
    mark_clip_selected,
    mark_clip_skipped,
)
from clipforge.utils.paths import ensure_directory


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


class ClipReviewError(RuntimeError):
    """Raised when manual clip review cannot complete."""


@dataclass(frozen=True)
class RenderOption:
    layout: str
    path: Path
    resolution: tuple[int, int] | None = None
    render_settings: FFmpegRenderSettings | None = None


@dataclass(frozen=True)
class SelectedExport:
    export_path: Path
    final_render_path: Path
    final_resolution: tuple[int, int] | None
    reused_preview: bool


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
            render_options = _render_options_from_metadata(metadata_path)
        except Exception as exc:
            mark_clip_failed(
                clip.clip_id,
                error_message=str(exc),
                db_path=config.state_db_path,
            )
            raise
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

        mark_clip_selected(
            clip.clip_id,
            selected_render_layout=choice.layout,
            selected_render_path=choice.path,
            db_path=config.state_db_path,
        )
        try:
            selected_export = _export_selected_render(
                clip=clip,
                streamer_login=streamer_login,
                metadata_path=metadata_path,
                selected=choice,
                force=force,
                config=config,
            )
        except Exception as exc:
            mark_clip_failed(
                clip.clip_id,
                error_message=str(exc),
                db_path=config.state_db_path,
            )
            raise
        mark_clip_exported(
            clip.clip_id,
            selected_render_layout=choice.layout,
            selected_render_path=choice.path,
            export_path=selected_export.export_path,
            db_path=config.state_db_path,
        )
        _write_selected_export_metadata(
            metadata_path,
            selected=choice,
            selected_export=selected_export,
        )
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
            options.append(
                RenderOption(
                    layout=layout,
                    path=Path(path),
                    resolution=_output_resolution(output.get("resolution")),
                    render_settings=_output_render_settings(output),
                )
            )

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
) -> RenderOption | ReviewAction:
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


def _export_selected_render(
    *,
    clip: ClipState,
    streamer_login: str,
    metadata_path: Path,
    selected: RenderOption,
    force: bool,
    config: ClipforgeConfig,
) -> SelectedExport:
    source_path = selected.path
    export_path = selected_export_path(
        config,
        streamer=streamer_login,
        title=clip.title,
        clip_id=clip.clip_id,
        layout=selected.layout,
    )
    if export_path.exists() and not force:
        raise ClipReviewError(f"Export already exists: {export_path}. Re-run with --force.")

    ensure_directory(export_path.parent)
    final_resolution = _final_resolution_from_metadata(
        metadata_path,
        selected_layout=selected.layout,
    )
    if _selected_preview_matches_final(
        selected,
        final_resolution=final_resolution,
        config=config,
    ):
        _copy_selected_render(source_path=source_path, export_path=export_path)
        return SelectedExport(
            export_path=export_path,
            final_render_path=source_path,
            final_resolution=final_resolution or selected.resolution,
            reused_preview=True,
        )

    try:
        render_selected_layout_from_metadata(
            metadata_path,
            selected_layout=selected.layout,
            output_path=export_path,
            channel=streamer_login,
            config=config,
        )
    except Exception as exc:
        raise ClipReviewError(
            f"Could not render selected layout {selected.layout!r} to {export_path}: {exc}"
        ) from exc
    return SelectedExport(
        export_path=export_path,
        final_render_path=export_path,
        final_resolution=final_resolution,
        reused_preview=False,
    )


def _copy_selected_render(*, source_path: Path, export_path: Path) -> None:
    try:
        shutil.copy2(source_path, export_path)
    except OSError as exc:
        raise ClipReviewError(
            f"Could not export selected render to {export_path}: {exc}"
        ) from exc


def _selected_preview_matches_final(
    selected: RenderOption,
    *,
    final_resolution: tuple[int, int] | None,
    config: ClipforgeConfig,
) -> bool:
    if selected.resolution is None and selected.render_settings is None:
        return True
    if final_resolution is not None and selected.resolution != final_resolution:
        return False
    if selected.render_settings is None:
        return True
    return selected.render_settings == config.render_settings_for(review=False)


def _final_resolution_from_metadata(
    metadata_path: Path,
    *,
    selected_layout: str,
) -> tuple[int, int] | None:
    payload = _read_metadata_payload(metadata_path)
    layouts = payload.get("layouts")
    if isinstance(layouts, list):
        for layout in layouts:
            if not isinstance(layout, dict) or layout.get("name") != selected_layout:
                continue
            resolution = _output_resolution(layout.get("output"))
            if resolution is not None:
                return resolution
    return _output_resolution(payload.get("target_resolution"))


def _write_selected_export_metadata(
    metadata_path: Path,
    *,
    selected: RenderOption,
    selected_export: SelectedExport,
) -> None:
    payload = _read_metadata_payload(metadata_path)
    payload["selected_export"] = {
        "layout": selected.layout,
        "preview_candidate": {
            "path": str(selected.path),
            "resolution": _resolution_payload(selected.resolution),
        },
        "final_render": {
            "path": str(selected_export.final_render_path),
            "resolution": _resolution_payload(selected_export.final_resolution),
        },
        "export": {
            "path": str(selected_export.export_path),
            "resolution": _resolution_payload(selected_export.final_resolution),
        },
        "reused_preview": selected_export.reused_preview,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_metadata_payload(metadata_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClipReviewError(
            f"Could not read pipeline metadata {metadata_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ClipReviewError(f"Pipeline metadata is not valid JSON: {metadata_path}") from exc
    if not isinstance(payload, dict):
        raise ClipReviewError(f"Pipeline metadata must be a JSON object: {metadata_path}")
    return payload


def _output_resolution(value: object) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    width = value.get("width")
    height = value.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    return (width, height)


def _resolution_payload(value: tuple[int, int] | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"width": value[0], "height": value[1]}


def _output_render_settings(value: dict[str, object]) -> FFmpegRenderSettings | None:
    settings = value.get("render_settings")
    if not isinstance(settings, dict):
        return None
    supported_keys = FFmpegRenderSettings().__dict__.keys()
    return FFmpegRenderSettings(
        **{key: settings[key] for key in supported_keys if key in settings}
    ).normalized()


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

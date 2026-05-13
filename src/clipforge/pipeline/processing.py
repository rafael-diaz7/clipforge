"""Reusable saved-clip processing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.workflows import process_clip
from clipforge.storage.state import (
    UNPROCESSED_STATUSES,
    ClipState,
    get_clip,
    get_unprocessed_clips,
    mark_clip_failed,
)


class SavedClipProcessingError(RuntimeError):
    """Raised when saved clips cannot be selected for processing."""


@dataclass(frozen=True)
class ProcessedClip:
    clip: ClipState
    metadata_path: Path | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_message is None


ProcessClipFn = Callable[..., Path]


def select_saved_clips_for_processing(
    *,
    top: int | None,
    clip_id: str | None,
    force: bool,
    rerender: bool,
    config: ClipforgeConfig,
) -> tuple[ClipState, ...]:
    """Resolve saved clips eligible for the CLI/process worker path."""

    if top is not None:
        if force:
            raise SavedClipProcessingError("--force can only be used with --clip-id.")
        if rerender:
            raise SavedClipProcessingError("--rerender can only be used with --clip-id.")
        clips = get_unprocessed_clips(db_path=config.state_db_path, limit=top)
        if not clips:
            raise SavedClipProcessingError("No unprocessed clips found.")
        return clips

    if clip_id is None:
        raise SavedClipProcessingError("A saved clip process command requires a clip ID.")
    clip = get_clip(clip_id, db_path=config.state_db_path)
    if clip is None:
        raise SavedClipProcessingError(f"Clip not found: {clip_id}.")
    if clip.status == "rendered" and not (force or rerender):
        raise SavedClipProcessingError(
            f"Clip is already rendered: {clip_id}. "
            "Re-run with --force to reprocess it or --rerender to rebuild "
            "visual artifacts while reusing captions."
        )
    if clip.status not in UNPROCESSED_STATUSES and not (
        clip.status == "rendered" and (force or rerender)
    ):
        raise SavedClipProcessingError(f"Clip is not unprocessed: {clip_id}.")
    return (clip,)


def process_saved_clips(
    clips: Sequence[ClipState],
    *,
    config: ClipforgeConfig,
    process_kwargs: dict[str, object],
    continue_on_error: bool,
    process_clip_fn: ProcessClipFn = process_clip,
) -> tuple[ProcessedClip, ...]:
    """Process selected clips and persist failure state consistently."""

    results: list[ProcessedClip] = []
    for clip in clips:
        clip_kwargs = dict(process_kwargs)
        if clip.status == "needs_rerender":
            clip_kwargs["force"] = True
        try:
            metadata_path = process_clip_fn(clip.url, **clip_kwargs)
        except Exception as exc:
            error_message = str(exc)
            mark_clip_failed(
                clip.clip_id,
                error_message=error_message,
                db_path=config.state_db_path,
            )
            results.append(ProcessedClip(clip=clip, error_message=error_message))
            if not continue_on_error:
                break
        else:
            results.append(ProcessedClip(clip=clip, metadata_path=metadata_path))
    return tuple(results)

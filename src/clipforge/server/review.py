"""Review queue service for the local web review server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.pipeline.exports import export_review_selection
from clipforge.pipeline.metadata import RenderCandidate, render_candidates_from_metadata
from clipforge.pipeline.workflows import render_selected_layout_from_metadata
from clipforge.storage.state import (
    ClipState,
    get_review_eligible_clips,
    mark_clip_failed,
    mark_clip_needs_rerender,
    mark_clip_skipped,
)


class ReviewServerError(RuntimeError):
    """Raised when a web review operation cannot complete."""


class ReviewItemNotFound(ReviewServerError):
    """Raised when a clip is not available in the normal review queue."""


class UnsafeRenderPath(ReviewServerError):
    """Raised when metadata points outside the rendered clip directory."""


@dataclass(frozen=True)
class ReviewItem:
    clip: ClipState
    metadata_path: Path
    candidates: tuple[RenderCandidate, ...]


ExportSelectionFn = Callable[..., object]


class ReviewQueueService:
    """Business operations for draining prepared rendered clips."""

    def __init__(
        self,
        *,
        config: ClipforgeConfig | None = None,
        export_selection: ExportSelectionFn = export_review_selection,
    ) -> None:
        self.config = config or load_config()
        self._export_selection = export_selection

    def next_item(self) -> ReviewItem | None:
        clips = get_review_eligible_clips(db_path=self.config.state_db_path, limit=1)
        if not clips:
            return None
        return self._review_item(clips[0])

    def item_for_clip(self, clip_id: str) -> ReviewItem:
        for clip in get_review_eligible_clips(db_path=self.config.state_db_path):
            if clip.clip_id == clip_id:
                return self._review_item(clip)
        raise ReviewItemNotFound(f"Clip is not in the normal review queue: {clip_id}.")

    def candidate_path(self, *, clip_id: str, layout: str) -> Path:
        if _has_path_separator(layout):
            raise UnsafeRenderPath(f"Invalid layout identifier: {layout!r}.")

        item = self.item_for_clip(clip_id)
        candidate = self._candidate_for_layout(item, layout)
        candidate_path = candidate.path.resolve()
        if not candidate_path.is_file():
            raise ReviewItemNotFound(f"Rendered candidate not found: {layout}.")

        render_dir = item.clip.render_dir
        if render_dir is None:
            raise ReviewItemNotFound(f"Clip has no render directory: {clip_id}.")
        render_root = Path(render_dir).resolve()
        if not candidate_path.is_relative_to(render_root):
            raise UnsafeRenderPath(
                f"Rendered candidate is outside the clip render directory: {layout}."
            )
        return candidate_path

    def approve(self, *, clip_id: str, layout: str, force: bool = False) -> None:
        item = self.item_for_clip(clip_id)
        selected = self._candidate_for_layout(item, layout)
        streamer_login = item.clip.streamer_login or "unknown_streamer"
        try:
            self._export_selection(
                clip=item.clip,
                streamer_login=streamer_login,
                metadata_path=item.metadata_path,
                selected=selected,
                force=force,
                config=self.config,
                render_selected=render_selected_layout_from_metadata,
            )
        except Exception as exc:
            mark_clip_failed(
                item.clip.clip_id,
                error_message=str(exc),
                db_path=self.config.state_db_path,
            )
            raise ReviewServerError(str(exc)) from exc

    def skip(self, *, clip_id: str) -> None:
        self.item_for_clip(clip_id)
        mark_clip_skipped(
            clip_id,
            skip_reason="web review skipped after candidates generated",
            db_path=self.config.state_db_path,
        )

    def mark_needs_rerender(self, *, clip_id: str) -> None:
        self.item_for_clip(clip_id)
        mark_clip_needs_rerender(
            clip_id,
            skip_reason="web review requested rerender after candidates generated",
            db_path=self.config.state_db_path,
        )

    def _review_item(self, clip: ClipState) -> ReviewItem:
        if clip.metadata_path is None:
            raise ReviewItemNotFound(f"Clip has no metadata path: {clip.clip_id}.")
        metadata_path = Path(clip.metadata_path)
        try:
            candidates = render_candidates_from_metadata(metadata_path)
        except Exception as exc:
            raise ReviewServerError(str(exc)) from exc
        return ReviewItem(clip=clip, metadata_path=metadata_path, candidates=candidates)

    @staticmethod
    def _candidate_for_layout(item: ReviewItem, layout: str) -> RenderCandidate:
        for candidate in item.candidates:
            if candidate.layout == layout:
                return candidate
        raise ReviewItemNotFound(
            f"Clip {item.clip.clip_id} has no rendered candidate for layout: {layout}."
        )


def _has_path_separator(value: str) -> bool:
    return "/" in value or "\\" in value or value in {"", ".", ".."}

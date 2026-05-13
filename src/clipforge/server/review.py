"""Review queue service for the local web review server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.pipeline.exports import (
    export_selected_candidate,
    write_selected_export_metadata,
)
from clipforge.pipeline.metadata import RenderCandidate, render_candidates_from_metadata
from clipforge.storage.state import (
    ClipState,
    get_clip,
    get_mobile_review_clips,
    mark_clip_exported,
    mark_clip_failed,
    mark_clip_needs_rerender,
    mark_clip_skipped,
)


class ReviewServerError(RuntimeError):
    """Raised when a web review operation cannot complete."""


class ReviewItemNotFound(ReviewServerError):
    """Raised when a clip is not available in the mobile review queue."""


class UnsafeRenderPath(ReviewServerError):
    """Raised when metadata points outside the rendered clip directory."""


class UnsafeExportPath(ReviewServerError):
    """Raised when a download request points outside the exports directory."""


@dataclass(frozen=True)
class ReviewItem:
    clip: ClipState
    metadata_path: Path
    candidates: tuple[RenderCandidate, ...]


@dataclass(frozen=True)
class ApprovedExport:
    export_path: Path
    download_url: str


class ReviewQueueService:
    """Business operations for draining prepared rendered clips."""

    def __init__(
        self,
        *,
        config: ClipforgeConfig | None = None,
    ) -> None:
        self.config = config or load_config()

    def next_item(self) -> ReviewItem | None:
        clips = get_mobile_review_clips(db_path=self.config.state_db_path, limit=1)
        if not clips:
            return None
        return self._review_item(clips[0])

    def item_for_clip(self, clip_id: str) -> ReviewItem:
        for clip in get_mobile_review_clips(db_path=self.config.state_db_path):
            if clip.clip_id == clip_id:
                return self._review_item(clip)
        raise ReviewItemNotFound(f"Clip is not in the mobile review queue: {clip_id}.")

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

    def approve(self, *, clip_id: str, layout: str) -> ApprovedExport:
        existing_export = self._existing_approved_export(clip_id=clip_id, layout=layout)
        if existing_export is not None:
            return existing_export

        item = self.item_for_clip(clip_id)
        selected = self._candidate_for_layout(item, layout)
        streamer_login = item.clip.streamer_login or "unknown_streamer"
        if item.clip.render_dir is None:
            raise ReviewItemNotFound(f"Clip has no render directory: {clip_id}.")
        try:
            selected_export = export_selected_candidate(
                clip=item.clip,
                streamer_login=streamer_login,
                selected=selected,
                config=self.config,
                render_root=Path(item.clip.render_dir),
            )
            mark_clip_exported(
                item.clip.clip_id,
                selected_render_layout=selected.layout,
                selected_render_path=selected.path,
                export_path=selected_export.export_path,
                db_path=self.config.state_db_path,
            )
            write_selected_export_metadata(
                item.metadata_path,
                selected=selected,
                selected_export=selected_export,
            )
        except Exception as exc:
            mark_clip_failed(
                item.clip.clip_id,
                error_message=str(exc),
                db_path=self.config.state_db_path,
            )
            raise ReviewServerError(str(exc)) from exc
        return self._approved_export(selected_export.export_path)

    def export_file_path(self, *, relative_parts: tuple[str, ...]) -> Path:
        if not relative_parts or any(_unsafe_path_part(part) for part in relative_parts):
            raise UnsafeExportPath("Invalid export path.")

        exports_root = self.config.exports_dir.resolve()
        export_path = exports_root.joinpath(*relative_parts).resolve()
        if not export_path.is_relative_to(exports_root):
            raise UnsafeExportPath("Export path is outside the exports directory.")
        if not export_path.is_file():
            raise ReviewItemNotFound("Export not found.")
        return export_path

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

    def _existing_approved_export(
        self,
        *,
        clip_id: str,
        layout: str,
    ) -> ApprovedExport | None:
        clip = get_clip(clip_id, db_path=self.config.state_db_path)
        if clip is None or clip.status != "exported":
            return None
        if clip.selected_render_layout != layout:
            raise ReviewServerError(
                f"Clip was already approved with layout: {clip.selected_render_layout}."
            )
        if clip.export_path is None:
            return None
        export_path = Path(clip.export_path)
        try:
            resolved_path = self.export_file_path(
                relative_parts=export_path.resolve().relative_to(
                    self.config.exports_dir.resolve()
                ).parts
            )
        except ValueError as exc:
            raise UnsafeExportPath("Stored export path is outside the exports directory.") from exc
        except ReviewItemNotFound:
            return None
        return self._approved_export(resolved_path)

    def _approved_export(self, export_path: Path) -> ApprovedExport:
        relative = export_path.resolve().relative_to(self.config.exports_dir.resolve())
        download_url = "/exports/" + "/".join(
            quote(part, safe="") for part in relative.parts
        )
        return ApprovedExport(export_path=export_path, download_url=download_url)


def _has_path_separator(value: str) -> bool:
    return "/" in value or "\\" in value or value in {"", ".", ".."}


def _unsafe_path_part(value: str) -> bool:
    return not value or "/" in value or "\\" in value or value in {".", ".."}

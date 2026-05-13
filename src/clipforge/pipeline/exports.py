"""Helpers for exporting a reviewed render candidate."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.metadata import (
    RenderCandidate,
    final_resolution_for_layout,
    read_pipeline_metadata,
    resolution_payload,
)
from clipforge.pipeline.workflows import render_selected_layout_from_metadata
from clipforge.storage.paths import export_path as selected_export_path
from clipforge.storage.state import ClipState, mark_clip_exported, mark_clip_selected
from clipforge.utils.paths import ensure_directory


class ClipExportError(RuntimeError):
    """Raised when a selected render cannot be exported."""


@dataclass(frozen=True)
class SelectedExport:
    export_path: Path
    final_render_path: Path
    final_resolution: tuple[int, int] | None
    reused_preview: bool


def export_review_selection(
    *,
    clip: ClipState,
    streamer_login: str,
    metadata_path: Path,
    selected: RenderCandidate,
    force: bool,
    config: ClipforgeConfig,
    render_selected: Callable[..., Path] = render_selected_layout_from_metadata,
) -> SelectedExport:
    """Persist selection state, write the export, and mark the clip exported."""

    mark_clip_selected(
        clip.clip_id,
        selected_render_layout=selected.layout,
        selected_render_path=selected.path,
        db_path=config.state_db_path,
    )
    selected_export = export_selected_render(
        clip=clip,
        streamer_login=streamer_login,
        metadata_path=metadata_path,
        selected=selected,
        force=force,
        config=config,
        render_selected=render_selected,
    )
    mark_clip_exported(
        clip.clip_id,
        selected_render_layout=selected.layout,
        selected_render_path=selected.path,
        export_path=selected_export.export_path,
        db_path=config.state_db_path,
    )
    write_selected_export_metadata(
        metadata_path,
        selected=selected,
        selected_export=selected_export,
    )
    return selected_export


def export_selected_render(
    *,
    clip: ClipState,
    streamer_login: str,
    metadata_path: Path,
    selected: RenderCandidate,
    force: bool,
    config: ClipforgeConfig,
    render_selected: Callable[..., Path] = render_selected_layout_from_metadata,
) -> SelectedExport:
    """Copy or final-render one selected review candidate to its export path."""

    export_path = selected_export_path(
        config,
        streamer=streamer_login,
        title=clip.title,
        clip_id=clip.clip_id,
        layout=selected.layout,
    )
    if export_path.exists() and not force:
        raise ClipExportError(f"Export already exists: {export_path}. Re-run with --force.")

    ensure_directory(export_path.parent)
    payload = read_pipeline_metadata(metadata_path)
    final_resolution = final_resolution_for_layout(
        payload,
        selected_layout=selected.layout,
    )
    if selected_preview_matches_final(
        selected,
        final_resolution=final_resolution,
        config=config,
    ):
        copy_selected_render(source_path=selected.path, export_path=export_path)
        return SelectedExport(
            export_path=export_path,
            final_render_path=selected.path,
            final_resolution=final_resolution or selected.resolution,
            reused_preview=True,
        )

    try:
        render_selected(
            metadata_path,
            selected_layout=selected.layout,
            output_path=export_path,
            channel=streamer_login,
            config=config,
        )
    except Exception as exc:
        raise ClipExportError(
            f"Could not render selected layout {selected.layout!r} to {export_path}: {exc}"
        ) from exc
    return SelectedExport(
        export_path=export_path,
        final_render_path=export_path,
        final_resolution=final_resolution,
        reused_preview=False,
    )


def copy_selected_render(*, source_path: Path, export_path: Path) -> None:
    try:
        shutil.copy2(source_path, export_path)
    except OSError as exc:
        raise ClipExportError(
            f"Could not export selected render to {export_path}: {exc}"
        ) from exc


def selected_preview_matches_final(
    selected: RenderCandidate,
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


def write_selected_export_metadata(
    metadata_path: Path,
    *,
    selected: RenderCandidate,
    selected_export: SelectedExport,
) -> None:
    payload = read_pipeline_metadata(metadata_path)
    payload["selected_export"] = {
        "layout": selected.layout,
        "preview_candidate": {
            "path": str(selected.path),
            "resolution": resolution_payload(selected.resolution),
        },
        "final_render": {
            "path": str(selected_export.final_render_path),
            "resolution": resolution_payload(selected_export.final_resolution),
        },
        "export": {
            "path": str(selected_export.export_path),
            "resolution": resolution_payload(selected_export.final_resolution),
        },
        "reused_preview": selected_export.reused_preview,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

"""Shared pipeline metadata readers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipforge.media.layouts import Layout, parse_layout
from clipforge.media.render_settings import FFmpegRenderSettings


class PipelineMetadataError(RuntimeError):
    """Raised when persisted pipeline metadata is missing or malformed."""


@dataclass(frozen=True)
class RenderCandidate:
    layout: str
    path: Path
    resolution: tuple[int, int] | None = None
    render_settings: FFmpegRenderSettings | None = None


def read_pipeline_metadata(metadata_path: Path) -> dict[str, Any]:
    """Read a pipeline metadata JSON object."""

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PipelineMetadataError(
            f"Could not read pipeline metadata {metadata_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PipelineMetadataError(
            f"Pipeline metadata is not valid JSON: {metadata_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PipelineMetadataError(
            f"Pipeline metadata must be a JSON object: {metadata_path}"
        )
    return payload


def render_candidates_from_metadata(metadata_path: Path) -> tuple[RenderCandidate, ...]:
    """Load usable review render candidates from pipeline metadata."""

    payload = read_pipeline_metadata(metadata_path)
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise PipelineMetadataError(
            f"Pipeline metadata has no render outputs: {metadata_path}."
        )

    candidates: list[RenderCandidate] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        layout = output.get("layout")
        path = output.get("path")
        if isinstance(layout, str) and isinstance(path, str):
            candidates.append(
                RenderCandidate(
                    layout=layout,
                    path=Path(path),
                    resolution=output_resolution(output.get("resolution")),
                    render_settings=output_render_settings(output),
                )
            )

    if not candidates:
        raise PipelineMetadataError(
            f"Pipeline metadata has no usable render outputs: {metadata_path}."
        )
    return tuple(candidates)


def metadata_source_path(payload: dict[str, Any], *, metadata_path: Path) -> Path:
    value = payload.get("source_path")
    if not isinstance(value, str) or not value:
        raise PipelineMetadataError(
            f"Pipeline metadata is missing source_path: {metadata_path}"
        )
    return Path(value)


def metadata_layout(payload: dict[str, Any], *, selected_layout: str) -> Layout:
    layouts = payload.get("layouts")
    if not isinstance(layouts, list):
        raise PipelineMetadataError("Pipeline metadata is missing layouts.")
    for layout_payload in layouts:
        if not isinstance(layout_payload, dict):
            continue
        if layout_payload.get("name") == selected_layout:
            return parse_layout(layout_payload)
    raise PipelineMetadataError(
        f"Pipeline metadata does not contain selected layout: {selected_layout}."
    )


def metadata_optional_path(payload: dict[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def final_resolution_for_layout(
    payload: dict[str, Any],
    *,
    selected_layout: str,
) -> tuple[int, int] | None:
    layouts = payload.get("layouts")
    if isinstance(layouts, list):
        for layout in layouts:
            if not isinstance(layout, dict) or layout.get("name") != selected_layout:
                continue
            resolution = output_resolution(layout.get("output"))
            if resolution is not None:
                return resolution
    return output_resolution(payload.get("target_resolution"))


def output_resolution(value: object) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    width = value.get("width")
    height = value.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    return (width, height)


def resolution_payload(value: tuple[int, int] | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"width": value[0], "height": value[1]}


def output_render_settings(value: dict[str, object]) -> FFmpegRenderSettings | None:
    settings = value.get("render_settings")
    if not isinstance(settings, dict):
        return None
    supported_keys = FFmpegRenderSettings().__dict__.keys()
    return FFmpegRenderSettings(
        **{key: settings[key] for key in supported_keys if key in settings}
    ).normalized()

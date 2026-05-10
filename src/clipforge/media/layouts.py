"""Load and validate editable clip layout templates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipforge.core.config import ANALYSIS_DIR, EXAMPLE_LAYOUTS_DIR
from clipforge.utils.paths import clip_analysis_dir, ensure_directory, safe_filename
from clipforge.utils.json_validation import required_int
from clipforge.utils.json_validation import required_list
from clipforge.utils.json_validation import required_number
from clipforge.utils.json_validation import required_object
from clipforge.utils.json_validation import required_string


class LayoutError(RuntimeError):
    """Raised when a layout file is missing, malformed, or invalid."""


DEFAULT_LAYOUT_NAMES = ("center_gameplay", "facecam_focus", "hybrid")
GENERATED_LAYOUT_NAMES = ("detected_streamer_focus", "detected_hybrid")
DYNAMIC_LAYOUT_CONFIDENCE_THRESHOLD = 0.58
SUPPORTED_REGION_EFFECTS = frozenset({"blur"})
HYBRID_STREAMER_REGION_NAMES = frozenset({"facecam", "streamer"})
HYBRID_GAMEPLAY_REGION_NAME = "gameplay"
HYBRID_STREAMER_OUTPUT_HEIGHT = 0.34
HYBRID_CAPTION_BAND_HEIGHT = 0.10


@dataclass(frozen=True)
class NormalizedRect:
    """A rectangle described with normalized coordinates from 0 to 1."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class OutputSize:
    """Target render dimensions in pixels."""

    width: int
    height: int


@dataclass(frozen=True)
class LayoutRegion:
    """A named mapping from a source crop to an output canvas region."""

    name: str
    source_region: NormalizedRect
    output_region: NormalizedRect
    effect: str | None = None


@dataclass(frozen=True)
class Layout:
    """Validated layout data that the renderer can consume."""

    name: str
    description: str
    output: OutputSize
    regions: tuple[LayoutRegion, ...]
    caption_region: NormalizedRect | None = None


def load_layout(path: Path) -> Layout:
    """Load one layout JSON file and return validated layout data."""

    try:
        raw_layout = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LayoutError(f"Layout file not found: {path}") from exc
    except OSError as exc:
        raise LayoutError(f"Could not read layout file {path}: {exc}") from exc

    try:
        payload = json.loads(raw_layout)
    except json.JSONDecodeError as exc:
        raise LayoutError(f"Invalid JSON in layout file {path}: {exc.msg}") from exc

    try:
        return parse_layout(payload)
    except LayoutError as exc:
        raise LayoutError(f"Invalid layout file {path}: {exc}") from exc


def load_example_layout(
    name: str,
    *,
    layouts_dir: Path = EXAMPLE_LAYOUTS_DIR,
) -> Layout:
    """Load one committed example layout by name."""

    return load_layout(layouts_dir / f"{name}.json")


def load_example_layouts(
    names: tuple[str, ...] = DEFAULT_LAYOUT_NAMES,
    *,
    layouts_dir: Path = EXAMPLE_LAYOUTS_DIR,
) -> tuple[Layout, ...]:
    """Load the default MVP example layouts in a stable order."""

    return tuple(load_example_layout(name, layouts_dir=layouts_dir) for name in names)


def generate_detected_layout_candidates(
    *,
    clip_id: str,
    analysis_dir: Path = ANALYSIS_DIR,
    example_layouts_dir: Path = EXAMPLE_LAYOUTS_DIR,
    confidence_threshold: float = DYNAMIC_LAYOUT_CONFIDENCE_THRESHOLD,
) -> tuple[Path, ...]:
    """Generate deterministic layout JSON candidates from overlay analysis metadata."""

    safe_clip_id = _safe_clip_id(clip_id)
    analysis_clip_dir = clip_analysis_dir(analysis_dir, safe_clip_id)
    overlay_path = analysis_clip_dir / "overlay.json"
    if not overlay_path.is_file():
        raise LayoutError(f"Overlay metadata not found: {overlay_path}")

    overlay_metadata = _read_overlay_metadata(overlay_path)
    layouts_dir = ensure_directory(analysis_clip_dir / "layouts")
    if _can_generate_dynamic_layouts(
        overlay_metadata,
        confidence_threshold=confidence_threshold,
    ):
        payloads = _dynamic_layout_payloads(
            overlay_metadata,
            overlay_path=overlay_path,
            example_layouts_dir=example_layouts_dir,
        )
    else:
        payloads = _fallback_layout_payloads(
            overlay_metadata,
            overlay_path=overlay_path,
            example_layouts_dir=example_layouts_dir,
            confidence_threshold=confidence_threshold,
        )

    paths: list[Path] = []
    for name in GENERATED_LAYOUT_NAMES:
        payload = payloads[name]
        parse_layout(payload)
        path = layouts_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def parse_layout(payload: Any) -> Layout:
    """Validate a decoded layout object."""

    if not isinstance(payload, dict):
        raise LayoutError("layout root must be an object.")

    name = required_string(payload, "name", context="layout", error_cls=LayoutError)
    description = required_string(
        payload,
        "description",
        context="layout",
        error_cls=LayoutError,
    )
    output = _parse_output(
        required_object(payload, "output", context="layout", error_cls=LayoutError)
    )
    regions_payload = required_list(
        payload,
        "regions",
        context="layout",
        error_cls=LayoutError,
    )
    if not regions_payload:
        raise LayoutError("layout.regions must contain at least one region.")

    regions = tuple(
        _parse_region(region_payload, index=index)
        for index, region_payload in enumerate(regions_payload)
    )
    caption_region = _parse_caption_region(payload, regions)

    return Layout(
        name=name,
        description=description,
        output=output,
        regions=regions,
        caption_region=caption_region,
    )


def _parse_output(payload: dict[str, Any]) -> OutputSize:
    width = required_int(payload, "width", context="layout.output", error_cls=LayoutError)
    height = required_int(payload, "height", context="layout.output", error_cls=LayoutError)

    if width <= 0 or height <= 0:
        raise LayoutError("layout.output width and height must be positive integers.")

    return OutputSize(width=width, height=height)


def _parse_region(payload: Any, *, index: int) -> LayoutRegion:
    context = f"layout.regions[{index}]"
    if not isinstance(payload, dict):
        raise LayoutError(f"{context} must be an object.")

    return LayoutRegion(
        name=required_string(payload, "name", context=context, error_cls=LayoutError),
        source_region=_parse_rect(
            required_object(
                payload,
                "source_region",
                context=context,
                error_cls=LayoutError,
            ),
            context=f"{context}.source_region",
        ),
        output_region=_parse_rect(
            required_object(
                payload,
                "output_region",
                context=context,
                error_cls=LayoutError,
            ),
            context=f"{context}.output_region",
        ),
        effect=_parse_region_effect(payload.get("effect"), context=f"{context}.effect"),
    )


def _parse_caption_region(
    payload: dict[str, Any],
    regions: tuple[LayoutRegion, ...],
) -> NormalizedRect | None:
    explicit_caption_region = payload.get("caption_region")
    if explicit_caption_region is not None:
        if not isinstance(explicit_caption_region, dict):
            raise LayoutError("layout.caption_region must be an object.")
        caption_region = _parse_rect(
            explicit_caption_region,
            context="layout.caption_region",
        )
    else:
        caption_region = _derived_caption_region(regions)

    return caption_region


def _derived_caption_region(
    regions: tuple[LayoutRegion, ...],
) -> NormalizedRect | None:
    streamer_regions = [
        region.output_region
        for region in regions
        if region.name in HYBRID_STREAMER_REGION_NAMES
    ]
    gameplay_regions = [
        region.output_region
        for region in regions
        if region.name == HYBRID_GAMEPLAY_REGION_NAME
    ]
    if not streamer_regions or not gameplay_regions:
        return None

    streamer_bottom = max(region.y + region.height for region in streamer_regions)
    gameplay_top = min(region.y for region in gameplay_regions)
    gameplay_bottom = max(region.y + region.height for region in gameplay_regions)
    if gameplay_top < streamer_bottom:
        return None

    if gameplay_top == streamer_bottom:
        caption_height = min(HYBRID_CAPTION_BAND_HEIGHT, gameplay_bottom - streamer_bottom)
    else:
        caption_height = gameplay_top - streamer_bottom
    if caption_height <= 0:
        return None

    return NormalizedRect(
        x=0.0,
        y=_round(streamer_bottom),
        width=1.0,
        height=_round(caption_height),
    )


def _parse_region_effect(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LayoutError(f"{context} must be a string.")
    if value not in SUPPORTED_REGION_EFFECTS:
        supported = ", ".join(sorted(SUPPORTED_REGION_EFFECTS))
        raise LayoutError(f"{context} must be one of: {supported}.")
    return value


def _parse_rect(payload: dict[str, Any], *, context: str) -> NormalizedRect:
    rect = NormalizedRect(
        x=required_number(payload, "x", context=context, error_cls=LayoutError),
        y=required_number(payload, "y", context=context, error_cls=LayoutError),
        width=required_number(payload, "width", context=context, error_cls=LayoutError),
        height=required_number(payload, "height", context=context, error_cls=LayoutError),
    )
    _validate_rect_bounds(rect, context=context)
    return rect


def _validate_rect_bounds(rect: NormalizedRect, *, context: str) -> None:
    values = {
        "x": rect.x,
        "y": rect.y,
        "width": rect.width,
        "height": rect.height,
    }

    for key, value in values.items():
        if not math.isfinite(value):
            raise LayoutError(f"{context}.{key} must be a finite number.")

        if value < 0 or value > 1:
            raise LayoutError(f"{context}.{key} must be between 0 and 1.")

    if rect.width <= 0 or rect.height <= 0:
        raise LayoutError(f"{context} width and height must be greater than 0.")

    if rect.x + rect.width > 1:
        raise LayoutError(f"{context} x + width must be less than or equal to 1.")

    if rect.y + rect.height > 1:
        raise LayoutError(f"{context} y + height must be less than or equal to 1.")


def _dynamic_layout_payloads(
    overlay_metadata: dict[str, object],
    *,
    overlay_path: Path,
    example_layouts_dir: Path,
) -> dict[str, dict[str, object]]:
    overlay_rect = _overlay_rect_from_metadata(overlay_metadata)
    center_gameplay = load_example_layout(
        "center_gameplay",
        layouts_dir=example_layouts_dir,
    )
    gameplay_source = center_gameplay.regions[0].source_region
    output = center_gameplay.output
    metadata = _generation_metadata(
        overlay_metadata,
        overlay_path=overlay_path,
        fallback_generated=False,
    )

    full_source = NormalizedRect(x=0.0, y=0.0, width=1.0, height=1.0)
    full_output = NormalizedRect(x=0.0, y=0.0, width=1.0, height=1.0)
    focus_streamer_output = NormalizedRect(x=0.0, y=0.19, width=1.0, height=0.62)
    hybrid_streamer_output = NormalizedRect(
        x=0.0,
        y=0.0,
        width=1.0,
        height=HYBRID_STREAMER_OUTPUT_HEIGHT,
    )
    hybrid_gameplay_output = NormalizedRect(
        x=0.0,
        y=HYBRID_STREAMER_OUTPUT_HEIGHT,
        width=1.0,
        height=1.0 - HYBRID_STREAMER_OUTPUT_HEIGHT,
    )

    return {
        "detected_streamer_focus": {
            "name": "detected_streamer_focus",
            "description": (
                "Generated streamer-focused layout from detected overlay analysis."
            ),
            "output": _output_to_payload(output),
            "metadata": {
                **metadata,
                "layout_goal": "streamer_focus",
            },
            "regions": [
                {
                    "name": "background",
                    "source_region": _rect_to_payload(full_source),
                    "output_region": _rect_to_payload(full_output),
                    "effect": "blur",
                },
                {
                    "name": "streamer",
                    "source_region": _rect_to_payload(overlay_rect),
                    "output_region": _rect_to_payload(focus_streamer_output),
                },
            ],
        },
        "detected_hybrid": {
            "name": "detected_hybrid",
            "description": "Generated balanced layout from detected overlay analysis.",
            "output": _output_to_payload(output),
            "metadata": {
                **metadata,
                "layout_goal": "hybrid",
            },
            "regions": [
                {
                    "name": "gameplay",
                    "source_region": _rect_to_payload(gameplay_source),
                    "output_region": _rect_to_payload(hybrid_gameplay_output),
                },
                {
                    "name": "streamer",
                    "source_region": _rect_to_payload(overlay_rect),
                    "output_region": _rect_to_payload(hybrid_streamer_output),
                },
            ],
        },
    }


def _fallback_layout_payloads(
    overlay_metadata: dict[str, object],
    *,
    overlay_path: Path,
    example_layouts_dir: Path,
    confidence_threshold: float,
) -> dict[str, dict[str, object]]:
    metadata = _generation_metadata(
        overlay_metadata,
        overlay_path=overlay_path,
        fallback_generated=True,
        confidence_threshold=confidence_threshold,
    )
    focus_template = load_example_layout(
        "facecam_focus",
        layouts_dir=example_layouts_dir,
    )
    hybrid_template = load_example_layout("hybrid", layouts_dir=example_layouts_dir)

    return {
        "detected_streamer_focus": _renamed_layout_payload(
            focus_template,
            name="detected_streamer_focus",
            description=(
                "Fallback streamer-focused layout generated from static facecam template."
            ),
            metadata={
                **metadata,
                "layout_goal": "streamer_focus",
                "source_template": "facecam_focus",
            },
        ),
        "detected_hybrid": _renamed_layout_payload(
            hybrid_template,
            name="detected_hybrid",
            description="Fallback hybrid layout generated from static hybrid template.",
            metadata={
                **metadata,
                "layout_goal": "hybrid",
                "source_template": "hybrid",
            },
        ),
    }


def _renamed_layout_payload(
    layout: Layout,
    *,
    name: str,
    description: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "output": _output_to_payload(layout.output),
        "metadata": metadata,
        "regions": [_region_to_payload(region) for region in layout.regions],
    }


def _region_to_payload(region: LayoutRegion) -> dict[str, object]:
    return {
        "name": region.name,
        "source_region": _rect_to_payload(region.source_region),
        "output_region": _rect_to_payload(region.output_region),
        **({"effect": region.effect} if region.effect is not None else {}),
    }


def _output_to_payload(output: OutputSize) -> dict[str, int]:
    return {
        "width": output.width,
        "height": output.height,
    }


def _generation_metadata(
    overlay_metadata: dict[str, object],
    *,
    overlay_path: Path,
    fallback_generated: bool,
    confidence_threshold: float | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "generated_by": "clipforge analyze layouts",
        "generation_source": "overlay_analysis",
        "overlay_path": str(overlay_path),
        "overlay_confidence": _overlay_confidence(overlay_metadata),
        "overlay_fallback": bool(overlay_metadata.get("fallback")),
        "fallback_generated": fallback_generated,
        "overlay_reason": str(overlay_metadata.get("reason") or ""),
    }
    selected_overlay_rect = overlay_metadata.get("selected_overlay_rect")
    if isinstance(selected_overlay_rect, dict):
        metadata["selected_overlay_rect"] = selected_overlay_rect
    if confidence_threshold is not None:
        metadata["confidence_threshold"] = confidence_threshold
    return metadata


def _can_generate_dynamic_layouts(
    overlay_metadata: dict[str, object],
    *,
    confidence_threshold: float,
) -> bool:
    if bool(overlay_metadata.get("fallback")):
        return False
    if _overlay_confidence(overlay_metadata) < confidence_threshold:
        return False
    return overlay_metadata.get("selected_overlay_rect") is not None


def _overlay_rect_from_metadata(payload: dict[str, object]) -> NormalizedRect:
    rect_payload = payload.get("selected_overlay_rect")
    if not isinstance(rect_payload, dict):
        raise LayoutError("Overlay metadata must include selected_overlay_rect.")
    return _rect_from_payload(rect_payload, context="selected_overlay_rect")


def _rect_from_payload(payload: dict[str, object], *, context: str) -> NormalizedRect:
    rect = NormalizedRect(
        x=_number_from_payload(payload.get("x"), context=f"{context}.x"),
        y=_number_from_payload(payload.get("y"), context=f"{context}.y"),
        width=_number_from_payload(payload.get("width"), context=f"{context}.width"),
        height=_number_from_payload(payload.get("height"), context=f"{context}.height"),
    )
    _validate_rect_bounds(rect, context=context)
    return rect


def _read_overlay_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutError(f"Overlay metadata is not valid JSON: {path}") from exc
    except OSError as exc:
        raise LayoutError(f"Could not read overlay metadata {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise LayoutError(f"Overlay metadata must be a JSON object: {path}")
    return payload


def _overlay_confidence(payload: dict[str, object]) -> float:
    value = payload.get("confidence", 0.0)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _number_from_payload(value: object, *, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise LayoutError(f"Overlay metadata contains an invalid {context}.")
    return float(value)


def _rect_to_payload(rect: NormalizedRect) -> dict[str, float]:
    return {
        "x": _round(rect.x),
        "y": _round(rect.y),
        "width": _round(rect.width),
        "height": _round(rect.height),
    }


def _safe_clip_id(clip_id: str) -> str:
    if not clip_id.strip():
        raise LayoutError("clip_id must not be empty.")
    return safe_filename(clip_id)


def _round(value: float) -> float:
    return round(value, 6)


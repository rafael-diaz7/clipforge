"""Load and validate editable clip layout templates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipforge.core.config import EXAMPLE_LAYOUTS_DIR


class LayoutError(RuntimeError):
    """Raised when a layout file is missing, malformed, or invalid."""


DEFAULT_LAYOUT_NAMES = ("center_gameplay", "facecam_focus", "hybrid")


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


@dataclass(frozen=True)
class Layout:
    """Validated layout data that the renderer can consume."""

    name: str
    description: str
    output: OutputSize
    regions: tuple[LayoutRegion, ...]


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


def parse_layout(payload: Any) -> Layout:
    """Validate a decoded layout object."""

    if not isinstance(payload, dict):
        raise LayoutError("layout root must be an object.")

    name = _required_string(payload, "name", context="layout")
    description = _required_string(payload, "description", context="layout")
    output = _parse_output(_required_object(payload, "output", context="layout"))
    regions_payload = _required_list(payload, "regions", context="layout")
    if not regions_payload:
        raise LayoutError("layout.regions must contain at least one region.")

    regions = tuple(
        _parse_region(region_payload, index=index)
        for index, region_payload in enumerate(regions_payload)
    )

    return Layout(
        name=name,
        description=description,
        output=output,
        regions=regions,
    )


def _parse_output(payload: dict[str, Any]) -> OutputSize:
    width = _required_int(payload, "width", context="layout.output")
    height = _required_int(payload, "height", context="layout.output")

    if width <= 0 or height <= 0:
        raise LayoutError("layout.output width and height must be positive integers.")

    return OutputSize(width=width, height=height)


def _parse_region(payload: Any, *, index: int) -> LayoutRegion:
    context = f"layout.regions[{index}]"
    if not isinstance(payload, dict):
        raise LayoutError(f"{context} must be an object.")

    return LayoutRegion(
        name=_required_string(payload, "name", context=context),
        source_region=_parse_rect(
            _required_object(payload, "source_region", context=context),
            context=f"{context}.source_region",
        ),
        output_region=_parse_rect(
            _required_object(payload, "output_region", context=context),
            context=f"{context}.output_region",
        ),
    )


def _parse_rect(payload: dict[str, Any], *, context: str) -> NormalizedRect:
    rect = NormalizedRect(
        x=_required_number(payload, "x", context=context),
        y=_required_number(payload, "y", context=context),
        width=_required_number(payload, "width", context=context),
        height=_required_number(payload, "height", context=context),
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


# TODO: Consider shared JSON validation helpers with integrations.twitch once
# their caller-specific error wording can be preserved.
def _required_object(payload: dict[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = _required_value(payload, key, context=context)
    if not isinstance(value, dict):
        raise LayoutError(f"{context}.{key} must be an object.")
    return value


def _required_list(payload: dict[str, Any], key: str, *, context: str) -> list[Any]:
    value = _required_value(payload, key, context=context)
    if not isinstance(value, list):
        raise LayoutError(f"{context}.{key} must be a list.")
    return value


def _required_string(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = _required_value(payload, key, context=context)
    if not isinstance(value, str) or not value.strip():
        raise LayoutError(f"{context}.{key} must be a non-empty string.")
    return value


def _required_int(payload: dict[str, Any], key: str, *, context: str) -> int:
    value = _required_value(payload, key, context=context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LayoutError(f"{context}.{key} must be an integer.")
    return value


def _required_number(payload: dict[str, Any], key: str, *, context: str) -> float:
    value = _required_value(payload, key, context=context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayoutError(f"{context}.{key} must be a number.")
    return float(value)


def _required_value(payload: dict[str, Any], key: str, *, context: str) -> Any:
    try:
        return payload[key]
    except KeyError as exc:
        raise LayoutError(f"{context}.{key} is required.") from exc

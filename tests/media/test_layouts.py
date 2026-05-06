from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.layouts import (
    DEFAULT_LAYOUT_NAMES,
    LayoutError,
    NormalizedRect,
    load_example_layouts,
    load_layout,
    parse_layout,
)


def test_load_layout_returns_validated_layout(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "name": "center_gameplay",
                "description": "Centered crop.",
                "output": {"width": 1080, "height": 1920},
                "regions": [
                    {
                        "name": "gameplay",
                        "source_region": {
                            "x": 0.25,
                            "y": 0.0,
                            "width": 0.5,
                            "height": 1.0,
                        },
                        "output_region": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 1.0,
                            "height": 1.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    layout = load_layout(path)

    assert layout.name == "center_gameplay"
    assert layout.output.width == 1080
    assert layout.output.height == 1920
    assert layout.regions[0].name == "gameplay"
    assert layout.regions[0].source_region == NormalizedRect(
        x=0.25,
        y=0.0,
        width=0.5,
        height=1.0,
    )


def test_load_example_layouts_loads_three_mvp_templates() -> None:
    layouts = load_example_layouts()

    assert [layout.name for layout in layouts] == list(DEFAULT_LAYOUT_NAMES)
    assert all(layout.output.width == 1080 for layout in layouts)
    assert all(layout.output.height == 1920 for layout in layouts)
    assert len(layouts[2].regions) == 2


def test_load_layout_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LayoutError, match="Layout file not found"):
        load_layout(tmp_path / "missing.json")


def test_load_layout_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(LayoutError, match="Invalid JSON"):
        load_layout(path)


def test_parse_layout_requires_fields() -> None:
    with pytest.raises(LayoutError, match="layout.name is required"):
        parse_layout({"description": "missing name"})


def test_parse_layout_rejects_empty_regions() -> None:
    with pytest.raises(LayoutError, match="must contain at least one region"):
        parse_layout(
            {
                "name": "empty",
                "description": "No regions.",
                "output": {"width": 1080, "height": 1920},
                "regions": [],
            }
        )


def test_parse_layout_rejects_out_of_bounds_coordinates() -> None:
    with pytest.raises(LayoutError, match="x \\+ width"):
        parse_layout(
            {
                "name": "bad",
                "description": "Bad bounds.",
                "output": {"width": 1080, "height": 1920},
                "regions": [
                    {
                        "name": "gameplay",
                        "source_region": {
                            "x": 0.75,
                            "y": 0.0,
                            "width": 0.5,
                            "height": 1.0,
                        },
                        "output_region": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 1.0,
                            "height": 1.0,
                        },
                    }
                ],
            }
        )


def test_parse_layout_rejects_non_numeric_coordinates() -> None:
    with pytest.raises(LayoutError, match="must be a number"):
        parse_layout(
            {
                "name": "bad",
                "description": "Bad type.",
                "output": {"width": 1080, "height": 1920},
                "regions": [
                    {
                        "name": "gameplay",
                        "source_region": {
                            "x": "left",
                            "y": 0.0,
                            "width": 0.5,
                            "height": 1.0,
                        },
                        "output_region": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 1.0,
                            "height": 1.0,
                        },
                    }
                ],
            }
        )


def test_parse_layout_rejects_non_finite_coordinates() -> None:
    with pytest.raises(LayoutError, match="finite number"):
        parse_layout(
            {
                "name": "bad",
                "description": "Bad numeric value.",
                "output": {"width": 1080, "height": 1920},
                "regions": [
                    {
                        "name": "gameplay",
                        "source_region": {
                            "x": float("nan"),
                            "y": 0.0,
                            "width": 0.5,
                            "height": 1.0,
                        },
                        "output_region": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 1.0,
                            "height": 1.0,
                        },
                    }
                ],
            }
        )

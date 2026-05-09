from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.media.layouts import (
    DEFAULT_LAYOUT_NAMES,
    LayoutError,
    NormalizedRect,
    generate_detected_layout_candidates,
    load_example_layout,
    load_example_layouts,
    load_layout,
    parse_layout,
)
from clipforge.media.render import build_filter_complex


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


def test_parse_layout_rejects_unsupported_region_effect() -> None:
    with pytest.raises(LayoutError, match="must be one of: blur"):
        parse_layout(
            {
                "name": "bad",
                "description": "Bad effect.",
                "output": {"width": 1080, "height": 1920},
                "regions": [
                    {
                        "name": "background",
                        "source_region": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 1.0,
                            "height": 1.0,
                        },
                        "output_region": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 1.0,
                            "height": 1.0,
                        },
                        "effect": "sharpen",
                    }
                ],
            }
        )


def test_generate_detected_layouts_from_high_confidence_overlay(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    overlay_rect = {"x": 0.02, "y": 0.08, "width": 0.28, "height": 0.34}
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        selected_overlay_rect=overlay_rect,
        confidence=0.82,
        fallback=False,
    )

    paths = generate_detected_layout_candidates(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
    )

    assert paths == (
        analysis_dir / "clip-123" / "layouts" / "detected_streamer_focus.json",
        analysis_dir / "clip-123" / "layouts" / "detected_hybrid.json",
    )
    layouts = tuple(load_layout(path) for path in paths)
    assert [layout.name for layout in layouts] == [
        "detected_streamer_focus",
        "detected_hybrid",
    ]
    focus_background = layouts[0].regions[0]
    focus_streamer = layouts[0].regions[1]
    hybrid_gameplay = layouts[1].regions[0]
    hybrid_streamer = layouts[1].regions[1]

    assert focus_background.name == "background"
    assert focus_background.source_region == NormalizedRect(
        x=0.0,
        y=0.0,
        width=1.0,
        height=1.0,
    )
    assert focus_background.effect == "blur"
    assert focus_streamer.name == "streamer"
    assert focus_streamer.source_region == NormalizedRect(**overlay_rect)
    assert focus_streamer.source_region != focus_background.source_region
    assert focus_streamer.output_region.y > 0.0
    assert focus_streamer.output_region.y + focus_streamer.output_region.height < 1.0
    assert hybrid_streamer.source_region == NormalizedRect(**overlay_rect)
    assert hybrid_streamer.output_region == NormalizedRect(
        x=0.0,
        y=0.0,
        width=1.0,
        height=0.4,
    )
    assert hybrid_gameplay.output_region == NormalizedRect(
        x=0.0,
        y=0.4,
        width=1.0,
        height=0.6,
    )
    assert focus_streamer.output_region.height > hybrid_streamer.output_region.height

    payload = _read_json(paths[0])
    assert payload["metadata"]["generation_source"] == "overlay_analysis"
    assert payload["metadata"]["overlay_confidence"] == 0.82
    assert payload["metadata"]["fallback_generated"] is False
    assert payload["regions"][0]["effect"] == "blur"


def test_fallback_overlay_generates_static_layout_candidates(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        selected_overlay_rect=None,
        confidence=0.31,
        fallback=True,
        reason="fallback: no face detections found in sampled frames",
    )

    paths = generate_detected_layout_candidates(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
    )

    focus = load_layout(paths[0])
    hybrid = load_layout(paths[1])
    assert focus.name == "detected_streamer_focus"
    assert hybrid.name == "detected_hybrid"
    assert focus.regions == load_example_layout("facecam_focus").regions
    assert hybrid.regions == load_example_layout("hybrid").regions

    payload = _read_json(paths[1])
    assert payload["metadata"]["fallback_generated"] is True
    assert payload["metadata"]["source_template"] == "hybrid"
    assert payload["metadata"]["confidence_threshold"] == 0.58


def test_generated_layout_json_matches_renderer_schema(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        selected_overlay_rect={"x": 0.64, "y": 0.04, "width": 0.24, "height": 0.28},
        confidence=0.9,
        fallback=False,
    )

    paths = generate_detected_layout_candidates(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
    )

    for path in paths:
        layout = load_layout(path)
        filter_complex = build_filter_complex(layout)
        assert filter_complex.endswith("[out]")

    assert "[0:v]split=2[src0][src1]" in build_filter_complex(load_layout(paths[1]))
    focus_filter_complex = build_filter_complex(load_layout(paths[0]))
    assert "[0:v]split=2[src0][src1]" in focus_filter_complex
    assert "boxblur=20:1" in focus_filter_complex


def test_generated_layouts_are_deterministic(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        selected_overlay_rect={"x": 0.04, "y": 0.12, "width": 0.2, "height": 0.3},
        confidence=0.75,
        fallback=False,
    )

    first_paths = generate_detected_layout_candidates(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
    )
    first_contents = tuple(path.read_text(encoding="utf-8") for path in first_paths)
    second_paths = generate_detected_layout_candidates(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
    )
    second_contents = tuple(path.read_text(encoding="utf-8") for path in second_paths)

    assert second_paths == first_paths
    assert second_contents == first_contents


def test_low_confidence_overlay_uses_stable_fallback_paths(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        selected_overlay_rect={"x": 0.04, "y": 0.12, "width": 0.2, "height": 0.3},
        confidence=0.2,
        fallback=False,
    )

    paths = generate_detected_layout_candidates(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
    )

    assert paths == (
        analysis_dir / "clip-123" / "layouts" / "detected_streamer_focus.json",
        analysis_dir / "clip-123" / "layouts" / "detected_hybrid.json",
    )
    assert _read_json(paths[0])["metadata"]["fallback_generated"] is True


def _write_overlay_metadata(
    analysis_dir: Path,
    *,
    clip_id: str,
    selected_overlay_rect: dict[str, float] | None,
    confidence: float,
    fallback: bool,
    reason: str = "selected stable edge/corner face cluster",
) -> Path:
    path = analysis_dir / clip_id / "overlay.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "selected_rect": selected_overlay_rect,
                "selected_face_rect": None,
                "selected_overlay_rect": selected_overlay_rect,
                "confidence": confidence,
                "fallback": fallback,
                "reason": reason,
                "candidate_clusters": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload

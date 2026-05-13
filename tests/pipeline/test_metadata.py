from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.media.render_settings import FFmpegRenderSettings
from clipforge.pipeline.metadata import (
    PipelineMetadataError,
    final_resolution_for_layout,
    read_pipeline_metadata,
    render_candidates_from_metadata,
)


def test_render_candidates_from_metadata_loads_paths_resolution_and_settings(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "layout": "hybrid",
                        "path": str(tmp_path / "hybrid.mp4"),
                        "resolution": {"width": 720, "height": 1280},
                        "render_settings": {
                            "encoder": "libx264",
                            "preset": "veryfast",
                            "crf": 28,
                            "quality": None,
                            "threads": 0,
                            "fallback_to_software": True,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = render_candidates_from_metadata(metadata_path)

    assert len(candidates) == 1
    assert candidates[0].layout == "hybrid"
    assert candidates[0].path == tmp_path / "hybrid.mp4"
    assert candidates[0].resolution == (720, 1280)
    assert candidates[0].render_settings == FFmpegRenderSettings(
        preset="veryfast",
        crf=28,
    )


def test_render_candidates_from_metadata_requires_usable_outputs(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"outputs": [{"layout": "hybrid"}]}), encoding="utf-8")

    with pytest.raises(PipelineMetadataError, match="no usable render outputs"):
        render_candidates_from_metadata(metadata_path)


def test_final_resolution_prefers_selected_layout_output(tmp_path: Path) -> None:
    payload = read_pipeline_metadata(
        _write_metadata(
            tmp_path,
            {
                "target_resolution": {"width": 1080, "height": 1920},
                "layouts": [
                    {
                        "name": "hybrid",
                        "output": {"width": 1440, "height": 2560},
                    }
                ],
            },
        )
    )

    assert final_resolution_for_layout(payload, selected_layout="hybrid") == (1440, 2560)


def _write_metadata(tmp_path: Path, payload: dict[str, object]) -> Path:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path

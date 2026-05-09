from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.media.analyze import AnalysisError
from clipforge.media.overlay import (
    FaceDetection,
    NormalizedRect,
    OverlayDetectorUnavailable,
    analyze_overlay,
)


class SyntheticDetector:
    def __init__(self, detections: dict[str, tuple[FaceDetection, ...]]) -> None:
        self._detections = detections

    def detect(self, frame_path: Path) -> tuple[FaceDetection, ...]:
        return self._detections.get(frame_path.name, ())


def test_stable_corner_face_wins_over_huge_central_face(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detections = {
        name: (
            _detection(0.02 + index * 0.002, 0.60, 0.22, 0.22),
            _detection(0.20, 0.12, 0.58, 0.58),
        )
        for index, name in enumerate(frame_names)
    }

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector(detections),
    )

    payload = _read_json(overlay_path)
    assert payload["fallback"] is False
    assert payload["confidence"] > 0.7
    assert payload["selected_rect"]["x"] < 0.05
    assert payload["selected_rect"]["y"] > 0.55
    assert payload["candidate_clusters"][0]["component_scores"]["edge_proximity"] > 0.7


def test_huge_central_face_is_penalized_to_fallback(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.18, 0.10, 0.62, 0.62),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    assert payload["fallback"] is True
    assert payload["selected_rect"] is None
    cluster = payload["candidate_clusters"][0]
    assert cluster["component_scores"]["huge_central_penalty"] > 0.2


def test_tiny_avatar_like_face_is_penalized_to_fallback(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.02, 0.02, 0.055, 0.055),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    assert payload["fallback"] is True
    assert payload["selected_rect"] is None
    cluster = payload["candidate_clusters"][0]
    assert cluster["component_scores"]["tiny_penalty"] > 0.3


def test_no_detections_writes_fallback_overlay_json(tmp_path: Path) -> None:
    analysis_dir, _frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=4)

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector({}),
    )

    payload = _read_json(overlay_path)
    assert payload == {
        "clip_id": "clip-123",
        "selected_rect": None,
        "confidence": 0.0,
        "fallback": True,
        "reason": "fallback: no face detections found in sampled frames",
        "candidate_clusters": [],
    }


def test_competing_stable_candidates_reduce_confidence(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detections = {
        name: (
            _detection(0.02, 0.58, 0.22, 0.22),
            _detection(0.74, 0.58, 0.22, 0.22),
        )
        for name in frame_names
    }

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector(detections),
    )

    payload = _read_json(overlay_path)
    assert payload["fallback"] is False
    assert len(payload["candidate_clusters"]) == 2
    selected_cluster = payload["candidate_clusters"][0]
    assert selected_cluster["component_scores"]["competition_penalty"] > 0
    assert selected_cluster["confidence"] < selected_cluster["raw_score"]


def test_detector_unavailable_writes_fallback_overlay_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_dir, _frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=2)

    def fail_detector() -> None:
        raise OverlayDetectorUnavailable("cascade missing")

    monkeypatch.setattr("clipforge.media.overlay.OpenCVFaceDetector", fail_detector)

    overlay_path = analyze_overlay(clip_id="clip-123", analysis_dir=analysis_dir)

    payload = _read_json(overlay_path)
    assert payload["fallback"] is True
    assert payload["selected_rect"] is None
    assert "detector unavailable" in payload["reason"]
    assert "cascade missing" in payload["reason"]


def test_analyze_overlay_requires_existing_frames_metadata(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="Frame metadata not found"):
        analyze_overlay(
            clip_id="clip-123",
            analysis_dir=tmp_path / "analysis",
            detector=SyntheticDetector({}),
        )


def test_analyze_overlay_requires_existing_frame_files(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    clip_dir = analysis_dir / "clip-123"
    clip_dir.mkdir(parents=True)
    (clip_dir / "frames.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-123",
                "frame_paths": [str(clip_dir / "frames" / "missing.jpg")],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AnalysisError, match="Sampled frame"):
        analyze_overlay(
            clip_id="clip-123",
            analysis_dir=analysis_dir,
            detector=SyntheticDetector({}),
        )


def test_analyze_overlay_reads_frame_metadata_and_writes_overlay_metadata(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=3)
    detector = SyntheticDetector(
        {name: (_detection(0.03, 0.62, 0.22, 0.22),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    assert overlay_path == analysis_dir / "clip-123" / "overlay.json"
    payload = _read_json(overlay_path)
    assert payload["clip_id"] == "clip-123"
    assert payload["fallback"] is False
    assert payload["selected_rect"] == {
        "x": 0.03,
        "y": 0.62,
        "width": 0.22,
        "height": 0.22,
    }


def _write_frames_metadata(
    tmp_path: Path,
    *,
    clip_id: str,
    count: int,
) -> tuple[Path, tuple[str, ...]]:
    analysis_dir = tmp_path / "analysis"
    clip_dir = analysis_dir / clip_id
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame_paths = tuple(frames_dir / f"frame_{index:04d}.jpg" for index in range(1, count + 1))
    for frame_path in frame_paths:
        frame_path.write_bytes(b"placeholder")
    (clip_dir / "frames.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "source_path": str(tmp_path / "source.mp4"),
                "sampled_timestamps": list(range(count)),
                "frame_paths": [str(path) for path in frame_paths],
                "sampling_mode": {
                    "type": "test",
                    "count": count,
                    "interval_seconds": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return analysis_dir, tuple(path.name for path in frame_paths)


def _detection(x: float, y: float, width: float, height: float) -> FaceDetection:
    return FaceDetection(rect=NormalizedRect(x=x, y=y, width=width, height=height))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))

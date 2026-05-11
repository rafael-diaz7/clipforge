from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.media.analyze import AnalysisError
from clipforge.media.overlay import (
    FaceDetection,
    HAAR_DETECTION_PASSES,
    NormalizedRect,
    OverlayDebugAnnotation,
    OverlayDetectorUnavailable,
    _HaarDetection,
    _deduplicate_haar_detections,
    analyze_overlay,
    write_overlay_debug_images,
)


TARGET_STREAMER_CROP_ASPECT_RATIO = ((9 / 16) / 0.40) / (16 / 9)


class SyntheticDetector:
    def __init__(self, detections: dict[str, tuple[FaceDetection, ...]]) -> None:
        self._detections = detections

    def detect(self, frame_path: Path) -> tuple[FaceDetection, ...]:
        return self._detections.get(frame_path.name, ())


class RecordingDebugWriter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write(
        self,
        *,
        frame_path: Path,
        output_path: Path,
        annotations: tuple[OverlayDebugAnnotation, ...],
        banner: str,
    ) -> None:
        self.calls.append(
            {
                "frame_path": frame_path,
                "output_path": output_path,
                "annotations": annotations,
                "banner": banner,
            }
        )
        output_path.write_bytes(b"debug")


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
    assert payload["selected_face_rect"]["x"] < 0.05
    assert payload["selected_face_rect"]["y"] > 0.55
    assert payload["selected_overlay_rect"]["x"] <= payload["selected_face_rect"]["x"]
    assert payload["selected_overlay_rect"]["width"] > payload["selected_face_rect"]["width"]
    assert payload["selected_overlay_rect"]["height"] > payload["selected_face_rect"]["height"]
    assert payload["candidate_clusters"][0]["component_scores"]["edge_proximity"] > 0.7
    assert "expanded into a streamer crop" in payload["reason"]


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
    assert payload["selected_overlay_rect"] is None
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
    assert payload["selected_overlay_rect"] is None
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
        "selected_face_rect": None,
        "selected_overlay_rect": None,
        "confidence": 0.0,
        "fallback": True,
        "reason": "fallback: no face detections found in sampled frames",
        "candidate_clusters": [],
    }


def test_multi_pass_haar_settings_match_debug_strategy() -> None:
    assert tuple(
        (
            pass_settings["scaleFactor"],
            pass_settings["minNeighbors"],
            pass_settings["minSize"],
        )
        for pass_settings in HAAR_DETECTION_PASSES
    ) == (
        (1.05, 3, (20, 20)),
        (1.05, 5, (20, 20)),
        (1.1, 3, (20, 20)),
        (1.1, 5, (20, 20)),
        (1.2, 3, (20, 20)),
    )


def test_overlapping_haar_pass_detections_are_deduped_with_pass_metadata() -> None:
    detections = [
        _haar_detection(10, 20, 40, 40, score=0.60, pass_index=1),
        _haar_detection(12, 21, 40, 40, score=0.72, pass_index=3),
        _haar_detection(110, 20, 35, 35, score=0.64, pass_index=5),
    ]

    merged = _deduplicate_haar_detections(detections)

    assert len(merged) == 2
    assert merged[0].score == pytest.approx(0.72)
    assert merged[0].merged_from_count == 2
    assert merged[0].pass_metadata["pass_index"] == 3
    assert [item["pass_index"] for item in merged[0].merged_passes] == [1, 3]
    assert merged[1].merged_from_count == 1


def test_recurring_face_detections_form_one_temporal_cluster(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=6)
    detections = {
        frame_names[0]: (_detection(0.04, 0.62, 0.18, 0.18, score=0.78),),
        frame_names[1]: (_detection(0.045, 0.615, 0.18, 0.18, score=0.74),),
        frame_names[3]: (_detection(0.035, 0.625, 0.19, 0.19, score=0.80),),
        frame_names[5]: (_detection(0.042, 0.618, 0.18, 0.18, score=0.76),),
    }

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector(detections),
        confidence_threshold=0.0,
    )

    payload = _read_json(overlay_path)
    clusters = payload["candidate_clusters"]
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["detection_count"] == 4
    assert cluster["sampled_frame_count"] == 6
    assert cluster["prevalence"] == pytest.approx(4 / 6)
    assert cluster["average_face_confidence"] == pytest.approx(0.77)
    assert cluster["max_face_confidence"] == pytest.approx(0.80)
    assert cluster["frame_indexes"] == [0, 1, 3, 5]
    assert cluster["timestamps"] == [0, 1, 3, 5]
    assert cluster["average_face_box"] == cluster["face_rect"]
    _assert_rect_contains(cluster["overlay_rect"], cluster["representative_face_box"])


def test_recurring_moderate_face_cluster_outranks_one_off_false_positive(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detections = {}
    for index, name in enumerate(frame_names):
        recurring_face = ()
        if index < 5:
            recurring_face = (_detection(0.04 + index * 0.004, 0.62, 0.18, 0.18, score=0.62),)
        one_off_false_positive = (
            (_detection(0.38, 0.24, 0.20, 0.20, score=0.99),) if index == 2 else ()
        )
        detections[name] = (*recurring_face, *one_off_false_positive)

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector(detections),
        confidence_threshold=0.0,
    )

    payload = _read_json(overlay_path)
    selected = payload["candidate_clusters"][0]
    rejected = payload["candidate_clusters"][1]
    assert selected["detection_count"] == 5
    assert selected["average_face_confidence"] == pytest.approx(0.62)
    assert rejected["detection_count"] == 1
    assert rejected["max_face_confidence"] == pytest.approx(0.99)
    assert selected["final_score"] > rejected["final_score"]
    assert selected["face_rect"]["x"] < 0.10


def test_one_off_central_gameplay_false_positive_is_penalized(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {frame_names[3]: (_detection(0.38, 0.28, 0.22, 0.22, score=1.0),)}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    cluster = payload["candidate_clusters"][0]
    assert payload["fallback"] is True
    assert cluster["detection_count"] == 1
    assert cluster["component_scores"]["one_frame_penalty"] > 0
    assert cluster["component_scores"]["central_penalty"] > 0
    assert cluster["final_score"] < 0.58
    assert payload["confidence"] < 0.58


def test_bottom_left_recurring_face_expands_to_clamped_webcam_rect(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.03, 0.82, 0.14, 0.14, score=0.86),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    face_rect = payload["selected_face_rect"]
    crop_rect = payload["selected_overlay_rect"]
    assert payload["fallback"] is False
    assert crop_rect["x"] == 0.0
    assert crop_rect["y"] + crop_rect["height"] == pytest.approx(1.0)
    assert crop_rect["width"] > face_rect["width"]
    assert crop_rect["height"] > face_rect["height"]
    _assert_rect_contains(crop_rect, face_rect)
    _assert_target_streamer_crop_aspect(crop_rect)


def test_overlay_fallback_still_works_when_no_face_clusters_exist(tmp_path: Path) -> None:
    analysis_dir, _frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector({}),
    )

    payload = _read_json(overlay_path)
    assert payload["fallback"] is True
    assert payload["selected_rect"] is None
    assert payload["candidate_clusters"] == []


def test_raw_detections_are_written_before_filtering_when_debug_enabled(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_valid_image_frames_metadata(
        tmp_path,
        clip_id="clip-123",
        count=2,
    )
    detector = SyntheticDetector(
        {
            frame_names[0]: (
                _detection(0.02, 0.82, 0.14, 0.14, score=0.80),
                _detection(-0.01, 0.40, 0.12, 0.12, score=0.95),
            )
        }
    )

    analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
        debug_raw_faces=True,
        confidence_threshold=0.0,
    )

    debug_dir = analysis_dir / "clip-123" / "debug" / "raw_faces"
    payload = _read_json(debug_dir / "raw_faces.json")
    assert (debug_dir / "frame_0001.png").is_file()
    assert payload["sampled_frame_count"] == 2
    assert payload["raw_face_detection_count"] == 2
    assert payload["filtered_detection_count"] == 1
    first_frame_detections = payload["frames"][0]["raw_detections"]
    assert [detection["filtered_out"] for detection in first_frame_detections] == [
        False,
        True,
    ]
    assert first_frame_detections[1]["filter_reason"] == "x_before_frame"


def test_bottom_left_edge_detection_is_not_discarded_by_raw_filter(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_valid_image_frames_metadata(
        tmp_path,
        clip_id="clip-123",
        count=1,
    )
    detector = SyntheticDetector(
        {frame_names[0]: (_detection(0.0, 0.84, 0.14, 0.14, score=0.77),)}
    )

    analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
        debug_raw_faces=True,
        confidence_threshold=0.0,
    )

    payload = _read_json(analysis_dir / "clip-123" / "debug" / "raw_faces" / "raw_faces.json")
    detection = payload["frames"][0]["raw_detections"][0]
    assert detection["filtered_out"] is False
    assert detection["filter_reason"] is None
    assert detection["cluster_id"] == 1


def test_face_evidence_cluster_is_reported_even_when_not_selected(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detections = {}
    for frame_name in frame_names:
        detections[frame_name] = (
            _detection(0.02, 0.82, 0.10, 0.10, score=0.65),
            _detection(0.72, 0.58, 0.22, 0.22, score=0.95),
            _detection(0.60, 0.58, 0.20, 0.20, score=0.90),
            _detection(0.48, 0.58, 0.20, 0.20, score=0.88),
            _detection(0.36, 0.58, 0.20, 0.20, score=0.86),
            _detection(0.24, 0.58, 0.20, 0.20, score=0.84),
            _detection(0.12, 0.20, 0.16, 0.16, score=0.82),
            _detection(0.36, 0.20, 0.16, 0.16, score=0.81),
            _detection(0.60, 0.20, 0.16, 0.16, score=0.80),
        )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector(detections),
        confidence_threshold=0.0,
    )

    payload = _read_json(overlay_path)
    assert payload["candidate_clusters"][0]["face_rect"]["x"] > 0.60
    assert any(
        cluster["face_rect"]["x"] < 0.05 and cluster["face_score"] > 0.0
        for cluster in payload["candidate_clusters"]
    )


def test_valid_face_cluster_heuristic_multiplier_has_nonzero_floor(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.18, 0.10, 0.62, 0.62, score=0.72),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    cluster = payload["candidate_clusters"][0]
    assert cluster["face_score"] > 0.0
    assert cluster["heuristic_multiplier"] >= 0.08
    assert cluster["final_score"] > 0.0


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


def test_stable_low_face_evidence_region_does_not_outrank_real_face(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detections = {}
    for index, name in enumerate(frame_names):
        stable_hud_region = _detection(0.02, 0.72, 0.16, 0.16, score=0.03)
        real_face = ()
        if index < 6:
            real_face = (
                _detection(
                    0.68 + (0.012 if index % 2 else 0.0),
                    0.54 + (0.010 if index % 3 == 0 else 0.0),
                    0.18,
                    0.18,
                    score=0.84,
                ),
            )
        detections[name] = (stable_hud_region, *real_face)

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector(detections),
        confidence_threshold=0.0,
    )

    payload = _read_json(overlay_path)
    selected = payload["candidate_clusters"][0]
    rejected = payload["candidate_clusters"][1]
    assert selected["face_rect"]["x"] > 0.60
    assert rejected["face_rect"]["x"] < 0.05
    assert selected["face_score"] > rejected["face_score"]
    assert selected["final_score"] > rejected["final_score"]
    assert selected["ranking_position"] == 1
    assert rejected["ranking_position"] == 2


def test_lower_stability_face_evidence_beats_stable_hud_region(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detections = {}
    for index, name in enumerate(frame_names):
        detections[name] = (
            _detection(0.78, 0.72, 0.14, 0.14, score=0.08),
            _detection(
                0.04 + index * 0.006,
                0.50 + (0.014 if index % 2 else 0.0),
                0.18,
                0.18,
                score=0.74,
            ),
        )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=SyntheticDetector(detections),
        confidence_threshold=0.0,
    )

    payload = _read_json(overlay_path)
    selected = payload["candidate_clusters"][0]
    rejected = payload["candidate_clusters"][1]
    assert selected["face_rect"]["x"] < 0.10
    assert selected["component_scores"]["position_stability"] < rejected["component_scores"][
        "position_stability"
    ]
    assert selected["face_score"] > rejected["face_score"]
    assert selected["final_score"] > rejected["final_score"]


def test_heuristics_do_not_create_high_score_from_near_zero_face_evidence(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.02, 0.62, 0.20, 0.20, score=0.01),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    cluster = payload["candidate_clusters"][0]
    assert payload["fallback"] is True
    assert cluster["face_score"] == pytest.approx(0.01)
    assert cluster["heuristic_multiplier"] <= 1.0
    assert cluster["final_score"] < 0.02
    assert cluster["confidence"] == cluster["final_score"]


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
    assert payload["selected_face_rect"] == {
        "x": 0.03,
        "y": 0.62,
        "width": 0.22,
        "height": 0.22,
    }
    assert payload["selected_overlay_rect"] == {
        "x": 0.0,
        "y": 0.58133,
        "width": 0.28,
        "height": 0.353975,
    }
    assert payload["selected_rect"] == payload["selected_overlay_rect"]
    assert payload["candidate_clusters"][0]["face_rect"] == payload["selected_face_rect"]
    assert payload["candidate_clusters"][0]["overlay_rect"] == payload["selected_overlay_rect"]
    _assert_rect_contains(payload["selected_overlay_rect"], payload["selected_face_rect"])
    _assert_target_streamer_crop_aspect(payload["selected_overlay_rect"])


def test_streamer_crop_expands_selected_face_rect(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.04, 0.56, 0.18, 0.18),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    face_rect = payload["selected_face_rect"]
    crop_rect = payload["selected_overlay_rect"]
    assert payload["fallback"] is False
    assert crop_rect["width"] > face_rect["width"]
    assert crop_rect["height"] > face_rect["height"]
    assert crop_rect["x"] <= face_rect["x"]
    assert crop_rect["y"] <= face_rect["y"]
    assert crop_rect["x"] + crop_rect["width"] >= face_rect["x"] + face_rect["width"]
    assert crop_rect["y"] + crop_rect["height"] >= face_rect["y"] + face_rect["height"]
    _assert_rect_contains(crop_rect, face_rect)
    _assert_rect_within_source_bounds(crop_rect)
    _assert_target_streamer_crop_aspect(crop_rect)


def test_streamer_crop_centers_horizontally_on_face_near_edge(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.03, 0.52, 0.20, 0.20),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    face_rect = payload["selected_face_rect"]
    crop_rect = payload["selected_overlay_rect"]
    face_center_x = face_rect["x"] + face_rect["width"] / 2
    crop_center_x = crop_rect["x"] + crop_rect["width"] / 2
    assert payload["fallback"] is False
    assert abs(face_center_x - crop_center_x) <= 0.005
    _assert_target_streamer_crop_aspect(crop_rect)


def test_streamer_crop_places_head_slightly_above_vertical_center(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.05, 0.36, 0.16, 0.16),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    face_rect = payload["selected_face_rect"]
    crop_rect = payload["selected_overlay_rect"]
    above_head = face_rect["y"] - crop_rect["y"]
    below_head = crop_rect["y"] + crop_rect["height"] - face_rect["y"] - face_rect["height"]
    face_center_y = face_rect["y"] + face_rect["height"] / 2
    crop_center_y = crop_rect["y"] + crop_rect["height"] / 2
    assert payload["fallback"] is False
    assert face_center_y < crop_center_y
    assert below_head > above_head
    _assert_rect_contains(crop_rect, face_rect)
    _assert_target_streamer_crop_aspect(crop_rect)


def test_streamer_crop_shrinks_to_preserve_aspect_near_source_bounds(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.03, 0.52, 0.20, 0.20),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    face_rect = payload["selected_face_rect"]
    crop_rect = payload["selected_overlay_rect"]
    face_center_x = face_rect["x"] + face_rect["width"] / 2
    crop_center_x = crop_rect["x"] + crop_rect["width"] / 2
    assert payload["fallback"] is False
    assert crop_rect["x"] == 0.0
    assert crop_rect["width"] == pytest.approx(face_center_x * 2)
    assert crop_center_x == pytest.approx(face_center_x)
    _assert_rect_contains(crop_rect, face_rect)
    _assert_rect_within_source_bounds(crop_rect)
    _assert_target_streamer_crop_aspect(crop_rect)


def test_streamer_crop_stays_tight_around_left_side_face(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.03, 0.52, 0.20, 0.20),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    crop_rect = payload["selected_overlay_rect"]
    assert payload["fallback"] is False
    assert crop_rect["x"] == 0.0
    assert crop_rect["width"] <= 0.32
    assert crop_rect["x"] + crop_rect["width"] <= 0.32
    _assert_target_streamer_crop_aspect(crop_rect)


def test_streamer_crop_does_not_expand_far_into_main_content(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.02, 0.44, 0.22, 0.22),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    crop_rect = payload["selected_overlay_rect"]
    assert payload["fallback"] is False
    assert crop_rect["width"] <= 0.36
    assert crop_rect["x"] + crop_rect["width"] < 0.40
    _assert_target_streamer_crop_aspect(crop_rect)


def test_overlay_expansion_clamps_at_frame_edges(tmp_path: Path) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=8)
    detector = SyntheticDetector(
        {name: (_detection(0.88, 0.86, 0.12, 0.12),) for name in frame_names}
    )

    overlay_path = analyze_overlay(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        detector=detector,
    )

    payload = _read_json(overlay_path)
    assert payload["fallback"] is False
    overlay_rect = payload["selected_overlay_rect"]
    assert overlay_rect["x"] >= 0.0
    assert overlay_rect["y"] >= 0.0
    assert overlay_rect["x"] + overlay_rect["width"] <= 1.0
    assert overlay_rect["y"] + overlay_rect["height"] <= 1.0
    assert overlay_rect["x"] + overlay_rect["width"] == 1.0
    assert overlay_rect["y"] + overlay_rect["height"] >= 0.99
    _assert_target_streamer_crop_aspect(overlay_rect)


def test_write_overlay_debug_images_writes_one_debug_image_per_sampled_frame(
    tmp_path: Path,
) -> None:
    analysis_dir, frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=2)
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        fallback=False,
        selected_rect={"x": 0.03, "y": 0.62, "width": 0.22, "height": 0.22},
        confidence=0.82,
        candidate_clusters=[
            {
                "cluster_id": 1,
                "rect": {"x": 0.03, "y": 0.62, "width": 0.22, "height": 0.22},
                "confidence": 0.82,
            },
            {
                "cluster_id": 2,
                "rect": {"x": 0.72, "y": 0.60, "width": 0.20, "height": 0.20},
                "confidence": 0.43,
            },
        ],
    )
    writer = RecordingDebugWriter()

    debug_dir = write_overlay_debug_images(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        image_writer=writer,
    )

    assert debug_dir == analysis_dir / "clip-123" / "debug"
    assert len(writer.calls) == 2
    assert [call["frame_path"].name for call in writer.calls] == list(frame_names)
    assert [call["output_path"].name for call in writer.calls] == [
        "frame_0001_overlay_debug.jpg",
        "frame_0002_overlay_debug.jpg",
    ]
    assert all(Path(call["output_path"]).is_file() for call in writer.calls)
    first_call = writer.calls[0]
    annotations = first_call["annotations"]
    assert first_call["banner"] == "overlay selected | confidence 0.820"
    assert [annotation.label for annotation in annotations] == [
        "cluster 1 | confidence 0.820 | selected",
        "cluster 2 | confidence 0.430 | candidate",
    ]


def test_write_overlay_debug_images_labels_fallback_candidates(tmp_path: Path) -> None:
    analysis_dir, _frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=1)
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        fallback=True,
        selected_rect=None,
        confidence=0.31,
        candidate_clusters=[
            {
                "cluster_id": 1,
                "rect": {"x": 0.18, "y": 0.10, "width": 0.62, "height": 0.62},
                "confidence": 0.31,
            }
        ],
    )
    writer = RecordingDebugWriter()

    write_overlay_debug_images(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        image_writer=writer,
    )

    assert writer.calls[0]["banner"] == "overlay fallback | confidence 0.310"
    annotations = writer.calls[0]["annotations"]
    assert [annotation.label for annotation in annotations] == [
        "cluster 1 | confidence 0.310 | fallback candidate"
    ]


def test_write_overlay_debug_images_labels_temporal_cluster_scores(tmp_path: Path) -> None:
    analysis_dir, _frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=1)
    _write_overlay_metadata(
        analysis_dir,
        clip_id="clip-123",
        fallback=False,
        selected_rect={"x": 0.0, "y": 0.62, "width": 0.24, "height": 0.30},
        confidence=0.71,
        candidate_clusters=[
            {
                "cluster_id": 7,
                "rect": {"x": 0.0, "y": 0.62, "width": 0.24, "height": 0.30},
                "confidence": 0.71,
                "detection_count": 5,
                "prevalence": 0.625,
                "average_face_confidence": 0.74,
                "heuristic_multiplier": 0.82,
                "final_score": 0.71,
            }
        ],
    )
    writer = RecordingDebugWriter()

    write_overlay_debug_images(
        clip_id="clip-123",
        analysis_dir=analysis_dir,
        image_writer=writer,
    )

    annotations = writer.calls[0]["annotations"]
    assert [annotation.label for annotation in annotations] == [
        (
            "cluster 7 | detections 5 | prevalence 0.625 | "
            "avg_face 0.740 | heuristic 0.820 | score 0.710 | selected"
        )
    ]


def test_write_overlay_debug_images_requires_overlay_metadata(tmp_path: Path) -> None:
    analysis_dir, _frame_names = _write_frames_metadata(tmp_path, clip_id="clip-123", count=1)

    with pytest.raises(AnalysisError, match="Overlay metadata not found"):
        write_overlay_debug_images(
            clip_id="clip-123",
            analysis_dir=analysis_dir,
            image_writer=RecordingDebugWriter(),
        )


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


def _write_valid_image_frames_metadata(
    tmp_path: Path,
    *,
    clip_id: str,
    count: int,
) -> tuple[Path, tuple[str, ...]]:
    import cv2
    import numpy as np

    analysis_dir, frame_names = _write_frames_metadata(
        tmp_path,
        clip_id=clip_id,
        count=count,
    )
    frames_dir = analysis_dir / clip_id / "frames"
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    for frame_name in frame_names:
        frame_path = frames_dir / frame_name
        assert cv2.imwrite(str(frame_path), image)
    return analysis_dir, frame_names


def _write_overlay_metadata(
    analysis_dir: Path,
    *,
    clip_id: str,
    fallback: bool,
    selected_rect: dict[str, float] | None,
    confidence: float,
    candidate_clusters: list[dict[str, object]],
) -> None:
    (analysis_dir / clip_id / "overlay.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "selected_rect": selected_rect,
                "confidence": confidence,
                "fallback": fallback,
                "reason": "test",
                "candidate_clusters": candidate_clusters,
            }
        ),
        encoding="utf-8",
    )


def _detection(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    score: float = 1.0,
) -> FaceDetection:
    return FaceDetection(rect=NormalizedRect(x=x, y=y, width=width, height=height), score=score)


def _haar_detection(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    score: float,
    pass_index: int,
) -> _HaarDetection:
    return _HaarDetection(
        box=(x, y, width, height),
        score=score,
        pass_metadata={
            "pass_index": pass_index,
            "scaleFactor": 1.05,
            "minNeighbors": 3,
            "minSize": [20, 20],
        },
    )


def _assert_rect_contains(
    outer: dict[str, float],
    inner: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> None:
    assert outer["x"] <= inner["x"] + tolerance
    assert outer["y"] <= inner["y"] + tolerance
    assert outer["x"] + outer["width"] >= inner["x"] + inner["width"] - tolerance
    assert outer["y"] + outer["height"] >= inner["y"] + inner["height"] - tolerance


def _assert_rect_within_source_bounds(
    rect: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> None:
    assert rect["x"] >= 0.0
    assert rect["y"] >= 0.0
    assert rect["x"] + rect["width"] <= 1.0 + tolerance
    assert rect["y"] + rect["height"] <= 1.0 + tolerance


def _assert_target_streamer_crop_aspect(
    rect: dict[str, float],
    *,
    tolerance: float = 1e-5,
) -> None:
    assert rect["width"] / rect["height"] == pytest.approx(
        TARGET_STREAMER_CROP_ASPECT_RATIO,
        abs=tolerance,
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))

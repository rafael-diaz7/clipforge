"""Infer likely streamer overlay regions from sampled analysis frames."""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from clipforge.core.config import ANALYSIS_DIR
from clipforge.media.analyze import AnalysisError
from clipforge.utils.paths import clip_analysis_dir, ensure_directory, safe_filename


LOGGER = logging.getLogger(__name__)
DEFAULT_OVERLAY_CONFIDENCE_THRESHOLD = 0.58
CLUSTER_CENTER_THRESHOLD = 0.12
CLUSTER_SIZE_THRESHOLD = 0.09
HAAR_DETECTION_PASSES = (
    {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (20, 20)},
    {"scaleFactor": 1.05, "minNeighbors": 5, "minSize": (20, 20)},
    {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (20, 20)},
    {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (20, 20)},
    {"scaleFactor": 1.2, "minNeighbors": 3, "minSize": (20, 20)},
)
HAAR_DEDUP_IOU_THRESHOLD = 0.35
STREAMER_CROP_HORIZONTAL_PADDING_RATIO = 0.28
STREAMER_CROP_TOP_PADDING_RATIO = 0.32
STREAMER_CROP_BOTTOM_PADDING_RATIO = 0.75
STREAMER_CROP_MAX_WIDTH_FACE_RATIO = 1.65
STREAMER_CROP_MAX_HEIGHT_FACE_RATIO = 2.10
STREAMER_CROP_MAX_NORMALIZED_WIDTH = 0.36
STREAMER_CROP_FACE_CENTER_Y_RATIO = 0.42
STREAMER_CROP_SIDE_MIN_NORMALIZED_HEIGHT = 0.30
STREAMER_CROP_SIDE_FACE_CENTER_Y_RATIO = 0.12
SOURCE_ASPECT_RATIO = 16 / 9
TARGET_ASPECT_RATIO = 9 / 16
HYBRID_STREAMER_OUTPUT_HEIGHT_RATIO = 0.40
MAX_REPORTED_FACE_CLUSTERS = 8
MIN_VALID_FACE_HEURISTIC_MULTIPLIER = 0.08
HIGH_CONFIDENCE_CLUSTER_THRESHOLD = DEFAULT_OVERLAY_CONFIDENCE_THRESHOLD
MIN_RENDER_ELIGIBLE_FACE_SCORE = 0.02
# Convert the hybrid top region's display aspect into normalized source coordinates.
STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO = (
    TARGET_ASPECT_RATIO / HYBRID_STREAMER_OUTPUT_HEIGHT_RATIO
) / SOURCE_ASPECT_RATIO


class OverlayDetectorUnavailable(RuntimeError):
    """Raised when a local face detector cannot be initialized or used."""


@dataclass(frozen=True)
class NormalizedRect:
    """A normalized rectangle in frame coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x": _round(self.x),
            "y": _round(self.y),
            "width": _round(self.width),
            "height": _round(self.height),
        }


@dataclass(frozen=True)
class FaceDetection:
    """A detector-provided face candidate."""

    rect: NormalizedRect
    score: float = 1.0


@dataclass(frozen=True)
class FaceDetectorDebugInfo:
    """Inspectable settings for a detector run."""

    name: str
    settings: dict[str, object]


class FaceDetector(Protocol):
    """Small detector abstraction so OpenCV can be swapped for MediaPipe later."""

    def detect(self, frame_path: Path) -> tuple[FaceDetection, ...]:
        """Return normalized face detections for one frame."""


@dataclass(frozen=True)
class OverlayDebugAnnotation:
    """One debug rectangle annotation to draw on a sampled frame."""

    cluster_id: int
    rect: NormalizedRect
    confidence: float
    state: str
    detection_count: int | None = None
    prevalence: float | None = None
    average_face_confidence: float | None = None
    heuristic_multiplier: float | None = None
    final_score: float | None = None
    source: str | None = None
    edge_corner_prior: float | None = None
    central_gameplay_penalty: float | None = None
    expanded_box_quality: float | None = None

    @property
    def label(self) -> str:
        if self.detection_count is None or self.prevalence is None:
            return (
                f"cluster {self.cluster_id} | confidence {self.confidence:.3f} | "
                f"{self.state}"
            )
        if (
            self.edge_corner_prior is not None
            and self.central_gameplay_penalty is not None
            and self.expanded_box_quality is not None
        ):
            source = f"{self.source} | " if self.source else ""
            return (
                f"cluster {self.cluster_id} | {source}conf {self.confidence:.3f} | "
                f"prevalence {self.prevalence:.3f} | "
                f"edge {self.edge_corner_prior:.3f} | "
                f"central_penalty {self.central_gameplay_penalty:.3f} | "
                f"box {self.expanded_box_quality:.3f} | "
                f"score {self.final_score or self.confidence:.3f} | {self.state}"
            )
        return (
            f"cluster {self.cluster_id} | detections {self.detection_count} | "
            f"prevalence {self.prevalence:.3f} | "
            f"avg_face {self.average_face_confidence or 0.0:.3f} | "
            f"heuristic {self.heuristic_multiplier or 0.0:.3f} | "
            f"score {self.final_score or self.confidence:.3f} | {self.state}"
        )


class OverlayDebugImageWriter(Protocol):
    """Draw overlay debug annotations onto one sampled frame."""

    def write(
        self,
        *,
        frame_path: Path,
        output_path: Path,
        annotations: tuple[OverlayDebugAnnotation, ...],
        banner: str,
    ) -> None:
        """Write a debug image for one sampled frame."""


@dataclass(frozen=True)
class _FrameDetection:
    frame_index: int
    timestamp: float | None
    rect: NormalizedRect
    score: float
    class_label: str = "face"
    detector_name: str = "haar_face"


@dataclass(frozen=True)
class _RawFrameDetection:
    frame_index: int
    timestamp: float | None
    frame_path: Path
    rect: NormalizedRect
    score: float
    detector_name: str
    pass_metadata: dict[str, object] | None
    merged_from_count: int
    merged_passes: tuple[dict[str, object], ...]
    filtered_out: bool
    filter_reason: str | None


@dataclass(frozen=True)
class _HaarDetection:
    box: tuple[int, int, int, int]
    score: float
    pass_metadata: dict[str, object]


@dataclass(frozen=True)
class _MergedHaarDetection:
    box: tuple[int, int, int, int]
    score: float
    pass_metadata: dict[str, object]
    merged_from_count: int
    merged_passes: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _FaceDetectionReport:
    detections: tuple[_FrameDetection, ...]
    raw_detections: tuple[_RawFrameDetection, ...]
    detector_info: FaceDetectorDebugInfo
    sampled_timestamps: tuple[float | None, ...]


@dataclass(frozen=True)
class _SubjectDetectionReport:
    detections: tuple[_FrameDetection, ...]
    raw_detections: tuple[_FrameDetection, ...]
    detector_info: object
    sampled_timestamps: tuple[float | None, ...]


@dataclass
class _Cluster:
    detections: list[_FrameDetection]

    def add(self, detection: _FrameDetection) -> None:
        self.detections.append(detection)


@dataclass(frozen=True)
class _ScoredCluster:
    cluster_id: int
    source: str
    class_label: str
    detector_name: str
    face_rect: NormalizedRect
    overlay_rect: NormalizedRect
    representative_face_rect: NormalizedRect
    average_face_rect: NormalizedRect
    detection_count: int
    sampled_frame_count: int
    frame_count: int
    prevalence: float
    average_face_confidence: float
    max_face_confidence: float
    frame_indexes: tuple[int, ...]
    timestamps: tuple[float, ...]
    component_scores: dict[str, float]
    face_score: float
    heuristic_multiplier: float
    final_score: float
    ranking_position: int | None
    raw_score: float
    confidence: float

    @property
    def rect(self) -> NormalizedRect:
        return self.overlay_rect

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "source": self.source,
            "class_label": self.class_label,
            "detector_name": self.detector_name,
            "rect": self.overlay_rect.to_dict(),
            "subject_rect": self.face_rect.to_dict(),
            "person_rect": self.face_rect.to_dict() if self.class_label == "person" else None,
            "face_rect": self.face_rect.to_dict(),
            "overlay_rect": self.overlay_rect.to_dict(),
            "representative_face_box": self.representative_face_rect.to_dict(),
            "average_face_box": self.average_face_rect.to_dict(),
            "detection_count": self.detection_count,
            "sampled_frame_count": self.sampled_frame_count,
            "frame_count": self.frame_count,
            "prevalence": _round(self.prevalence),
            "average_face_confidence": _round(self.average_face_confidence),
            "max_face_confidence": _round(self.max_face_confidence),
            "frame_indexes": list(self.frame_indexes),
            "timestamps": [_round(timestamp) for timestamp in self.timestamps],
            "component_scores": {
                name: _round(value) for name, value in self.component_scores.items()
            },
            "detector_confidence": _round(
                self.component_scores.get("detector_confidence", self.average_face_confidence)
            ),
            "recurrence": _round(self.component_scores.get("recurrence", self.prevalence)),
            "edge_corner_prior": _round(self.component_scores.get("edge_corner_prior", 0.0)),
            "webcam_location_prior": _round(
                self.component_scores.get("webcam_location_prior", 0.0)
            ),
            "central_gameplay_penalty": _round(
                self.component_scores.get("central_gameplay_penalty", 0.0)
            ),
            "hud_penalty": _round(self.component_scores.get("hud_penalty", 0.0)),
            "expanded_box_quality": _round(
                self.component_scores.get("expanded_box_quality", 0.0)
            ),
            "overlay_support_score": _round(
                self.component_scores.get("overlay_support_score", 0.0)
            ),
            "face_score": _round(self.face_score),
            "heuristic_multiplier": _round(self.heuristic_multiplier),
            "final_score": _round(self.final_score),
            "high_confidence": self.confidence >= HIGH_CONFIDENCE_CLUSTER_THRESHOLD,
            "render_eligible": (
                self.face_score >= MIN_RENDER_ELIGIBLE_FACE_SCORE
                and self.component_scores.get("expanded_box_quality", 0.0) > 0.0
            ),
            "ranking_position": self.ranking_position,
            "raw_score": _round(self.raw_score),
            "confidence": _round(self.confidence),
        }


class OpenCVFaceDetector:
    """OpenCV Haar cascade face detector isolated behind FaceDetector."""

    def __init__(self, *, cascade_path: Path | None = None) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise OverlayDetectorUnavailable(
                "OpenCV is not installed; install opencv-python to detect overlays."
            ) from exc

        self._cv2 = cv2
        if cascade_path is None:
            try:
                cascade_path = (
                    Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
                )
            except AttributeError as exc:
                raise OverlayDetectorUnavailable(
                    "OpenCV Haar cascade data path is unavailable."
                ) from exc

        self._cascade_path = cascade_path
        self._last_detection_methods: tuple[str, ...] = ()
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise OverlayDetectorUnavailable(f"OpenCV face cascade could not load: {cascade_path}")

    def detect(self, frame_path: Path) -> tuple[FaceDetection, ...]:
        return tuple(
            FaceDetection(rect=raw.rect, score=raw.score)
            for raw in self.detect_raw(frame_path)
            if not raw.filtered_out
        )

    def detect_raw(self, frame_path: Path) -> tuple[_RawFrameDetection, ...]:
        image = self._cv2.imread(str(frame_path))
        if image is None:
            return ()

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return ()

        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces_with_scores(gray)
        detections: list[_RawFrameDetection] = []
        for face in faces:
            x, y, face_width, face_height = face.box
            rect = NormalizedRect(
                x=float(x) / width,
                y=float(y) / height,
                width=float(face_width) / width,
                height=float(face_height) / height,
            )
            filter_reason = _detection_filter_reason(rect)
            detections.append(
                _RawFrameDetection(
                    frame_index=-1,
                    timestamp=None,
                    frame_path=frame_path,
                    rect=rect,
                    score=_clamp01(face.score),
                    detector_name=self.debug_info.name,
                    pass_metadata=face.pass_metadata,
                    merged_from_count=face.merged_from_count,
                    merged_passes=face.merged_passes,
                    filtered_out=filter_reason is not None,
                    filter_reason=filter_reason,
                )
            )
        return tuple(detections)

    @property
    def debug_info(self) -> FaceDetectorDebugInfo:
        return FaceDetectorDebugInfo(
            name="opencv_haar_frontalface_default",
            settings={
                "cascade_path": str(self._cascade_path),
                "passes": [_haar_pass_to_dict(pass_settings) for pass_settings in HAAR_DETECTION_PASSES],
                "dedup_iou_threshold": HAAR_DEDUP_IOU_THRESHOLD,
                "preprocessing": "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
                "equalize_hist": False,
                "detect_methods": list(self._last_detection_methods),
                "confidence_normalization": "(level_weight + 3.0) / 7.0 clamped to [0, 1]",
            },
        )

    def _detect_faces_with_scores(self, gray: object) -> tuple[_MergedHaarDetection, ...]:
        detections: list[_HaarDetection] = []
        methods: list[str] = []
        for pass_index, pass_settings in enumerate(HAAR_DETECTION_PASSES, start=1):
            pass_detections, method = self._detect_faces_for_pass(
                gray,
                pass_index=pass_index,
                pass_settings=pass_settings,
            )
            methods.append(method)
            detections.extend(pass_detections)
        self._last_detection_methods = tuple(methods)
        return _deduplicate_haar_detections(detections)

    def _detect_faces_for_pass(
        self,
        gray: object,
        *,
        pass_index: int,
        pass_settings: dict[str, object],
    ) -> tuple[tuple[_HaarDetection, ...], str]:
        pass_metadata = {
            "pass_index": pass_index,
            **_haar_pass_to_dict(pass_settings),
        }
        try:
            method = "detectMultiScale3"
            faces, _reject_levels, level_weights = self._cascade.detectMultiScale3(
                gray,
                scaleFactor=float(pass_settings["scaleFactor"]),
                minNeighbors=int(pass_settings["minNeighbors"]),
                minSize=tuple(pass_settings["minSize"]),
                outputRejectLevels=True,
            )
        except (AttributeError, TypeError):
            method = "detectMultiScale"
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=float(pass_settings["scaleFactor"]),
                minNeighbors=int(pass_settings["minNeighbors"]),
                minSize=tuple(pass_settings["minSize"]),
            )
            scores = tuple(1.0 for _face in faces)
        else:
            scores = tuple(_opencv_face_score(float(weight)) for weight in level_weights)

        return (
            tuple(
                _HaarDetection(
                    box=(
                        int(face[0]),
                        int(face[1]),
                        int(face[2]),
                        int(face[3]),
                    ),
                    score=_clamp01(score),
                    pass_metadata=pass_metadata,
                )
                for face, score in zip(faces, scores, strict=True)
            ),
            method,
        )


def _haar_pass_to_dict(pass_settings: dict[str, object]) -> dict[str, object]:
    return {
        "scaleFactor": float(pass_settings["scaleFactor"]),
        "minNeighbors": int(pass_settings["minNeighbors"]),
        "minSize": list(pass_settings["minSize"]),
    }


def _deduplicate_haar_detections(
    detections: list[_HaarDetection],
) -> tuple[_MergedHaarDetection, ...]:
    groups: list[list[_HaarDetection]] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        matching_group: list[_HaarDetection] | None = None
        matching_iou = 0.0
        for group in groups:
            iou = _box_iou(_representative_haar_box(group), detection.box)
            if iou > matching_iou:
                matching_group = group
                matching_iou = iou

        if matching_group is not None and matching_iou >= HAAR_DEDUP_IOU_THRESHOLD:
            matching_group.append(detection)
        else:
            groups.append([detection])

    return tuple(
        _merged_haar_detection(group)
        for group in sorted(
            groups,
            key=lambda item: max(detection.score for detection in item),
            reverse=True,
        )
    )


def _merged_haar_detection(group: list[_HaarDetection]) -> _MergedHaarDetection:
    best = max(group, key=lambda detection: detection.score)
    return _MergedHaarDetection(
        box=_average_haar_box(group),
        score=best.score,
        pass_metadata=best.pass_metadata,
        merged_from_count=len(group),
        merged_passes=tuple(
            detection.pass_metadata
            for detection in sorted(
                group,
                key=lambda detection: int(detection.pass_metadata["pass_index"]),
            )
        ),
    )


def _representative_haar_box(group: list[_HaarDetection]) -> tuple[int, int, int, int]:
    return max(group, key=lambda detection: detection.score).box


def _average_haar_box(group: list[_HaarDetection]) -> tuple[int, int, int, int]:
    return (
        round(statistics.fmean(detection.box[0] for detection in group)),
        round(statistics.fmean(detection.box[1] for detection in group)),
        round(statistics.fmean(detection.box[2] for detection in group)),
        round(statistics.fmean(detection.box[3] for detection in group)),
    )


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection_width = max(0, right - left)
    intersection_height = max(0, bottom - top)
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0

    first_area = first_width * first_height
    second_area = second_width * second_height
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class OpenCVOverlayDebugImageWriter:
    """OpenCV-backed writer for overlay debug frame images."""

    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise AnalysisError(
                "OpenCV is not installed; install opencv-python to write overlay debug images."
            ) from exc

        self._cv2 = cv2

    def write(
        self,
        *,
        frame_path: Path,
        output_path: Path,
        annotations: tuple[OverlayDebugAnnotation, ...],
        banner: str,
    ) -> None:
        image = self._cv2.imread(str(frame_path))
        if image is None:
            raise AnalysisError(f"Could not read sampled frame for debug image: {frame_path}")

        height, width = image.shape[:2]
        self._draw_label(image, banner, (8, 24), color=(255, 255, 255), background=(32, 32, 32))
        for annotation in annotations:
            color = _debug_color(annotation.state)
            left, top, right, bottom = _rect_pixels(annotation.rect, width=width, height=height)
            self._cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            self._draw_label(
                image,
                annotation.label,
                (left, max(18, top - 8)),
                color=(255, 255, 255),
                background=color,
            )

        ensure_directory(output_path.parent)
        if not self._cv2.imwrite(str(output_path), image):
            raise AnalysisError(f"Could not write overlay debug image: {output_path}")

    def _draw_label(
        self,
        image: object,
        text: str,
        origin: tuple[int, int],
        *,
        color: tuple[int, int, int],
        background: tuple[int, int, int],
    ) -> None:
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1
        (text_width, text_height), baseline = self._cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )
        x, y = origin
        self._cv2.rectangle(
            image,
            (x - 4, y - text_height - baseline - 4),
            (x + text_width + 4, y + baseline + 4),
            background,
            -1,
        )
        self._cv2.putText(
            image,
            text,
            (x, y),
            font,
            scale,
            color,
            thickness,
            self._cv2.LINE_AA,
        )


def _draw_debug_label(
    cv2: object,
    image: object,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    background: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )
    x, y = origin
    cv2.rectangle(
        image,
        (x - 4, y - text_height - baseline - 4),
        (x + text_width + 4, y + baseline + 4),
        background,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def analyze_overlay(
    *,
    clip_id: str,
    analysis_dir: Path = ANALYSIS_DIR,
    detector: FaceDetector | None = None,
    subject_detector: object | None = None,
    confidence_threshold: float = DEFAULT_OVERLAY_CONFIDENCE_THRESHOLD,
    debug_raw_faces: bool = False,
) -> Path:
    """Analyze saved sampled frames and write overlay inference metadata."""

    safe_clip_id = _safe_clip_id(clip_id)
    analysis_clip_dir = clip_analysis_dir(analysis_dir, safe_clip_id)
    frames_metadata_path = analysis_clip_dir / "frames.json"
    if not frames_metadata_path.is_file():
        raise AnalysisError(f"Frame metadata not found: {frames_metadata_path}")

    frames_metadata = _read_frames_metadata(frames_metadata_path)
    metadata_clip_id = str(frames_metadata.get("clip_id") or clip_id)
    frame_paths = _frame_paths_from_metadata(frames_metadata, base_path=frames_metadata_path.parent)
    sampled_timestamps = _sampled_timestamps_from_metadata(
        frames_metadata,
        expected_count=len(frame_paths),
    )
    _require_existing_frames(frame_paths)
    output_path = analysis_clip_dir / "overlay.json"

    subject_unavailable_reason: str | None = None
    subject_scored_clusters: tuple[_ScoredCluster, ...] = ()
    if detector is None or subject_detector is not None:
        try:
            subject_detector_instance = subject_detector or _build_subject_detector_from_env()
        except OverlayDetectorUnavailable as exc:
            subject_detector_instance = None
            subject_unavailable_reason = str(exc)
        if subject_detector_instance is not None:
            try:
                subject_report = _detect_subjects(
                    frame_paths,
                    sampled_timestamps=sampled_timestamps,
                    detector=subject_detector_instance,
                )
            except OverlayDetectorUnavailable as exc:
                subject_report = None
                subject_unavailable_reason = str(exc)
            else:
                subject_clusters = _cluster_detections(subject_report.detections)
                subject_scored_clusters = _score_clusters(
                    subject_clusters,
                    total_frames=len(frame_paths),
                )
                if debug_raw_faces:
                    _write_raw_subject_debug(
                        analysis_clip_dir=analysis_clip_dir,
                        frame_paths=frame_paths,
                        detection_report=subject_report,
                        clusters=subject_clusters,
                        scored_clusters=subject_scored_clusters,
                        selected_mode=(
                            "yolo_person"
                            if subject_scored_clusters
                            and subject_scored_clusters[0].confidence >= confidence_threshold
                            else "haar_face_fallback"
                        ),
                    )
                if subject_scored_clusters and subject_scored_clusters[0].confidence >= confidence_threshold:
                    selected_subject = subject_scored_clusters[0]
                    return _write_overlay_result(
                        output_path,
                        clip_id=metadata_clip_id,
                        selected_face_rect=selected_subject.face_rect,
                        selected_overlay_rect=selected_subject.overlay_rect,
                        confidence=selected_subject.confidence,
                        fallback=False,
                        reason=_selection_reason(selected_subject, subject_scored_clusters),
                        candidate_clusters=subject_scored_clusters,
                        selected_source="yolo_person",
                        subject_detector_error=None,
                    )

    detector_instance: FaceDetector
    try:
        detector_instance = detector or OpenCVFaceDetector()
    except OverlayDetectorUnavailable as exc:
        return _write_overlay_result(
            output_path,
            clip_id=metadata_clip_id,
            selected_face_rect=None,
            selected_overlay_rect=None,
            confidence=0.0,
            fallback=True,
            reason=f"fallback: detector unavailable: {exc}",
            candidate_clusters=subject_scored_clusters,
            selected_source="overlay_fallback",
            subject_detector_error=subject_unavailable_reason,
        )

    try:
        detection_report = _detect_faces(
            frame_paths,
            sampled_timestamps=sampled_timestamps,
            detector=detector_instance,
        )
    except OverlayDetectorUnavailable as exc:
        return _write_overlay_result(
            output_path,
            clip_id=metadata_clip_id,
            selected_face_rect=None,
            selected_overlay_rect=None,
            confidence=0.0,
            fallback=True,
            reason=f"fallback: detector failed: {exc}",
            candidate_clusters=subject_scored_clusters,
            selected_source="overlay_fallback",
            subject_detector_error=subject_unavailable_reason,
        )

    clusters = _cluster_detections(detection_report.detections)
    scored_clusters = _score_clusters(clusters, total_frames=len(frame_paths))
    if not scored_clusters:
        _log_face_detection_diagnostics(
            detection_report=detection_report,
            sampled_frame_count=len(frame_paths),
            clusters=clusters,
            selected_mode="overlay_fallback",
            fallback_reason="fallback: no face detections found in sampled frames",
        )
        if debug_raw_faces:
            _write_raw_face_debug(
                analysis_clip_dir=analysis_clip_dir,
                frame_paths=frame_paths,
                detection_report=detection_report,
                clusters=clusters,
                scored_clusters=scored_clusters,
                selected_mode="overlay_fallback",
                fallback_reason="fallback: no face detections found in sampled frames",
            )
        return _write_overlay_result(
            output_path,
            clip_id=metadata_clip_id,
            selected_face_rect=None,
            selected_overlay_rect=None,
            confidence=0.0,
            fallback=True,
            reason="fallback: no face detections found in sampled frames",
            candidate_clusters=subject_scored_clusters,
            selected_source="overlay_fallback",
            subject_detector_error=subject_unavailable_reason,
        )

    selected = scored_clusters[0]
    if selected.confidence < confidence_threshold:
        fallback_reason = (
            "fallback: best candidate confidence "
            f"{selected.confidence:.3f} is below threshold {confidence_threshold:.3f}"
        )
        _log_face_detection_diagnostics(
            detection_report=detection_report,
            sampled_frame_count=len(frame_paths),
            clusters=clusters,
            selected_mode="overlay_fallback",
            fallback_reason=fallback_reason,
        )
        if debug_raw_faces:
            _write_raw_face_debug(
                analysis_clip_dir=analysis_clip_dir,
                frame_paths=frame_paths,
                detection_report=detection_report,
                clusters=clusters,
                scored_clusters=scored_clusters,
                selected_mode="overlay_fallback",
                fallback_reason=fallback_reason,
            )
        return _write_overlay_result(
            output_path,
            clip_id=metadata_clip_id,
            selected_face_rect=None,
            selected_overlay_rect=None,
            confidence=selected.confidence,
            fallback=True,
            reason=fallback_reason,
            candidate_clusters=scored_clusters,
            selected_source="overlay_fallback",
            subject_detector_error=subject_unavailable_reason,
        )

    _log_face_detection_diagnostics(
        detection_report=detection_report,
        sampled_frame_count=len(frame_paths),
        clusters=clusters,
        selected_mode="face_cluster",
        fallback_reason=None,
    )
    if debug_raw_faces:
        _write_raw_face_debug(
            analysis_clip_dir=analysis_clip_dir,
            frame_paths=frame_paths,
            detection_report=detection_report,
            clusters=clusters,
            scored_clusters=scored_clusters,
            selected_mode="face_cluster",
            fallback_reason=None,
        )
    return _write_overlay_result(
        output_path,
        clip_id=metadata_clip_id,
        selected_face_rect=selected.face_rect,
        selected_overlay_rect=selected.overlay_rect,
        confidence=selected.confidence,
        fallback=False,
        reason=_selection_reason(selected, scored_clusters),
        candidate_clusters=scored_clusters,
        selected_source="haar_face",
        subject_detector_error=subject_unavailable_reason,
    )


def write_overlay_debug_images(
    *,
    clip_id: str,
    analysis_dir: Path = ANALYSIS_DIR,
    image_writer: OverlayDebugImageWriter | None = None,
) -> Path:
    """Draw overlay inference candidates on sampled frames and return the debug dir."""

    safe_clip_id = _safe_clip_id(clip_id)
    analysis_clip_dir = clip_analysis_dir(analysis_dir, safe_clip_id)
    frames_metadata_path = analysis_clip_dir / "frames.json"
    overlay_path = analysis_clip_dir / "overlay.json"
    if not frames_metadata_path.is_file():
        raise AnalysisError(f"Frame metadata not found: {frames_metadata_path}")
    if not overlay_path.is_file():
        raise AnalysisError(f"Overlay metadata not found: {overlay_path}")

    frames_metadata = _read_frames_metadata(frames_metadata_path)
    frame_paths = _frame_paths_from_metadata(frames_metadata, base_path=frames_metadata_path.parent)
    _require_existing_frames(frame_paths)
    overlay_metadata = _read_overlay_metadata(overlay_path)
    annotations = _debug_annotations(overlay_metadata)
    banner = _debug_banner(overlay_metadata)
    writer = image_writer or OpenCVOverlayDebugImageWriter()
    debug_dir = ensure_directory(analysis_clip_dir / "debug")

    for index, frame_path in enumerate(frame_paths, start=1):
        output_path = debug_dir / f"{frame_path.stem}_overlay_debug{frame_path.suffix or '.jpg'}"
        writer.write(
            frame_path=frame_path,
            output_path=output_path,
            annotations=annotations,
            banner=banner,
        )

    return debug_dir


def _detect_faces(
    frame_paths: tuple[Path, ...],
    *,
    sampled_timestamps: tuple[float | None, ...],
    detector: FaceDetector,
) -> _FaceDetectionReport:
    detections: list[_FrameDetection] = []
    raw_detections: list[_RawFrameDetection] = []
    for frame_index, (frame_path, timestamp) in enumerate(
        zip(frame_paths, sampled_timestamps, strict=True)
    ):
        try:
            frame_raw_detections = _raw_detections_for_frame(
                detector,
                frame_path=frame_path,
                frame_index=frame_index,
                timestamp=timestamp,
            )
        except Exception as exc:
            raise OverlayDetectorUnavailable(str(exc)) from exc

        raw_detections.extend(frame_raw_detections)
        for detection in frame_raw_detections:
            if not detection.filtered_out:
                detections.append(
                    _FrameDetection(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        rect=detection.rect,
                        score=_clamp01(detection.score),
                    )
                )
    return _FaceDetectionReport(
        detections=tuple(detections),
        raw_detections=tuple(raw_detections),
        detector_info=_detector_debug_info(detector),
        sampled_timestamps=sampled_timestamps,
    )


def _detect_subjects(
    frame_paths: tuple[Path, ...],
    *,
    sampled_timestamps: tuple[float | None, ...],
    detector: object,
) -> _SubjectDetectionReport:
    detections: list[_FrameDetection] = []
    for frame_index, (frame_path, timestamp) in enumerate(
        zip(frame_paths, sampled_timestamps, strict=True)
    ):
        try:
            frame_detections = detector.detect(
                frame_path,
                frame_index=frame_index,
                timestamp=timestamp,
            )
        except Exception as exc:
            raise OverlayDetectorUnavailable(str(exc)) from exc
        for detection in frame_detections:
            if detection.class_label != "person":
                continue
            if not _is_valid_rect(detection.rect):
                continue
            detections.append(
                _FrameDetection(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    rect=detection.rect,
                    score=_clamp01(detection.confidence),
                    class_label=detection.class_label,
                    detector_name=detection.detector_name,
                )
            )
    return _SubjectDetectionReport(
        detections=tuple(detections),
        raw_detections=tuple(detections),
        detector_info=_subject_detector_debug_info(detector),
        sampled_timestamps=sampled_timestamps,
    )


def _subject_detector_debug_info(detector: object) -> object:
    info = getattr(detector, "debug_info", None)
    if info is not None:
        return info
    return {
        "name": detector.__class__.__name__,
        "settings": {"source": "SubjectDetector.detect(frame_path)"},
    }


def _build_subject_detector_from_env() -> object | None:
    detector_name = os.getenv("CLIPFORGE_SUBJECT_DETECTOR", "yolo").strip().lower()
    if detector_name in {"", "none", "off", "disabled", "haar", "haar_face"}:
        return None
    if detector_name != "yolo":
        raise OverlayDetectorUnavailable(
            "Unsupported subject detector "
            f"{detector_name!r}; supported values are yolo, haar, or none."
        )

    from clipforge.media.subject import (
        DEFAULT_YOLO_CONFIDENCE_THRESHOLD,
        DEFAULT_YOLO_DEVICE,
        DEFAULT_YOLO_MODEL,
        YOLOPersonDetector,
    )

    threshold_value = os.getenv("CLIPFORGE_YOLO_CONFIDENCE_THRESHOLD")
    try:
        threshold = (
            float(threshold_value)
            if threshold_value is not None and threshold_value.strip()
            else DEFAULT_YOLO_CONFIDENCE_THRESHOLD
        )
    except ValueError as exc:
        raise OverlayDetectorUnavailable(
            "Invalid CLIPFORGE_YOLO_CONFIDENCE_THRESHOLD; expected a number."
        ) from exc

    return YOLOPersonDetector(
        model=os.getenv("CLIPFORGE_YOLO_MODEL", DEFAULT_YOLO_MODEL),
        device=os.getenv("CLIPFORGE_YOLO_DEVICE", DEFAULT_YOLO_DEVICE).strip().lower(),
        confidence_threshold=threshold,
    )


def _raw_detections_for_frame(
    detector: FaceDetector,
    *,
    frame_path: Path,
    frame_index: int,
    timestamp: float | None,
) -> tuple[_RawFrameDetection, ...]:
    detect_raw = getattr(detector, "detect_raw", None)
    if callable(detect_raw):
        raw_detections = detect_raw(frame_path)
        return tuple(
            _RawFrameDetection(
                frame_index=frame_index,
                timestamp=timestamp,
                frame_path=frame_path,
                rect=detection.rect,
                score=_clamp01(detection.score),
                detector_name=detection.detector_name,
                pass_metadata=detection.pass_metadata,
                merged_from_count=detection.merged_from_count,
                merged_passes=detection.merged_passes,
                filtered_out=detection.filtered_out,
                filter_reason=detection.filter_reason,
            )
            for detection in raw_detections
        )

    detector_name = _detector_debug_info(detector).name
    return tuple(
        _raw_detection_from_public_detection(
            detection,
            detector_name=detector_name,
            frame_path=frame_path,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        for detection in detector.detect(frame_path)
    )


def _raw_detection_from_public_detection(
    detection: FaceDetection,
    *,
    detector_name: str,
    frame_path: Path,
    frame_index: int,
    timestamp: float | None,
) -> _RawFrameDetection:
    filter_reason = _detection_filter_reason(detection.rect)
    return _RawFrameDetection(
        frame_index=frame_index,
        timestamp=timestamp,
        frame_path=frame_path,
        rect=detection.rect,
        score=_clamp01(detection.score),
        detector_name=detector_name,
        pass_metadata=None,
        merged_from_count=1,
        merged_passes=(),
        filtered_out=filter_reason is not None,
        filter_reason=filter_reason,
    )


def _detector_debug_info(detector: FaceDetector) -> FaceDetectorDebugInfo:
    info = getattr(detector, "debug_info", None)
    if isinstance(info, FaceDetectorDebugInfo):
        return info
    return FaceDetectorDebugInfo(
        name=detector.__class__.__name__,
        settings={
            "source": "FaceDetector.detect(frame_path)",
            "raw_detection_method": "public_detect_adapter",
        },
    )


def _log_face_detection_diagnostics(
    *,
    detection_report: _FaceDetectionReport,
    sampled_frame_count: int,
    clusters: tuple[_Cluster, ...],
    selected_mode: str,
    fallback_reason: str | None,
) -> None:
    filtered_count = sum(1 for detection in detection_report.raw_detections if detection.filtered_out)
    filter_reasons = Counter(
        detection.filter_reason or "kept"
        for detection in detection_report.raw_detections
        if detection.filtered_out
    )
    LOGGER.info(
        "overlay face diagnostics: sampled_frames=%s raw_face_detections=%s "
        "clusters_formed=%s detections_filtered=%s top_filter_reasons=%s "
        "selected_mode=%s fallback_reason=%s",
        sampled_frame_count,
        len(detection_report.raw_detections),
        len(clusters),
        filtered_count,
        dict(filter_reasons.most_common()),
        selected_mode,
        fallback_reason,
    )


def _write_raw_face_debug(
    *,
    analysis_clip_dir: Path,
    frame_paths: tuple[Path, ...],
    detection_report: _FaceDetectionReport,
    clusters: tuple[_Cluster, ...],
    scored_clusters: tuple[_ScoredCluster, ...],
    selected_mode: str,
    fallback_reason: str | None,
) -> Path:
    debug_dir = ensure_directory(analysis_clip_dir / "debug" / "raw_faces")
    _write_raw_face_debug_images(
        debug_dir=debug_dir,
        frame_paths=frame_paths,
        raw_detections=detection_report.raw_detections,
    )
    payload = _raw_face_debug_payload(
        detection_report=detection_report,
        frame_paths=frame_paths,
        clusters=clusters,
        scored_clusters=scored_clusters,
        selected_mode=selected_mode,
        fallback_reason=fallback_reason,
    )
    output_path = debug_dir / "raw_faces.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _write_raw_subject_debug(
    *,
    analysis_clip_dir: Path,
    frame_paths: tuple[Path, ...],
    detection_report: _SubjectDetectionReport,
    clusters: tuple[_Cluster, ...],
    scored_clusters: tuple[_ScoredCluster, ...],
    selected_mode: str,
) -> Path:
    debug_dir = ensure_directory(analysis_clip_dir / "debug" / "raw_subjects")
    _write_raw_subject_debug_images(
        debug_dir=debug_dir,
        frame_paths=frame_paths,
        raw_detections=detection_report.raw_detections,
    )
    payload = _raw_subject_debug_payload(
        detection_report=detection_report,
        frame_paths=frame_paths,
        clusters=clusters,
        scored_clusters=scored_clusters,
        selected_mode=selected_mode,
    )
    output_path = debug_dir / "raw_subjects.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _write_raw_subject_debug_images(
    *,
    debug_dir: Path,
    frame_paths: tuple[Path, ...],
    raw_detections: tuple[_FrameDetection, ...],
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise AnalysisError(
            "OpenCV is not installed; install opencv-python to write raw subject debug images."
        ) from exc

    detections_by_frame: dict[int, list[_FrameDetection]] = {
        index: [] for index, _path in enumerate(frame_paths)
    }
    for detection in raw_detections:
        detections_by_frame.setdefault(detection.frame_index, []).append(detection)

    for frame_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise AnalysisError(f"Could not read sampled frame for raw subject debug: {frame_path}")

        height, width = image.shape[:2]
        banner = (
            f"raw YOLO person detections | frame {frame_index} | "
            f"count {len(detections_by_frame[frame_index])}"
        )
        _draw_debug_label(
            cv2,
            image,
            banner,
            (8, 24),
            color=(255, 255, 255),
            background=(32, 32, 32),
        )
        for detection in detections_by_frame[frame_index]:
            left, top, right, bottom = _rect_pixels(detection.rect, width=width, height=height)
            color = (255, 180, 0)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            label = (
                f"f{detection.frame_index} {detection.detector_name} "
                f"{detection.class_label} conf {detection.score:.3f} "
                f"x{detection.rect.x:.3f} y{detection.rect.y:.3f} "
                f"w{detection.rect.width:.3f} h{detection.rect.height:.3f}"
            )
            _draw_debug_label(
                cv2,
                image,
                label,
                (left, max(18, top - 8)),
                color=(255, 255, 255),
                background=color,
            )

        output_path = debug_dir / f"{frame_path.stem}.png"
        if not cv2.imwrite(str(output_path), image):
            raise AnalysisError(f"Could not write raw subject debug image: {output_path}")


def _raw_subject_debug_payload(
    *,
    detection_report: _SubjectDetectionReport,
    frame_paths: tuple[Path, ...],
    clusters: tuple[_Cluster, ...],
    scored_clusters: tuple[_ScoredCluster, ...],
    selected_mode: str,
) -> dict[str, object]:
    cluster_lookup = _raw_detection_cluster_lookup(clusters, scored_clusters)
    frames = []
    for frame_index, frame_path in enumerate(frame_paths):
        frame_detections = [
            detection
            for detection in detection_report.raw_detections
            if detection.frame_index == frame_index
        ]
        frames.append(
            {
                "frame_index": frame_index,
                "frame_path": str(frame_path),
                "frame_timestamp": detection_report.sampled_timestamps[frame_index],
                "raw_detections": [
                    {
                        "x": _round(detection.rect.x),
                        "y": _round(detection.rect.y),
                        "w": _round(detection.rect.width),
                        "h": _round(detection.rect.height),
                        "confidence": _round(detection.score),
                        "class_label": detection.class_label,
                        "detector_name": detection.detector_name,
                        "cluster_id": (
                            cluster_lookup[_raw_detection_key(detection)]["cluster_id"]
                            if _raw_detection_key(detection) in cluster_lookup
                            else None
                        ),
                    }
                    for detection in frame_detections
                ],
            }
        )

    info_name = getattr(detection_report.detector_info, "name", None)
    info_settings = getattr(detection_report.detector_info, "settings", None)
    if info_name is None and isinstance(detection_report.detector_info, dict):
        info_name = detection_report.detector_info.get("name")
        info_settings = detection_report.detector_info.get("settings")

    return {
        "sampled_frame_count": len(frame_paths),
        "raw_subject_detection_count": len(detection_report.raw_detections),
        "clusters_formed": len(clusters),
        "reported_clusters": len(scored_clusters),
        "selected_mode": selected_mode,
        "detector": {
            "name": info_name,
            "settings": info_settings or {},
        },
        "frames": frames,
        "clusters": [cluster.to_dict() for cluster in scored_clusters],
    }


def _write_raw_face_debug_images(
    *,
    debug_dir: Path,
    frame_paths: tuple[Path, ...],
    raw_detections: tuple[_RawFrameDetection, ...],
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise AnalysisError(
            "OpenCV is not installed; install opencv-python to write raw face debug images."
        ) from exc

    detections_by_frame: dict[int, list[_RawFrameDetection]] = {
        index: [] for index, _path in enumerate(frame_paths)
    }
    for detection in raw_detections:
        detections_by_frame.setdefault(detection.frame_index, []).append(detection)

    for frame_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise AnalysisError(f"Could not read sampled frame for raw face debug: {frame_path}")

        height, width = image.shape[:2]
        banner = f"raw face detections | frame {frame_index} | count {len(detections_by_frame[frame_index])}"
        _draw_debug_label(
            cv2,
            image,
            banner,
            (8, 24),
            color=(255, 255, 255),
            background=(32, 32, 32),
        )
        for detection in detections_by_frame[frame_index]:
            color = (0, 80, 255) if detection.filtered_out else (40, 220, 40)
            left, top, right, bottom = _rect_pixels(detection.rect, width=width, height=height)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            label = (
                f"f{detection.frame_index} {detection.detector_name} "
                f"conf {detection.score:.3f} "
                f"x{detection.rect.x:.3f} y{detection.rect.y:.3f} "
                f"w{detection.rect.width:.3f} h{detection.rect.height:.3f}"
            )
            if detection.pass_metadata is not None:
                label += (
                    f" sf {detection.pass_metadata['scaleFactor']} "
                    f"mn {detection.pass_metadata['minNeighbors']} "
                    f"merged {detection.merged_from_count}"
                )
            if detection.filtered_out:
                label += f" filtered {detection.filter_reason}"
            _draw_debug_label(
                cv2,
                image,
                label,
                (left, max(18, top - 8)),
                color=(255, 255, 255),
                background=color,
            )

        output_path = debug_dir / f"{frame_path.stem}.png"
        if not cv2.imwrite(str(output_path), image):
            raise AnalysisError(f"Could not write raw face debug image: {output_path}")


def _raw_face_debug_payload(
    *,
    detection_report: _FaceDetectionReport,
    frame_paths: tuple[Path, ...],
    clusters: tuple[_Cluster, ...],
    scored_clusters: tuple[_ScoredCluster, ...],
    selected_mode: str,
    fallback_reason: str | None,
) -> dict[str, object]:
    cluster_lookup = _raw_detection_cluster_lookup(clusters, scored_clusters)
    raw_detection_count = len(detection_report.raw_detections)
    filtered_count = sum(1 for detection in detection_report.raw_detections if detection.filtered_out)
    filter_reasons = Counter(
        detection.filter_reason or "kept"
        for detection in detection_report.raw_detections
        if detection.filtered_out
    )
    frames = []
    for frame_index, frame_path in enumerate(frame_paths):
        frame_detections = [
            detection
            for detection in detection_report.raw_detections
            if detection.frame_index == frame_index
        ]
        frames.append(
            {
                "frame_index": frame_index,
                "frame_path": str(frame_path),
                "frame_timestamp": detection_report.sampled_timestamps[frame_index],
                "raw_detections": [
                    _raw_detection_to_dict(
                        detection,
                        cluster_lookup=cluster_lookup,
                    )
                    for detection in frame_detections
                ],
            }
        )

    return {
        "sampled_frame_count": len(frame_paths),
        "raw_face_detection_count": raw_detection_count,
        "filtered_detection_count": filtered_count,
        "clusters_formed": len(clusters),
        "reported_clusters": len(scored_clusters),
        "top_filter_reasons": dict(filter_reasons.most_common()),
        "selected_mode": selected_mode,
        "fallback_reason": fallback_reason,
        "detector": {
            "name": detection_report.detector_info.name,
            "settings": detection_report.detector_info.settings,
        },
        "frames": frames,
        "clusters": [cluster.to_dict() for cluster in scored_clusters],
    }


def _raw_detection_cluster_lookup(
    clusters: tuple[_Cluster, ...],
    scored_clusters: tuple[_ScoredCluster, ...],
) -> dict[tuple[int, float, float, float, float, float], dict[str, int | None]]:
    rank_by_cluster_id = {
        cluster.cluster_id: cluster.ranking_position for cluster in scored_clusters
    }
    lookup: dict[tuple[int, float, float, float, float, float], dict[str, int | None]] = {}
    for cluster_id, cluster in enumerate(clusters, start=1):
        for detection in cluster.detections:
            lookup[_raw_detection_key(detection)] = {
                "cluster_id": cluster_id,
                "ranking_position": rank_by_cluster_id.get(cluster_id),
            }
    return lookup


def _raw_detection_to_dict(
    detection: _RawFrameDetection,
    *,
    cluster_lookup: dict[tuple[int, float, float, float, float, float], dict[str, int | None]],
) -> dict[str, object]:
    cluster_info = cluster_lookup.get(_raw_detection_key(detection))
    return {
        "x": _round(detection.rect.x),
        "y": _round(detection.rect.y),
        "w": _round(detection.rect.width),
        "h": _round(detection.rect.height),
        "confidence": _round(detection.score),
        "detector_name": detection.detector_name,
        "pass_metadata": detection.pass_metadata,
        "scaleFactor": (
            detection.pass_metadata["scaleFactor"] if detection.pass_metadata else None
        ),
        "minNeighbors": (
            detection.pass_metadata["minNeighbors"] if detection.pass_metadata else None
        ),
        "minSize": detection.pass_metadata["minSize"] if detection.pass_metadata else None,
        "merged_from_count": detection.merged_from_count,
        "merged_passes": list(detection.merged_passes),
        "filtered_out": detection.filtered_out,
        "filter_reason": detection.filter_reason,
        "cluster_id": cluster_info["cluster_id"] if cluster_info else None,
        "ranking_position": cluster_info["ranking_position"] if cluster_info else None,
    }


def _raw_detection_key(
    detection: _FrameDetection | _RawFrameDetection,
) -> tuple[int, float, float, float, float, float]:
    return (
        detection.frame_index,
        round(detection.rect.x, 8),
        round(detection.rect.y, 8),
        round(detection.rect.width, 8),
        round(detection.rect.height, 8),
        round(detection.score, 8),
    )


def _cluster_detections(detections: tuple[_FrameDetection, ...]) -> tuple[_Cluster, ...]:
    clusters: list[_Cluster] = []
    for detection in detections:
        nearest_cluster: _Cluster | None = None
        nearest_distance = math.inf
        for cluster in clusters:
            distance = _cluster_distance(_cluster_rect(cluster), detection.rect)
            if distance < nearest_distance:
                nearest_cluster = cluster
                nearest_distance = distance

        if nearest_cluster is not None and nearest_distance <= 1.0:
            nearest_cluster.add(detection)
        else:
            clusters.append(_Cluster(detections=[detection]))

    return tuple(clusters)


def _score_clusters(
    clusters: tuple[_Cluster, ...],
    *,
    total_frames: int,
) -> tuple[_ScoredCluster, ...]:
    scored = [
        _score_cluster(cluster, cluster_id=index + 1, total_frames=total_frames)
        for index, cluster in enumerate(clusters)
    ]
    scored.sort(key=lambda cluster: cluster.raw_score, reverse=True)

    adjusted: list[_ScoredCluster] = []
    for index, cluster in enumerate(scored):
        penalty = _competition_penalty(cluster, scored[:index] + scored[index + 1 :])
        competition_multiplier = _clamp01(1.0 - penalty)
        heuristic_multiplier = _clamp(
            cluster.heuristic_multiplier * competition_multiplier,
            lower=0.0,
            upper=1.85,
        )
        final_score = _clamp01(cluster.raw_score * competition_multiplier)
        adjusted.append(
            _ScoredCluster(
                cluster_id=cluster.cluster_id,
                source=cluster.source,
                class_label=cluster.class_label,
                detector_name=cluster.detector_name,
                face_rect=cluster.face_rect,
                overlay_rect=cluster.overlay_rect,
                representative_face_rect=cluster.representative_face_rect,
                average_face_rect=cluster.average_face_rect,
                detection_count=cluster.detection_count,
                sampled_frame_count=cluster.sampled_frame_count,
                frame_count=cluster.frame_count,
                prevalence=cluster.prevalence,
                average_face_confidence=cluster.average_face_confidence,
                max_face_confidence=cluster.max_face_confidence,
                frame_indexes=cluster.frame_indexes,
                timestamps=cluster.timestamps,
                component_scores={
                    **cluster.component_scores,
                    "competition_penalty": penalty,
                    "competition_multiplier": competition_multiplier,
                },
                face_score=cluster.face_score,
                heuristic_multiplier=heuristic_multiplier,
                final_score=final_score,
                ranking_position=None,
                raw_score=cluster.raw_score,
                confidence=final_score,
            )
        )
    adjusted.sort(key=lambda cluster: cluster.confidence, reverse=True)
    reported = _reported_clusters(adjusted)
    return tuple(
        _ScoredCluster(
            cluster_id=cluster.cluster_id,
            source=cluster.source,
            class_label=cluster.class_label,
            detector_name=cluster.detector_name,
            face_rect=cluster.face_rect,
            overlay_rect=cluster.overlay_rect,
            representative_face_rect=cluster.representative_face_rect,
            average_face_rect=cluster.average_face_rect,
            detection_count=cluster.detection_count,
            sampled_frame_count=cluster.sampled_frame_count,
            frame_count=cluster.frame_count,
            prevalence=cluster.prevalence,
            average_face_confidence=cluster.average_face_confidence,
            max_face_confidence=cluster.max_face_confidence,
            frame_indexes=cluster.frame_indexes,
            timestamps=cluster.timestamps,
            component_scores=cluster.component_scores,
            face_score=cluster.face_score,
            heuristic_multiplier=cluster.heuristic_multiplier,
            final_score=cluster.final_score,
            ranking_position=index,
            raw_score=cluster.raw_score,
            confidence=cluster.confidence,
        )
        for index, cluster in enumerate(reported, start=1)
    )


def _reported_clusters(scored_clusters: list[_ScoredCluster]) -> list[_ScoredCluster]:
    reported: list[_ScoredCluster] = []

    def add(cluster: _ScoredCluster) -> None:
        if all(existing.cluster_id != cluster.cluster_id for existing in reported):
            reported.append(cluster)

    for cluster in scored_clusters[: min(3, MAX_REPORTED_FACE_CLUSTERS)]:
        add(cluster)

    if scored_clusters:
        add(max(scored_clusters, key=lambda cluster: cluster.face_score))

    for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
        corner_clusters = [
            cluster
            for cluster in scored_clusters
            if _corner_affinity(cluster.overlay_rect, corner) > 0.0
        ]
        if corner_clusters:
            add(
                max(
                    corner_clusters,
                    key=lambda cluster: (
                        cluster.face_score * _corner_affinity(cluster.overlay_rect, corner),
                        cluster.confidence,
                    ),
                )
            )

    for cluster in scored_clusters:
        add(cluster)
        if len(reported) >= MAX_REPORTED_FACE_CLUSTERS:
            break

    return sorted(reported[:MAX_REPORTED_FACE_CLUSTERS], key=lambda cluster: cluster.confidence, reverse=True)


def _score_cluster(
    cluster: _Cluster,
    *,
    cluster_id: int,
    total_frames: int,
) -> _ScoredCluster:
    subject_label = _cluster_class_label(cluster)
    detector_name = _cluster_detector_name(cluster)
    average_face_rect = _cluster_rect(cluster)
    representative_face_rect = _representative_rect(cluster, average_face_rect)
    face_rect = average_face_rect
    if subject_label == "person":
        overlay_rect = _expand_person_rect_to_streamer_crop_rect(face_rect)
    else:
        overlay_rect = _expand_face_rect_to_streamer_crop_rect(face_rect)
    detection_count = len(cluster.detections)
    frame_indexes = tuple(sorted({detection.frame_index for detection in cluster.detections}))
    timestamps = tuple(
        detection.timestamp
        for detection in sorted(cluster.detections, key=lambda item: item.frame_index)
        if detection.timestamp is not None
    )
    frame_count = len(frame_indexes)
    prevalence = frame_count / total_frames if total_frames else 0.0
    average_face_confidence = statistics.fmean(
        detection.score for detection in cluster.detections
    )
    max_face_confidence = max(detection.score for detection in cluster.detections)

    position_stability = _position_stability(cluster)
    size_stability = _size_stability(cluster)
    edge_proximity = _edge_proximity(overlay_rect)
    edge_corner_prior = _edge_corner_prior(overlay_rect)
    webcam_location_prior = _webcam_location_prior(overlay_rect)
    center_avoidance = _center_avoidance(overlay_rect)
    size_score = _overlay_size_score(overlay_rect.area)
    expanded_aspect_ratio_score = _rect_aspect_ratio_score(
        overlay_rect,
        target_ratio=STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
    )
    expanded_box_quality = _clamp01(size_score * 0.62 + expanded_aspect_ratio_score * 0.38)
    face_aspect_ratio_score = _face_aspect_ratio_score(face_rect)
    confidence_score = (
        average_face_confidence * 0.72
        + max_face_confidence * 0.18
        + min(average_face_confidence, max_face_confidence) * 0.10
    )
    recurrence_multiplier = 0.35 + prevalence * 0.65
    face_score = _clamp01(confidence_score * recurrence_multiplier)

    one_frame_penalty = 0.32 if frame_count <= 1 and total_frames > 1 else 0.0
    brief_penalty = max(0.0, 0.45 - prevalence) * 0.70
    central_penalty = (1.0 - center_avoidance) * 0.50
    hud_penalty = _hud_penalty(overlay_rect, webcam_location_prior=webcam_location_prior)
    huge_central_penalty = _huge_central_penalty(overlay_rect, center_avoidance)
    tiny_penalty = _tiny_penalty(overlay_rect.area, size_score)

    if subject_label == "person":
        recurrence_multiplier = 0.30 + prevalence * 0.70
        face_score = _clamp01(confidence_score * recurrence_multiplier)
        one_frame_penalty = 0.42 if frame_count <= 1 and total_frames > 1 else 0.0
        brief_penalty = max(0.0, 0.38 - prevalence) * 0.55
        central_penalty = (1.0 - center_avoidance) * (0.72 - webcam_location_prior * 0.22)
        hud_penalty = _hud_penalty(overlay_rect, webcam_location_prior=webcam_location_prior) * 0.55
        huge_central_penalty = _huge_central_penalty(overlay_rect, center_avoidance) * 0.55
        tiny_penalty = 0.0

    prevalence_multiplier = 0.45 + prevalence * 0.55
    stability_multiplier = 0.70 + (
        position_stability * 0.58 + size_stability * 0.42
    ) * 0.30
    placement_multiplier = 0.70 + edge_corner_prior * 0.30
    size_multiplier = 0.25 + expanded_box_quality * 0.75
    aspect_ratio_multiplier = 0.70 + expanded_aspect_ratio_score * 0.30
    penalty_score = _clamp(
        one_frame_penalty
        + brief_penalty
        + central_penalty
        + hud_penalty
        + huge_central_penalty
        + tiny_penalty,
        lower=0.0,
        upper=1.35,
    )
    penalty_multiplier = _clamp01(
        1.0
        - one_frame_penalty
        - brief_penalty
        - central_penalty
        - hud_penalty
        - huge_central_penalty
        - tiny_penalty
    )
    overlay_support_score = _clamp01(
        position_stability * 0.36
        + size_stability * 0.24
        + prevalence * 0.40
    )
    context_support_score = _clamp01(
        prevalence * 0.18
        + edge_corner_prior * 0.20
        + webcam_location_prior * 0.22
        + expanded_box_quality * 0.18
        + overlay_support_score * 0.17
        + confidence_score * 0.05
    )
    heuristic_multiplier = _clamp(
        0.45 + context_support_score * 1.45 - penalty_score,
        lower=0.0,
        upper=1.85,
    )
    heuristic_floor_applied = 0.0
    if face_score > 0.0 and heuristic_multiplier < MIN_VALID_FACE_HEURISTIC_MULTIPLIER:
        heuristic_floor_applied = MIN_VALID_FACE_HEURISTIC_MULTIPLIER
        heuristic_multiplier = MIN_VALID_FACE_HEURISTIC_MULTIPLIER
    raw_score = _clamp01(face_score * heuristic_multiplier)
    if subject_label == "person" and webcam_location_prior < 0.18 and center_avoidance < 0.45:
        raw_score = min(raw_score, 0.50)
    if tiny_penalty > 0.30 or huge_central_penalty > 0.20:
        raw_score = min(raw_score, 0.45)

    component_scores = {
        "prevalence": prevalence,
        "recurrence": prevalence,
        "detection_count": float(detection_count),
        "sampled_frame_count": float(total_frames),
        "position_stability": position_stability,
        "size_stability": size_stability,
        "edge_proximity": edge_proximity,
        "edge_corner_prior": edge_corner_prior,
        "webcam_location_prior": webcam_location_prior,
        "center_avoidance": center_avoidance,
        "central_gameplay_penalty": central_penalty,
        "overlay_size": size_score,
        "expanded_aspect_ratio": expanded_aspect_ratio_score,
        "expanded_box_quality": expanded_box_quality,
        "face_aspect_ratio": face_aspect_ratio_score,
        "average_face_confidence": average_face_confidence,
        "max_face_confidence": max_face_confidence,
        "confidence_score": confidence_score,
        "recurrence_multiplier": recurrence_multiplier,
        "detector_confidence": average_face_confidence,
        "overlay_support_score": overlay_support_score,
        "context_support_score": context_support_score,
        "prevalence_multiplier": prevalence_multiplier,
        "stability_multiplier": stability_multiplier,
        "placement_multiplier": placement_multiplier,
        "size_multiplier": size_multiplier,
        "aspect_ratio_multiplier": aspect_ratio_multiplier,
        "penalty_score": penalty_score,
        "penalty_multiplier": penalty_multiplier,
        "heuristic_floor_applied": heuristic_floor_applied,
        "one_frame_penalty": one_frame_penalty,
        "brief_penalty": brief_penalty,
        "central_penalty": central_penalty,
        "hud_penalty": hud_penalty,
        "huge_central_penalty": huge_central_penalty,
        "tiny_penalty": tiny_penalty,
    }
    return _ScoredCluster(
        cluster_id=cluster_id,
        source="yolo_person" if subject_label == "person" else "haar_face",
        class_label=subject_label,
        detector_name=detector_name,
        face_rect=face_rect,
        overlay_rect=overlay_rect,
        representative_face_rect=representative_face_rect,
        average_face_rect=average_face_rect,
        detection_count=detection_count,
        sampled_frame_count=total_frames,
        frame_count=frame_count,
        prevalence=prevalence,
        average_face_confidence=average_face_confidence,
        max_face_confidence=max_face_confidence,
        frame_indexes=frame_indexes,
        timestamps=timestamps,
        component_scores=component_scores,
        face_score=face_score,
        heuristic_multiplier=heuristic_multiplier,
        final_score=raw_score,
        ranking_position=None,
        raw_score=raw_score,
        confidence=raw_score,
    )


def _cluster_distance(rect: NormalizedRect, candidate: NormalizedRect) -> float:
    center_distance = math.hypot(
        (rect.center_x - candidate.center_x) / CLUSTER_CENTER_THRESHOLD,
        (rect.center_y - candidate.center_y) / CLUSTER_CENTER_THRESHOLD,
    )
    size_distance = math.hypot(
        (rect.width - candidate.width) / CLUSTER_SIZE_THRESHOLD,
        (rect.height - candidate.height) / CLUSTER_SIZE_THRESHOLD,
    )
    return max(center_distance, size_distance)


def _cluster_rect(cluster: _Cluster) -> NormalizedRect:
    return NormalizedRect(
        x=statistics.fmean(detection.rect.x for detection in cluster.detections),
        y=statistics.fmean(detection.rect.y for detection in cluster.detections),
        width=statistics.fmean(detection.rect.width for detection in cluster.detections),
        height=statistics.fmean(detection.rect.height for detection in cluster.detections),
    )


def _representative_rect(cluster: _Cluster, average_rect: NormalizedRect) -> NormalizedRect:
    return min(
        (detection.rect for detection in cluster.detections),
        key=lambda rect: _cluster_distance(average_rect, rect),
    )


def _cluster_class_label(cluster: _Cluster) -> str:
    labels = Counter(detection.class_label for detection in cluster.detections)
    return labels.most_common(1)[0][0] if labels else "face"


def _cluster_detector_name(cluster: _Cluster) -> str:
    names = Counter(detection.detector_name for detection in cluster.detections)
    return names.most_common(1)[0][0] if names else "haar_face"


def _expand_person_rect_to_streamer_crop_rect(person_rect: NormalizedRect) -> NormalizedRect:
    min_height = 0.28 if _edge_proximity(person_rect) > 0.45 else 0.20
    natural_height = max(person_rect.height * 1.18, min_height)
    natural_width = max(
        person_rect.width * 1.60,
        natural_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
        0.20 if _edge_proximity(person_rect) > 0.45 else 0.14,
    )
    natural_height = max(natural_height, natural_width / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO)

    max_height = 0.62 if _edge_proximity(person_rect) > 0.45 else 0.46
    natural_height = min(natural_height, max_height)
    natural_width = min(
        natural_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
        0.46,
    )
    natural_height = natural_width / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO

    left_edge = person_rect.x <= 0.14 or person_rect.center_x <= 0.28
    right_edge = person_rect.x + person_rect.width >= 0.86 or person_rect.center_x >= 0.72
    bottom_edge = person_rect.y + person_rect.height >= 0.78
    if bottom_edge and natural_height < 1.0 - person_rect.y:
        natural_height = min(1.0 - person_rect.y, max_height)
        natural_width = min(
            natural_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
            0.46,
        )
        natural_height = natural_width / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO

    if left_edge:
        x = 0.0
    elif right_edge:
        x = 1.0 - natural_width
    else:
        x = person_rect.center_x - natural_width / 2

    if bottom_edge:
        y = 1.0 - natural_height
    else:
        y = person_rect.center_y - natural_height * STREAMER_CROP_FACE_CENTER_Y_RATIO
        y = max(person_rect.y + person_rect.height - natural_height, min(y, person_rect.y))

    return _fit_rect_to_frame(
        x=x,
        y=y,
        width=natural_width,
        height=natural_height,
    )


def _expand_face_rect_to_streamer_crop_rect(face_rect: NormalizedRect) -> NormalizedRect:
    natural_width = min(
        face_rect.width * STREAMER_CROP_MAX_WIDTH_FACE_RATIO,
        STREAMER_CROP_MAX_NORMALIZED_WIDTH,
    )
    natural_width = min(
        natural_width,
        face_rect.width * (1.0 + STREAMER_CROP_HORIZONTAL_PADDING_RATIO * 2),
    )
    natural_width = max(face_rect.width, natural_width)

    natural_height = max(
        face_rect.height,
        face_rect.height * STREAMER_CROP_MAX_HEIGHT_FACE_RATIO,
    )
    natural_height = min(
        natural_height,
        face_rect.height
        * (
            1.0
            + STREAMER_CROP_TOP_PADDING_RATIO
            + STREAMER_CROP_BOTTOM_PADDING_RATIO
        ),
    )
    natural_height = max(
        face_rect.height,
        natural_height,
    )

    target_height = max(
        natural_height,
        natural_width / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
        face_rect.width / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
    )
    target_width = target_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO

    max_frame_height = min(1.0, 1.0 / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO)
    target_height = min(target_height, max_frame_height)
    target_width = target_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO

    min_face_height = max(
        face_rect.height,
        face_rect.width / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
    )
    anchored_height = min(
        _centered_rect_extent(face_rect.center_x)
        / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
        _biased_rect_extent(
            face_rect.center_y,
            target_ratio=STREAMER_CROP_FACE_CENTER_Y_RATIO,
        ),
    )
    if min_face_height <= anchored_height < target_height:
        target_height = anchored_height
        target_width = target_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO

    return _fit_streamer_crop_to_face(
        face_rect,
        width=target_width,
        height=target_height,
    )


def _fit_streamer_crop_to_face(
    face_rect: NormalizedRect,
    *,
    width: float,
    height: float,
) -> NormalizedRect:
    edge_anchored = _edge_anchored_streamer_crop(face_rect, width=width, height=height)
    if edge_anchored is not None:
        return edge_anchored

    x = face_rect.center_x - width / 2
    y = face_rect.center_y - height * STREAMER_CROP_FACE_CENTER_Y_RATIO

    x = max(face_rect.x + face_rect.width - width, min(x, face_rect.x))
    y = max(face_rect.y + face_rect.height - height, min(y, face_rect.y))
    return _fit_rect_to_frame(x=x, y=y, width=width, height=height)


def _edge_anchored_streamer_crop(
    face_rect: NormalizedRect,
    *,
    width: float,
    height: float,
) -> NormalizedRect | None:
    bottom_aligned_face = face_rect.y + face_rect.height >= 0.90
    left_aligned_face = face_rect.x <= 0.05
    right_aligned_face = face_rect.x + face_rect.width >= 0.95
    if not left_aligned_face and not right_aligned_face:
        return None
    if not bottom_aligned_face:
        side_fragment_face = (
            face_rect.width <= 0.08
            and face_rect.height <= 0.14
            and face_rect.y >= 0.08
        )
        if not side_fragment_face:
            return None
        return _side_anchored_streamer_crop(
            face_rect,
            width=width,
            height=height,
            align_right=right_aligned_face,
        )

    if left_aligned_face:
        minimum_width = face_rect.x + face_rect.width
        x = 0.0
    else:
        minimum_width = 1.0 - face_rect.x
        x = 1.0 - width

    minimum_height = max(
        height,
        1.0 - face_rect.y,
        minimum_width / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO,
    )
    anchored_height = min(
        minimum_height,
        min(1.0, 1.0 / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO),
    )
    anchored_width = anchored_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO
    if right_aligned_face:
        x = 1.0 - anchored_width

    return _fit_rect_to_frame(
        x=x,
        y=1.0 - anchored_height,
        width=anchored_width,
        height=anchored_height,
    )


def _side_anchored_streamer_crop(
    face_rect: NormalizedRect,
    *,
    width: float,
    height: float,
    align_right: bool,
) -> NormalizedRect:
    anchored_height = max(height, STREAMER_CROP_SIDE_MIN_NORMALIZED_HEIGHT)
    anchored_height = min(
        anchored_height,
        min(1.0, 1.0 / STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO),
    )
    anchored_width = max(width, anchored_height * STREAMER_CROP_TARGET_NORMALIZED_ASPECT_RATIO)
    anchored_width = min(anchored_width, 1.0)
    x = 1.0 - anchored_width if align_right else 0.0
    y = face_rect.center_y - anchored_height * STREAMER_CROP_SIDE_FACE_CENTER_Y_RATIO
    y = max(face_rect.y + face_rect.height - anchored_height, min(y, face_rect.y))

    return _fit_rect_to_frame(
        x=x,
        y=y,
        width=anchored_width,
        height=anchored_height,
    )


def _centered_rect_extent(center: float) -> float:
    return max(0.0, min(center, 1.0 - center) * 2)


def _biased_rect_extent(center: float, *, target_ratio: float) -> float:
    return max(
        0.0,
        min(
            center / target_ratio,
            (1.0 - center) / (1.0 - target_ratio),
        ),
    )


def _fit_rect_to_frame(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> NormalizedRect:
    width = _clamp01(width)
    height = _clamp01(height)
    return NormalizedRect(
        x=max(0.0, min(1.0 - width, x)),
        y=max(0.0, min(1.0 - height, y)),
        width=width,
        height=height,
    )


def _position_stability(cluster: _Cluster) -> float:
    if len(cluster.detections) <= 1:
        return 1.0
    center_x_deviation = statistics.pstdev(
        detection.rect.center_x for detection in cluster.detections
    )
    center_y_deviation = statistics.pstdev(
        detection.rect.center_y for detection in cluster.detections
    )
    return _clamp01(1.0 - math.hypot(center_x_deviation, center_y_deviation) / 0.08)


def _size_stability(cluster: _Cluster) -> float:
    if len(cluster.detections) <= 1:
        return 1.0
    width_deviation = statistics.pstdev(detection.rect.width for detection in cluster.detections)
    height_deviation = statistics.pstdev(
        detection.rect.height for detection in cluster.detections
    )
    return _clamp01(1.0 - math.hypot(width_deviation, height_deviation) / 0.06)


def _edge_proximity(rect: NormalizedRect) -> float:
    distances = sorted((rect.x, rect.y, 1.0 - rect.x - rect.width, 1.0 - rect.y - rect.height))
    edge_score = _clamp01(1.0 - distances[0] / 0.22)
    corner_score = _clamp01(1.0 - distances[1] / 0.25)
    return _clamp01(edge_score * 0.75 + corner_score * 0.25)


def _edge_corner_prior(rect: NormalizedRect) -> float:
    corner_score = max(
        _corner_affinity(rect, corner)
        for corner in ("top_left", "top_right", "bottom_left", "bottom_right")
    )
    return _clamp01(_edge_proximity(rect) * 0.46 + corner_score * 0.54)


def _corner_affinity(rect: NormalizedRect, corner: str) -> float:
    left = rect.x
    top = rect.y
    right = 1.0 - rect.x - rect.width
    bottom = 1.0 - rect.y - rect.height
    distances = {
        "top_left": (left, top),
        "top_right": (right, top),
        "bottom_left": (left, bottom),
        "bottom_right": (right, bottom),
    }
    horizontal_distance, vertical_distance = distances[corner]
    return _clamp01(
        (1.0 - horizontal_distance / 0.28)
        * (1.0 - vertical_distance / 0.28)
    )


def _webcam_location_prior(rect: NormalizedRect) -> float:
    corner_score = max(
        _corner_affinity(rect, corner)
        for corner in ("top_left", "top_right", "bottom_left", "bottom_right")
    )
    return _clamp01(corner_score * (0.68 + _overlay_size_score(rect.area) * 0.32))


def _center_avoidance(rect: NormalizedRect) -> float:
    center_distance = math.hypot(rect.center_x - 0.5, rect.center_y - 0.5)
    return _clamp01((center_distance - 0.12) / 0.38)


def _overlay_size_score(area: float) -> float:
    if area <= 0.008 or area >= 0.24:
        return 0.0
    if area < 0.025:
        return _clamp01((area - 0.008) / (0.025 - 0.008))
    if area <= 0.12:
        return 1.0
    return _clamp01(1.0 - (area - 0.12) / (0.24 - 0.12))


def _rect_aspect_ratio_score(rect: NormalizedRect, *, target_ratio: float) -> float:
    if rect.height <= 0 or target_ratio <= 0:
        return 0.0
    return _clamp01(1.0 - abs(rect.width / rect.height - target_ratio) / target_ratio)


def _face_aspect_ratio_score(rect: NormalizedRect) -> float:
    if rect.height <= 0:
        return 0.0
    ratio = (rect.width * SOURCE_ASPECT_RATIO) / rect.height
    return _clamp01(1.0 - abs(ratio - 1.0) / 0.55)


def _hud_penalty(rect: NormalizedRect, *, webcam_location_prior: float) -> float:
    if webcam_location_prior >= 0.35:
        return 0.0

    bottom_hud = _clamp01((rect.center_y - 0.68) / 0.22)
    bottom_center = _clamp01(1.0 - abs(rect.center_x - 0.50) / 0.34)
    ability_bar_penalty = bottom_hud * bottom_center * 0.30

    minimap_like = _clamp01((rect.center_y - 0.70) / 0.20) * _clamp01(
        (rect.center_x - 0.66) / 0.20
    )
    minimap_penalty = minimap_like * 0.18
    return max(ability_bar_penalty, minimap_penalty)


def _huge_central_penalty(rect: NormalizedRect, center_avoidance: float) -> float:
    huge_score = _clamp01((rect.area - 0.16) / 0.12)
    return huge_score * (1.0 - center_avoidance) * 0.38


def _tiny_penalty(area: float, size_score: float) -> float:
    if area >= 0.025:
        return 0.0
    return (1.0 - size_score) * 0.45


def _competition_penalty(
    cluster: _ScoredCluster,
    other_clusters: list[_ScoredCluster],
) -> float:
    penalty = 0.0
    for other in other_clusters:
        if other.raw_score < 0.55:
            continue
        if other.component_scores["prevalence"] < 0.45:
            continue
        penalty += 0.10
        if abs(cluster.raw_score - other.raw_score) <= 0.12:
            penalty += 0.08
    return min(0.35, penalty)


def _selection_reason(
    selected: _ScoredCluster,
    scored_clusters: tuple[_ScoredCluster, ...],
) -> str:
    components = selected.component_scores
    subject_name = "YOLO person" if selected.source == "yolo_person" else "face"
    reason = (
        f"selected stable edge/corner {subject_name} cluster expanded into a streamer crop "
        f"seen in {selected.frame_count} sampled frame(s); "
        f"prevalence={components['prevalence']:.3f}, "
        f"position_stability={components['position_stability']:.3f}, "
        f"size_stability={components['size_stability']:.3f}, "
        f"edge_proximity={components['edge_proximity']:.3f}"
    )
    competing = sum(
        1
        for cluster in scored_clusters
        if cluster.cluster_id != selected.cluster_id
        and cluster.raw_score >= 0.55
        and cluster.component_scores["prevalence"] >= 0.45
    )
    if competing:
        reason += f"; confidence reduced by {competing} competing stable cluster(s)"
    return reason


def _write_overlay_result(
    output_path: Path,
    *,
    clip_id: str,
    selected_face_rect: NormalizedRect | None,
    selected_overlay_rect: NormalizedRect | None,
    confidence: float,
    fallback: bool,
    reason: str,
    candidate_clusters: tuple[_ScoredCluster, ...],
    selected_source: str | None = None,
    subject_detector_error: str | None = None,
) -> Path:
    ensure_directory(output_path.parent)
    payload = {
        "clip_id": clip_id,
        "selected_rect": selected_overlay_rect.to_dict() if selected_overlay_rect else None,
        "selected_face_rect": selected_face_rect.to_dict() if selected_face_rect else None,
        "selected_overlay_rect": (
            selected_overlay_rect.to_dict() if selected_overlay_rect else None
        ),
        "confidence": _round(confidence),
        "fallback": fallback,
        "selected_source": selected_source or ("overlay_fallback" if fallback else "haar_face"),
        "reason": reason,
        "candidate_clusters": [cluster.to_dict() for cluster in candidate_clusters],
    }
    if subject_detector_error:
        payload["subject_detector_error"] = subject_detector_error
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _read_frames_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Frame metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AnalysisError(f"Frame metadata must be a JSON object: {path}")
    return payload


def _read_overlay_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Overlay metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AnalysisError(f"Overlay metadata must be a JSON object: {path}")
    return payload


def _frame_paths_from_metadata(
    payload: dict[str, object],
    *,
    base_path: Path,
) -> tuple[Path, ...]:
    frame_paths = payload.get("frame_paths")
    if not isinstance(frame_paths, list) or not frame_paths:
        raise AnalysisError("Frame metadata must include a non-empty frame_paths list.")

    paths: list[Path] = []
    for value in frame_paths:
        if not isinstance(value, str) or not value.strip():
            raise AnalysisError("Frame metadata contains an invalid frame path.")
        path = Path(value)
        paths.append(path if path.is_absolute() else base_path / path)
    return tuple(paths)


def _sampled_timestamps_from_metadata(
    payload: dict[str, object],
    *,
    expected_count: int,
) -> tuple[float | None, ...]:
    sampled_timestamps = payload.get("sampled_timestamps")
    if sampled_timestamps is None:
        return tuple(None for _index in range(expected_count))
    if not isinstance(sampled_timestamps, list) or len(sampled_timestamps) != expected_count:
        raise AnalysisError(
            "Frame metadata sampled_timestamps must match the frame_paths length."
        )

    timestamps: list[float | None] = []
    for value in sampled_timestamps:
        if not isinstance(value, (int, float)):
            raise AnalysisError("Frame metadata contains an invalid sampled timestamp.")
        timestamps.append(float(value))
    return tuple(timestamps)


def _debug_annotations(payload: dict[str, object]) -> tuple[OverlayDebugAnnotation, ...]:
    clusters = payload.get("candidate_clusters")
    if not isinstance(clusters, list):
        raise AnalysisError("Overlay metadata must include a candidate_clusters list.")

    selected_rect = _optional_rect_from_payload(payload.get("selected_rect"))
    fallback = bool(payload.get("fallback"))
    annotations: list[OverlayDebugAnnotation] = []
    for cluster in clusters[:MAX_REPORTED_FACE_CLUSTERS]:
        if not isinstance(cluster, dict):
            raise AnalysisError("Overlay metadata contains an invalid candidate cluster.")
        rect = _rect_from_payload(cluster.get("rect"), context="candidate cluster rect")
        cluster_id = _int_from_payload(cluster.get("cluster_id"), context="cluster_id")
        confidence = _float_from_payload(cluster.get("confidence"), context="cluster confidence")
        state = _debug_state(rect=rect, selected_rect=selected_rect, fallback=fallback)
        annotations.append(
            OverlayDebugAnnotation(
                cluster_id=cluster_id,
                rect=rect,
                confidence=confidence,
                state=state,
                detection_count=_optional_int_from_payload(cluster.get("detection_count")),
                prevalence=_optional_float_from_payload(cluster.get("prevalence")),
                average_face_confidence=_optional_float_from_payload(
                    cluster.get("average_face_confidence")
                ),
                heuristic_multiplier=_optional_float_from_payload(
                    cluster.get("heuristic_multiplier")
                ),
                final_score=_optional_float_from_payload(cluster.get("final_score")),
                source=str(cluster.get("source")) if cluster.get("source") else None,
                edge_corner_prior=_optional_float_from_payload(
                    cluster.get("edge_corner_prior")
                ),
                central_gameplay_penalty=_optional_float_from_payload(
                    cluster.get("central_gameplay_penalty")
                ),
                expanded_box_quality=_optional_float_from_payload(
                    cluster.get("expanded_box_quality")
                ),
            )
        )
    return tuple(annotations)


def _debug_banner(payload: dict[str, object]) -> str:
    confidence = _float_from_payload(payload.get("confidence"), context="overlay confidence")
    fallback = bool(payload.get("fallback"))
    state = "fallback" if fallback else "selected"
    selected_source = payload.get("selected_source")
    if isinstance(selected_source, str) and selected_source:
        return f"overlay {state} | source {selected_source} | confidence {confidence:.3f}"
    return f"overlay {state} | confidence {confidence:.3f}"


def _debug_state(
    *,
    rect: NormalizedRect,
    selected_rect: NormalizedRect | None,
    fallback: bool,
) -> str:
    if fallback:
        return "fallback candidate"
    if selected_rect is not None and _rects_close(rect, selected_rect):
        return "selected"
    return "candidate"


def _optional_rect_from_payload(value: object) -> NormalizedRect | None:
    if value is None:
        return None
    return _rect_from_payload(value, context="selected_rect")


def _rect_from_payload(value: object, *, context: str) -> NormalizedRect:
    if not isinstance(value, dict):
        raise AnalysisError(f"Overlay metadata contains an invalid {context}.")
    rect = NormalizedRect(
        x=_float_from_payload(value.get("x"), context=f"{context}.x"),
        y=_float_from_payload(value.get("y"), context=f"{context}.y"),
        width=_float_from_payload(value.get("width"), context=f"{context}.width"),
        height=_float_from_payload(value.get("height"), context=f"{context}.height"),
    )
    if not _is_valid_rect(rect):
        raise AnalysisError(f"Overlay metadata contains an out-of-bounds {context}.")
    return rect


def _float_from_payload(value: object, *, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise AnalysisError(f"Overlay metadata contains an invalid {context}.")
    return float(value)


def _optional_float_from_payload(value: object) -> float | None:
    if value is None or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int_from_payload(value: object, *, context: str) -> int:
    if not isinstance(value, int):
        raise AnalysisError(f"Overlay metadata contains an invalid {context}.")
    return value


def _optional_int_from_payload(value: object) -> int | None:
    if value is None or not isinstance(value, int):
        return None
    return value


def _require_existing_frames(frame_paths: tuple[Path, ...]) -> None:
    missing = tuple(path for path in frame_paths if not path.is_file())
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise AnalysisError(f"Sampled frame file(s) not found: {formatted}")


def _rect_pixels(
    rect: NormalizedRect,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = round(rect.x * width)
    top = round(rect.y * height)
    right = round((rect.x + rect.width) * width)
    bottom = round((rect.y + rect.height) * height)
    return (
        max(0, min(width - 1, left)),
        max(0, min(height - 1, top)),
        max(0, min(width - 1, right)),
        max(0, min(height - 1, bottom)),
    )


def _debug_color(state: str) -> tuple[int, int, int]:
    if state == "selected":
        return (40, 220, 40)
    if state == "fallback candidate":
        return (0, 180, 255)
    return (255, 180, 0)


def _rects_close(first: NormalizedRect, second: NormalizedRect) -> bool:
    return (
        abs(first.x - second.x) <= 0.0005
        and abs(first.y - second.y) <= 0.0005
        and abs(first.width - second.width) <= 0.0005
        and abs(first.height - second.height) <= 0.0005
    )


def _detection_filter_reason(rect: NormalizedRect) -> str | None:
    if rect.width <= 0:
        return "non_positive_width"
    if rect.height <= 0:
        return "non_positive_height"
    if rect.x < 0:
        return "x_before_frame"
    if rect.y < 0:
        return "y_before_frame"
    if rect.x + rect.width > 1.0:
        return "x_after_frame"
    if rect.y + rect.height > 1.0:
        return "y_after_frame"
    return None


def _is_valid_rect(rect: NormalizedRect) -> bool:
    return _detection_filter_reason(rect) is None


def _safe_clip_id(clip_id: str) -> str:
    if not clip_id.strip():
        raise AnalysisError("clip_id must not be empty.")
    return safe_filename(clip_id)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _opencv_face_score(level_weight: float) -> float:
    return _clamp01((level_weight + 3.0) / 7.0)


def _round(value: float) -> float:
    return round(value, 6)

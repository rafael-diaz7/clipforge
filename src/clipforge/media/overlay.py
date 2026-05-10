"""Infer likely streamer overlay regions from sampled analysis frames."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from clipforge.core.config import ANALYSIS_DIR
from clipforge.media.analyze import AnalysisError
from clipforge.utils.paths import clip_analysis_dir, ensure_directory, safe_filename


DEFAULT_OVERLAY_CONFIDENCE_THRESHOLD = 0.58
CLUSTER_CENTER_THRESHOLD = 0.12
CLUSTER_SIZE_THRESHOLD = 0.09
STREAMER_CROP_HORIZONTAL_PADDING_RATIO = 0.28
STREAMER_CROP_TOP_PADDING_RATIO = 0.32
STREAMER_CROP_BOTTOM_PADDING_RATIO = 0.75
STREAMER_CROP_MAX_WIDTH_FACE_RATIO = 1.65
STREAMER_CROP_MAX_HEIGHT_FACE_RATIO = 2.10
STREAMER_CROP_MAX_NORMALIZED_WIDTH = 0.36
STREAMER_CROP_FACE_CENTER_Y_RATIO = 0.42
SOURCE_ASPECT_RATIO = 16 / 9
TARGET_ASPECT_RATIO = 9 / 16
HYBRID_STREAMER_OUTPUT_HEIGHT_RATIO = 0.40
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

    @property
    def label(self) -> str:
        return f"cluster {self.cluster_id} | confidence {self.confidence:.3f} | {self.state}"


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
    rect: NormalizedRect
    score: float


@dataclass
class _Cluster:
    detections: list[_FrameDetection]

    def add(self, detection: _FrameDetection) -> None:
        self.detections.append(detection)


@dataclass(frozen=True)
class _ScoredCluster:
    cluster_id: int
    face_rect: NormalizedRect
    overlay_rect: NormalizedRect
    frame_count: int
    component_scores: dict[str, float]
    raw_score: float
    confidence: float

    @property
    def rect(self) -> NormalizedRect:
        return self.overlay_rect

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "rect": self.overlay_rect.to_dict(),
            "face_rect": self.face_rect.to_dict(),
            "overlay_rect": self.overlay_rect.to_dict(),
            "frame_count": self.frame_count,
            "component_scores": {
                name: _round(value) for name, value in self.component_scores.items()
            },
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

        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise OverlayDetectorUnavailable(f"OpenCV face cascade could not load: {cascade_path}")

    def detect(self, frame_path: Path) -> tuple[FaceDetection, ...]:
        image = self._cv2.imread(str(frame_path))
        if image is None:
            return ()

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return ()

        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        detections: list[FaceDetection] = []
        for x, y, face_width, face_height in faces:
            rect = NormalizedRect(
                x=float(x) / width,
                y=float(y) / height,
                width=float(face_width) / width,
                height=float(face_height) / height,
            )
            if _is_valid_rect(rect):
                detections.append(FaceDetection(rect=rect))
        return tuple(detections)


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


def analyze_overlay(
    *,
    clip_id: str,
    analysis_dir: Path = ANALYSIS_DIR,
    detector: FaceDetector | None = None,
    confidence_threshold: float = DEFAULT_OVERLAY_CONFIDENCE_THRESHOLD,
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
    _require_existing_frames(frame_paths)
    output_path = analysis_clip_dir / "overlay.json"

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
            candidate_clusters=(),
        )

    try:
        detections = _detect_faces(frame_paths, detector=detector_instance)
    except OverlayDetectorUnavailable as exc:
        return _write_overlay_result(
            output_path,
            clip_id=metadata_clip_id,
            selected_face_rect=None,
            selected_overlay_rect=None,
            confidence=0.0,
            fallback=True,
            reason=f"fallback: detector failed: {exc}",
            candidate_clusters=(),
        )

    clusters = _cluster_detections(detections)
    scored_clusters = _score_clusters(clusters, total_frames=len(frame_paths))
    if not scored_clusters:
        return _write_overlay_result(
            output_path,
            clip_id=metadata_clip_id,
            selected_face_rect=None,
            selected_overlay_rect=None,
            confidence=0.0,
            fallback=True,
            reason="fallback: no face detections found in sampled frames",
            candidate_clusters=(),
        )

    selected = scored_clusters[0]
    if selected.confidence < confidence_threshold:
        return _write_overlay_result(
            output_path,
            clip_id=metadata_clip_id,
            selected_face_rect=None,
            selected_overlay_rect=None,
            confidence=selected.confidence,
            fallback=True,
            reason=(
                "fallback: best candidate confidence "
                f"{selected.confidence:.3f} is below threshold {confidence_threshold:.3f}"
            ),
            candidate_clusters=scored_clusters,
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
    detector: FaceDetector,
) -> tuple[_FrameDetection, ...]:
    detections: list[_FrameDetection] = []
    for frame_index, frame_path in enumerate(frame_paths):
        try:
            frame_detections = detector.detect(frame_path)
        except Exception as exc:
            raise OverlayDetectorUnavailable(str(exc)) from exc

        for detection in frame_detections:
            if _is_valid_rect(detection.rect):
                detections.append(
                    _FrameDetection(
                        frame_index=frame_index,
                        rect=detection.rect,
                        score=_clamp01(detection.score),
                    )
                )
    return tuple(detections)


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
        adjusted.append(
            _ScoredCluster(
                cluster_id=cluster.cluster_id,
                face_rect=cluster.face_rect,
                overlay_rect=cluster.overlay_rect,
                frame_count=cluster.frame_count,
                component_scores={**cluster.component_scores, "competition_penalty": penalty},
                raw_score=cluster.raw_score,
                confidence=_clamp01(cluster.raw_score - penalty),
            )
        )
    adjusted.sort(key=lambda cluster: cluster.confidence, reverse=True)
    return tuple(adjusted)


def _score_cluster(
    cluster: _Cluster,
    *,
    cluster_id: int,
    total_frames: int,
) -> _ScoredCluster:
    face_rect = _cluster_rect(cluster)
    overlay_rect = _expand_face_rect_to_streamer_crop_rect(face_rect)
    frame_count = len({detection.frame_index for detection in cluster.detections})
    prevalence = frame_count / total_frames if total_frames else 0.0

    position_stability = _position_stability(cluster)
    size_stability = _size_stability(cluster)
    edge_proximity = _edge_proximity(overlay_rect)
    center_avoidance = _center_avoidance(overlay_rect)
    size_score = _overlay_size_score(overlay_rect.area)
    detection_score = statistics.fmean(detection.score for detection in cluster.detections)

    brief_penalty = max(0.0, 0.45 - prevalence) * 0.75
    central_penalty = (1.0 - center_avoidance) * 0.22
    huge_central_penalty = _huge_central_penalty(overlay_rect, center_avoidance)
    tiny_penalty = max(
        _tiny_penalty(overlay_rect.area, size_score),
        _tiny_face_penalty(face_rect.area),
    )

    raw_score = _clamp01(
        prevalence * 0.30
        + position_stability * 0.16
        + size_stability * 0.12
        + edge_proximity * 0.18
        + center_avoidance * 0.12
        + size_score * 0.10
        + detection_score * 0.02
        - brief_penalty
        - central_penalty
        - huge_central_penalty
        - tiny_penalty
    )

    component_scores = {
        "prevalence": prevalence,
        "position_stability": position_stability,
        "size_stability": size_stability,
        "edge_proximity": edge_proximity,
        "center_avoidance": center_avoidance,
        "overlay_size": size_score,
        "detector_confidence": detection_score,
        "brief_penalty": brief_penalty,
        "central_penalty": central_penalty,
        "huge_central_penalty": huge_central_penalty,
        "tiny_penalty": tiny_penalty,
    }
    return _ScoredCluster(
        cluster_id=cluster_id,
        face_rect=face_rect,
        overlay_rect=overlay_rect,
        frame_count=frame_count,
        component_scores=component_scores,
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
    x = face_rect.center_x - width / 2
    y = face_rect.center_y - height * STREAMER_CROP_FACE_CENTER_Y_RATIO

    x = max(face_rect.x + face_rect.width - width, min(x, face_rect.x))
    y = max(face_rect.y + face_rect.height - height, min(y, face_rect.y))
    return _fit_rect_to_frame(x=x, y=y, width=width, height=height)


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


def _huge_central_penalty(rect: NormalizedRect, center_avoidance: float) -> float:
    huge_score = _clamp01((rect.area - 0.16) / 0.12)
    return huge_score * (1.0 - center_avoidance) * 0.38


def _tiny_penalty(area: float, size_score: float) -> float:
    if area >= 0.025:
        return 0.0
    return (1.0 - size_score) * 0.45


def _tiny_face_penalty(area: float) -> float:
    if area >= 0.008:
        return 0.0
    return _clamp01((0.008 - area) / 0.005) * 0.45


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
    reason = (
        "selected stable edge/corner face cluster expanded into a streamer crop "
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
        "reason": reason,
        "candidate_clusters": [cluster.to_dict() for cluster in candidate_clusters],
    }
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


def _debug_annotations(payload: dict[str, object]) -> tuple[OverlayDebugAnnotation, ...]:
    clusters = payload.get("candidate_clusters")
    if not isinstance(clusters, list):
        raise AnalysisError("Overlay metadata must include a candidate_clusters list.")

    selected_rect = _optional_rect_from_payload(payload.get("selected_rect"))
    fallback = bool(payload.get("fallback"))
    annotations: list[OverlayDebugAnnotation] = []
    for cluster in clusters:
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
            )
        )
    return tuple(annotations)


def _debug_banner(payload: dict[str, object]) -> str:
    confidence = _float_from_payload(payload.get("confidence"), context="overlay confidence")
    fallback = bool(payload.get("fallback"))
    state = "fallback" if fallback else "selected"
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


def _int_from_payload(value: object, *, context: str) -> int:
    if not isinstance(value, int):
        raise AnalysisError(f"Overlay metadata contains an invalid {context}.")
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


def _is_valid_rect(rect: NormalizedRect) -> bool:
    return (
        rect.width > 0
        and rect.height > 0
        and rect.x >= 0
        and rect.y >= 0
        and rect.x + rect.width <= 1.0
        and rect.y + rect.height <= 1.0
    )


def _safe_clip_id(clip_id: str) -> str:
    if not clip_id.strip():
        raise AnalysisError("clip_id must not be empty.")
    return safe_filename(clip_id)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float) -> float:
    return round(value, 6)

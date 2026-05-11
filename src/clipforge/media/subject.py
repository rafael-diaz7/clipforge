"""Pluggable subject detectors for sampled analysis frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from clipforge.media.overlay import NormalizedRect, OverlayDetectorUnavailable


DEFAULT_SUBJECT_DETECTOR = "yolo"
DEFAULT_YOLO_MODEL = "yolo11n.pt"
DEFAULT_YOLO_DEVICE = "auto"
DEFAULT_YOLO_CONFIDENCE_THRESHOLD = 0.30
YOLO_PERSON_CLASS_ID = 0
YOLO_PERSON_CLASS_LABEL = "person"


@dataclass(frozen=True)
class SubjectDetection:
    """A normalized detector output for one sampled frame."""

    rect: NormalizedRect
    confidence: float
    class_label: str
    detector_name: str
    frame_index: int
    timestamp: float | None


@dataclass(frozen=True)
class SubjectDetectorDebugInfo:
    """Inspectable settings for a subject detector run."""

    name: str
    settings: dict[str, object]


class SubjectDetector(Protocol):
    """Detector interface for sampled-frame subject analysis."""

    def detect(
        self,
        frame_path: Path,
        *,
        frame_index: int,
        timestamp: float | None,
    ) -> tuple[SubjectDetection, ...]:
        """Return normalized subject detections for one sampled frame."""


class YOLOPersonDetector:
    """Ultralytics YOLO detector filtered to person detections only."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_YOLO_MODEL,
        device: str = DEFAULT_YOLO_DEVICE,
        confidence_threshold: float = DEFAULT_YOLO_CONFIDENCE_THRESHOLD,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise OverlayDetectorUnavailable(
                "Ultralytics is not installed; install ultralytics to use YOLO person detection."
            ) from exc

        self.model_name = model
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self._model = YOLO(model)

    def detect(
        self,
        frame_path: Path,
        *,
        frame_index: int,
        timestamp: float | None,
    ) -> tuple[SubjectDetection, ...]:
        device = None if self.device == "auto" else self.device
        results = self._model.predict(
            source=str(frame_path),
            classes=[YOLO_PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            device=device,
            verbose=False,
        )
        detections: list[SubjectDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            image_shape = getattr(result, "orig_shape", None)
            if not image_shape or len(image_shape) < 2:
                continue
            image_height = float(image_shape[0])
            image_width = float(image_shape[1])
            if image_width <= 0 or image_height <= 0:
                continue

            for box in boxes:
                class_id = int(_scalar(getattr(box, "cls", None), default=-1))
                if class_id != YOLO_PERSON_CLASS_ID:
                    continue
                confidence = float(_scalar(getattr(box, "conf", None), default=0.0))
                if confidence < self.confidence_threshold:
                    continue
                coordinates = _xyxy(box)
                if coordinates is None:
                    continue
                left, top, right, bottom = coordinates
                rect = NormalizedRect(
                    x=left / image_width,
                    y=top / image_height,
                    width=(right - left) / image_width,
                    height=(bottom - top) / image_height,
                )
                if rect.width <= 0 or rect.height <= 0:
                    continue
                detections.append(
                    SubjectDetection(
                        rect=_clamp_rect(rect),
                        confidence=max(0.0, min(1.0, confidence)),
                        class_label=YOLO_PERSON_CLASS_LABEL,
                        detector_name=self.debug_info.name,
                        frame_index=frame_index,
                        timestamp=timestamp,
                    )
                )
        return tuple(detections)

    @property
    def debug_info(self) -> SubjectDetectorDebugInfo:
        return SubjectDetectorDebugInfo(
            name="yolo_person",
            settings={
                "model": self.model_name,
                "device": self.device,
                "confidence_threshold": self.confidence_threshold,
                "classes": [YOLO_PERSON_CLASS_LABEL],
                "backend": "ultralytics",
            },
        )


def _xyxy(box: object) -> tuple[float, float, float, float] | None:
    xyxy = getattr(box, "xyxy", None)
    if xyxy is None:
        return None
    values = _as_flat_list(xyxy)
    if len(values) < 4:
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _scalar(value: object, *, default: float) -> float:
    values = _as_flat_list(value)
    if not values:
        return default
    return float(values[0])


def _as_flat_list(value: object) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_as_flat_list(item))
        return flattened
    if isinstance(value, tuple):
        flattened = []
        for item in value:
            flattened.extend(_as_flat_list(item))
        return flattened
    return []


def _clamp_rect(rect: NormalizedRect) -> NormalizedRect:
    left = max(0.0, min(1.0, rect.x))
    top = max(0.0, min(1.0, rect.y))
    right = max(left, min(1.0, rect.x + rect.width))
    bottom = max(top, min(1.0, rect.y + rect.height))
    return NormalizedRect(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
    )

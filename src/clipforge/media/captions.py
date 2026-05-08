"""Caption metadata schema and JSON helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from clipforge.core.config import ClipforgeConfig
from clipforge.utils import ensure_directory, safe_filename
from clipforge.utils.json_validation import required_int
from clipforge.utils.json_validation import required_list
from clipforge.utils.json_validation import required_number
from clipforge.utils.json_validation import required_string


CAPTION_METADATA_TYPE = "clipforge.caption_metadata"
CAPTION_METADATA_VERSION = 1


class CaptionError(RuntimeError):
    """Raised when caption metadata is missing, malformed, or invalid."""


class CaptionTranscriptionError(CaptionError):
    """Raised when caption generation cannot complete."""


class CaptionTranscriber(Protocol):
    """Adapter that transcribes one source clip into caption metadata."""

    def transcribe(self, source_path: Path, *, clip_id: str) -> "CaptionMetadata":
        """Return caption metadata for one local source clip."""


@dataclass(frozen=True)
class CaptionSegment:
    """One timed caption segment."""

    start_time: float
    end_time: float
    text: str

    def __post_init__(self) -> None:
        start_time = _coerce_time(self.start_time, field_name="start_time")
        end_time = _coerce_time(self.end_time, field_name="end_time")
        text = self.text.strip()

        if end_time <= start_time:
            raise CaptionError("caption segment end_time must be greater than start_time.")
        if not text:
            raise CaptionError("caption segment text must be a non-empty string.")

        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class CaptionMetadata:
    """Validated caption metadata for one clip."""

    clip_id: str
    segments: tuple[CaptionSegment, ...]

    def __post_init__(self) -> None:
        clip_id = self.clip_id.strip()
        if not clip_id:
            raise CaptionError("caption metadata clip_id must be a non-empty string.")

        segments = tuple(self.segments)
        sorted_segments = tuple(
            sorted(
                segments,
                key=lambda segment: (
                    segment.start_time,
                    segment.end_time,
                    segment.text,
                ),
            )
        )

        object.__setattr__(self, "clip_id", clip_id)
        object.__setattr__(self, "segments", sorted_segments)


def caption_metadata_path(clip_id: str, *, config: ClipforgeConfig) -> Path:
    """Return the deterministic metadata path for one clip's captions."""

    safe_clip_id = safe_filename(clip_id)
    return config.metadata_dir / "captions" / f"{safe_clip_id}.json"


def save_caption_metadata(
    metadata: CaptionMetadata,
    *,
    config: ClipforgeConfig,
    output_path: Path | None = None,
) -> Path:
    """Write caption metadata as readable JSON and return the output path."""

    path = output_path or caption_metadata_path(metadata.clip_id, config=config)
    ensure_directory(path.parent)
    # TODO: Store the caption metadata path in SQLite state once captions become
    # a first-class processing stage in the queued workflow.
    path.write_text(
        json.dumps(_caption_metadata_payload(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def save_captions(
    *,
    clip_id: str,
    segments: Sequence[CaptionSegment],
    config: ClipforgeConfig,
    output_path: Path | None = None,
) -> Path:
    """Build and save caption metadata for one clip."""

    return save_caption_metadata(
        CaptionMetadata(clip_id=clip_id, segments=tuple(segments)),
        config=config,
        output_path=output_path,
    )


def generate_caption_metadata(
    source_path: Path,
    *,
    clip_id: str,
    config: ClipforgeConfig,
    transcriber: CaptionTranscriber | None = None,
    output_path: Path | None = None,
) -> Path:
    """Transcribe one local source clip and save caption metadata."""

    from clipforge.integrations.openai import OpenAITranscriptionClient

    client = transcriber or OpenAITranscriptionClient.from_config(config)
    metadata = client.transcribe(source_path, clip_id=clip_id)
    return save_caption_metadata(metadata, config=config, output_path=output_path)


def load_caption_metadata(path: Path) -> CaptionMetadata:
    """Load caption metadata JSON from disk."""

    try:
        raw_metadata = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CaptionError(f"Caption metadata file not found: {path}") from exc
    except OSError as exc:
        raise CaptionError(f"Could not read caption metadata file {path}: {exc}") from exc

    try:
        payload = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise CaptionError(f"Invalid JSON in caption metadata file {path}: {exc.msg}") from exc

    try:
        return parse_caption_metadata(payload)
    except CaptionError as exc:
        raise CaptionError(f"Invalid caption metadata file {path}: {exc}") from exc


def parse_caption_metadata(payload: Any) -> CaptionMetadata:
    """Validate a decoded caption metadata object."""

    if not isinstance(payload, dict):
        raise CaptionError("caption metadata root must be an object.")

    metadata_type = required_string(
        payload,
        "type",
        context="caption metadata",
        error_cls=CaptionError,
    )
    if metadata_type != CAPTION_METADATA_TYPE:
        raise CaptionError(
            "caption metadata type must be "
            f"{CAPTION_METADATA_TYPE!r}, got {metadata_type!r}."
        )

    version = required_int(
        payload,
        "version",
        context="caption metadata",
        error_cls=CaptionError,
    )
    if version != CAPTION_METADATA_VERSION:
        raise CaptionError(f"unsupported caption metadata version: {version}.")

    clip_id = required_string(
        payload,
        "clip_id",
        context="caption metadata",
        error_cls=CaptionError,
    )
    segments_payload = required_list(
        payload,
        "segments",
        context="caption metadata",
        error_cls=CaptionError,
    )
    segments = tuple(
        _parse_caption_segment(segment_payload, index=index)
        for index, segment_payload in enumerate(segments_payload)
    )
    return CaptionMetadata(clip_id=clip_id, segments=segments)


def _caption_metadata_payload(metadata: CaptionMetadata) -> dict[str, Any]:
    return {
        "type": CAPTION_METADATA_TYPE,
        "version": CAPTION_METADATA_VERSION,
        "clip_id": metadata.clip_id,
        "segments": [
            {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "text": segment.text,
            }
            for segment in metadata.segments
        ],
    }


def _parse_caption_segment(payload: Any, *, index: int) -> CaptionSegment:
    context = f"caption metadata.segments[{index}]"
    if not isinstance(payload, dict):
        raise CaptionError(f"{context} must be an object.")

    return CaptionSegment(
        start_time=required_number(
            payload,
            "start_time",
            context=context,
            error_cls=CaptionError,
        ),
        end_time=required_number(
            payload,
            "end_time",
            context=context,
            error_cls=CaptionError,
        ),
        text=required_string(
            payload,
            "text",
            context=context,
            error_cls=CaptionError,
        ),
    )


def _coerce_time(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptionError(f"caption segment {field_name} must be a number.")

    coerced = float(value)
    if not math.isfinite(coerced):
        raise CaptionError(f"caption segment {field_name} must be a finite number.")
    if coerced < 0:
        raise CaptionError(f"caption segment {field_name} must be greater than or equal to 0.")
    return coerced

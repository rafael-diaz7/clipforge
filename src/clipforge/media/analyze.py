"""Extract lightweight media samples for later analysis passes."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from clipforge.core.config import DATA_DIR
from clipforge.utils import ensure_directory, safe_filename


DEFAULT_FRAME_SAMPLE_COUNT = 12
DEFAULT_FRAME_SAMPLE_INTERVAL_SECONDS = 2.0
ANALYSIS_DIR = DATA_DIR / "analysis"


class AnalysisError(RuntimeError):
    """Raised when media analysis samples cannot be created."""


@dataclass(frozen=True)
class FrameSampleMetadata:
    """Metadata describing saved analysis frame samples."""

    clip_id: str
    source_path: str
    sampled_timestamps: tuple[float, ...]
    frame_paths: tuple[str, ...]
    sampling_mode: dict[str, float | int | str]

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "source_path": self.source_path,
            "sampled_timestamps": list(self.sampled_timestamps),
            "frame_paths": list(self.frame_paths),
            "sampling_mode": self.sampling_mode,
        }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def sample_timestamps(
    *,
    count: int = DEFAULT_FRAME_SAMPLE_COUNT,
    interval_seconds: float | None = None,
) -> tuple[float, ...]:
    """Return deterministic frame timestamps in seconds."""

    _validate_positive_int(count, "count")
    interval = _sample_interval(interval_seconds)
    return tuple(round(index * interval, 3) for index in range(count))


def build_frame_sample_commands(
    source_path: Path,
    frame_paths: tuple[Path, ...],
    sampled_timestamps: tuple[float, ...],
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> tuple[list[str], ...]:
    """Build safe FFmpeg argv lists for extracting still frames."""

    if len(frame_paths) != len(sampled_timestamps):
        raise AnalysisError("Frame paths and sampled timestamps must have equal length.")

    return tuple(
        [
            ffmpeg_binary,
            "-y",
            "-ss",
            _format_timestamp(timestamp),
            "-i",
            str(source_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        for frame_path, timestamp in zip(frame_paths, sampled_timestamps, strict=True)
    )


def sample_frames(
    source_path: Path,
    *,
    clip_id: str,
    count: int = DEFAULT_FRAME_SAMPLE_COUNT,
    interval_seconds: float | None = None,
    analysis_dir: Path = ANALYSIS_DIR,
    ffmpeg_binary: str = "ffmpeg",
    runner: Runner = subprocess.run,
) -> Path:
    """Save representative frames and return the metadata JSON path."""

    if not source_path.is_file():
        raise AnalysisError(f"Source video not found: {source_path}")

    safe_clip_id = _safe_clip_id(clip_id)
    sampled_timestamps = sample_timestamps(
        count=count,
        interval_seconds=interval_seconds,
    )
    clip_analysis_dir = analysis_dir / safe_clip_id
    frames_dir = ensure_directory(clip_analysis_dir / "frames")
    frame_paths = _frame_paths(frames_dir, len(sampled_timestamps))
    commands = build_frame_sample_commands(
        source_path,
        frame_paths,
        sampled_timestamps,
        ffmpeg_binary=ffmpeg_binary,
    )

    try:
        for command in commands:
            _run_ffmpeg_command(command, runner=runner)
        _require_frame_outputs(frame_paths)
    except AnalysisError as exc:
        raise AnalysisError(f"Could not sample frames from {source_path}: {exc}") from exc
    metadata = FrameSampleMetadata(
        clip_id=clip_id,
        source_path=str(source_path),
        sampled_timestamps=sampled_timestamps,
        frame_paths=tuple(str(path) for path in frame_paths),
        sampling_mode=_sampling_mode(
            count=count,
            interval_seconds=interval_seconds,
        ),
    )
    metadata_path = clip_analysis_dir / "frames.json"
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )
    return metadata_path


def _sampling_mode(
    *,
    count: int,
    interval_seconds: float | None,
) -> dict[str, float | int | str]:
    interval = _sample_interval(interval_seconds)
    mode_type = "interval_seconds" if interval_seconds is not None else "default_interval"
    return {
        "type": mode_type,
        "count": count,
        "interval_seconds": interval,
    }


def _sample_interval(interval_seconds: float | None) -> float:
    if interval_seconds is None:
        return DEFAULT_FRAME_SAMPLE_INTERVAL_SECONDS
    if interval_seconds <= 0:
        raise AnalysisError("interval_seconds must be greater than 0.")
    return float(interval_seconds)


def _frame_paths(frames_dir: Path, count: int) -> tuple[Path, ...]:
    return tuple(frames_dir / f"frame_{index:04d}.jpg" for index in range(1, count + 1))


def _run_ffmpeg_command(command: list[str], *, runner: Runner) -> None:
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        binary = command[0] if command else "ffmpeg"
        raise AnalysisError(
            f"{binary} was not found. Install FFmpeg and make sure it is available in PATH."
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        detail = f": {stderr}" if stderr else "."
        raise AnalysisError(f"FFmpeg failed with exit code {completed.returncode}{detail}")


def _require_frame_outputs(frame_paths: tuple[Path, ...]) -> None:
    missing_paths = tuple(path for path in frame_paths if not path.is_file())
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise AnalysisError(f"FFmpeg completed but did not create frame output(s): {missing}")


def _safe_clip_id(clip_id: str) -> str:
    if not clip_id.strip():
        raise AnalysisError("clip_id must not be empty.")
    return safe_filename(clip_id)


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise AnalysisError(f"{name} must be greater than 0.")


def _format_timestamp(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"

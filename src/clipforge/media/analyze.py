"""Extract lightweight media samples for later analysis passes."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from clipforge.core.config import ANALYSIS_DIR
from clipforge.utils.paths import clip_analysis_dir, ensure_directory, safe_filename


DEFAULT_FRAME_SAMPLE_COUNT = 12
DEFAULT_FRAME_SAMPLE_INTERVAL_SECONDS = 2.0
FRAME_SAMPLE_EXTENSION = ".jpg"
FRAME_SAMPLE_EOF_MARGIN_SECONDS = 0.05


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
DurationProbe = Callable[[Path], float]


def sample_timestamps(
    *,
    count: int | None = None,
    interval_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> tuple[float, ...]:
    """Return deterministic frame timestamps in seconds."""

    if count is None:
        count = _default_frame_sample_count()
    _validate_positive_int(count, "count")
    interval = _sample_interval(interval_seconds)
    if duration_seconds is not None:
        _validate_positive_float(duration_seconds, "duration_seconds")
        max_timestamp = max(duration_seconds - FRAME_SAMPLE_EOF_MARGIN_SECONDS, 0.0)
        default_last_timestamp = (count - 1) * interval
        if count > 1 and max_timestamp < default_last_timestamp:
            step = max_timestamp / (count - 1)
            return tuple(round(index * step, 3) for index in range(count))
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
    count: int | None = None,
    interval_seconds: float | None = None,
    analysis_dir: Path = ANALYSIS_DIR,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    duration_seconds: float | None = None,
    duration_probe: DurationProbe | None = None,
    runner: Runner = subprocess.run,
    probe_runner: Runner = subprocess.run,
) -> Path:
    """Save representative frames and return the metadata JSON path."""

    if not source_path.is_file():
        raise AnalysisError(f"Source video not found: {source_path}")

    safe_clip_id = _safe_clip_id(clip_id)
    source_duration_seconds = duration_seconds
    if source_duration_seconds is None:
        probe = duration_probe or (
            lambda path: _probe_media_duration_seconds(
                path,
                ffprobe_binary=ffprobe_binary,
                runner=probe_runner,
            )
        )
        source_duration_seconds = probe(source_path)
    sampled_timestamps = sample_timestamps(
        count=count,
        interval_seconds=interval_seconds,
        duration_seconds=source_duration_seconds,
    )
    analysis_clip_dir = clip_analysis_dir(analysis_dir, safe_clip_id)
    frames_dir = ensure_directory(analysis_clip_dir / "frames")
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
    metadata_path = analysis_clip_dir / "frames.json"
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )
    return metadata_path


def _sampling_mode(
    *,
    count: int | None,
    interval_seconds: float | None,
) -> dict[str, float | int | str]:
    sample_count = count if count is not None else _default_frame_sample_count()
    interval = _sample_interval(interval_seconds)
    mode_type = "interval_seconds" if interval_seconds is not None else "default_interval"
    return {
        "type": mode_type,
        "count": sample_count,
        "interval_seconds": interval,
    }


def _default_frame_sample_count() -> int:
    value = os.getenv("CLIPFORGE_SUBJECT_SAMPLE_COUNT")
    if value is None or not value.strip():
        return DEFAULT_FRAME_SAMPLE_COUNT
    try:
        count = int(value)
    except ValueError as exc:
        raise AnalysisError("CLIPFORGE_SUBJECT_SAMPLE_COUNT must be an integer.") from exc
    _validate_positive_int(count, "CLIPFORGE_SUBJECT_SAMPLE_COUNT")
    return count


def _sample_interval(interval_seconds: float | None) -> float:
    if interval_seconds is None:
        return DEFAULT_FRAME_SAMPLE_INTERVAL_SECONDS
    if interval_seconds <= 0:
        raise AnalysisError("interval_seconds must be greater than 0.")
    return float(interval_seconds)


def _frame_paths(frames_dir: Path, count: int) -> tuple[Path, ...]:
    return tuple(
        frames_dir / f"frame_{index:04d}{FRAME_SAMPLE_EXTENSION}"
        for index in range(1, count + 1)
    )


def _probe_media_duration_seconds(
    source_path: Path,
    *,
    ffprobe_binary: str,
    runner: Runner,
) -> float:
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]
    try:
        completed = runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(
            f"Could not determine source clip duration with ffprobe for {source_path}: "
            f"{_process_error_excerpt(exc)}"
        ) from exc

    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise AnalysisError(
            f"ffprobe returned an invalid duration for {source_path}: "
            f"{completed.stdout.strip()!r}."
        ) from exc
    _validate_positive_float(duration, "ffprobe duration")
    return duration


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


def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0:
        raise AnalysisError(f"{name} must be greater than 0.")


def _format_timestamp(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _process_error_excerpt(
    exc: OSError | subprocess.CalledProcessError,
    *,
    limit: int = 320,
) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        output = (exc.stderr or exc.stdout or "").strip().replace("\n", " ")
        if not output:
            output = f"exit code {exc.returncode}"
    else:
        output = str(exc)
    if len(output) > limit:
        return f"{output[:limit]}..."
    return output

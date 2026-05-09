from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from clipforge.media.analyze import (
    AnalysisError,
    build_frame_sample_commands,
    sample_frames,
    sample_timestamps,
)


def test_sample_timestamps_defaults_to_twelve_short_clip_samples() -> None:
    assert sample_timestamps() == (
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
    )


def test_sample_timestamps_uses_custom_count_and_interval() -> None:
    assert sample_timestamps(count=4, interval_seconds=1.5) == (0, 1.5, 3, 4.5)


def test_build_frame_sample_commands_returns_safe_ffmpeg_argv_lists(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    frame_paths = (tmp_path / "frame_0001.jpg", tmp_path / "frame_0002.jpg")

    commands = build_frame_sample_commands(
        source_path,
        frame_paths,
        (0, 2.5),
        ffmpeg_binary="ffmpeg-test",
    )

    assert commands == (
        [
            "ffmpeg-test",
            "-y",
            "-ss",
            "0",
            "-i",
            str(source_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_paths[0]),
        ],
        [
            "ffmpeg-test",
            "-y",
            "-ss",
            "2.5",
            "-i",
            str(source_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_paths[1]),
        ],
    )
    assert all(isinstance(command, list) for command in commands)
    assert all(isinstance(part, str) for command in commands for part in command)


def test_build_frame_sample_commands_requires_matching_paths_and_timestamps(
    tmp_path: Path,
) -> None:
    with pytest.raises(AnalysisError, match="equal length"):
        build_frame_sample_commands(
            tmp_path / "source.mp4",
            (tmp_path / "frame_0001.jpg",),
            (0, 2),
        )


def test_sample_frames_writes_frames_metadata(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    metadata_path = sample_frames(
        source_path,
        clip_id="clip-123",
        count=3,
        interval_seconds=1.25,
        analysis_dir=tmp_path / "analysis",
        runner=fake_runner,
    )

    assert metadata_path == tmp_path / "analysis" / "clip-123" / "frames.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload == {
        "clip_id": "clip-123",
        "source_path": str(source_path),
        "sampled_timestamps": [0, 1.25, 2.5],
        "frame_paths": [
            str(tmp_path / "analysis" / "clip-123" / "frames" / "frame_0001.jpg"),
            str(tmp_path / "analysis" / "clip-123" / "frames" / "frame_0002.jpg"),
            str(tmp_path / "analysis" / "clip-123" / "frames" / "frame_0003.jpg"),
        ],
        "sampling_mode": {
            "type": "interval_seconds",
            "count": 3,
            "interval_seconds": 1.25,
        },
    }
    assert len(calls) == 3
    assert all(Path(path).is_file() for path in payload["frame_paths"])


def test_sample_frames_records_default_sampling_mode(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0)

    metadata_path = sample_frames(
        source_path,
        clip_id="clip-123",
        count=1,
        analysis_dir=tmp_path / "analysis",
        runner=fake_runner,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["sampling_mode"] == {
        "type": "default_interval",
        "count": 1,
        "interval_seconds": 2.0,
    }


def test_sample_frames_uses_safe_clip_directory(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0)

    metadata_path = sample_frames(
        source_path,
        clip_id=" My Clip!? ",
        count=1,
        analysis_dir=tmp_path / "analysis",
        runner=fake_runner,
    )

    assert metadata_path == tmp_path / "analysis" / "My_Clip" / "frames.json"


def test_sample_frames_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="Source video not found"):
        sample_frames(
            tmp_path / "missing.mp4",
            clip_id="clip-123",
            analysis_dir=tmp_path / "analysis",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"clip_id": ""}, "clip_id"),
        ({"clip_id": "clip-123", "count": 0}, "count"),
        ({"clip_id": "clip-123", "interval_seconds": 0}, "interval_seconds"),
    ),
)
def test_sample_frames_rejects_invalid_inputs(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")

    with pytest.raises(AnalysisError, match=message):
        sample_frames(
            source_path,
            analysis_dir=tmp_path / "analysis",
            **kwargs,
        )


def test_sample_frames_wraps_missing_ffmpeg_binary(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")

    def fake_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    with pytest.raises(AnalysisError, match="not found"):
        sample_frames(
            source_path,
            clip_id="clip-123",
            count=1,
            analysis_dir=tmp_path / "analysis",
            runner=fake_runner,
        )


def test_sample_frames_reports_ffmpeg_failure_and_skips_metadata(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad input")

    with pytest.raises(AnalysisError, match="bad input"):
        sample_frames(
            source_path,
            clip_id="clip-123",
            count=1,
            analysis_dir=tmp_path / "analysis",
            runner=fake_runner,
        )

    assert not (tmp_path / "analysis" / "clip-123" / "frames.json").exists()


def test_sample_frames_reports_missing_frame_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(AnalysisError, match="did not create"):
        sample_frames(
            source_path,
            clip_id="clip-123",
            count=1,
            analysis_dir=tmp_path / "analysis",
            runner=fake_runner,
        )

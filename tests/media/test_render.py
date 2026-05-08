from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.media.layouts import Layout, LayoutRegion, NormalizedRect, OutputSize
from clipforge.media.render import (
    RenderError,
    build_ffmpeg_command,
    build_filter_complex,
    rect_to_pixels,
    render_layout,
    run_ffmpeg_command,
)


def _layout(*regions: LayoutRegion) -> Layout:
    return Layout(
        name="test_layout",
        description="Test layout.",
        output=OutputSize(width=1080, height=1920),
        regions=regions,
    )


def _region(
    name: str = "gameplay",
    source_region: NormalizedRect | None = None,
    output_region: NormalizedRect | None = None,
) -> LayoutRegion:
    return LayoutRegion(
        name=name,
        source_region=source_region
        or NormalizedRect(x=0.21875, y=0.0, width=0.5625, height=1.0),
        output_region=output_region
        or NormalizedRect(x=0.0, y=0.0, width=1.0, height=1.0),
    )


def test_rect_to_pixels_converts_normalized_output_region() -> None:
    rect = NormalizedRect(x=0.0, y=0.36, width=1.0, height=0.64)

    pixels = rect_to_pixels(rect, OutputSize(width=1080, height=1920))

    assert pixels.x == 0
    assert pixels.y == 691
    assert pixels.width == 1080
    assert pixels.height == 1229


def test_build_filter_complex_builds_single_region_graph() -> None:
    filter_complex = build_filter_complex(_layout(_region()))

    assert "color=c=black:s=1080x1920:r=30[base]" in filter_complex
    assert "[0:v]split=1[src0]" in filter_complex
    assert "crop=iw*0.5625:ih*1:iw*0.21875:ih*0" in filter_complex
    assert "scale=1080:1920:force_original_aspect_ratio=increase" in filter_complex
    assert "[base][region0]overlay=0:0:format=auto:shortest=1[out]" in filter_complex


def test_build_filter_complex_overlays_regions_in_layout_order() -> None:
    layout = _layout(
        _region(
            name="gameplay",
            output_region=NormalizedRect(x=0.0, y=0.36, width=1.0, height=0.64),
        ),
        _region(
            name="facecam",
            source_region=NormalizedRect(x=0.0, y=0.0, width=0.375, height=0.375),
            output_region=NormalizedRect(x=0.0, y=0.0, width=1.0, height=0.36),
        ),
    )

    filter_complex = build_filter_complex(layout)

    assert "[0:v]split=2[src0][src1]" in filter_complex
    assert (
        "[base][region0]overlay=0:691:format=auto:shortest=1[composed0]"
        in filter_complex
    )
    assert (
        "[composed0][region1]overlay=0:0:format=auto:shortest=1[out]"
        in filter_complex
    )


def test_build_ffmpeg_command_returns_argument_list(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "render.mp4"

    command = build_ffmpeg_command(source, output, _layout(_region()))

    assert command[0] == "ffmpeg"
    assert "-filter_complex" in command
    assert "-map" in command
    assert "[out]" in command
    assert "0:a?" in command
    assert "-shortest" in command
    assert "-s" in command
    assert "1080x1920" in command
    assert command[-1] == str(output)
    assert all(isinstance(part, str) for part in command)


def test_run_ffmpeg_command_raises_clear_error_when_binary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr("clipforge.media.render.subprocess.run", fake_run)

    with pytest.raises(RenderError, match="not found"):
        run_ffmpeg_command(["ffmpeg", "-version"])


def test_run_ffmpeg_command_raises_clear_error_for_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 1
        stderr = "bad filter"

    def fake_run(*args: object, **kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr("clipforge.media.render.subprocess.run", fake_run)

    with pytest.raises(RenderError, match="bad filter"):
        run_ffmpeg_command(["ffmpeg", "-i", "source.mp4"])


def test_render_layout_runs_command_and_returns_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")

    def fake_run(command: list[str]) -> None:
        calls.append(command)

    monkeypatch.setattr("clipforge.media.render.run_ffmpeg_command", fake_run)

    output_path = tmp_path / "render.mp4"

    assert render_layout(source_path, output_path, _layout(_region())) == output_path
    assert calls


def test_render_layout_reports_missing_source_path(tmp_path: Path) -> None:
    with pytest.raises(RenderError, match="Source video not found"):
        render_layout(tmp_path / "missing.mp4", tmp_path / "render.mp4", _layout(_region()))


def test_render_layout_adds_context_to_ffmpeg_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "render.mp4"
    source_path.write_bytes(b"video")

    def fake_run(command: list[str]) -> None:
        raise RenderError("bad filter")

    monkeypatch.setattr("clipforge.media.render.run_ffmpeg_command", fake_run)

    with pytest.raises(RenderError) as exc_info:
        render_layout(source_path, output_path, _layout(_region()))

    message = str(exc_info.value)
    assert "test_layout" in message
    assert str(source_path) in message
    assert str(output_path) in message
    assert "bad filter" in message

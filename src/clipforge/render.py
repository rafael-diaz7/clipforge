"""Build and run FFmpeg commands for layout-based vertical renders."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from clipforge.layouts import Layout, LayoutRegion, NormalizedRect, OutputSize


class RenderError(RuntimeError):
    """Raised when an FFmpeg command cannot be built or executed."""


@dataclass(frozen=True)
class PixelRect:
    """A rectangle described in output pixels."""

    x: int
    y: int
    width: int
    height: int


def build_ffmpeg_command(
    source_path: Path,
    output_path: Path,
    layout: Layout,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build an FFmpeg argv list that renders one layout to an MP4."""

    output_size = layout.output
    filter_complex = build_filter_complex(layout)

    return [
        ffmpeg_binary,
        "-y",
        "-i",
        str(source_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
        "-s",
        f"{output_size.width}x{output_size.height}",
        "-f",
        "mp4",
        str(output_path),
    ]


def build_filter_complex(layout: Layout) -> str:
    """Build the FFmpeg filter graph for a layout."""

    if not layout.regions:
        raise RenderError(f"Layout {layout.name!r} must contain at least one region.")

    output_size = layout.output
    filter_parts = [
        f"color=c=black:s={output_size.width}x{output_size.height}:r=30[base]",
        f"[0:v]split={len(layout.regions)}{_split_labels(len(layout.regions))}",
    ]

    overlay_input = "[base]"
    for index, region in enumerate(layout.regions):
        region_label = f"region{index}"
        composed_label = "out" if index == len(layout.regions) - 1 else f"composed{index}"
        output_rect = rect_to_pixels(region.output_region, output_size)

        filter_parts.append(
            f"[src{index}]{_region_filter(region, output_rect)}[{region_label}]"
        )
        filter_parts.append(
            f"{overlay_input}[{region_label}]overlay="
            f"{output_rect.x}:{output_rect.y}:format=auto:shortest=1[{composed_label}]"
        )
        overlay_input = f"[{composed_label}]"

    return ";".join(filter_parts)


def rect_to_pixels(rect: NormalizedRect, output_size: OutputSize) -> PixelRect:
    """Convert a normalized output rectangle to integer pixel coordinates."""

    x = round(rect.x * output_size.width)
    y = round(rect.y * output_size.height)
    width = round(rect.width * output_size.width)
    height = round(rect.height * output_size.height)

    if x + width > output_size.width:
        width = output_size.width - x
    if y + height > output_size.height:
        height = output_size.height - y

    if width <= 0 or height <= 0:
        raise RenderError("Output region resolved to an empty pixel rectangle.")

    return PixelRect(x=x, y=y, width=width, height=height)


def run_ffmpeg_command(command: list[str]) -> None:
    """Run an FFmpeg argv list and raise clear errors for common failures."""

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        binary = command[0] if command else "ffmpeg"
        raise RenderError(
            f"{binary} was not found. Install FFmpeg and make sure it is available in PATH."
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        detail = f": {stderr}" if stderr else "."
        raise RenderError(f"FFmpeg failed with exit code {completed.returncode}{detail}")


def render_layout(
    source_path: Path,
    output_path: Path,
    layout: Layout,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Render one layout and return the output path."""

    command = build_ffmpeg_command(
        source_path,
        output_path,
        layout,
        ffmpeg_binary=ffmpeg_binary,
    )
    run_ffmpeg_command(command)
    return output_path


def _split_labels(count: int) -> str:
    return "".join(f"[src{index}]" for index in range(count))


def _region_filter(region: LayoutRegion, output_rect: PixelRect) -> str:
    source = region.source_region
    return (
        f"crop=iw*{_fmt(source.width)}:ih*{_fmt(source.height)}:"
        f"iw*{_fmt(source.x)}:ih*{_fmt(source.y)},"
        f"scale={output_rect.width}:{output_rect.height}:"
        "force_original_aspect_ratio=increase,"
        f"crop={output_rect.width}:{output_rect.height},"
        "setsar=1"
    )


def _fmt(value: float) -> str:
    return f"{value:.10g}"

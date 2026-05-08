"""Build and run FFmpeg commands for layout-based vertical renders."""

from __future__ import annotations

import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

from clipforge.media.captions import CaptionMetadata, CaptionSegment
from clipforge.media.layouts import Layout, LayoutRegion, NormalizedRect, OutputSize


class RenderError(RuntimeError):
    """Raised when an FFmpeg command cannot be built or executed."""


@dataclass(frozen=True)
class PixelRect:
    """A rectangle described in output pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class CaptionStyle:
    """Caption overlay style for vertical mobile renders."""

    font_color: str = "white"
    font_size: int = 56
    font_file: Path | None = None
    box_color: str = "black@0.68"
    box_border_width: int = 20
    line_spacing: int = 8
    safe_margin_x: int = 96
    safe_margin_bottom: int = 220
    max_chars_per_line: int | None = None
    max_lines: int = 2
    min_display_seconds: float = 0.75
    max_display_seconds: float = 3.2
    seconds_per_word: float = 0.36
    display_padding_seconds: float = 0.45


@dataclass(frozen=True)
class CaptionCue:
    """One render-ready caption cue."""

    start_time: float
    end_time: float
    lines: tuple[str, ...]


DEFAULT_CAPTION_STYLE = CaptionStyle()


def build_ffmpeg_command(
    source_path: Path,
    output_path: Path,
    layout: Layout,
    *,
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = DEFAULT_CAPTION_STYLE,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build an FFmpeg argv list that renders one layout to an MP4."""

    output_size = layout.output
    filter_complex = build_filter_complex(
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
    )

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


def build_filter_complex(
    layout: Layout,
    *,
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = DEFAULT_CAPTION_STYLE,
) -> str:
    """Build the FFmpeg filter graph for a layout."""

    if not layout.regions:
        raise RenderError(f"Layout {layout.name!r} must contain at least one region.")

    output_size = layout.output
    caption_segments = caption_metadata.segments if caption_metadata is not None else ()
    filter_parts = [
        f"color=c=black:s={output_size.width}x{output_size.height}:r=30[base]",
        f"[0:v]split={len(layout.regions)}{_split_labels(len(layout.regions))}",
    ]

    overlay_input = "[base]"
    for index, region in enumerate(layout.regions):
        region_label = f"region{index}"
        if index == len(layout.regions) - 1:
            composed_label = "captionbase" if caption_segments else "out"
        else:
            composed_label = f"composed{index}"
        output_rect = rect_to_pixels(region.output_region, output_size)

        filter_parts.append(
            f"[src{index}]{_region_filter(region, output_rect)}[{region_label}]"
        )
        filter_parts.append(
            f"{overlay_input}[{region_label}]overlay="
            f"{output_rect.x}:{output_rect.y}:format=auto:shortest=1[{composed_label}]"
        )
        overlay_input = f"[{composed_label}]"

    if caption_segments:
        filter_parts.extend(
            _caption_filters(
                caption_segments,
                caption_style=caption_style,
                output_size=output_size,
            )
        )

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
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = DEFAULT_CAPTION_STYLE,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Render one layout and return the output path."""

    if not source_path.is_file():
        raise RenderError(f"Source video not found: {source_path}")

    command = build_ffmpeg_command(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        ffmpeg_binary=ffmpeg_binary,
    )
    try:
        run_ffmpeg_command(command)
    except RenderError as exc:
        raise RenderError(
            f"Could not render layout {layout.name!r} from {source_path} "
            f"to {output_path}: {exc}"
        ) from exc

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


def _caption_filters(
    segments: tuple[CaptionSegment, ...],
    *,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> list[str]:
    filters = []
    input_label = "captionbase"
    cues = _caption_cues(segments, caption_style=caption_style, output_size=output_size)
    for index, cue in enumerate(cues):
        for line_index, line in enumerate(cue.lines):
            is_last_filter = index == len(cues) - 1 and line_index == len(cue.lines) - 1
            output_label = (
                "out"
                if is_last_filter
                else f"caption{index}_{line_index}"
            )
            filters.append(
                f"[{input_label}]"
                f"drawtext=text={_escape_drawtext_text(line)}:"
                f"x=max({caption_style.safe_margin_x}\\,(w-text_w)/2):"
                f"y={_caption_line_y(cue.lines, line_index, caption_style)}:"
                f"{_caption_font_option(caption_style)}"
                f"fontcolor={caption_style.font_color}:"
                f"fontsize={caption_style.font_size}:"
                "box=1:"
                f"boxcolor={caption_style.box_color}:"
                f"boxborderw={caption_style.box_border_width}:"
                f"enable='between(t\\,{_fmt(cue.start_time)}\\,{_fmt(cue.end_time)})'"
                f"[{output_label}]"
            )
            input_label = output_label
    return filters


def _caption_cues(
    segments: tuple[CaptionSegment, ...],
    *,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> tuple[CaptionCue, ...]:
    cues: list[CaptionCue] = []
    for segment in segments:
        chunks = _caption_chunks(segment.text, caption_style, output_size)
        if not chunks:
            continue

        durations = [_caption_chunk_duration(chunk, caption_style) for chunk in chunks]
        available_duration = segment.end_time - segment.start_time
        requested_duration = sum(durations)
        if requested_duration > available_duration:
            scale = available_duration / requested_duration
            durations = [duration * scale for duration in durations]

        cursor = segment.start_time
        for chunk, duration in zip(chunks, durations, strict=True):
            end_time = min(segment.end_time, cursor + duration)
            if end_time > cursor:
                cues.append(
                    CaptionCue(
                        start_time=cursor,
                        end_time=end_time,
                        lines=tuple(
                            _wrapped_caption_lines(chunk, caption_style, output_size)
                        ),
                    )
                )
            cursor = end_time
    return tuple(cues)


def _caption_chunks(
    text: str,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> tuple[str, ...]:
    words = text.split()
    if not words:
        return ()

    chunks: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate_words = [*current_words, word]
        candidate = " ".join(candidate_words)
        if current_words and len(_wrapped_caption_lines(candidate, caption_style, output_size)) > caption_style.max_lines:
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate_words

    if current_words:
        chunks.append(" ".join(current_words))
    return tuple(chunks)


def _wrapped_caption_lines(
    text: str,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> list[str]:
    return textwrap.wrap(
        " ".join(text.split()),
        width=_caption_chars_per_line(caption_style, output_size),
        break_long_words=True,
        break_on_hyphens=False,
    )


def _caption_line_y(
    lines: tuple[str, ...],
    line_index: int,
    caption_style: CaptionStyle,
) -> str:
    block_height = (
        len(lines) * caption_style.font_size
        + (len(lines) - 1) * caption_style.line_spacing
    )
    y = (
        f"h-{block_height}-{caption_style.safe_margin_bottom}"
        if line_index == 0
        else f"h-{block_height}-{caption_style.safe_margin_bottom}+"
        f"{line_index * (caption_style.font_size + caption_style.line_spacing)}"
    )
    return f"max({caption_style.safe_margin_x}\\,{y})"


def _caption_chars_per_line(caption_style: CaptionStyle, output_size: OutputSize) -> int:
    if caption_style.max_chars_per_line is not None:
        return caption_style.max_chars_per_line

    text_width = output_size.width - (
        2 * (caption_style.safe_margin_x + caption_style.box_border_width)
    )
    average_character_width = caption_style.font_size * 0.58
    return max(8, int(text_width / average_character_width))


def _caption_chunk_duration(
    text: str,
    caption_style: CaptionStyle,
) -> float:
    word_count = len(text.split())
    readable_duration = (
        word_count * caption_style.seconds_per_word
        + caption_style.display_padding_seconds
    )
    return min(
        caption_style.max_display_seconds,
        max(caption_style.min_display_seconds, readable_duration),
    )


def _caption_font_option(caption_style: CaptionStyle) -> str:
    if caption_style.font_file is None:
        return ""
    font_file = str(caption_style.font_file).replace("\\", "/")
    return f"fontfile='{_escape_drawtext_option(font_file)}':"


def _escape_drawtext_option(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _escape_drawtext_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(" ", "\\ ")
        .replace("'", "\\\\\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )

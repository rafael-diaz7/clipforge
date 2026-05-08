"""Build and run FFmpeg commands for layout-based vertical renders."""

from __future__ import annotations

import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from clipforge.media.captions import CaptionMetadata, CaptionSegment
from clipforge.media.layouts import Layout, LayoutRegion, NormalizedRect, OutputSize


class RenderError(RuntimeError):
    """Raised when an FFmpeg command cannot be built or executed."""


DEFAULT_CAPTION_RENDERER_BACKEND = "drawtext"
CAPTION_RENDERER_ASS = "ass"
CAPTION_RENDERER_DRAWTEXT = "drawtext"
SUPPORTED_CAPTION_RENDERER_BACKENDS = frozenset(
    {CAPTION_RENDERER_ASS, CAPTION_RENDERER_DRAWTEXT}
)


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

    font_family: str | None = None
    font_color: str = "white"
    font_size: int = 56
    font_file: Path | None = None
    font_fallbacks: tuple[str, ...] = ("Arial",)
    box_color: str = "black@0.68"
    box_border_width: int = 20
    line_spacing: int = 8
    outline_width: int = 4
    shadow_offset: int = 2
    safe_margin_x: int = 96
    safe_margin_bottom: int = 220
    max_chars_per_line: int | None = None
    max_lines: int = 2
    min_display_seconds: float = 0.75
    max_display_seconds: float = 3.2
    seconds_per_word: float = 0.36
    display_padding_seconds: float = 0.45
    uppercase: bool = False
    ass_style_name: str = "Default"


@dataclass(frozen=True)
class CaptionCue:
    """One render-ready caption cue."""

    start_time: float
    end_time: float
    lines: tuple[str, ...]


DEFAULT_CAPTION_STYLE = CaptionStyle()


class CaptionRenderer(Protocol):
    """Renderer backend that turns prepared cues into FFmpeg filter parts."""

    def render_filter_parts(
        self,
        cues: tuple[CaptionCue, ...],
        *,
        caption_style: CaptionStyle,
        output_size: OutputSize,
        input_label: str,
        output_label: str,
    ) -> tuple[str, ...]:
        """Return FFmpeg filter graph parts that burn the supplied cues."""


@dataclass(frozen=True)
class DrawtextCaptionRenderer:
    """FFmpeg drawtext caption renderer."""

    def render_filter_parts(
        self,
        cues: tuple[CaptionCue, ...],
        *,
        caption_style: CaptionStyle,
        output_size: OutputSize,
        input_label: str,
        output_label: str,
    ) -> tuple[str, ...]:
        filters = []
        current_input_label = input_label
        for index, cue in enumerate(cues):
            for line_index, line in enumerate(cue.lines):
                is_last_filter = index == len(cues) - 1 and line_index == len(cue.lines) - 1
                current_output_label = (
                    output_label
                    if is_last_filter
                    else f"caption{index}_{line_index}"
                )
                filters.append(
                    f"[{current_input_label}]"
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
                    f"[{current_output_label}]"
                )
                current_input_label = current_output_label
        return tuple(filters)


@dataclass(frozen=True)
class AssCaptionRenderer:
    """FFmpeg libass caption renderer backed by a temporary ASS subtitle file."""

    subtitle_path: Path

    def render_filter_parts(
        self,
        cues: tuple[CaptionCue, ...],
        *,
        caption_style: CaptionStyle,
        output_size: OutputSize,
        input_label: str,
        output_label: str,
    ) -> tuple[str, ...]:
        self.subtitle_path.parent.mkdir(parents=True, exist_ok=True)
        self.subtitle_path.write_text(
            generate_ass_subtitle(
                cues,
                caption_style=caption_style,
                output_size=output_size,
            ),
            encoding="utf-8",
        )
        return (
            f"[{input_label}]"
            f"ass=filename='{_escape_ffmpeg_filter_option(_ffmpeg_path(self.subtitle_path))}'"
            f"{_ass_fontsdir_option(caption_style)}"
            f"[{output_label}]",
        )


def build_ffmpeg_command(
    source_path: Path,
    output_path: Path,
    layout: Layout,
    *,
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = DEFAULT_CAPTION_STYLE,
    caption_renderer_backend: str = DEFAULT_CAPTION_RENDERER_BACKEND,
    ass_temp_dir: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build an FFmpeg argv list that renders one layout to an MP4."""

    output_size = layout.output
    ass_subtitle_path = _ass_subtitle_path(
        output_path,
        ass_temp_dir=ass_temp_dir,
        caption_metadata=caption_metadata,
        caption_renderer_backend=caption_renderer_backend,
    )
    filter_complex = build_filter_complex(
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        caption_renderer_backend=caption_renderer_backend,
        ass_subtitle_path=ass_subtitle_path,
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
    caption_renderer_backend: str = DEFAULT_CAPTION_RENDERER_BACKEND,
    ass_subtitle_path: Path | None = None,
) -> str:
    """Build the FFmpeg filter graph for a layout."""

    if not layout.regions:
        raise RenderError(f"Layout {layout.name!r} must contain at least one region.")

    _require_caption_renderer_backend(caption_renderer_backend)

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
                caption_renderer_backend=caption_renderer_backend,
                ass_subtitle_path=ass_subtitle_path,
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
    caption_renderer_backend: str = DEFAULT_CAPTION_RENDERER_BACKEND,
    ass_temp_dir: Path | None = None,
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
        caption_renderer_backend=caption_renderer_backend,
        ass_temp_dir=ass_temp_dir,
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
    caption_renderer_backend: str,
    ass_subtitle_path: Path | None,
) -> tuple[str, ...]:
    cues = _caption_cues(segments, caption_style=caption_style, output_size=output_size)
    renderer = _caption_renderer(
        caption_renderer_backend,
        ass_subtitle_path=ass_subtitle_path,
    )
    return renderer.render_filter_parts(
        cues,
        caption_style=caption_style,
        output_size=output_size,
        input_label="captionbase",
        output_label="out",
    )


def _caption_cues(
    segments: tuple[CaptionSegment, ...],
    *,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> tuple[CaptionCue, ...]:
    cues: list[CaptionCue] = []
    for segment in segments:
        segment_text = segment.text.upper() if caption_style.uppercase else segment.text
        chunks = _caption_chunks(segment_text, caption_style, output_size)
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


def _require_caption_renderer_backend(caption_renderer_backend: str) -> None:
    if caption_renderer_backend not in SUPPORTED_CAPTION_RENDERER_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_CAPTION_RENDERER_BACKENDS))
        raise RenderError(
            "Invalid caption renderer backend: "
            f"{caption_renderer_backend!r}. Supported values: {supported}."
        )


def _caption_renderer(
    caption_renderer_backend: str,
    *,
    ass_subtitle_path: Path | None,
) -> CaptionRenderer:
    _require_caption_renderer_backend(caption_renderer_backend)
    if caption_renderer_backend == CAPTION_RENDERER_DRAWTEXT:
        return DrawtextCaptionRenderer()
    if ass_subtitle_path is None:
        raise RenderError("ASS caption rendering requires an ASS subtitle path.")
    return AssCaptionRenderer(ass_subtitle_path)


def _ass_subtitle_path(
    output_path: Path,
    *,
    ass_temp_dir: Path | None,
    caption_metadata: CaptionMetadata | None,
    caption_renderer_backend: str,
) -> Path | None:
    if caption_renderer_backend != CAPTION_RENDERER_ASS:
        return None
    if caption_metadata is None or not caption_metadata.segments:
        return None

    subtitle_dir = ass_temp_dir or output_path.parent
    return subtitle_dir / f"{output_path.stem}.ass"


def generate_ass_subtitle(
    cues: tuple[CaptionCue, ...],
    *,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> str:
    """Build an Advanced SubStation Alpha subtitle file for render-ready cues."""

    parts = [
        _ass_script_info(output_size),
        _ass_style_block(caption_style),
        _ass_events_block(cues, caption_style=caption_style, output_size=output_size),
    ]
    return "\n\n".join(parts) + "\n"


def _ass_script_info(output_size: OutputSize) -> str:
    return "\n".join(
        (
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {output_size.width}",
            f"PlayResY: {output_size.height}",
        )
    )


def _ass_style_block(caption_style: CaptionStyle) -> str:
    return "\n".join(
        (
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding"
            ),
            (
                "Style: "
                f"{_ass_field(caption_style.ass_style_name)},"
                f"{_ass_field(_ass_font_name(caption_style))},"
                f"{caption_style.font_size},"
                f"{_ass_color(caption_style.font_color, default='&H00FFFFFF')},"
                f"{_ass_color(caption_style.font_color, default='&H00FFFFFF')},"
                "&H00000000,"
                "&H80000000,"
                "1,0,0,0,"
                "100,100,0,0,"
                "1,"
                f"{caption_style.outline_width},"
                f"{caption_style.shadow_offset},"
                "2,"
                f"{caption_style.safe_margin_x},"
                f"{caption_style.safe_margin_x},"
                f"{caption_style.safe_margin_bottom},"
                "1"
            ),
        )
    )


def _ass_events_block(
    cues: tuple[CaptionCue, ...],
    *,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> str:
    lines = [
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        lines.extend(_ass_dialogue_lines(cue, caption_style, output_size))
    return "\n".join(lines)


def _ass_dialogue_lines(
    cue: CaptionCue,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> tuple[str, ...]:
    dialogue_lines = []
    line_count = len(cue.lines)
    for line_index, line in enumerate(cue.lines):
        y = _ass_caption_line_y(line_count, line_index, caption_style, output_size)
        text = line.upper() if caption_style.uppercase else line
        dialogue_lines.append(
            "Dialogue: "
            f"0,{_format_ass_time(cue.start_time)},{_format_ass_time(cue.end_time)},"
            f"{_ass_field(caption_style.ass_style_name)},,"
            f"{caption_style.safe_margin_x},{caption_style.safe_margin_x},"
            f"{caption_style.safe_margin_bottom},,"
            f"{{\\an2\\pos({output_size.width // 2},{y})}}{_escape_ass_text(text)}"
        )
    return tuple(dialogue_lines)


def _ass_caption_line_y(
    line_count: int,
    line_index: int,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> int:
    line_step = caption_style.font_size + caption_style.line_spacing
    bottom_y = output_size.height - caption_style.safe_margin_bottom
    return bottom_y - (line_count - line_index - 1) * line_step


def _format_ass_time(seconds: float) -> str:
    total_centiseconds = max(0, int(round(seconds * 100)))
    centiseconds = total_centiseconds % 100
    total_seconds = total_centiseconds // 100
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours}:{minutes:02d}:{seconds_part:02d}.{centiseconds:02d}"


def _ass_font_name(caption_style: CaptionStyle) -> str:
    if caption_style.font_family is not None and caption_style.font_family.strip():
        return caption_style.font_family.strip()
    if caption_style.font_file is not None:
        return caption_style.font_file.stem
    if caption_style.font_fallbacks:
        first_fallback = caption_style.font_fallbacks[0].strip()
        if first_fallback:
            return first_fallback
    return "Arial"


def _ass_field(value: str) -> str:
    return value.replace(",", " ").strip()


def _ass_color(value: str, *, default: str) -> str:
    normalized = value.strip().lower()
    named_colors = {
        "white": "&H00FFFFFF",
        "black": "&H00000000",
        "yellow": "&H0000FFFF",
        "red": "&H000000FF",
        "green": "&H00008000",
        "blue": "&H00FF0000",
    }
    if normalized in named_colors:
        return named_colors[normalized]
    if normalized.startswith("#") and len(normalized) == 7:
        red = normalized[1:3]
        green = normalized[3:5]
        blue = normalized[5:7]
        return f"&H00{blue}{green}{red}".upper()
    return default


def _escape_ass_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\N")
    )


def _ass_fontsdir_option(caption_style: CaptionStyle) -> str:
    if caption_style.font_file is None:
        return ""
    fonts_dir = _ffmpeg_path(caption_style.font_file.parent)
    return f":fontsdir='{_escape_ffmpeg_filter_option(fonts_dir)}'"


def _ffmpeg_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _escape_ffmpeg_filter_option(value: str) -> str:
    return _escape_drawtext_option(value)


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

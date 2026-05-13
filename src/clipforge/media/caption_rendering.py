"""Caption cue timing and FFmpeg subtitle renderer helpers."""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from clipforge.media.caption_escaping import (
    escape_ass_text,
    escape_drawtext_option,
    escape_drawtext_text,
)
from clipforge.media.captions import CaptionSegment, CaptionWord
from clipforge.media.layouts import OutputSize
from clipforge.utils.paths import ensure_directory


class CaptionRenderingError(RuntimeError):
    """Raised when caption filters or subtitles cannot be built."""


DEFAULT_CAPTION_RENDERER_BACKEND = "drawtext"
CAPTION_RENDERER_ASS = "ass"
CAPTION_RENDERER_DRAWTEXT = "drawtext"
SUPPORTED_CAPTION_RENDERER_BACKENDS = frozenset(
    {CAPTION_RENDERER_ASS, CAPTION_RENDERER_DRAWTEXT}
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptionVerticalSafeArea:
    """Vertical caption bounds in output pixels."""

    top: int = 0
    bottom: int = 220
    center: bool = False

    def __post_init__(self) -> None:
        if self.top < 0 or self.bottom < 0:
            raise CaptionRenderingError(
                "Caption vertical safe area values must be non-negative."
            )

    def to_dict(self) -> dict[str, int]:
        payload = {
            "top": self.top,
            "bottom": self.bottom,
        }
        if self.center:
            payload["center"] = self.center
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaptionVerticalSafeArea":
        return cls(
            top=int(payload.get("top", 0)),
            bottom=int(payload.get("bottom", 220)),
            center=bool(payload.get("center", False)),
        )


class CaptionAnimationPreset(str, Enum):
    """Named caption animation hooks for future renderer support."""

    NONE = "none"
    SCALE_POP = "scale_pop"
    ACTIVE_WORD = "active_word"
    KARAOKE = "karaoke"


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
    outline_thickness: int | None = None
    shadow_offset: int = 2
    shadow_strength: int | None = None
    safe_margin_x: int = 96
    safe_margin_bottom: int = 220
    vertical_safe_area: CaptionVerticalSafeArea | None = None
    max_chars_per_line: int | None = None
    max_lines: int = 2
    min_display_seconds: float = 0.75
    max_display_seconds: float = 3.2
    seconds_per_word: float = 0.36
    seconds_per_character: float = 0.025
    punctuation_pause_seconds: float = 0.14
    min_cue_seconds: float | None = None
    max_hold_seconds: float | None = None
    display_padding_seconds: float = 0.45
    uppercase: bool = False
    highlight_color: str = "yellow"
    active_word_color: str = "yellow"
    ass_active_word_activation_delay_seconds: float = 0.04
    ass_active_word_min_display_seconds: float = 0.14
    ass_active_word_gap_tolerance_seconds: float = 0.12
    animation_preset: CaptionAnimationPreset = CaptionAnimationPreset.NONE
    ass_style_name: str = "Default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "font_family": self.font_family,
            "font_color": self.font_color,
            "font_size": self.font_size,
            "font_file": str(self.font_file) if self.font_file is not None else None,
            "font_fallbacks": list(self.font_fallbacks),
            "box_color": self.box_color,
            "box_border_width": self.box_border_width,
            "line_spacing": self.line_spacing,
            "outline_width": self.outline_width,
            "outline_thickness": self.outline_thickness,
            "shadow_offset": self.shadow_offset,
            "shadow_strength": self.shadow_strength,
            "safe_margin_x": self.safe_margin_x,
            "safe_margin_bottom": self.safe_margin_bottom,
            "vertical_safe_area": (
                self.vertical_safe_area.to_dict()
                if self.vertical_safe_area is not None
                else None
            ),
            "max_chars_per_line": self.max_chars_per_line,
            "max_lines": self.max_lines,
            "min_display_seconds": self.min_display_seconds,
            "max_display_seconds": self.max_display_seconds,
            "seconds_per_word": self.seconds_per_word,
            "seconds_per_character": self.seconds_per_character,
            "punctuation_pause_seconds": self.punctuation_pause_seconds,
            "min_cue_seconds": self.min_cue_seconds,
            "max_hold_seconds": self.max_hold_seconds,
            "display_padding_seconds": self.display_padding_seconds,
            "uppercase": self.uppercase,
            "highlight_color": self.highlight_color,
            "active_word_color": self.active_word_color,
            "ass_active_word_activation_delay_seconds": (
                self.ass_active_word_activation_delay_seconds
            ),
            "ass_active_word_min_display_seconds": (
                self.ass_active_word_min_display_seconds
            ),
            "ass_active_word_gap_tolerance_seconds": (
                self.ass_active_word_gap_tolerance_seconds
            ),
            "animation_preset": self.animation_preset.value,
            "ass_style_name": self.ass_style_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaptionStyle":
        values = dict(payload)
        if values.get("font_file") is not None:
            values["font_file"] = Path(str(values["font_file"]))
        if "font_fallbacks" in values:
            values["font_fallbacks"] = tuple(values["font_fallbacks"])
        if "vertical_safe_area" in values and values["vertical_safe_area"] is not None:
            values["vertical_safe_area"] = CaptionVerticalSafeArea.from_dict(
                values["vertical_safe_area"]
            )
        if "animation_preset" in values:
            values["animation_preset"] = CaptionAnimationPreset(values["animation_preset"])
        supported_keys = cls().to_dict().keys()
        return cls(**{key: value for key, value in values.items() if key in supported_keys})


@dataclass(frozen=True)
class CaptionCue:
    """One render-ready caption cue."""

    start_time: float
    end_time: float
    lines: tuple[str, ...]
    words: tuple[CaptionWord, ...] = ()


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
        del output_size
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
                    f"drawtext=text={escape_drawtext_text(line)}:"
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
        ensure_directory(self.subtitle_path.parent)
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


def caption_filter_parts(
    segments: tuple[CaptionSegment, ...],
    *,
    caption_style: CaptionStyle,
    output_size: OutputSize,
    caption_renderer_backend: str,
    ass_subtitle_path: Path | None,
    input_label: str,
    output_label: str,
    logger: logging.Logger | None = None,
) -> tuple[str, ...]:
    """Build caption-specific FFmpeg filter graph parts."""

    cues = _caption_cues(segments, caption_style=caption_style, output_size=output_size)
    log = logger or LOGGER
    log.info(
        "Rendering captions with %s backend using %s.",
        caption_renderer_backend,
        _caption_font_log_value(caption_style),
    )
    renderer = _caption_renderer(
        caption_renderer_backend,
        ass_subtitle_path=ass_subtitle_path,
    )
    return renderer.render_filter_parts(
        cues,
        caption_style=caption_style,
        output_size=output_size,
        input_label=input_label,
        output_label=output_label,
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

        available_duration = segment.end_time - segment.start_time
        durations = _caption_chunk_durations(
            chunks,
            available_duration=available_duration,
            caption_style=caption_style,
        )
        fills_segment = sum(durations) >= available_duration

        cursor = segment.start_time
        for index, (chunk, duration) in enumerate(zip(chunks, durations, strict=True)):
            is_last_chunk = index == len(chunks) - 1
            end_time = (
                segment.end_time
                if fills_segment and is_last_chunk
                else min(segment.end_time, cursor + duration)
            )
            if end_time > cursor:
                cues.append(
                    CaptionCue(
                        start_time=cursor,
                        end_time=end_time,
                        lines=tuple(
                            _wrapped_caption_lines(chunk, caption_style, output_size)
                        ),
                        words=_caption_words_for_cue(segment.words, cursor, end_time),
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
        if (
            current_words
            and len(_wrapped_caption_lines(candidate, caption_style, output_size))
            > caption_style.max_lines
        ):
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
    safe_margin_bottom = _caption_safe_margin_bottom(caption_style)
    block_height = _caption_block_height(len(lines), caption_style)
    line_offset = line_index * (caption_style.font_size + caption_style.line_spacing)
    if _caption_should_center_vertically(caption_style):
        safe_margin_top = _caption_safe_margin_top(caption_style)
        y = (
            f"{safe_margin_top}+(h-{safe_margin_top}-{safe_margin_bottom}-"
            f"{block_height})/2"
        )
        if line_offset:
            y += f"+{line_offset}"
        return f"max({safe_margin_top}\\,{y})"

    y = (
        f"h-{block_height}-{safe_margin_bottom}"
        if line_index == 0
        else f"h-{block_height}-{safe_margin_bottom}+"
        f"{line_offset}"
    )
    min_y = (
        _caption_safe_margin_top(caption_style)
        if caption_style.vertical_safe_area is not None
        else caption_style.safe_margin_x
    )
    return f"max({min_y}\\,{y})"


def _caption_words_for_cue(
    words: tuple[CaptionWord, ...],
    start_time: float,
    end_time: float,
) -> tuple[CaptionWord, ...]:
    return tuple(
        word
        for word in words
        if word.start_time < end_time and word.end_time > start_time
    )


def _caption_chars_per_line(caption_style: CaptionStyle, output_size: OutputSize) -> int:
    if caption_style.max_chars_per_line is not None:
        return caption_style.max_chars_per_line

    text_width = output_size.width - (
        2 * (caption_style.safe_margin_x + caption_style.box_border_width)
    )
    average_character_width = caption_style.font_size * 0.58
    return max(8, int(text_width / average_character_width))


def _caption_chunk_durations(
    chunks: tuple[str, ...],
    *,
    available_duration: float,
    caption_style: CaptionStyle,
) -> tuple[float, ...]:
    requested_durations = tuple(
        _caption_chunk_duration(chunk, caption_style) for chunk in chunks
    )
    requested_duration = sum(requested_durations)
    if requested_duration <= available_duration:
        return requested_durations

    cue_count = len(chunks)
    min_duration = _min_caption_cue_seconds(caption_style)
    if available_duration >= cue_count * min_duration:
        flexible_weights = tuple(
            max(0.0, duration - min_duration) for duration in requested_durations
        )
        flexible_duration = available_duration - cue_count * min_duration
        flexible_total = sum(flexible_weights)
        if flexible_total == 0:
            extra_duration = flexible_duration / cue_count
            return tuple(min_duration + extra_duration for _chunk in chunks)
        return tuple(
            min_duration + flexible_duration * weight / flexible_total
            for weight in flexible_weights
        )

    if requested_duration == 0:
        return tuple(available_duration / cue_count for _chunk in chunks)
    return tuple(
        available_duration * duration / requested_duration
        for duration in requested_durations
    )


def _caption_chunk_duration(
    text: str,
    caption_style: CaptionStyle,
) -> float:
    word_count = len(text.split())
    character_count = len("".join(text.split()))
    punctuation_count = sum(1 for character in text if character in ",.;:!?")
    readable_duration = (
        word_count * caption_style.seconds_per_word
        + character_count * caption_style.seconds_per_character
        + punctuation_count * caption_style.punctuation_pause_seconds
        + caption_style.display_padding_seconds
    )
    return min(
        _max_caption_hold_seconds(caption_style),
        max(_min_caption_cue_seconds(caption_style), readable_duration),
    )


def _min_caption_cue_seconds(caption_style: CaptionStyle) -> float:
    if caption_style.min_cue_seconds is not None:
        return caption_style.min_cue_seconds
    return caption_style.min_display_seconds


def _max_caption_hold_seconds(caption_style: CaptionStyle) -> float:
    if caption_style.max_hold_seconds is not None:
        return caption_style.max_hold_seconds
    return caption_style.max_display_seconds


def require_caption_renderer_backend(caption_renderer_backend: str) -> None:
    """Validate the configured caption renderer backend."""

    if caption_renderer_backend not in SUPPORTED_CAPTION_RENDERER_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_CAPTION_RENDERER_BACKENDS))
        raise CaptionRenderingError(
            "Invalid caption renderer backend: "
            f"{caption_renderer_backend!r}. Supported values: {supported}."
        )


def _caption_renderer(
    caption_renderer_backend: str,
    *,
    ass_subtitle_path: Path | None,
) -> CaptionRenderer:
    require_caption_renderer_backend(caption_renderer_backend)
    if caption_renderer_backend == CAPTION_RENDERER_DRAWTEXT:
        return DrawtextCaptionRenderer()
    if ass_subtitle_path is None:
        raise CaptionRenderingError(
            "ASS caption rendering requires an ASS subtitle path."
        )
    return AssCaptionRenderer(ass_subtitle_path)


def ass_subtitle_path(
    output_path: Path,
    *,
    ass_temp_dir: Path | None,
    caption_segments: tuple[CaptionSegment, ...],
    caption_renderer_backend: str,
) -> Path | None:
    """Return the temporary ASS subtitle path required by the ASS backend."""

    if caption_renderer_backend != CAPTION_RENDERER_ASS:
        return None
    if not caption_segments:
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
                "&HFF000000,"
                "1,0,0,0,"
                "100,100,0,0,"
                "1,"
                f"{_caption_outline_width(caption_style)},"
                f"{_caption_shadow_strength(caption_style)},"
                "2,"
                f"{caption_style.safe_margin_x},"
                f"{caption_style.safe_margin_x},"
                f"{_caption_safe_margin_bottom(caption_style)},"
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
    if cue.words:
        return _ass_active_word_dialogue_lines(cue, caption_style, output_size)

    return _ass_plain_dialogue_lines(cue, caption_style, output_size)


def _ass_plain_dialogue_lines(
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
            f"{_caption_safe_margin_bottom(caption_style)},,"
            f"{{{_ass_caption_override_tags(output_size.width // 2, y)}}}"
            f"{escape_ass_text(text)}"
        )
    return tuple(dialogue_lines)


def _ass_active_word_dialogue_lines(
    cue: CaptionCue,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> tuple[str, ...]:
    dialogue_lines = []
    line_word_offsets = _caption_line_word_offsets(cue.lines)
    for start_time, end_time, active_word_index in _caption_word_intervals(
        cue,
        caption_style,
    ):
        for line_index, line in enumerate(cue.lines):
            y = _ass_caption_line_y(len(cue.lines), line_index, caption_style, output_size)
            active_line_word_index = _active_line_word_index(
                active_word_index,
                line_word_offsets[line_index],
                line,
            )
            dialogue_lines.append(
                _ass_dialogue_line(
                    start_time=start_time,
                    end_time=end_time,
                    line=line,
                    active_line_word_index=active_line_word_index,
                    y=y,
                    caption_style=caption_style,
                    output_size=output_size,
                )
            )
    return tuple(dialogue_lines)


def _ass_dialogue_line(
    *,
    start_time: float,
    end_time: float,
    line: str,
    active_line_word_index: int | None,
    y: int,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> str:
    text = line.upper() if caption_style.uppercase else line
    return (
        "Dialogue: "
        f"0,{_format_ass_time(start_time)},{_format_ass_time(end_time)},"
        f"{_ass_field(caption_style.ass_style_name)},,"
        f"{caption_style.safe_margin_x},{caption_style.safe_margin_x},"
        f"{_caption_safe_margin_bottom(caption_style)},,"
        f"{{{_ass_caption_override_tags(output_size.width // 2, y)}}}"
        f"{_ass_caption_text(text, active_line_word_index, caption_style)}"
    )


def _caption_word_intervals(
    cue: CaptionCue,
    caption_style: CaptionStyle,
) -> tuple[tuple[float, float, int | None], ...]:
    word_spans = tuple(
        (index, max(cue.start_time, word.start_time), min(cue.end_time, word.end_time))
        for index, word in enumerate(cue.words)
        if word.start_time < cue.end_time and word.end_time > cue.start_time
    )
    if not word_spans:
        return ((cue.start_time, cue.end_time, None),)

    activation_delay = max(0.0, caption_style.ass_active_word_activation_delay_seconds)
    min_display = max(0.0, caption_style.ass_active_word_min_display_seconds)
    gap_tolerance = max(0.0, caption_style.ass_active_word_gap_tolerance_seconds)

    intervals: list[tuple[float, float, int | None]] = []
    cursor = cue.start_time
    for position, (word_index, raw_start_time, raw_end_time) in enumerate(word_spans):
        next_raw_start_time = (
            word_spans[position + 1][1] if position + 1 < len(word_spans) else None
        )
        start_time = min(cue.end_time, raw_start_time + activation_delay)
        if start_time > cursor:
            intervals.append((cursor, start_time, None))
        start_time = max(start_time, cursor)

        if next_raw_start_time is None:
            latest_end_time = min(cue.end_time, raw_end_time + gap_tolerance)
        else:
            next_start_time = min(cue.end_time, next_raw_start_time + activation_delay)
            latest_end_time = next_start_time

        target_end_time = max(raw_end_time, start_time + min_display)
        if (
            next_raw_start_time is not None
            and next_raw_start_time - raw_end_time <= gap_tolerance
        ):
            target_end_time = max(target_end_time, latest_end_time)

        end_time = min(latest_end_time, target_end_time)
        if end_time > start_time:
            intervals.append((start_time, end_time, word_index))
            cursor = end_time

    if cursor < cue.end_time:
        intervals.append((cursor, cue.end_time, None))
    return tuple(intervals)


def _caption_line_word_offsets(lines: tuple[str, ...]) -> tuple[int, ...]:
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(_caption_line_word_spans(line))
    return tuple(offsets)


def _active_line_word_index(
    active_word_index: int | None,
    line_word_offset: int,
    line: str,
) -> int | None:
    if active_word_index is None:
        return None
    line_word_index = active_word_index - line_word_offset
    if 0 <= line_word_index < len(_caption_line_word_spans(line)):
        return line_word_index
    return None


def _ass_caption_text(
    text: str,
    active_word_index: int | None,
    caption_style: CaptionStyle,
) -> str:
    if active_word_index is None:
        return escape_ass_text(text)

    spans = _caption_line_word_spans(text)
    if active_word_index >= len(spans):
        return escape_ass_text(text)

    start, end = spans[active_word_index]
    highlight_color = _ass_inline_color(
        caption_style.active_word_color,
        default="&H0000FFFF",
    )
    base_color = _ass_inline_color(
        caption_style.font_color,
        default="&H00FFFFFF",
    )
    return "".join(
        (
            escape_ass_text(text[:start]),
            f"{{\\c{highlight_color}}}",
            escape_ass_text(text[start:end]),
            f"{{\\c{base_color}}}",
            escape_ass_text(text[end:]),
        )
    )


def _caption_line_word_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


def _ass_caption_line_y(
    line_count: int,
    line_index: int,
    caption_style: CaptionStyle,
    output_size: OutputSize,
) -> int:
    line_step = caption_style.font_size + caption_style.line_spacing
    if _caption_should_center_vertically(caption_style):
        safe_margin_top = _caption_safe_margin_top(caption_style)
        safe_area_height = (
            output_size.height
            - safe_margin_top
            - _caption_safe_margin_bottom(caption_style)
        )
        block_height = _caption_block_height(line_count, caption_style)
        bottom_y = safe_margin_top + round((safe_area_height + block_height) / 2)
    else:
        bottom_y = output_size.height - _caption_safe_margin_bottom(caption_style)
    raw_y = bottom_y - (line_count - line_index - 1) * line_step
    return max(_caption_safe_margin_top(caption_style), raw_y)


def _ass_caption_override_tags(x: int, y: int) -> str:
    return f"\\an2\\pos({x},{y})"


def _caption_outline_width(caption_style: CaptionStyle) -> int:
    if caption_style.outline_thickness is not None:
        return caption_style.outline_thickness
    return caption_style.outline_width


def _caption_shadow_strength(caption_style: CaptionStyle) -> int:
    if caption_style.shadow_strength is not None:
        return caption_style.shadow_strength
    return caption_style.shadow_offset


def _caption_safe_margin_top(caption_style: CaptionStyle) -> int:
    if caption_style.vertical_safe_area is None:
        return 0
    return caption_style.vertical_safe_area.top


def _caption_safe_margin_bottom(caption_style: CaptionStyle) -> int:
    if caption_style.vertical_safe_area is None:
        return caption_style.safe_margin_bottom
    return caption_style.vertical_safe_area.bottom


def _caption_should_center_vertically(caption_style: CaptionStyle) -> bool:
    return (
        caption_style.vertical_safe_area is not None
        and caption_style.vertical_safe_area.center
    )


def _caption_block_height(line_count: int, caption_style: CaptionStyle) -> int:
    return (
        line_count * caption_style.font_size
        + (line_count - 1) * caption_style.line_spacing
    )


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


def _ass_inline_color(value: str, *, default: str) -> str:
    color = _ass_color(value, default=default)
    return f"&H{color[-6:]}&"


def _ass_fontsdir_option(caption_style: CaptionStyle) -> str:
    if caption_style.font_file is None:
        return ""
    fonts_dir = _ffmpeg_path(caption_style.font_file.parent)
    return f":fontsdir='{_escape_ffmpeg_filter_option(fonts_dir)}'"


def _ffmpeg_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _escape_ffmpeg_filter_option(value: str) -> str:
    return escape_drawtext_option(value)


def _caption_font_option(caption_style: CaptionStyle) -> str:
    if caption_style.font_file is None:
        return ""
    font_file = str(caption_style.font_file).replace("\\", "/")
    return f"fontfile='{escape_drawtext_option(font_file)}':"


def _caption_font_log_value(caption_style: CaptionStyle) -> str:
    if caption_style.font_file is not None:
        return f"font file {caption_style.font_file}"
    if caption_style.font_family is not None and caption_style.font_family.strip():
        return f"font family {caption_style.font_family.strip()}"
    if caption_style.font_fallbacks:
        return f"font fallback {caption_style.font_fallbacks[0]}"
    return "default font fallback"


def _fmt(value: float) -> str:
    return f"{value:.10g}"

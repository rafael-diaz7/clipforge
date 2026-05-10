from __future__ import annotations

import logging
from pathlib import Path

import pytest

from clipforge.media.captions import CaptionMetadata, CaptionSegment, CaptionWord
from clipforge.media.layouts import Layout, LayoutRegion, NormalizedRect, OutputSize
from clipforge.media.render import (
    CaptionAnimationPreset,
    CaptionCue,
    CaptionStyle,
    CaptionVerticalSafeArea,
    RenderError,
    _caption_chunk_duration,
    _caption_cues,
    build_ffmpeg_command,
    build_filter_complex,
    generate_ass_subtitle,
    rect_to_pixels,
    render_layout,
    run_ffmpeg_command,
)


def _layout(
    *regions: LayoutRegion,
    caption_region: NormalizedRect | None = None,
) -> Layout:
    return Layout(
        name="test_layout",
        description="Test layout.",
        output=OutputSize(width=1080, height=1920),
        regions=regions,
        caption_region=caption_region,
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


def test_build_filter_complex_applies_optional_region_blur_effect() -> None:
    filter_complex = build_filter_complex(
        _layout(
            LayoutRegion(
                name="background",
                source_region=NormalizedRect(x=0.0, y=0.0, width=1.0, height=1.0),
                output_region=NormalizedRect(x=0.0, y=0.0, width=1.0, height=1.0),
                effect="blur",
            )
        )
    )

    assert "crop=iw*1:ih*1:iw*0:ih*0" in filter_complex
    assert "scale=1080:1920:force_original_aspect_ratio=increase" in filter_complex
    assert "setsar=1,boxblur=20:1[region0]" in filter_complex


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


def test_build_filter_complex_can_append_caption_overlays() -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(
            CaptionSegment(start_time=0.5, end_time=1.75, text="Let's go: win now"),
            CaptionSegment(start_time=2, end_time=3, text="Second caption"),
        ),
    )

    filter_complex = build_filter_complex(
        _layout(_region()),
        caption_metadata=caption_metadata,
        caption_style=CaptionStyle(
            font_size=52,
            safe_margin_bottom=180,
            max_chars_per_line=16,
        ),
    )

    assert "[base][region0]overlay=0:0:format=auto:shortest=1[captionbase]" in filter_complex
    assert "drawtext=text=Let\\\\\\'s\\ go\\:\\ win" in filter_complex
    assert "drawtext=text=now" in filter_complex
    assert "\\n" not in filter_complex
    assert "fontsize=52" in filter_complex
    assert "y=max(96\\,h-112-180)" in filter_complex
    assert "enable='between(t\\,0.5\\,1.75)'" in filter_complex
    assert "enable='between(t\\,2\\,3)'" in filter_complex
    assert filter_complex.endswith("[out]")


def test_build_filter_complex_centers_captions_in_layout_caption_region() -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
    )
    layout = _layout(
        _region(
            name="gameplay",
            output_region=NormalizedRect(x=0.0, y=0.34, width=1.0, height=0.66),
        ),
        _region(
            name="facecam",
            output_region=NormalizedRect(x=0.0, y=0.0, width=1.0, height=0.34),
        ),
        caption_region=NormalizedRect(x=0.0, y=0.34, width=1.0, height=0.1),
    )

    filter_complex = build_filter_complex(layout, caption_metadata=caption_metadata)

    assert "drawtext=text=hello" in filter_complex
    assert "y=max(653\\,653+(h-653-1075-56)/2)" in filter_complex


def test_build_filter_complex_limits_long_caption_display_time() -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(
            CaptionSegment(
                start_time=4,
                end_time=12,
                text="short caption",
            ),
        ),
    )

    filter_complex = build_filter_complex(
        _layout(_region()),
        caption_metadata=caption_metadata,
    )

    assert "enable='between(t\\,4\\,5.47)'" in filter_complex


def test_build_filter_complex_splits_long_caption_segments_into_multiple_cues() -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(
            CaptionSegment(
                start_time=0,
                end_time=6,
                text="Hey case happy Cinco de Mayo was wondering if you wanted to come over again like last",
            ),
        ),
    )

    filter_complex = build_filter_complex(
        _layout(_region()),
        caption_metadata=caption_metadata,
    )

    assert filter_complex.count("drawtext=") > 1
    assert "wanted\\ to\\ come\\ over\\ again" in filter_complex
    assert "drawtext=text=like\\ last" in filter_complex
    assert "\\n" not in filter_complex
    assert "..." not in filter_complex


def test_caption_timing_uses_character_weighting() -> None:
    cues = _caption_cues(
        (
            CaptionSegment(
                start_time=0,
                end_time=8,
                text="short extraordinarilylongword",
            ),
        ),
        caption_style=CaptionStyle(
            max_chars_per_line=12,
            max_lines=1,
            max_hold_seconds=8,
        ),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert len(cues) == 2
    short_duration = cues[0].end_time - cues[0].start_time
    long_duration = cues[1].end_time - cues[1].start_time
    assert long_duration > short_duration


def test_caption_timing_adds_punctuation_pause_weighting() -> None:
    plain_duration = _caption_chunk_duration("wait really now", CaptionStyle())
    punctuated_duration = _caption_chunk_duration("wait, really? now!", CaptionStyle())

    assert punctuated_duration > plain_duration


def test_caption_timing_enforces_minimum_cue_duration() -> None:
    duration = _caption_chunk_duration(
        "go",
        CaptionStyle(
            min_cue_seconds=1.25,
            seconds_per_word=0,
            seconds_per_character=0,
            punctuation_pause_seconds=0,
            display_padding_seconds=0,
        ),
    )

    assert duration == 1.25


def test_caption_timing_preserves_chunk_boundaries_inside_segment() -> None:
    cues = _caption_cues(
        (
            CaptionSegment(
                start_time=10,
                end_time=11,
                text="alpha beta gamma delta",
            ),
        ),
        caption_style=CaptionStyle(
            max_chars_per_line=6,
            max_lines=1,
            min_cue_seconds=0.4,
            max_hold_seconds=3,
        ),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert len(cues) == 4
    assert cues[0].start_time == 10
    assert cues[-1].end_time == 11
    assert all(cue.end_time > cue.start_time for cue in cues)
    assert all(
        previous.end_time == current.start_time
        for previous, current in zip(cues[:-1], cues[1:], strict=True)
    )


def test_caption_cues_carry_optional_word_timing_metadata() -> None:
    words = (
        CaptionWord(start_time=0, end_time=0.4, text="hello"),
        CaptionWord(start_time=0.5, end_time=1, text="world"),
    )

    cues = _caption_cues(
        (
            CaptionSegment(
                start_time=0,
                end_time=1,
                text="hello world",
                words=words,
            ),
        ),
        caption_style=CaptionStyle(max_chars_per_line=24),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert len(cues) == 1
    assert cues[0].words == words


def test_caption_style_serializes_future_caption_fields() -> None:
    style = CaptionStyle(
        font_file=Path("C:/Windows/Fonts/arial.ttf"),
        font_fallbacks=("Inter", "Segoe UI Emoji"),
        outline_thickness=7,
        shadow_strength=5,
        uppercase=True,
        highlight_color="#ffee00",
        active_word_color="#00e5ff",
        animation_preset=CaptionAnimationPreset.SCALE_POP,
        vertical_safe_area=CaptionVerticalSafeArea(top=140, bottom=260),
    )

    payload = style.to_dict()

    assert payload["font_file"] == "C:\\Windows\\Fonts\\arial.ttf"
    assert payload["font_fallbacks"] == ["Inter", "Segoe UI Emoji"]
    assert payload["outline_thickness"] == 7
    assert payload["shadow_strength"] == 5
    assert payload["uppercase"] is True
    assert payload["highlight_color"] == "#ffee00"
    assert payload["active_word_color"] == "#00e5ff"
    assert payload["ass_active_word_activation_delay_seconds"] == 0.04
    assert payload["ass_active_word_min_display_seconds"] == 0.14
    assert payload["ass_active_word_gap_tolerance_seconds"] == 0.12
    assert payload["animation_preset"] == "scale_pop"
    assert payload["vertical_safe_area"] == {"top": 140, "bottom": 260}
    assert CaptionStyle.from_dict(payload) == style


def test_generate_ass_subtitle_uses_explicit_polish_fields() -> None:
    ass_text = generate_ass_subtitle(
        (CaptionCue(start_time=0, end_time=1, lines=("hello",)),),
        caption_style=CaptionStyle(
            outline_thickness=8,
            shadow_strength=6,
            vertical_safe_area=CaptionVerticalSafeArea(top=150, bottom=300),
        ),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert ",1,8,6,2,96,96,300,1" in ass_text
    assert "{\\an2\\pos(540,1620)}hello" in ass_text


def test_generate_ass_subtitle_keeps_captions_without_words_unchanged() -> None:
    ass_text = generate_ass_subtitle(
        (CaptionCue(start_time=0, end_time=1, lines=("hello world",)),),
        caption_style=CaptionStyle(),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,96,96,220,," in ass_text
    assert "{\\an2\\pos(540,1700)}hello world" in ass_text
    assert "\\c" not in ass_text


def test_generate_ass_subtitle_highlights_active_word_by_timing() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=0,
                end_time=2,
                lines=("hello world",),
                words=(
                    CaptionWord(start_time=0.25, end_time=0.75, text="hello"),
                    CaptionWord(start_time=1.0, end_time=1.5, text="world"),
                ),
            ),
        ),
        caption_style=CaptionStyle(active_word_color="#ffee00"),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert (
        "Dialogue: 0,0:00:00.29,0:00:00.75,Default,,96,96,220,,"
        "{\\an2\\pos(540,1700)}{\\c&H00EEFF&}hello{\\c&HFFFFFF&} world"
    ) in ass_text
    assert (
        "Dialogue: 0,0:00:01.04,0:00:01.50,Default,,96,96,220,,"
        "{\\an2\\pos(540,1700)}hello {\\c&H00EEFF&}world{\\c&HFFFFFF&}"
    ) in ass_text


def test_generate_ass_subtitle_keeps_short_active_word_windows_visible() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=0,
                end_time=1,
                lines=("a b",),
                words=(
                    CaptionWord(start_time=0.0, end_time=0.03, text="a"),
                    CaptionWord(start_time=0.5, end_time=0.9, text="b"),
                ),
            ),
        ),
        caption_style=CaptionStyle(),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert (
        "Dialogue: 0,0:00:00.04,0:00:00.18,Default,,96,96,220,,"
        "{\\an2\\pos(540,1700)}{\\c&H00FFFF&}a{\\c&HFFFFFF&} b"
    ) in ass_text


def test_generate_ass_subtitle_bridges_tiny_active_word_gaps() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=0,
                end_time=1,
                lines=("one two",),
                words=(
                    CaptionWord(start_time=0.0, end_time=0.2, text="one"),
                    CaptionWord(start_time=0.22, end_time=0.5, text="two"),
                ),
            ),
        ),
        caption_style=CaptionStyle(),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert (
        "Dialogue: 0,0:00:00.04,0:00:00.26,Default,,96,96,220,,"
        "{\\an2\\pos(540,1700)}{\\c&H00FFFF&}one{\\c&HFFFFFF&} two"
    ) in ass_text
    assert (
        "Dialogue: 0,0:00:00.26,0:00:00.50,Default,,96,96,220,,"
        "{\\an2\\pos(540,1700)}one {\\c&H00FFFF&}two{\\c&HFFFFFF&}"
    ) in ass_text


def test_generate_ass_subtitle_highlights_repeated_word_by_occurrence() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=0,
                end_time=2,
                lines=("go go go",),
                words=(
                    CaptionWord(start_time=0.0, end_time=0.4, text="go"),
                    CaptionWord(start_time=0.8, end_time=1.2, text="go"),
                    CaptionWord(start_time=1.6, end_time=2.0, text="go"),
                ),
            ),
        ),
        caption_style=CaptionStyle(),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert (
        "Dialogue: 0,0:00:00.84,0:00:01.20,Default,,96,96,220,,"
        "{\\an2\\pos(540,1700)}go {\\c&H00FFFF&}go{\\c&HFFFFFF&} go"
    ) in ass_text


def test_generate_ass_subtitle_does_not_highlight_outside_word_timings() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=0,
                end_time=2,
                lines=("hello",),
                words=(CaptionWord(start_time=0.5, end_time=1.0, text="hello"),),
            ),
        ),
        caption_style=CaptionStyle(),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert "Dialogue: 0,0:00:00.00,0:00:00.54" in ass_text
    assert "Dialogue: 0,0:00:01.00,0:00:02.00" in ass_text
    assert "{\\an2\\pos(540,1700)}hello\n" in ass_text
    assert ass_text.count("\\c&H00FFFF&") == 1


def test_generate_ass_subtitle_caps_final_active_word_extension() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=0,
                end_time=1.2,
                lines=("go",),
                words=(CaptionWord(start_time=0.9, end_time=0.93, text="go"),),
            ),
        ),
        caption_style=CaptionStyle(),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert (
        "Dialogue: 0,0:00:00.94,0:00:01.05,Default,,96,96,220,,"
        "{\\an2\\pos(540,1700)}{\\c&H00FFFF&}go{\\c&HFFFFFF&}"
    ) in ass_text
    assert "Dialogue: 0,0:00:01.05,0:00:01.20" in ass_text


def test_ass_active_word_caption_placement_stays_in_caption_region(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "render.mp4"
    ass_temp_dir = tmp_path / "ass"
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(
            CaptionSegment(
                start_time=0,
                end_time=1,
                text="hello",
                words=(CaptionWord(start_time=0, end_time=1, text="hello"),),
            ),
        ),
    )
    layout = _layout(
        _region(
            name="gameplay",
            output_region=NormalizedRect(x=0.0, y=0.34, width=1.0, height=0.66),
        ),
        _region(
            name="facecam",
            output_region=NormalizedRect(x=0.0, y=0.0, width=1.0, height=0.34),
        ),
        caption_region=NormalizedRect(x=0.0, y=0.34, width=1.0, height=0.1),
    )

    build_ffmpeg_command(
        source,
        output,
        layout,
        caption_metadata=caption_metadata,
        caption_renderer_backend="ass",
        ass_temp_dir=ass_temp_dir,
    )

    ass_text = (ass_temp_dir / "render.ass").read_text(encoding="utf-8")
    assert "&HFF000000" in ass_text
    assert "{\\an2\\pos(540,777)}{\\c&H00FFFF&}hello{\\c&HFFFFFF&}" in ass_text


def test_build_filter_complex_can_use_custom_caption_font_file() -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
    )

    filter_complex = build_filter_complex(
        _layout(_region()),
        caption_metadata=caption_metadata,
        caption_style=CaptionStyle(font_file=Path("C:/Windows/Fonts/arial.ttf")),
    )

    assert "fontfile='C\\:/Windows/Fonts/arial.ttf'" in filter_complex


def test_build_filter_complex_logs_caption_renderer_and_font(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
    )

    with caplog.at_level(logging.INFO, logger="clipforge.media.render"):
        build_filter_complex(
            _layout(_region()),
            caption_metadata=caption_metadata,
            caption_style=CaptionStyle(font_file=Path("C:/Windows/Fonts/arial.ttf")),
        )

    assert (
        "Rendering captions with drawtext backend using "
        "font file C:\\Windows\\Fonts\\arial.ttf."
    ) in caplog.messages


def test_build_filter_complex_normalizes_caption_apostrophes_for_drawtext() -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(
            CaptionSegment(start_time=0, end_time=1, text="You'd win that true."),
            CaptionSegment(start_time=1, end_time=2, text="But I don't have them."),
        ),
    )

    filter_complex = build_filter_complex(
        _layout(_region()),
        caption_metadata=caption_metadata,
    )

    assert "You\\\\\\'d\\ win\\ that\\ true." in filter_complex
    assert "But\\ I\\ don\\\\\\'t\\ have\\ them." in filter_complex


def test_build_filter_complex_ignores_empty_caption_metadata() -> None:
    filter_complex = build_filter_complex(
        _layout(_region()),
        caption_metadata=CaptionMetadata(clip_id="clip-123", segments=()),
    )

    assert "drawtext=" not in filter_complex
    assert filter_complex.endswith("[out]")


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


def test_build_ffmpeg_command_accepts_caption_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "render.mp4"
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
    )

    command = build_ffmpeg_command(
        source,
        output,
        _layout(_region()),
        caption_metadata=caption_metadata,
    )

    filter_complex = command[command.index("-filter_complex") + 1]
    assert "drawtext=" in filter_complex
    assert command[-1] == str(output)
    assert all(isinstance(part, str) for part in command)


def test_build_ffmpeg_command_can_use_ass_caption_backend(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "render.mp4"
    ass_temp_dir = tmp_path / "ass"
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(
            CaptionSegment(
                start_time=0.5,
                end_time=1.75,
                text="Hello Twitch chat, café time",
            ),
        ),
    )

    command = build_ffmpeg_command(
        source,
        output,
        _layout(_region()),
        caption_metadata=caption_metadata,
        caption_style=CaptionStyle(
            font_family="Inter",
            font_size=48,
            line_spacing=12,
            safe_margin_x=80,
            safe_margin_bottom=200,
            max_chars_per_line=18,
        ),
        caption_renderer_backend="ass",
        ass_temp_dir=ass_temp_dir,
    )

    filter_complex = command[command.index("-filter_complex") + 1]
    assert "drawtext=" not in filter_complex
    assert "ass=filename=" in filter_complex
    assert "[base][region0]overlay=0:0:format=auto:shortest=1[captionbase]" in filter_complex
    assert filter_complex.endswith("[out]")

    ass_path = ass_temp_dir / "render.ass"
    assert ass_path.is_file()
    ass_text = ass_path.read_text(encoding="utf-8")
    assert "[Script Info]" in ass_text
    assert "Style: Default,Inter,48" in ass_text
    assert "Dialogue: 0,0:00:00.50,0:00:01.75,Default" in ass_text
    assert "café" in ass_text


def test_generate_ass_subtitle_builds_structure_and_style_block() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=61.234,
                end_time=62.999,
                lines=("First line", "Unicode snowman ☃"),
            ),
        ),
        caption_style=CaptionStyle(
            font_family="Aptos",
            font_size=42,
            font_color="#ffee00",
            line_spacing=10,
            outline_width=5,
            shadow_offset=3,
            safe_margin_x=72,
            safe_margin_bottom=180,
        ),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text
    assert "Format: Name, Fontname, Fontsize" in ass_text
    assert "Style: Default,Aptos,42,&H0000EEFF,&H0000EEFF,&H00000000" in ass_text
    assert ",5,3,2,72,72,180,1" in ass_text
    assert "Dialogue: 0,0:01:01.23,0:01:03.00,Default" in ass_text
    assert "{\\an2\\pos(540,1688)}First line" in ass_text
    assert "{\\an2\\pos(540,1740)}Unicode snowman ☃" in ass_text


def test_generate_ass_subtitle_escapes_text_and_can_uppercase() -> None:
    ass_text = generate_ass_subtitle(
        (
            CaptionCue(
                start_time=0,
                end_time=1,
                lines=("slash \\ and {tag}", "emoji 😀"),
            ),
        ),
        caption_style=CaptionStyle(uppercase=True),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert "SLASH \\\\ AND \\{TAG\\}" in ass_text
    assert "EMOJI 😀" in ass_text


def test_generate_ass_subtitle_uses_custom_font_references() -> None:
    ass_text = generate_ass_subtitle(
        (CaptionCue(start_time=0, end_time=1, lines=("hello",)),),
        caption_style=CaptionStyle(
            font_file=Path("C:/Windows/Fonts/arial.ttf"),
            font_fallbacks=("Segoe UI Emoji", "Arial"),
        ),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert "Style: Default,arial,56" in ass_text


def test_build_filter_complex_requires_ass_subtitle_path_for_ass_backend() -> None:
    caption_metadata = CaptionMetadata(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
    )

    with pytest.raises(RenderError, match="ASS subtitle path"):
        build_filter_complex(
            _layout(_region()),
            caption_metadata=caption_metadata,
            caption_renderer_backend="ass",
        )


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

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from clipforge.integrations.openai import (
    OpenAITranscriptionClient,
    extract_transcription_audio,
    media_duration_seconds,
    parse_openai_transcription_payload,
)
from clipforge.media.captions import (
    CaptionMetadata,
    CaptionSegment,
    CaptionTranscriptionError,
    CaptionWord,
)
from clipforge.media.caption_rendering import (
    CaptionStyle,
    _caption_cues,
    generate_ass_subtitle,
)
from clipforge.media.layouts import OutputSize
from tests.constants import TWITCH_CLIP_SLUG


CAPTION_CLIP_ID = TWITCH_CLIP_SLUG


class FakeOpenAIResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: object | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload if payload is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> object:
        return self.payload


class FakeOpenAISession:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        data: list[tuple[str, str]],
        files: dict[str, tuple[str, object]],
        timeout: int,
    ) -> FakeOpenAIResponse:
        filename, file_obj = files["file"]
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "data": data,
                "filename": filename,
                "file_bytes": file_obj.read(),
                "timeout": timeout,
            }
        )
        return self.response


def test_parse_openai_transcription_payload_normalizes_segments() -> None:
    metadata = parse_openai_transcription_payload(
        {
            "text": "hello world",
            "segments": [
                {"start": 1.25, "end": 2.5, "text": " world "},
                {"start": 0, "end": 1, "text": "hello"},
                {"start": 3, "end": 4, "text": "   "},
            ],
        },
        clip_id=CAPTION_CLIP_ID,
    )

    assert metadata == CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(
            CaptionSegment(start_time=0, end_time=1, text="hello"),
            CaptionSegment(start_time=1.25, end_time=2.5, text="world"),
        ),
    )


def test_parse_openai_transcription_payload_populates_segment_words() -> None:
    metadata = parse_openai_transcription_payload(
        {
            "text": "hello world again",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello world"},
                {"start": 1.0, "end": 2.0, "text": "again"},
            ],
            "words": [
                {"start": 0.1, "end": 0.3, "word": "hello"},
                {"start": 0.4, "end": 0.9, "word": "world"},
                {"start": 1.1, "end": 1.5, "word": "again"},
            ],
        },
        clip_id=CAPTION_CLIP_ID,
    )

    assert metadata == CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(
            CaptionSegment(
                start_time=0.0,
                end_time=1.0,
                text="hello world",
                words=(
                    CaptionWord(start_time=0.1, end_time=0.3, text="hello"),
                    CaptionWord(start_time=0.4, end_time=0.9, text="world"),
                ),
            ),
            CaptionSegment(
                start_time=1.0,
                end_time=2.0,
                text="again",
                words=(CaptionWord(start_time=1.1, end_time=1.5, text="again"),),
            ),
        ),
    )


def test_parse_openai_transcription_payload_assigns_overlapping_words_once() -> None:
    metadata = parse_openai_transcription_payload(
        {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
            "words": [
                {"start": 0.8, "end": 1.4, "word": "world"},
            ],
        },
        clip_id=CAPTION_CLIP_ID,
    )

    assert metadata.segments[0].words == ()
    assert metadata.segments[1].words == (
        CaptionWord(start_time=1.0, end_time=1.4, text="world"),
    )


def test_parse_openai_transcription_payload_handles_empty_words() -> None:
    metadata = parse_openai_transcription_payload(
        {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
            ],
            "words": [
                {"start": 0.1, "end": 0.2, "word": "   "},
            ],
        },
        clip_id=CAPTION_CLIP_ID,
    )

    assert metadata == CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(CaptionSegment(start_time=0.0, end_time=1.0, text="hello"),),
    )


def test_generated_openai_words_can_drive_ass_active_word_rendering() -> None:
    metadata = parse_openai_transcription_payload(
        {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello world"},
            ],
            "words": [
                {"start": 0.0, "end": 0.4, "word": "hello"},
                {"start": 0.5, "end": 1.0, "word": "world"},
            ],
        },
        clip_id=CAPTION_CLIP_ID,
    )

    ass_text = generate_ass_subtitle(
        _caption_cues(
            metadata.segments,
            caption_style=CaptionStyle(max_chars_per_line=24),
            output_size=OutputSize(width=1080, height=1920),
        ),
        caption_style=CaptionStyle(max_chars_per_line=24),
        output_size=OutputSize(width=1080, height=1920),
    )

    assert "{\\c&H00FFFF&}hello{\\c&HFFFFFF&} world" in ass_text
    assert "hello {\\c&H00FFFF&}world{\\c&HFFFFFF&}" in ass_text


def test_parse_openai_transcription_payload_can_use_full_clip_fallback() -> None:
    metadata = parse_openai_transcription_payload(
        {"text": "whole clip transcript"},
        clip_id=CAPTION_CLIP_ID,
        fallback_duration_seconds=12.5,
    )

    assert metadata == CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(CaptionSegment(start_time=0, end_time=12.5, text="whole clip transcript"),),
    )


def test_parse_openai_transcription_payload_rejects_missing_segments_without_fallback() -> None:
    with pytest.raises(CaptionTranscriptionError, match="timestamp segments"):
        parse_openai_transcription_payload({"text": "hello"}, clip_id=CAPTION_CLIP_ID)


def test_openai_transcription_client_uses_json_for_gpt_transcription_model(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video bytes")
    response = FakeOpenAIResponse(
        payload={"text": "full transcript"},
        headers={"x-request-id": "req_123"},
    )
    session = FakeOpenAISession(response)
    client = OpenAITranscriptionClient(
        api_key="test-openai-key",
        model="gpt-4o-mini-transcribe",
        session=session,
        timeout_seconds=30,
        duration_probe=lambda path: 9.25,
        audio_extractor=lambda source, output: output.write_bytes(b"audio bytes"),
    )

    metadata = client.transcribe(source_path, clip_id=CAPTION_CLIP_ID)

    assert metadata == CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(CaptionSegment(start_time=0.0, end_time=9.25, text="full transcript"),),
    )
    assert session.calls == [
        {
            "url": "https://api.openai.com/v1/audio/transcriptions",
            "headers": {"Authorization": "Bearer test-openai-key"},
            "data": [
                ("model", "gpt-4o-mini-transcribe"),
                ("response_format", "json"),
            ],
            "filename": "source.mp3",
            "file_bytes": b"audio bytes",
            "timeout": 30,
        }
    ]


def test_openai_transcription_client_uses_verbose_json_for_whisper(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video bytes")
    session = FakeOpenAISession(
        FakeOpenAIResponse(
            payload={
                "segments": [
                    {"start": 0.0, "end": 1.5, "text": " first caption "},
                ],
            }
        )
    )
    client = OpenAITranscriptionClient(
        api_key="test-openai-key",
        model="whisper-1",
        session=session,
        audio_extractor=lambda source, output: output.write_bytes(b"audio bytes"),
    )

    metadata = client.transcribe(source_path, clip_id=CAPTION_CLIP_ID)

    assert metadata == CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(CaptionSegment(start_time=0.0, end_time=1.5, text="first caption"),),
    )
    assert session.calls[0]["data"] == [
        ("model", "whisper-1"),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
    ]
    assert session.calls[0]["filename"] == "source.mp3"
    assert session.calls[0]["file_bytes"] == b"audio bytes"


def test_openai_transcription_client_redacts_api_key_from_errors(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video bytes")
    session = FakeOpenAISession(
        FakeOpenAIResponse(
            status_code=401,
            text="invalid key test-openai-key",
            headers={"x-request-id": "req_401"},
        )
    )
    client = OpenAITranscriptionClient(
        api_key="test-openai-key",
        session=session,
        audio_extractor=lambda source, output: output.write_bytes(b"audio bytes"),
    )

    with pytest.raises(CaptionTranscriptionError) as exc_info:
        client.transcribe(source_path, clip_id=CAPTION_CLIP_ID)

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "request_id=req_401" in message
    assert "test-openai-key" not in message
    assert "[redacted]" in message


def test_extract_transcription_audio_uses_speech_optimized_ffmpeg_command(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "source.mp3"
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output_path.write_bytes(b"audio bytes")
        return subprocess.CompletedProcess(command, 0)

    assert extract_transcription_audio(source_path, output_path, runner=fake_runner) == output_path
    assert calls == [
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "mp3",
            "-b:a",
            "32k",
            str(output_path),
        ]
    ]


def test_media_duration_seconds_uses_ffprobe(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="12.5\n")

    assert media_duration_seconds(source_path, runner=fake_runner) == 12.5
    assert calls == [
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ]
    ]

"""OpenAI API integration helpers."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from clipforge.core.config import DEFAULT_OPENAI_TRANSCRIPTION_MODEL, ClipforgeConfig
from clipforge.integrations.retry import RetryDecision, RetryPolicy, retry_call
from clipforge.media.captions import (
    CaptionMetadata,
    CaptionSegment,
    CaptionTranscriptionError,
    CaptionWord,
)
from clipforge.utils.json_validation import required_list, required_number
from clipforge.utils.paths import response_text_excerpt


OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS = 120
DEFAULT_OPENAI_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=1.0,
    max_delay_seconds=30.0,
    jitter_seconds=0.25,
)
GPT_TRANSCRIBE_JSON_MODELS = ("gpt-4o-transcribe", "gpt-4o-mini-transcribe")
LOGGER = logging.getLogger(__name__)

DurationProbe = Callable[[Path], float]
AudioExtractor = Callable[[Path, Path], None]
SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class _OpenAITranscriptionWord:
    start_time: float
    end_time: float
    text: str


class _OpenAIHTTPStatusError(CaptionTranscriptionError):
    def __init__(
        self,
        *,
        response: requests.Response,
        source_path: Path,
        api_key: str,
    ) -> None:
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")
        super().__init__(
            _openai_error_message(
                response,
                source_path=source_path,
                api_key=api_key,
                request_id=self.request_id,
            )
        )


def classify_openai_retry_error(exc: BaseException) -> RetryDecision:
    """Classify OpenAI integration errors for retry."""

    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return RetryDecision(retryable=True, reason="transient network error")

    status_code = _openai_status_code(exc)
    if status_code is not None:
        if status_code in {408, 409, 429} or status_code >= 500:
            return RetryDecision(
                retryable=True,
                reason=f"retryable HTTP status {status_code}",
            )
        return RetryDecision(
            retryable=False,
            reason=f"non-retryable HTTP status {status_code}",
        )

    error_name = type(exc).__name__
    if error_name in {"RateLimitError", "APITimeoutError", "APIConnectionError"}:
        return RetryDecision(retryable=True, reason=error_name)
    if error_name in {
        "AuthenticationError",
        "BadRequestError",
        "PermissionDeniedError",
        "NotFoundError",
        "APIResponseValidationError",
    }:
        return RetryDecision(retryable=False, reason=error_name)

    return RetryDecision(retryable=False, reason="not classified as retryable")


def _openai_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


@dataclass(frozen=True)
class OpenAITranscriptionClient:
    """Small OpenAI transcription adapter focused on caption metadata."""

    api_key: str
    model: str = DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    session: requests.Session | None = None
    timeout_seconds: int = DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS
    duration_probe: DurationProbe | None = None
    audio_extractor: AudioExtractor | None = None
    retry_policy: RetryPolicy = DEFAULT_OPENAI_RETRY_POLICY

    @classmethod
    def from_config(cls, config: ClipforgeConfig) -> "OpenAITranscriptionClient":
        return cls(
            api_key=config.require_openai_api_key(),
            model=config.require_openai_transcription_model(),
        )

    def transcribe(self, source_path: Path, *, clip_id: str) -> CaptionMetadata:
        if not source_path.is_file():
            raise CaptionTranscriptionError(f"Caption source clip not found: {source_path}")

        try:
            with tempfile.TemporaryDirectory(prefix="clipforge-transcription-") as temp_dir:
                upload_path = Path(temp_dir) / f"{source_path.stem}.mp3"
                extractor = self.audio_extractor or extract_transcription_audio
                extractor(source_path, upload_path)
                with upload_path.open("rb") as source_file:
                    response = retry_call(
                        operation_name="transcription request",
                        provider="OpenAI",
                        operation=lambda: self._post_transcription_request(
                            upload_path=upload_path,
                            source_file=source_file,
                            source_path=source_path,
                        ),
                        policy=self.retry_policy,
                        classify_error=classify_openai_retry_error,
                    )
        except _OpenAIHTTPStatusError as exc:
            LOGGER.warning(
                "OpenAI transcription failed for %s with HTTP %s%s.",
                source_path,
                exc.status_code,
                f" (request_id={exc.request_id})" if exc.request_id else "",
            )
            raise
        except CaptionTranscriptionError:
            raise
        except requests.RequestException as exc:
            raise CaptionTranscriptionError(
                f"OpenAI transcription request failed for {source_path}: {exc}"
            ) from exc
        except OSError as exc:
            raise CaptionTranscriptionError(
                f"Could not prepare caption audio for {source_path}: {exc}"
            ) from exc

        request_id = response.headers.get("x-request-id")
        if request_id:
            LOGGER.info(
                "OpenAI transcription completed for %s (request_id=%s).",
                source_path,
                request_id,
            )
        payload = _decode_openai_transcription_response(response)
        fallback_duration = _fallback_duration_seconds(
            payload,
            source_path=source_path,
            duration_probe=self.duration_probe,
        )
        return parse_openai_transcription_payload(
            payload,
            clip_id=clip_id,
            fallback_duration_seconds=fallback_duration,
        )

    def _post_transcription_request(
        self,
        *,
        upload_path: Path,
        source_file: Any,
        source_path: Path,
    ) -> requests.Response:
        client = self.session or requests
        source_file.seek(0)
        response = client.post(
            OPENAI_TRANSCRIPTIONS_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=_transcription_request_data(self.model),
            files={"file": (upload_path.name, source_file)},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise _OpenAIHTTPStatusError(
                response=response,
                source_path=source_path,
                api_key=self.api_key,
            )
        return response


def extract_transcription_audio(
    source_path: Path,
    output_path: Path,
    *,
    runner: SubprocessRunner = subprocess.run,
) -> Path:
    """Extract compressed speech-oriented audio for transcription upload."""

    command = [
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
    try:
        runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CaptionTranscriptionError(
            f"Could not extract transcription audio with FFmpeg for {source_path}: "
            f"{_process_error_excerpt(exc)}"
        ) from exc

    try:
        if output_path.stat().st_size <= 0:
            raise CaptionTranscriptionError(
                f"FFmpeg produced an empty transcription audio file for {source_path}."
            )
    except OSError as exc:
        raise CaptionTranscriptionError(
            f"FFmpeg did not create transcription audio for {source_path}: {exc}"
        ) from exc
    return output_path


def normalize_openai_transcription_response(
    response: requests.Response,
    *,
    clip_id: str,
    fallback_duration_seconds: float | None = None,
) -> CaptionMetadata:
    """Normalize an OpenAI transcription response object."""

    payload = _decode_openai_transcription_response(response)
    return parse_openai_transcription_payload(
        payload,
        clip_id=clip_id,
        fallback_duration_seconds=fallback_duration_seconds,
    )


def parse_openai_transcription_payload(
    payload: Any,
    *,
    clip_id: str,
    fallback_duration_seconds: float | None = None,
) -> CaptionMetadata:
    """Normalize a decoded OpenAI transcription payload into caption metadata."""

    if not isinstance(payload, dict):
        raise CaptionTranscriptionError("OpenAI transcription response must be an object.")

    if "segments" in payload:
        segments_payload = required_list(
            payload,
            "segments",
            context="OpenAI transcription response",
            error_cls=CaptionTranscriptionError,
        )
        segments = tuple(
            segment
            for index, segment_payload in enumerate(segments_payload)
            if (
                segment := _parse_openai_transcription_segment(
                    segment_payload,
                    index=index,
                )
            )
            is not None
        )
        words = _parse_openai_transcription_words(payload)
        if words:
            segments = _segments_with_words(segments, words)
        return CaptionMetadata(clip_id=clip_id, segments=segments)

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise CaptionTranscriptionError(
            "OpenAI transcription response did not include timestamp segments or text."
        )
    if fallback_duration_seconds is None:
        raise CaptionTranscriptionError(
            "OpenAI transcription response did not include timestamp segments."
        )
    return CaptionMetadata(
        clip_id=clip_id,
        segments=(
            CaptionSegment(
                start_time=0.0,
                end_time=fallback_duration_seconds,
                text=text,
            ),
        ),
    )


def media_duration_seconds(
    source_path: Path,
    *,
    runner: SubprocessRunner = subprocess.run,
) -> float:
    """Return source media duration using ffprobe."""

    command = [
        "ffprobe",
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
        raise CaptionTranscriptionError(
            f"Could not determine source clip duration with ffprobe for {source_path}: "
            f"{_process_error_excerpt(exc)}"
        ) from exc

    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise CaptionTranscriptionError(
            f"ffprobe returned an invalid duration for {source_path}: "
            f"{completed.stdout.strip()!r}."
        ) from exc
    if duration <= 0:
        raise CaptionTranscriptionError(
            f"ffprobe returned a non-positive duration for {source_path}: {duration}."
        )
    return duration


def _transcription_request_data(model: str) -> list[tuple[str, str]]:
    model = model.strip()
    if _json_only_transcription_model(model):
        return [
            ("model", model),
            ("response_format", "json"),
        ]
    return [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
    ]


def _json_only_transcription_model(model: str) -> bool:
    if model.startswith("gpt-4o-transcribe-diarize"):
        return False
    return any(model.startswith(prefix) for prefix in GPT_TRANSCRIBE_JSON_MODELS)


def _decode_openai_transcription_response(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise CaptionTranscriptionError(
            "OpenAI transcription response was not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise CaptionTranscriptionError("OpenAI transcription response must be an object.")
    return payload


def _fallback_duration_seconds(
    payload: dict[str, Any],
    *,
    source_path: Path,
    duration_probe: DurationProbe | None,
) -> float | None:
    if "segments" in payload:
        return None
    if not isinstance(payload.get("text"), str) or not payload["text"].strip():
        return None
    probe = duration_probe or media_duration_seconds
    return probe(source_path)


def _parse_openai_transcription_segment(
    payload: Any,
    *,
    index: int,
) -> CaptionSegment | None:
    context = f"OpenAI transcription response.segments[{index}]"
    if not isinstance(payload, dict):
        raise CaptionTranscriptionError(f"{context} must be an object.")

    text = payload.get("text")
    if not isinstance(text, str):
        raise CaptionTranscriptionError(f"{context}.text must be a string.")
    if not text.strip():
        return None

    return CaptionSegment(
        start_time=required_number(
            payload,
            "start",
            context=context,
            error_cls=CaptionTranscriptionError,
        ),
        end_time=required_number(
            payload,
            "end",
            context=context,
            error_cls=CaptionTranscriptionError,
        ),
        text=text,
    )


def _parse_openai_transcription_words(
    payload: dict[str, Any],
) -> tuple[_OpenAITranscriptionWord, ...]:
    if "words" not in payload or payload["words"] is None:
        return ()

    words_payload = required_list(
        payload,
        "words",
        context="OpenAI transcription response",
        error_cls=CaptionTranscriptionError,
    )
    return tuple(
        word
        for index, word_payload in enumerate(words_payload)
        if (
            word := _parse_openai_transcription_word(
                word_payload,
                index=index,
            )
        )
        is not None
    )


def _parse_openai_transcription_word(
    payload: Any,
    *,
    index: int,
) -> _OpenAITranscriptionWord | None:
    context = f"OpenAI transcription response.words[{index}]"
    if not isinstance(payload, dict):
        raise CaptionTranscriptionError(f"{context} must be an object.")

    text = payload.get("word", payload.get("text"))
    if not isinstance(text, str):
        raise CaptionTranscriptionError(f"{context}.word must be a string.")
    if not text.strip():
        return None

    start_time = required_number(
        payload,
        "start",
        context=context,
        error_cls=CaptionTranscriptionError,
    )
    end_time = required_number(
        payload,
        "end",
        context=context,
        error_cls=CaptionTranscriptionError,
    )
    if end_time <= start_time:
        return None

    return _OpenAITranscriptionWord(
        start_time=start_time,
        end_time=end_time,
        text=text,
    )


def _segments_with_words(
    segments: tuple[CaptionSegment, ...],
    words: tuple[_OpenAITranscriptionWord, ...],
) -> tuple[CaptionSegment, ...]:
    sorted_segments = tuple(
        sorted(
            segments,
            key=lambda segment: (
                segment.start_time,
                segment.end_time,
                segment.text,
            ),
        )
    )
    segment_words: list[list[CaptionWord]] = [[] for _segment in sorted_segments]

    for word in words:
        segment_index = _word_segment_index(word, sorted_segments)
        if segment_index is None:
            continue

        segment = sorted_segments[segment_index]
        start_time = max(segment.start_time, word.start_time)
        end_time = min(segment.end_time, word.end_time)
        if end_time <= start_time:
            continue

        segment_words[segment_index].append(
            CaptionWord(
                start_time=start_time,
                end_time=end_time,
                text=word.text,
            )
        )

    return tuple(
        CaptionSegment(
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=segment.text,
            words=tuple(words_for_segment),
        )
        for segment, words_for_segment in zip(
            sorted_segments,
            segment_words,
            strict=True,
        )
    )


def _word_segment_index(
    word: _OpenAITranscriptionWord,
    segments: tuple[CaptionSegment, ...],
) -> int | None:
    best_index: int | None = None
    best_overlap = 0.0
    for index, segment in enumerate(segments):
        overlap = min(segment.end_time, word.end_time) - max(
            segment.start_time,
            word.start_time,
        )
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
    return best_index


def _openai_error_message(
    response: requests.Response,
    *,
    source_path: Path,
    api_key: str,
    request_id: str | None,
) -> str:
    excerpt = response_text_excerpt(response.text, secrets=(api_key,))
    message = (
        f"OpenAI transcription failed for {source_path} with "
        f"HTTP {response.status_code}"
    )
    if request_id:
        message = f"{message} (request_id={request_id})"
    if excerpt:
        message = f"{message}: {excerpt}"
    else:
        message = f"{message}."
    return message


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

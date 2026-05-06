from __future__ import annotations

from pathlib import Path

import pytest
import requests

from clipforge.clipr import (
    CLIPR_API_HOST,
    CliprAPIError,
    CliprClient,
    CliprDownloader,
    CliprError,
    CliprResponseError,
    extract_download_url,
    get_clip_download_url,
)
from clipforge.config import ClipforgeConfig


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.response


def test_client_requests_clip_by_twitch_slug_and_returns_download_url() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {"data": {"downloadUrl": "https://cdn.example.test/clip.mp4"}},
        )
    )
    client = CliprClient(api_key="test-key", session=session)

    download_url = client.get_download_url(
        "https://www.twitch.tv/example/clip/SeductiveKathishChipmunkDatBoi"
    )

    assert download_url == "https://cdn.example.test/clip.mp4"
    assert session.calls[0]["url"].endswith("/SeductiveKathishChipmunkDatBoi")
    assert session.calls[0]["headers"] == {
        "Content-Type": "application/json",
        "x-rapidapi-host": CLIPR_API_HOST,
        "x-rapidapi-key": "test-key",
    }


def test_get_clip_download_url_accepts_clips_twitch_url() -> None:
    session = FakeSession(
        FakeResponse(200, {"url": "https://cdn.example.test/source.mp4"})
    )

    download_url = get_clip_download_url(
        "https://clips.twitch.tv/SeductiveKathishChipmunkDatBoi",
        "test-key",
        session=session,
    )

    assert download_url == "https://cdn.example.test/source.mp4"


def test_client_can_be_created_from_config() -> None:
    config = ClipforgeConfig(clipr_api_key="test-key")

    client = CliprClient.from_config(config)

    assert client.api_key == "test-key"


def test_client_from_config_requires_clipr_api_key_only_for_clipr_use() -> None:
    with pytest.raises(CliprError, match="CLIPR_API_KEY"):
        CliprClient.from_config(ClipforgeConfig())


def test_clipr_downloader_resolves_and_downloads_with_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeClient:
        def get_download_url(self, twitch_clip_url: str) -> str:
            events.append(f"resolve:{twitch_clip_url}")
            return "https://cdn.example.test/source.mp4"

    def fake_download_clip(
        media_url: str,
        *,
        downloads_dir: Path,
        filename_stem: str | None,
    ) -> Path:
        events.append(f"download:{media_url}")
        assert downloads_dir == tmp_path
        assert filename_stem == "clip-123"
        return tmp_path / "clip-123.mp4"

    monkeypatch.setattr("clipforge.clipr.download_clip", fake_download_clip)
    downloader = CliprDownloader(client=FakeClient(), downloads_dir=tmp_path)

    result = downloader.download(
        "https://clips.twitch.tv/TallHelpfulClipKappa",
        clip_id="clip-123",
        on_media_url_resolved=lambda media_url: events.append(f"callback:{media_url}"),
    )

    assert result.source_path == tmp_path / "clip-123.mp4"
    assert result.backend == "clipr"
    assert result.media_url == "https://cdn.example.test/source.mp4"
    assert events == [
        "resolve:https://clips.twitch.tv/TallHelpfulClipKappa",
        "callback:https://cdn.example.test/source.mp4",
        "download:https://cdn.example.test/source.mp4",
    ]


def test_extract_download_url_finds_nested_media_url() -> None:
    payload = {
        "clip": {
            "assets": [
                {"type": "thumbnail", "url": "https://cdn.example.test/thumb.jpg"},
                {"type": "video", "mediaUrl": "https://cdn.example.test/video.mp4"},
            ]
        }
    }

    assert extract_download_url(payload) == "https://cdn.example.test/video.mp4"


def test_http_error_raises_clear_error_without_api_key() -> None:
    session = FakeSession(
        FakeResponse(401, {}, text="invalid key test-key for this request")
    )
    client = CliprClient(api_key="test-key", session=session)

    with pytest.raises(CliprAPIError) as exc_info:
        client.get_download_url("https://clips.twitch.tv/SeductiveKathishChipmunkDatBoi")

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "test-key" not in message
    assert "[redacted]" in message


def test_request_exception_raises_api_error() -> None:
    class RaisingSession:
        def get(self, *args: object, **kwargs: object) -> object:
            raise requests.Timeout("request timed out")

    client = CliprClient(api_key="test-key", session=RaisingSession())

    with pytest.raises(CliprAPIError, match="request timed out"):
        client.get_download_url("https://clips.twitch.tv/SeductiveKathishChipmunkDatBoi")


def test_malformed_json_raises_response_error() -> None:
    session = FakeSession(FakeResponse(200, ValueError("bad json"), text="not json"))
    client = CliprClient(api_key="test-key", session=session)

    with pytest.raises(CliprResponseError, match="malformed JSON"):
        client.get_download_url("https://clips.twitch.tv/SeductiveKathishChipmunkDatBoi")


def test_missing_download_url_raises_response_error() -> None:
    with pytest.raises(CliprResponseError, match="downloadable video URL"):
        extract_download_url({"clip": {"title": "No media here"}})


def test_invalid_twitch_clip_url_fails_before_api_call() -> None:
    session = FakeSession(
        FakeResponse(200, {"url": "https://cdn.example.test/source.mp4"})
    )
    client = CliprClient(api_key="test-key", session=session)

    with pytest.raises(ValueError, match="Unsupported Twitch clip URL"):
        client.get_download_url("https://example.com/not-a-twitch-clip")

    assert session.calls == []

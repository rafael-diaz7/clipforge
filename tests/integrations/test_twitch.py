from __future__ import annotations

from typing import Any

import pytest
import requests

from clipforge.config import ClipforgeConfig, ConfigError
from clipforge.twitch import (
    TwitchAPIError,
    TwitchClient,
    TwitchResponseError,
    list_channel_clips,
    twitch_channel_login_from_input,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object,
        *,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.posts.append({"url": url, **kwargs})
        return FakeResponse(200, {"access_token": "access-token", "expires_in": 3600})

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.gets.append({"url": url, **kwargs})
        if url.endswith("/users"):
            return FakeResponse(
                200,
                {
                    "data": [
                        {
                            "id": "1234",
                            "login": "example",
                            "display_name": "Example",
                        }
                    ]
                },
            )
        if url.endswith("/clips"):
            return FakeResponse(
                200,
                {
                    "data": [
                        {
                            "id": "clip-1",
                            "url": "https://clips.twitch.tv/clip-1",
                            "broadcaster_name": "Example",
                            "creator_name": "Viewer",
                            "title": "great clip",
                            "view_count": 42,
                            "created_at": "2026-05-01T00:00:00Z",
                            "duration": 28.5,
                            "thumbnail_url": "https://example.test/thumb.jpg",
                        }
                    ],
                    "pagination": {},
                },
            )
        raise AssertionError(f"unexpected URL: {url}")


def test_client_lists_channel_clips_with_filters() -> None:
    session = FakeSession()
    client = TwitchClient(
        client_id="client-id",
        client_secret="client-secret",
        session=session,  # type: ignore[arg-type]
    )

    clips = client.list_clips(
        channel_login="Example",
        limit=5,
        started_at="2026-05-01T00:00:00Z",
        ended_at="2026-05-06T00:00:00Z",
    )

    assert len(clips) == 1
    assert clips[0].url == "https://clips.twitch.tv/clip-1"
    assert clips[0].title == "great clip"
    assert session.posts[0]["data"] == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "grant_type": "client_credentials",
    }
    assert session.gets[0]["params"] == {"login": "example"}
    assert session.gets[1]["params"] == {
        "broadcaster_id": "1234",
        "first": "5",
        "started_at": "2026-05-01T00:00:00Z",
        "ended_at": "2026-05-06T00:00:00Z",
    }
    assert session.gets[1]["headers"] == {
        "Authorization": "Bearer access-token",
        "Client-Id": "client-id",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("EXAMPLE", "example"),
        ("https://www.twitch.tv/EXAMPLE", "example"),
        ("https://twitch.tv/example/RottenDogAppleBee12312213", "example"),
        ("twitch.tv/example/videos", "example"),
        ("https://m.twitch.tv/Example?desktop-redirect=true", "example"),
    ],
)
def test_twitch_channel_login_from_input_accepts_logins_and_channel_urls(
    value: str,
    expected: str,
) -> None:
    assert twitch_channel_login_from_input(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "https://clips.twitch.tv/RottenDogAppleBee12312213",
        "https://www.twitch.tv/directory/game/Just%20Chatting",
        "https://www.twitch.tv/videos/1234",
        "https://example.com/example",
    ],
)
def test_twitch_channel_login_from_input_rejects_non_channel_urls(value: str) -> None:
    with pytest.raises(TwitchResponseError):
        twitch_channel_login_from_input(value)


def test_client_reuses_app_access_token_for_multiple_helix_calls() -> None:
    session = FakeSession()
    client = TwitchClient(
        client_id="client-id",
        client_secret="client-secret",
        session=session,  # type: ignore[arg-type]
    )

    client.list_clips(channel_login="example", limit=1)

    assert len(session.posts) == 1
    assert len(session.gets) == 2


def test_client_reports_missing_channel() -> None:
    class MissingUserSession(FakeSession):
        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            self.gets.append({"url": url, **kwargs})
            return FakeResponse(200, {"data": []})

    client = TwitchClient(
        client_id="client-id",
        client_secret="client-secret",
        session=MissingUserSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(TwitchResponseError, match="Twitch channel not found"):
        client.list_clips(channel_login="missing")


def test_client_rejects_limit_above_twitch_page_size() -> None:
    client = TwitchClient(client_id="client-id", client_secret="client-secret")

    with pytest.raises(TwitchResponseError, match="100 or less"):
        client.list_clips(channel_login="example", limit=101)


def test_api_errors_do_not_expose_credentials() -> None:
    class ErrorSession(FakeSession):
        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            self.posts.append({"url": url, **kwargs})
            return FakeResponse(
                401,
                {"message": "bad secret"},
                text="bad client-id client-secret",
            )

    client = TwitchClient(
        client_id="client-id",
        client_secret="client-secret",
        session=ErrorSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(TwitchAPIError) as exc_info:
        client.list_clips(channel_login="example")

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "client-secret" not in message
    assert "client-id" not in message
    assert "[redacted]" in message


def test_request_exceptions_are_wrapped() -> None:
    class RaisingSession(FakeSession):
        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            raise requests.RequestException("network down")

    client = TwitchClient(
        client_id="client-id",
        client_secret="client-secret",
        session=RaisingSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(TwitchAPIError, match="Twitch auth request failed"):
        client.list_clips(channel_login="example")


def test_list_channel_clips_uses_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[ClipforgeConfig] = []

    class FakeClient:
        @classmethod
        def from_config(cls, config: ClipforgeConfig) -> "FakeClient":
            calls.append(config)
            return cls()

        def list_clips(
            self,
            *,
            channel_login: str,
            limit: int,
            started_at: str | None,
            ended_at: str | None,
        ) -> tuple[object, ...]:
            assert channel_login == "example"
            assert limit == 3
            assert started_at is None
            assert ended_at is None
            return ("clip",)

    monkeypatch.setattr("clipforge.twitch.TwitchClient", FakeClient)
    config = ClipforgeConfig(
        twitch_client_id="client-id",
        twitch_client_secret="client-secret",
    )

    clips = list_channel_clips("example", limit=3, config=config)

    assert clips == ("clip",)
    assert calls == [config]


def test_client_from_config_requires_twitch_credentials() -> None:
    with pytest.raises(ConfigError, match="TWITCH_CLIENT_ID"):
        TwitchClient.from_config(ClipforgeConfig())

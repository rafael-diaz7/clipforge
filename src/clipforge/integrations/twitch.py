"""Twitch Helix client helpers for clip discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

from clipforge.core.config import ClipforgeConfig
from clipforge.core.utils import normalized_host, response_text_excerpt
from clipforge.json_validation import required_int
from clipforge.json_validation import required_number
from clipforge.json_validation import required_string


TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_API_BASE_URL = "https://api.twitch.tv/helix"
DEFAULT_CLIP_LIMIT = 10
MAX_CLIP_LIMIT = 100
_TWITCH_LOGIN_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,25}$")
_NON_CHANNEL_TWITCH_PATHS = frozenset(
    {
        "about",
        "activate",
        "bits",
        "clip",
        "creatorcamp",
        "directory",
        "downloads",
        "drops",
        "jobs",
        "p",
        "popout",
        "products",
        "search",
        "settings",
        "subscriptions",
        "teams",
        "turbo",
        "videos",
    }
)


class TwitchError(RuntimeError):
    """Base error for Twitch discovery failures."""


class TwitchAPIError(TwitchError):
    """Raised when Twitch returns an error or cannot be reached."""


class TwitchResponseError(TwitchError):
    """Raised when Twitch returns an unexpected response shape."""


@dataclass(frozen=True)
class TwitchUser:
    id: str
    login: str
    display_name: str


@dataclass(frozen=True)
class TwitchClip:
    id: str
    url: str
    broadcaster_name: str
    creator_name: str
    title: str
    view_count: int
    created_at: str
    duration: float
    thumbnail_url: str


class TwitchClient:
    """Small Twitch Helix client focused on discovering clips."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        session: requests.Session | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = session or requests.Session()
        self._access_token: str | None = None

    @classmethod
    def from_config(cls, config: ClipforgeConfig) -> "TwitchClient":
        client_id, client_secret = config.require_twitch_credentials()
        return cls(client_id=client_id, client_secret=client_secret)

    def get_user_by_login(self, login: str) -> TwitchUser:
        clean_login = twitch_channel_login_from_input(login)
        if not clean_login:
            raise TwitchResponseError("Twitch channel login is required.")

        payload = self._helix_get("users", params={"login": clean_login})
        data = _response_data(payload, context=f"Twitch user {clean_login!r}")
        if not data:
            raise TwitchResponseError(f"Twitch channel not found: {clean_login}.")

        user = data[0]
        return TwitchUser(
            id=required_string(user, "id", context="Twitch user", error_cls=TwitchResponseError),
            login=required_string(
                user,
                "login",
                context="Twitch user",
                error_cls=TwitchResponseError,
            ),
            display_name=required_string(
                user,
                "display_name",
                context="Twitch user",
                error_cls=TwitchResponseError,
            ),
        )

    def list_clips(
        self,
        *,
        channel_login: str,
        limit: int = DEFAULT_CLIP_LIMIT,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> tuple[TwitchClip, ...]:
        if limit < 1:
            raise TwitchResponseError("Clip limit must be at least 1.")
        if limit > MAX_CLIP_LIMIT:
            raise TwitchResponseError(f"Clip limit must be {MAX_CLIP_LIMIT} or less.")

        user = self.get_user_by_login(channel_login)
        params = {
            "broadcaster_id": user.id,
            "first": str(limit),
        }
        if started_at:
            params["started_at"] = started_at
        if ended_at:
            params["ended_at"] = ended_at

        payload = self._helix_get("clips", params=params)
        return tuple(_parse_clip(item) for item in _response_data(payload, context="Twitch clips"))

    def _get_app_access_token(self) -> str:
        if self._access_token is not None:
            return self._access_token

        try:
            response = self.session.post(
                TWITCH_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                },
                timeout=15,
            )
        except requests.RequestException as exc:
            raise TwitchAPIError(f"Twitch auth request failed: {exc}") from exc

        if response.status_code >= 400:
            raise TwitchAPIError(
                _http_error_message(
                    response,
                    "Twitch auth request failed",
                    secrets=(self.client_id, self.client_secret),
                )
            )

        payload = _decode_json(response, context="Twitch auth response")
        token = required_string(
            payload,
            "access_token",
            context="Twitch auth response",
            error_cls=TwitchResponseError,
        )
        self._access_token = token
        return token

    def _helix_get(self, path: str, *, params: dict[str, str]) -> dict[str, Any]:
        token = self._get_app_access_token()
        try:
            response = self.session.get(
                f"{TWITCH_API_BASE_URL}/{path}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Client-Id": self.client_id,
                },
                params=params,
                timeout=15,
            )
        except requests.RequestException as exc:
            raise TwitchAPIError(f"Twitch API request failed: {exc}") from exc

        if response.status_code >= 400:
            raise TwitchAPIError(
                _http_error_message(
                    response,
                    "Twitch API request failed",
                    secrets=(self.client_id, token),
                )
            )

        return _decode_json(response, context="Twitch API response")


def list_channel_clips(
    channel_login: str,
    *,
    limit: int = DEFAULT_CLIP_LIMIT,
    started_at: str | None = None,
    ended_at: str | None = None,
    config: ClipforgeConfig | None = None,
) -> tuple[TwitchClip, ...]:
    """List clips for a Twitch channel without downloading or rendering them."""

    from clipforge.core.config import load_config

    client = TwitchClient.from_config(config or load_config())
    return client.list_clips(
        channel_login=channel_login,
        limit=limit,
        started_at=started_at,
        ended_at=ended_at,
    )


def twitch_channel_login_from_input(value: str) -> str:
    """Normalize a Twitch channel login or URL to a lowercase channel login."""

    candidate = value.strip()
    if not candidate:
        raise TwitchResponseError("Twitch channel login is required.")

    parsed = _parse_possible_twitch_url(candidate)
    if parsed.netloc:
        host = normalized_host(parsed.netloc)
        if host == "clips.twitch.tv":
            raise TwitchResponseError(
                "Twitch clip URLs do not include a channel login. "
                "Use a Twitch channel URL such as twitch.tv/<channel>."
            )
        if host != "twitch.tv":
            raise TwitchResponseError(f"Unsupported Twitch channel URL host: {parsed.netloc}.")

        path_parts = [part for part in parsed.path.split("/") if part]
        if not path_parts:
            raise TwitchResponseError("Twitch channel URL must include a channel login.")

        first_part = path_parts[0]
        if first_part.lower() in _NON_CHANNEL_TWITCH_PATHS:
            raise TwitchResponseError(
                f"Twitch URL does not point to a channel: {parsed.geturl()}."
            )
        candidate = first_part

    if not _TWITCH_LOGIN_PATTERN.fullmatch(candidate):
        raise TwitchResponseError(
            "Twitch channel login must contain only letters, numbers, and underscores."
        )
    return candidate.lower()


def _parse_possible_twitch_url(value: str):
    parsed = urlparse(value)
    if parsed.netloc:
        return parsed
    if value.lower().startswith(("twitch.tv/", "www.twitch.tv/", "m.twitch.tv/")):
        return urlparse(f"https://{value}")
    return parsed


def _parse_clip(payload: Any) -> TwitchClip:
    if not isinstance(payload, dict):
        raise TwitchResponseError("Twitch clips response contained a non-object item.")

    return TwitchClip(
        id=required_string(payload, "id", context="Twitch clip", error_cls=TwitchResponseError),
        url=required_string(payload, "url", context="Twitch clip", error_cls=TwitchResponseError),
        broadcaster_name=required_string(
            payload,
            "broadcaster_name",
            context="Twitch clip",
            error_cls=TwitchResponseError,
        ),
        creator_name=required_string(
            payload,
            "creator_name",
            context="Twitch clip",
            error_cls=TwitchResponseError,
        ),
        title=str(payload.get("title") or ""),
        view_count=required_int(
            payload,
            "view_count",
            context="Twitch clip",
            error_cls=TwitchResponseError,
        ),
        created_at=required_string(
            payload,
            "created_at",
            context="Twitch clip",
            error_cls=TwitchResponseError,
        ),
        duration=required_number(
            payload,
            "duration",
            context="Twitch clip",
            error_cls=TwitchResponseError,
        ),
        thumbnail_url=str(payload.get("thumbnail_url") or ""),
    )


def _response_data(payload: Any, *, context: str) -> list[Any]:
    if not isinstance(payload, dict):
        raise TwitchResponseError(f"{context} response was not a JSON object.")
    data = payload.get("data")
    if not isinstance(data, list):
        raise TwitchResponseError(f"{context} response did not include a data list.")
    return data


def _decode_json(response: requests.Response, *, context: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise TwitchResponseError(f"{context} was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise TwitchResponseError(f"{context} was not a JSON object.")
    return payload


def _http_error_message(
    response: requests.Response,
    prefix: str,
    *,
    secrets: tuple[str, ...],
) -> str:
    excerpt = response_text_excerpt(response.text, secrets=secrets)
    if excerpt:
        return f"{prefix}: HTTP {response.status_code}: {excerpt}"
    return f"{prefix}: HTTP {response.status_code}."

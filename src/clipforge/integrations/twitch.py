"""Twitch Helix client helpers for clip discovery."""

from __future__ import annotations

from email.utils import parsedate_to_datetime
import re
import time
from dataclasses import dataclass
from datetime import timezone
from typing import Any
from urllib.parse import urlparse

import requests

from clipforge.core.config import ClipforgeConfig
from clipforge.integrations.retry import RetryDecision, RetryPolicy, retry_call
from clipforge.utils.json_validation import required_int
from clipforge.utils.json_validation import required_number
from clipforge.utils.json_validation import required_string
from clipforge.utils.paths import normalized_host, response_text_excerpt


TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_API_BASE_URL = "https://api.twitch.tv/helix"
DEFAULT_CLIP_LIMIT = 10
MAX_CLIP_LIMIT = 100
DEFAULT_TWITCH_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=1.0,
    max_delay_seconds=30.0,
    jitter_seconds=0.25,
)
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


class _TwitchHTTPStatusError(TwitchAPIError):
    def __init__(
        self,
        *,
        response: requests.Response,
        prefix: str,
        secrets: tuple[str, ...],
    ) -> None:
        self.response = response
        self.status_code = response.status_code
        super().__init__(_http_error_message(response, prefix, secrets=secrets))


def classify_twitch_retry_error(exc: BaseException) -> RetryDecision:
    """Classify Twitch integration errors for retry."""

    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return RetryDecision(retryable=True, reason="transient network error")

    response = getattr(exc, "response", None)
    status_code = _status_code_from_error(exc)
    if status_code is not None:
        delay_override = _twitch_delay_override_seconds(response, status_code=status_code)
        if status_code in {408, 429} or status_code >= 500:
            return RetryDecision(
                retryable=True,
                reason=f"retryable HTTP status {status_code}",
                delay_override_seconds=delay_override,
            )
        return RetryDecision(
            retryable=False,
            reason=f"non-retryable HTTP status {status_code}",
        )

    return RetryDecision(retryable=False, reason="not classified as retryable")


def _status_code_from_error(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


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
        retry_policy: RetryPolicy = DEFAULT_TWITCH_RETRY_POLICY,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = session or requests.Session()
        self.retry_policy = retry_policy
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
            response = retry_call(
                operation_name="auth request",
                provider="Twitch",
                operation=self._post_app_access_token_request,
                policy=self.retry_policy,
                classify_error=classify_twitch_retry_error,
            )
        except TwitchAPIError:
            raise
        except requests.RequestException as exc:
            raise TwitchAPIError(f"Twitch auth request failed: {exc}") from exc

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
            response = retry_call(
                operation_name=f"Helix {path} request",
                provider="Twitch",
                operation=lambda: self._get_helix_response(
                    path,
                    params=params,
                    token=token,
                ),
                policy=self.retry_policy,
                classify_error=classify_twitch_retry_error,
            )
        except TwitchAPIError:
            raise
        except requests.RequestException as exc:
            raise TwitchAPIError(f"Twitch API request failed: {exc}") from exc

        return _decode_json(response, context="Twitch API response")

    def _post_app_access_token_request(self) -> requests.Response:
        response = self.session.post(
            TWITCH_TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
            timeout=15,
        )
        if response.status_code >= 400:
            raise _TwitchHTTPStatusError(
                response=response,
                prefix="Twitch auth request failed",
                secrets=(self.client_id, self.client_secret),
            )
        return response

    def _get_helix_response(
        self,
        path: str,
        *,
        params: dict[str, str],
        token: str,
    ) -> requests.Response:
        response = self.session.get(
            f"{TWITCH_API_BASE_URL}/{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Client-Id": self.client_id,
            },
            params=params,
            timeout=15,
        )
        if response.status_code >= 400:
            raise _TwitchHTTPStatusError(
                response=response,
                prefix="Twitch API request failed",
                secrets=(self.client_id, token),
            )
        return response


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


def _twitch_delay_override_seconds(
    response: Any,
    *,
    status_code: int,
) -> float | None:
    if response is None:
        return None

    retry_after = _header_value(response, "Retry-After")
    if retry_after is not None:
        return _parse_retry_after_seconds(retry_after)

    if status_code == 429:
        ratelimit_reset = _header_value(response, "Ratelimit-Reset")
        if ratelimit_reset is not None:
            return _parse_ratelimit_reset_seconds(ratelimit_reset)
    return None


def _header_value(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    value = headers.get(name)
    if value is not None:
        return str(value)

    lower_name = name.lower()
    for header_name, header_value in headers.items():
        if str(header_name).lower() == lower_name:
            return str(header_value)
    return None


def _parse_retry_after_seconds(value: str) -> float | None:
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        reset_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    return max(0.0, reset_at.timestamp() - time.time())


def _parse_ratelimit_reset_seconds(value: str) -> float | None:
    try:
        reset_epoch_seconds = float(value)
    except ValueError:
        return None
    return max(0.0, reset_epoch_seconds - time.time())


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

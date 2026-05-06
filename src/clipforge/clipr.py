"""Clipr API client for resolving Twitch clips to downloadable media URLs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from clipforge.config import ClipforgeConfig
from clipforge.download import DownloadResult, backend_download_dir, download_clip
from clipforge.utils import twitch_clip_slug_from_url


CLIPR_API_HOST = "clipr.p.rapidapi.com"
CLIPR_API_BASE_URL = f"https://{CLIPR_API_HOST}/api/v1/clips"
DEFAULT_TIMEOUT_SECONDS = 30

_DOWNLOAD_URL_KEYS = (
    "download_url",
    "downloadUrl",
    "download",
    "video_url",
    "videoUrl",
    "media_url",
    "mediaUrl",
    "source_url",
    "sourceUrl",
    "source",
    "mp4",
    "url",
)


class CliprError(RuntimeError):
    """Base error raised by Clipr API helpers."""


class CliprAPIError(CliprError):
    """Raised when Clipr returns an HTTP error or cannot be reached."""


class CliprResponseError(CliprError):
    """Raised when Clipr returns malformed data or no downloadable URL."""


@dataclass(frozen=True)
class CliprClient:
    """Small Clipr API wrapper that does not download video bytes."""

    api_key: str
    base_url: str = CLIPR_API_BASE_URL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    session: requests.Session | None = None

    @classmethod
    def from_config(cls, config: ClipforgeConfig) -> "CliprClient":
        return cls(api_key=_require_clipr_api_key(config))

    def get_download_url(self, twitch_clip_url: str) -> str:
        """Return a direct downloadable media URL for a Twitch clip URL."""

        slug = twitch_clip_slug_from_url(twitch_clip_url)
        response = self._get(slug)
        payload = self._decode_json(response)
        return extract_download_url(payload)

    def _get(self, clip_slug: str) -> requests.Response:
        client = self.session or requests
        url = f"{self.base_url.rstrip('/')}/{clip_slug}"
        headers = {
            "Content-Type": "application/json",
            "x-rapidapi-host": CLIPR_API_HOST,
            "x-rapidapi-key": self.api_key,
        }

        try:
            response = client.get(url, headers=headers, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise CliprAPIError(f"Clipr request failed for clip '{clip_slug}': {exc}") from exc

        if response.status_code >= 400:
            raise CliprAPIError(
                f"Clipr request failed for clip '{clip_slug}' with "
                f"HTTP {response.status_code}: "
                f"{_response_excerpt(response, secret=self.api_key)}"
            )

        return response

    @staticmethod
    def _decode_json(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise CliprResponseError("Clipr returned a malformed JSON response.") from exc


@dataclass(frozen=True)
class CliprDownloader:
    """Clipr-backed downloader that resolves and downloads Twitch clips."""

    client: CliprClient
    downloads_dir: Path
    backend_name: str = "clipr"

    @classmethod
    def from_config(cls, config: ClipforgeConfig) -> "CliprDownloader":
        return cls(
            client=CliprClient.from_config(config),
            downloads_dir=config.downloads_dir,
        )

    def download(
        self,
        twitch_clip_url: str,
        *,
        clip_id: str | None = None,
        on_media_url_resolved: Callable[[str], None] | None = None,
    ) -> DownloadResult:
        clip_slug = clip_id or twitch_clip_slug_from_url(twitch_clip_url)
        media_url = self.client.get_download_url(twitch_clip_url)
        if on_media_url_resolved is not None:
            on_media_url_resolved(media_url)
        source_path = download_clip(
            media_url,
            downloads_dir=backend_download_dir(
                self.downloads_dir,
                clip_id=clip_slug,
                backend=self.backend_name,
            ),
            filename_stem=clip_slug,
        )
        return DownloadResult(
            source_path=source_path,
            backend=self.backend_name,
            media_url=media_url,
        )


def get_clip_download_url(
    twitch_clip_url: str,
    api_key: str,
    *,
    session: requests.Session | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Resolve a Twitch clip URL to a downloadable media URL."""

    client = CliprClient(
        api_key=api_key,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    return client.get_download_url(twitch_clip_url)


def _require_clipr_api_key(config: ClipforgeConfig) -> str:
    if not config.clipr_api_key:
        raise CliprError(
            "Missing required configuration for Clipr downloader: CLIPR_API_KEY. "
            "Set it in your environment or in a local .env file."
        )
    return config.clipr_api_key


def extract_download_url(payload: Any) -> str:
    """Extract a downloadable media URL from a Clipr JSON payload."""

    for key in _DOWNLOAD_URL_KEYS:
        found = _find_url_for_key(payload, key)
        if found:
            return found

    found = _find_media_url(payload)
    if found:
        return found

    raise CliprResponseError("Clipr response did not include a downloadable video URL.")


def _find_url_for_key(value: Any, key: str) -> str | None:
    if isinstance(value, dict):
        for candidate_key, candidate_value in value.items():
            if candidate_key == key:
                found = _url_from_value(candidate_value)
                if found:
                    return found

        for candidate_value in value.values():
            found = _find_url_for_key(candidate_value, key)
            if found:
                return found

    if isinstance(value, list):
        for item in value:
            found = _find_url_for_key(item, key)
            if found:
                return found

    return None


def _find_media_url(value: Any) -> str | None:
    if isinstance(value, str) and _looks_like_media_url(value):
        return value

    if isinstance(value, dict):
        for candidate_value in value.values():
            found = _find_media_url(candidate_value)
            if found:
                return found

    if isinstance(value, list):
        for item in value:
            found = _find_media_url(item)
            if found:
                return found

    return None


def _url_from_value(value: Any) -> str | None:
    if isinstance(value, str) and _is_http_url(value):
        return value

    if isinstance(value, dict) or isinstance(value, list):
        return _find_media_url(value)

    return None


def _looks_like_media_url(value: str) -> bool:
    parsed = urlparse(value)
    path = parsed.path.lower()
    return _is_http_url(value) and (
        path.endswith(".mp4") or path.endswith(".m3u8") or "/video" in path
    )


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _response_excerpt(
    response: requests.Response,
    *,
    secret: str | None = None,
    limit: int = 240,
) -> str:
    text = response.text.strip().replace("\n", " ")
    if secret:
        text = text.replace(secret, "[redacted]")
    if not text:
        return "empty response body"
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text

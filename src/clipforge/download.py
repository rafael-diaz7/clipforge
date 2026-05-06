"""Local downloader for direct media URLs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import unquote, urlparse

import requests

from clipforge.config import DOWNLOADS_DIR, ClipforgeConfig, ConfigError
from clipforge.utils import ensure_directory, safe_filename


DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_EXTENSION = ".mp4"


class DownloadError(RuntimeError):
    """Raised when a clip cannot be downloaded safely."""


@dataclass(frozen=True)
class DownloadResult:
    """Result returned by a Twitch clip downloader backend."""

    source_path: Path
    backend: str
    media_url: str | None = None


class ClipDownloader(Protocol):
    """Downloader backend that turns a Twitch clip URL into a local media file."""

    backend_name: str

    def download(
        self,
        twitch_clip_url: str,
        *,
        clip_id: str | None = None,
        on_media_url_resolved: Callable[[str], None] | None = None,
    ) -> DownloadResult:
        """Download a Twitch clip URL and return the local output path."""


def create_downloader(config: ClipforgeConfig) -> ClipDownloader:
    """Create the configured Twitch clip downloader backend."""

    backend = config.require_downloader_backend()
    if backend == "clipr":
        from clipforge.clipr import CliprDownloader

        return CliprDownloader.from_config(config)

    raise ConfigError(f"Unsupported downloader backend: {backend}")


def download_twitch_clip(
    twitch_clip_url: str,
    *,
    clip_id: str | None = None,
    on_media_url_resolved: Callable[[str], None] | None = None,
    config: ClipforgeConfig,
) -> DownloadResult:
    """Download a Twitch clip URL with the configured backend."""

    return create_downloader(config).download(
        twitch_clip_url,
        clip_id=clip_id,
        on_media_url_resolved=on_media_url_resolved,
    )


def download_clip(
    media_url: str,
    *,
    downloads_dir: Path = DOWNLOADS_DIR,
    filename_stem: str | None = None,
    session: requests.Session | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> Path:
    """Stream a direct media URL into the local downloads directory."""

    _validate_media_url(media_url)
    downloads_dir = ensure_directory(downloads_dir)
    output_path = downloads_dir / _download_filename(media_url, filename_stem=filename_stem)
    partial_path = output_path.with_name(f"{output_path.name}.part")
    client = session or requests

    try:
        response = client.get(media_url, stream=True, timeout=timeout_seconds)
        try:
            _raise_for_status(response, media_url)
            bytes_written = _write_stream(response, partial_path, chunk_size=chunk_size)
            _verify_complete(response, bytes_written, media_url)
        finally:
            close = getattr(response, "close", None)
            if close:
                close()

        partial_path.replace(output_path)
        return output_path
    except requests.RequestException as exc:
        _remove_partial(partial_path)
        raise DownloadError(f"Download failed for {media_url}: {exc}") from exc
    except OSError as exc:
        _remove_partial(partial_path)
        raise DownloadError(f"Could not save download to {output_path}: {exc}") from exc
    except DownloadError:
        _remove_partial(partial_path)
        raise


def _validate_media_url(media_url: str) -> None:
    parsed = urlparse(media_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError("Media URL must be an http or https URL.")


def _download_filename(media_url: str, *, filename_stem: str | None = None) -> str:
    parsed = urlparse(media_url)
    path_name = unquote(Path(parsed.path).name)
    suffix = _safe_extension(Path(path_name).suffix)

    if filename_stem is not None:
        stem = safe_filename(filename_stem)
    else:
        stem = safe_filename(Path(path_name).stem, fallback="clip")

    return f"{stem}{suffix}"


def _safe_extension(extension: str) -> str:
    extension = extension.lower()
    if (
        extension.startswith(".")
        and 2 <= len(extension) <= 10
        and extension[1:].replace("-", "").replace("_", "").isalnum()
    ):
        return extension
    return DEFAULT_EXTENSION


def _raise_for_status(response: requests.Response, media_url: str) -> None:
    status_code = getattr(response, "status_code", 0)
    if status_code >= 400:
        raise DownloadError(f"Download failed for {media_url} with HTTP {status_code}.")


def _write_stream(
    response: requests.Response,
    partial_path: Path,
    *,
    chunk_size: int,
) -> int:
    bytes_written = 0
    with partial_path.open("wb") as output:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            output.write(chunk)
            bytes_written += len(chunk)

    if bytes_written == 0:
        raise DownloadError("Download did not contain any media bytes.")

    return bytes_written


def _verify_complete(
    response: requests.Response,
    bytes_written: int,
    media_url: str,
) -> None:
    content_length = response.headers.get("content-length")
    if content_length is None:
        return

    try:
        expected_bytes = int(content_length)
    except ValueError:
        return

    if expected_bytes != bytes_written:
        raise DownloadError(
            f"Incomplete download for {media_url}: expected "
            f"{expected_bytes} bytes, wrote {bytes_written} bytes."
        )


def _remove_partial(partial_path: Path) -> None:
    try:
        partial_path.unlink(missing_ok=True)
    except OSError:
        pass

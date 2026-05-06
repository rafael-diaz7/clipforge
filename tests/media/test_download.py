from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys

import pytest
import requests

from clipforge.clipr import CliprDownloader
from clipforge.config import ClipforgeConfig
from clipforge.download import DownloadError, YtDlpDownloader, create_downloader, download_clip
from tests.constants import TWITCH_CLIP_SLUG, TWITCH_CLIP_URL


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        chunks: list[bytes],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.chunks = chunks
        self.headers = headers or {}
        self.closed = False
        self.chunk_sizes: list[int] = []

    def iter_content(self, chunk_size: int) -> list[bytes]:
        self.chunk_sizes.append(chunk_size)
        return self.chunks

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        stream: bool,
        timeout: int,
    ) -> FakeResponse:
        self.calls.append({"url": url, "stream": stream, "timeout": timeout})
        return self.response


def test_download_clip_streams_media_to_downloads_dir(tmp_path: Path) -> None:
    response = FakeResponse(
        200,
        [b"first", b"", b"second"],
        headers={"content-length": "11"},
    )
    session = FakeSession(response)

    output_path = download_clip(
        "https://cdn.example.test/videos/source.mp4",
        downloads_dir=tmp_path,
        session=session,
        chunk_size=4,
    )

    assert output_path == tmp_path / "source.mp4"
    assert output_path.read_bytes() == b"firstsecond"
    assert session.calls == [
        {
            "url": "https://cdn.example.test/videos/source.mp4",
            "stream": True,
            "timeout": 60,
        }
    ]
    assert response.chunk_sizes == [4]
    assert response.closed
    assert not (tmp_path / "source.mp4.part").exists()


def test_download_clip_uses_safe_filename_stem(tmp_path: Path) -> None:
    response = FakeResponse(200, [b"video"], headers={"content-length": "5"})
    session = FakeSession(response)

    output_path = download_clip(
        "https://cdn.example.test/videos/source.mp4?token=abc",
        downloads_dir=tmp_path,
        filename_stem=" My Clip: wow!? ",
        session=session,
    )

    assert output_path == tmp_path / "My_Clip_wow.mp4"
    assert output_path.read_bytes() == b"video"


def test_download_clip_defaults_to_mp4_for_url_without_safe_extension(
    tmp_path: Path,
) -> None:
    response = FakeResponse(200, [b"video"])
    session = FakeSession(response)

    output_path = download_clip(
        "https://cdn.example.test/?id=123",
        downloads_dir=tmp_path,
        session=session,
    )

    assert output_path == tmp_path / "clip.mp4"


def test_download_clip_raises_clear_error_for_http_failure(tmp_path: Path) -> None:
    response = FakeResponse(404, [b"not found"])
    session = FakeSession(response)

    with pytest.raises(DownloadError, match="HTTP 404"):
        download_clip(
            "https://cdn.example.test/missing.mp4",
            downloads_dir=tmp_path,
            session=session,
        )

    assert response.closed
    assert not (tmp_path / "missing.mp4.part").exists()


def test_download_clip_cleans_up_incomplete_download(tmp_path: Path) -> None:
    response = FakeResponse(200, [b"short"], headers={"content-length": "10"})
    session = FakeSession(response)

    with pytest.raises(DownloadError, match="Incomplete download"):
        download_clip(
            "https://cdn.example.test/source.mp4",
            downloads_dir=tmp_path,
            session=session,
        )

    assert not (tmp_path / "source.mp4").exists()
    assert not (tmp_path / "source.mp4.part").exists()


def test_download_clip_wraps_request_exceptions(tmp_path: Path) -> None:
    class RaisingSession:
        def get(self, *args: object, **kwargs: object) -> object:
            raise requests.Timeout("request timed out")

    with pytest.raises(DownloadError, match="request timed out"):
        download_clip(
            "https://cdn.example.test/source.mp4",
            downloads_dir=tmp_path,
            session=RaisingSession(),
        )


def test_download_clip_rejects_non_http_urls(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="http or https"):
        download_clip("file:///tmp/source.mp4", downloads_dir=tmp_path)


def test_create_downloader_selects_clipr_backend(tmp_path: Path) -> None:
    config = ClipforgeConfig(
        clipr_api_key="test-key",
        downloader_backend="clipr",
        downloads_dir=tmp_path,
    )

    downloader = create_downloader(config)

    assert isinstance(downloader, CliprDownloader)
    assert downloader.backend_name == "clipr"
    assert downloader.downloads_dir == tmp_path


def test_create_downloader_selects_ytdlp_backend(tmp_path: Path) -> None:
    config = ClipforgeConfig(
        downloader_backend="ytdlp",
        downloads_dir=tmp_path,
    )

    downloader = create_downloader(config)

    assert isinstance(downloader, YtDlpDownloader)
    assert downloader.backend_name == "ytdlp"
    assert downloader.downloads_dir == tmp_path


def test_ytdlp_downloader_builds_command_and_returns_output_path(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="clipforge.download")
    output_path = (
        tmp_path
        / "downloads"
        / TWITCH_CLIP_SLUG
        / "ytdlp"
        / f"{TWITCH_CLIP_SLUG}.mp4"
    )
    calls: list[dict[str, object]] = []

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "command": command,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        output_path.write_bytes(b"video")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{output_path}\n",
            stderr="",
        )

    downloader = YtDlpDownloader(
        downloads_dir=tmp_path / "downloads",
        runner=fake_runner,
        module_resolver=lambda module: object(),
    )

    result = downloader.download(
        TWITCH_CLIP_URL,
        clip_id=TWITCH_CLIP_SLUG,
    )

    assert result.source_path == output_path
    assert result.backend == "ytdlp"
    assert result.media_url is None
    assert calls == [
        {
            "command": [
                sys.executable,
                "-m",
                "yt_dlp",
                "--quiet",
                "--no-warnings",
                "--no-playlist",
                "--paths",
                str(output_path.parent),
                "--output",
                f"{TWITCH_CLIP_SLUG}.%(ext)s",
                "--print",
                "after_move:filepath",
                TWITCH_CLIP_URL,
            ],
            "check": True,
            "capture_output": True,
            "text": True,
        }
    ]
    assert f"Starting yt-dlp processing for {TWITCH_CLIP_URL}." in caplog.text
    assert f"Starting yt-dlp download to {output_path.parent}." in caplog.text


def test_ytdlp_downloader_uses_safe_clip_id_for_output_template(tmp_path: Path) -> None:
    output_path = tmp_path / "downloads" / "My_Clip" / "ytdlp" / "My_Clip.mp4"
    commands: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output_path.write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, stdout=f"{output_path}\n")

    downloader = YtDlpDownloader(
        downloads_dir=tmp_path / "downloads",
        runner=fake_runner,
        module_resolver=lambda module: object(),
    )

    downloader.download(
        TWITCH_CLIP_URL,
        clip_id=" My Clip!? ",
    )

    assert commands[0][9] == "My_Clip.%(ext)s"


def test_ytdlp_downloader_raises_clear_error_when_missing(tmp_path: Path) -> None:
    downloader = YtDlpDownloader(
        downloads_dir=tmp_path,
        module_resolver=lambda module: None,
    )

    with pytest.raises(DownloadError, match="yt-dlp Python package"):
        downloader.download(TWITCH_CLIP_URL)


def test_ytdlp_downloader_wraps_process_failures(tmp_path: Path) -> None:
    def fake_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="Unsupported URL",
        )

    downloader = YtDlpDownloader(
        downloads_dir=tmp_path,
        runner=fake_runner,
        module_resolver=lambda module: object(),
    )

    with pytest.raises(DownloadError, match="Unsupported URL"):
        downloader.download(TWITCH_CLIP_URL)

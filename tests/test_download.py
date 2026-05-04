from __future__ import annotations

from pathlib import Path

import pytest
import requests

from clipforge.download import DownloadError, download_clip


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

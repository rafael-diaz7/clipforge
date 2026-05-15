from __future__ import annotations

from pathlib import Path

from clipforge.core.config import ClipforgeConfig
from clipforge.storage.paths import (
    backend_download_dir,
    clip_folder_name,
    download_dir,
    export_path,
    ready_export_filename,
    ready_export_path,
    render_path,
    sanitize_path_part,
)


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(
        downloads_dir=tmp_path / "downloads",
        renders_dir=tmp_path / "renders",
        exports_dir=tmp_path / "exports",
    )


def test_render_path_includes_streamer_clip_engine_and_layout(tmp_path: Path) -> None:
    config = _config(tmp_path)

    path = render_path(
        config,
        streamer="Example Streamer!",
        clip_id=" Clip:One! ",
        engine="yt-dlp",
        layout="hybrid",
    )

    assert path == (
        tmp_path / "renders" / "Example_Streamer" / "Clip_One" / "yt-dlp" / "hybrid.mp4"
    )


def test_export_path_uses_safe_title_plus_clip_id(tmp_path: Path) -> None:
    config = _config(tmp_path)

    path = export_path(
        config,
        streamer="example",
        title="Wild clutch: 1v5?!",
        clip_id="clip-1",
        layout="center gameplay",
    )

    assert path == (
        tmp_path
        / "exports"
        / "example"
        / "Wild_clutch_1v5__clip-1"
        / "center_gameplay.mp4"
    )


def test_ready_export_path_uses_ready_streamer_clip_layout(tmp_path: Path) -> None:
    config = _config(tmp_path)

    path = ready_export_path(
        config,
        streamer="Example Streamer!",
        title="WAS???",
        clip_id=" Clip:One! ",
        layout="center gameplay",
    )

    assert path == (
        tmp_path
        / "exports"
        / "ready"
        / "Example_Streamer"
        / "Clip_One"
        / "WAS-Clip_One.mp4"
    )


def test_ready_export_filename_uses_title_and_clip_id() -> None:
    filename = ready_export_filename(
        title="WAS???",
        clip_id="CreativeHilariousWrenchM4xHeh-e1mVQUh6JEati9qQ",
        layout="hybrid",
        extension="mp4",
    )

    assert (
        filename
        == "WAS-CreativeHilariousWrenchM4xHeh-e1mVQUh6JEati9qQ.mp4"
    )


def test_ready_export_filename_falls_back_and_limits_title() -> None:
    filename = ready_export_filename(
        title=" " + "A" * 90 + " ",
        clip_id="clip-1",
        layout="hybrid",
        extension=".mp4",
    )
    missing_title = ready_export_filename(
        title="...",
        clip_id="clip-1",
        layout="hybrid",
        extension="mp4",
    )

    assert filename == f"{'A' * 80}-clip-1.mp4"
    assert missing_title == "clip-clip-1.mp4"


def test_sanitize_path_part_is_safe_and_short() -> None:
    long_title = "A" * 90

    assert sanitize_path_part(" My Clip: wow!? ") == "My_Clip_wow"
    assert sanitize_path_part(long_title) == "A" * 72


def test_clip_folder_name_falls_back_for_empty_titles() -> None:
    assert clip_folder_name("...", "clip-1") == "untitled__clip-1"
    assert clip_folder_name(None, "clip-1") == "untitled__clip-1"


def test_download_paths_preserve_existing_backend_structure(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert download_dir(config, clip_id=" Clip:One! ", engine="yt-dlp") == (
        tmp_path / "downloads" / "Clip_One" / "yt-dlp"
    )
    assert backend_download_dir(tmp_path / "downloads", clip_id="clip-1", backend="clipr") == (
        tmp_path / "downloads" / "clip-1" / "clipr"
    )

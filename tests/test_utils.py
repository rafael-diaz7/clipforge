from pathlib import Path

from clipforge.utils import (
    clip_slug_from_url,
    ensure_directory,
    ensure_project_subdir,
    safe_filename,
    utc_timestamp,
)


def test_safe_filename_replaces_unsafe_characters() -> None:
    assert safe_filename(" My Clip: wow!? ") == "My_Clip_wow"


def test_safe_filename_uses_fallback_for_empty_values() -> None:
    assert safe_filename("...", fallback="fallback") == "fallback"


def test_clip_slug_from_twitch_clip_url() -> None:
    url = "https://www.twitch.tv/example/clip/TallHelpfulClipKappa"

    assert clip_slug_from_url(url) == "TallHelpfulClipKappa"


def test_clip_slug_falls_back_to_last_path_part() -> None:
    url = "https://clips.twitch.tv/TallHelpfulClipKappa?tt_medium=clips_api"

    assert clip_slug_from_url(url) == "TallHelpfulClipKappa"


def test_ensure_directory_creates_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "directory"

    assert ensure_directory(target) == target
    assert target.is_dir()


def test_ensure_project_subdir_creates_path_under_root(tmp_path: Path) -> None:
    target = ensure_project_subdir(tmp_path, "data", "downloads")

    assert target == (tmp_path / "data" / "downloads").resolve()
    assert target.is_dir()


def test_utc_timestamp_returns_utc_iso_string() -> None:
    timestamp = utc_timestamp()

    assert timestamp.endswith("+00:00")

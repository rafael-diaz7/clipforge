from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.cleanup import cleanup_local_artifacts
from clipforge.pipeline.cli import main
from clipforge.storage.state import (
    get_clip,
    mark_clip_mobile_review,
    mark_clip_rendered,
    upsert_discovered_clip,
)


NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(
        downloads_dir=tmp_path / "downloads",
        renders_dir=tmp_path / "renders",
        metadata_dir=tmp_path / "metadata",
        exports_dir=tmp_path / "exports",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
    )


def _write_file(path: Path, *, age: timedelta) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact", encoding="utf-8")
    timestamp = (NOW - age).timestamp()
    os.utime(path, (timestamp, timestamp))
    return path


def test_cleanup_dry_run_deletes_nothing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    old_download = _write_file(
        config.downloads_dir / "clip-old" / "ytdlp" / "clip-old.mp4",
        age=timedelta(hours=25),
    )

    result = cleanup_local_artifacts(apply=False, config=config, now=NOW)

    assert old_download.exists()
    assert result.deleted_files == (old_download,)


def test_cleanup_apply_deletes_old_downloads_renders_and_ready_exports(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    old_download = _write_file(
        config.downloads_dir / "clip-old" / "ytdlp" / "clip-old.mp4",
        age=timedelta(hours=25),
    )
    old_render = _write_file(
        config.renders_dir / "example" / "clip-old" / "ytdlp" / "hybrid.mp4",
        age=timedelta(hours=25),
    )
    old_ready_export = _write_file(
        config.exports_dir / "ready" / "example" / "clip-old" / "hybrid.mp4",
        age=timedelta(hours=25),
    )

    cleanup_local_artifacts(apply=True, config=config, now=NOW)

    assert not old_download.exists()
    assert not old_render.exists()
    assert not old_ready_export.exists()


def test_cleanup_apply_deletes_old_metadata_after_seven_days(tmp_path: Path) -> None:
    config = _config(tmp_path)
    old_metadata = _write_file(
        config.metadata_dir / "clip-old.json",
        age=timedelta(days=8),
    )
    old_caption_metadata = _write_file(
        config.metadata_dir / "captions" / "clip-old.json",
        age=timedelta(days=8),
    )

    cleanup_local_artifacts(apply=True, config=config, now=NOW)

    assert not old_metadata.exists()
    assert not old_caption_metadata.exists()


def test_cleanup_preserves_recent_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    recent_download = _write_file(
        config.downloads_dir / "clip-recent" / "ytdlp" / "clip-recent.mp4",
        age=timedelta(hours=23),
    )
    recent_render = _write_file(
        config.renders_dir / "example" / "clip-recent" / "ytdlp" / "hybrid.mp4",
        age=timedelta(hours=23),
    )
    recent_metadata = _write_file(
        config.metadata_dir / "clip-recent.json",
        age=timedelta(days=6),
    )

    cleanup_local_artifacts(apply=True, config=config, now=NOW)

    assert recent_download.exists()
    assert recent_render.exists()
    assert recent_metadata.exists()


def test_cleanup_preserves_mobile_review_candidate_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    render_dir = config.renders_dir / "example" / "clip-review" / "ytdlp"
    old_candidate = _write_file(
        render_dir / "hybrid.mp4",
        age=timedelta(hours=25),
    )
    old_metadata = _write_file(
        config.metadata_dir / "clip-review.json",
        age=timedelta(days=8),
    )
    upsert_discovered_clip(
        clip_id="clip-review",
        url="https://clips.twitch.tv/clip-review",
        db_path=config.state_db_path,
    )
    mark_clip_mobile_review(
        "clip-review",
        render_dir=render_dir,
        metadata_path=old_metadata,
        db_path=config.state_db_path,
    )

    cleanup_local_artifacts(apply=True, config=config, now=NOW)

    assert old_candidate.exists()
    assert old_metadata.exists()


def test_cleanup_leaves_db_rows_untouched(tmp_path: Path) -> None:
    config = _config(tmp_path)
    render_dir = config.renders_dir / "example" / "clip-rendered" / "ytdlp"
    metadata_path = config.metadata_dir / "clip-rendered.json"
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        title="rendered clip",
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=render_dir,
        metadata_path=metadata_path,
        db_path=config.state_db_path,
    )

    cleanup_local_artifacts(apply=True, config=config, now=NOW)

    state = get_clip("clip-rendered", db_path=config.state_db_path)
    assert state is not None
    assert state.status == "rendered"
    assert state.title == "rendered clip"
    assert state.render_dir == str(render_dir)
    assert state.metadata_path == str(metadata_path)


def test_cleanup_removes_empty_directories_when_safe(tmp_path: Path) -> None:
    config = _config(tmp_path)
    old_download = _write_file(
        config.downloads_dir / "clip-old" / "ytdlp" / "clip-old.mp4",
        age=timedelta(hours=25),
    )
    old_dir = old_download.parent
    old_clip_dir = old_dir.parent

    cleanup_local_artifacts(apply=True, config=config, now=NOW)

    assert not old_dir.exists()
    assert not old_clip_dir.exists()
    assert config.downloads_dir.exists()


def test_cleanup_repeated_apply_is_safe(tmp_path: Path) -> None:
    config = _config(tmp_path)
    old_download = _write_file(
        config.downloads_dir / "clip-old" / "ytdlp" / "clip-old.mp4",
        age=timedelta(hours=25),
    )

    first = cleanup_local_artifacts(apply=True, config=config, now=NOW)
    second = cleanup_local_artifacts(apply=True, config=config, now=NOW)

    assert first.deleted_files == (old_download,)
    assert second.deleted_files == ()


def test_cleanup_cli_routes_apply_mode(monkeypatch, capsys, tmp_path: Path) -> None:
    config = _config(tmp_path)
    old_download = _write_file(
        config.downloads_dir / "clip-old" / "ytdlp" / "clip-old.mp4",
        age=timedelta(hours=25),
    )

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)

    exit_code = main(["cleanup", "--apply"])

    assert exit_code == 0
    assert not old_download.exists()
    assert "deleted: 1 files" in capsys.readouterr().out

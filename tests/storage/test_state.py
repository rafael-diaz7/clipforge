from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.storage.state import (
    ClipStateError,
    get_clip,
    get_unprocessed_clips,
    init_db,
    mark_clip_downloaded,
    mark_clip_failed,
    mark_clip_rendered,
    mark_clip_skipped,
    upsert_discovered_clip,
)


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "clipforge.sqlite"


def test_init_db_creates_database_under_requested_path(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)

    created_path = init_db(db_path)

    assert created_path == db_path
    assert db_path.exists()


def test_upsert_discovered_clip_inserts_new_clip_with_discovered_status(
    tmp_path: Path,
) -> None:
    db_path = _db_path(tmp_path)

    clip = upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        streamer_login="example",
        title="first title",
        view_count=42,
        duration_seconds=28.5,
        rank_score=0.75,
        rank_breakdown={"views": 0.5, "age": 1.0},
        db_path=db_path,
        now="2026-05-01T00:00:00+00:00",
    )

    assert clip.clip_id == "clip-1"
    assert clip.status == "discovered"
    assert clip.url == "https://clips.twitch.tv/clip-1"
    assert clip.streamer_login == "example"
    assert clip.title == "first title"
    assert clip.view_count == 42
    assert clip.duration_seconds == 28.5
    assert clip.rank_score == 0.75
    assert clip.rank_breakdown == {"age": 1.0, "views": 0.5}
    assert clip.discovered_at == "2026-05-01T00:00:00+00:00"
    assert clip.last_seen_at == "2026-05-01T00:00:00+00:00"


def test_rediscovery_updates_metadata_but_preserves_status(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        title="old title",
        view_count=42,
        db_path=db_path,
        now="2026-05-01T00:00:00+00:00",
    )
    mark_clip_rendered(
        "clip-1",
        render_dir=tmp_path / "renders" / "clip-1",
        db_path=db_path,
    )

    clip = upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1-new",
        streamer_login="example",
        title="new title",
        view_count=100,
        duration_seconds=30,
        rank_score=0.9,
        rank_breakdown={"views": 0.8, "age": 1.0},
        db_path=db_path,
        now="2026-05-02T00:00:00+00:00",
    )

    assert clip.status == "rendered"
    assert clip.url == "https://clips.twitch.tv/clip-1-new"
    assert clip.streamer_login == "example"
    assert clip.title == "new title"
    assert clip.view_count == 100
    assert clip.duration_seconds == 30
    assert clip.rank_score == 0.9
    assert clip.rank_breakdown == {"age": 1.0, "views": 0.8}
    assert clip.discovered_at == "2026-05-01T00:00:00+00:00"
    assert clip.last_seen_at == "2026-05-02T00:00:00+00:00"
    assert clip.render_dir == str(tmp_path / "renders" / "clip-1")


def test_mark_clip_status_updates_processing_fields(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        db_path=db_path,
        now="2026-05-01T00:00:00+00:00",
    )

    downloaded = mark_clip_downloaded(
        "clip-1",
        download_path=tmp_path / "downloads" / "clip-1.mp4",
        metadata_path=tmp_path / "metadata" / "clip-1.json",
        db_path=db_path,
    )
    rendered = mark_clip_rendered(
        "clip-1",
        render_dir=tmp_path / "renders" / "clip-1",
        db_path=db_path,
    )

    assert downloaded.status == "downloaded"
    assert downloaded.download_path == str(tmp_path / "downloads" / "clip-1.mp4")
    assert downloaded.metadata_path == str(tmp_path / "metadata" / "clip-1.json")
    assert rendered.status == "rendered"
    assert rendered.render_dir == str(tmp_path / "renders" / "clip-1")
    assert rendered.download_path == str(tmp_path / "downloads" / "clip-1.mp4")


def test_mark_clip_skipped_and_failed_exclude_clips_from_unprocessed_query(
    tmp_path: Path,
) -> None:
    db_path = _db_path(tmp_path)
    for clip_id in ("clip-1", "clip-2", "clip-3"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            db_path=db_path,
            now=f"2026-05-0{clip_id[-1]}T00:00:00+00:00",
        )

    skipped = mark_clip_skipped("clip-2", skip_reason="not enough context", db_path=db_path)
    failed = mark_clip_failed("clip-3", error_message="download failed", db_path=db_path)

    assert skipped.status == "skipped"
    assert skipped.skip_reason == "not enough context"
    assert failed.status == "failed"
    assert failed.error_message == "download failed"
    assert [clip.clip_id for clip in get_unprocessed_clips(db_path=db_path)] == ["clip-1"]


def test_rendered_clip_is_not_returned_by_unprocessed_query(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        db_path=db_path,
        now="2026-05-01T00:00:00+00:00",
    )
    upsert_discovered_clip(
        clip_id="clip-2",
        url="https://clips.twitch.tv/clip-2",
        db_path=db_path,
        now="2026-05-02T00:00:00+00:00",
    )
    mark_clip_rendered("clip-1", render_dir=tmp_path / "renders" / "clip-1", db_path=db_path)

    assert [clip.clip_id for clip in get_unprocessed_clips(db_path=db_path)] == ["clip-2"]


def test_unprocessed_query_orders_ranked_clips_by_score(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        rank_score=0.2,
        rank_breakdown={"views": 0.2},
        db_path=db_path,
        now="2026-05-01T00:00:00+00:00",
    )
    upsert_discovered_clip(
        clip_id="clip-2",
        url="https://clips.twitch.tv/clip-2",
        rank_score=0.9,
        rank_breakdown={"views": 0.9},
        db_path=db_path,
        now="2026-05-02T00:00:00+00:00",
    )
    upsert_discovered_clip(
        clip_id="clip-3",
        url="https://clips.twitch.tv/clip-3",
        db_path=db_path,
        now="2026-05-03T00:00:00+00:00",
    )

    assert [clip.clip_id for clip in get_unprocessed_clips(db_path=db_path)] == [
        "clip-2",
        "clip-1",
        "clip-3",
    ]


def test_get_clip_returns_none_for_unknown_clip(tmp_path: Path) -> None:
    assert get_clip("missing", db_path=_db_path(tmp_path)) is None


def test_marking_unknown_clip_raises(tmp_path: Path) -> None:
    with pytest.raises(ClipStateError, match="Clip not found"):
        mark_clip_failed("missing", error_message="boom", db_path=_db_path(tmp_path))

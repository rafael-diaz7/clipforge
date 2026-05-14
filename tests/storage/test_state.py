from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.storage.state import (
    ClipStateError,
    get_clip,
    get_mobile_review_clips,
    get_prepare_candidate_clips,
    get_review_eligible_clips,
    get_unprocessed_clips,
    init_db,
    mark_clip_downloaded,
    mark_clip_exported,
    mark_clip_failed,
    mark_clip_needs_rerender,
    mark_clip_rendered,
    mark_clip_mobile_review,
    mark_clip_selected,
    mark_clip_skipped,
    reset_all_clips_to_discovered,
    reset_clip_to_discovered,
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
        created_at="2026-04-30T12:00:00Z",
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
    assert clip.created_at == "2026-04-30T12:00:00Z"
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
        created_at="2026-05-02T12:00:00Z",
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
    assert clip.created_at == "2026-05-02T12:00:00Z"
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
    assert failed.failed_at is not None
    assert [clip.clip_id for clip in get_unprocessed_clips(db_path=db_path)] == ["clip-1"]


def test_prepare_candidates_include_only_cooled_down_failures(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    for clip_id in ("clip-ready", "clip-old-failed", "clip-recent-failed"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            streamer_login="example",
            rank_score=1.0,
            db_path=db_path,
        )
    mark_clip_failed(
        "clip-old-failed",
        error_message="old failure",
        failed_at="2026-05-14T10:00:00+00:00",
        db_path=db_path,
    )
    mark_clip_failed(
        "clip-recent-failed",
        error_message="recent failure",
        failed_at="2026-05-14T11:30:00+00:00",
        db_path=db_path,
    )

    clips = get_prepare_candidate_clips(
        db_path=db_path,
        streamer_login="example",
        failed_before="2026-05-14T11:00:00+00:00",
    )

    assert [clip.clip_id for clip in clips] == ["clip-ready", "clip-old-failed"]


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


def test_skipped_clip_is_not_returned_by_review_eligibility(
    caplog,
    tmp_path: Path,
) -> None:
    db_path = _db_path(tmp_path)
    for clip_id in ("clip-skipped", "clip-eligible"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            streamer_login="example",
            rank_score=1.0,
            db_path=db_path,
        )
    mark_clip_skipped(
        "clip-skipped",
        skip_reason="already reviewed",
        db_path=db_path,
    )
    mark_clip_rendered(
        "clip-eligible",
        render_dir=tmp_path / "renders" / "clip-eligible",
        metadata_path=tmp_path / "metadata" / "clip-eligible.json",
        db_path=db_path,
    )

    with caplog.at_level("INFO", logger="clipforge.storage.state"):
        eligible = get_review_eligible_clips(
            db_path=db_path,
            streamer_login="example",
        )

    assert [clip.clip_id for clip in eligible] == ["clip-eligible"]
    assert "Excluding 1 skipped clip(s) from normal review eligibility." in caplog.text


def test_needs_rerender_is_processing_eligible_but_not_normal_review_eligible(
    tmp_path: Path,
) -> None:
    db_path = _db_path(tmp_path)
    for clip_id in ("clip-rerender", "clip-eligible"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            streamer_login="example",
            rank_score=1.0 if clip_id == "clip-rerender" else 0.5,
            db_path=db_path,
        )
    rerender = mark_clip_needs_rerender(
        "clip-rerender",
        skip_reason="layouts did not fit",
        db_path=db_path,
    )
    mark_clip_rendered(
        "clip-eligible",
        render_dir=tmp_path / "renders" / "clip-eligible",
        metadata_path=tmp_path / "metadata" / "clip-eligible.json",
        db_path=db_path,
    )

    assert rerender.status == "needs_rerender"
    assert rerender.skip_reason == "layouts did not fit"
    assert [clip.clip_id for clip in get_unprocessed_clips(db_path=db_path)] == [
        "clip-rerender",
    ]
    assert [clip.clip_id for clip in get_review_eligible_clips(db_path=db_path)] == [
        "clip-eligible"
    ]
    assert [
        clip.clip_id
        for clip in get_review_eligible_clips(
            db_path=db_path,
            include_needs_rerender=True,
        )
    ] == ["clip-eligible"]


def test_review_eligibility_requires_render_candidates(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    for clip_id in ("clip-discovered", "clip-rendered-without-metadata", "clip-ready"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            rank_score=1.0,
            db_path=db_path,
        )
    mark_clip_rendered(
        "clip-rendered-without-metadata",
        render_dir=tmp_path / "renders" / "clip-rendered-without-metadata",
        db_path=db_path,
    )
    mark_clip_rendered(
        "clip-ready",
        render_dir=tmp_path / "renders" / "clip-ready",
        metadata_path=tmp_path / "metadata" / "clip-ready.json",
        db_path=db_path,
    )

    assert [clip.clip_id for clip in get_review_eligible_clips(db_path=db_path)] == [
        "clip-ready"
    ]


def test_mobile_review_query_requires_prepare_state(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    for clip_id in ("clip-rendered", "clip-mobile"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            rank_score=1.0,
            db_path=db_path,
        )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        metadata_path=tmp_path / "metadata" / "clip-rendered.json",
        db_path=db_path,
    )
    mark_clip_mobile_review(
        "clip-mobile",
        render_dir=tmp_path / "renders" / "clip-mobile",
        metadata_path=tmp_path / "metadata" / "clip-mobile.json",
        db_path=db_path,
    )

    assert [clip.clip_id for clip in get_mobile_review_clips(db_path=db_path)] == [
        "clip-mobile"
    ]
    assert [clip.clip_id for clip in get_review_eligible_clips(db_path=db_path)] == [
        "clip-rendered"
    ]


def test_selected_and_exported_metadata_control_review_eligibility(
    tmp_path: Path,
) -> None:
    db_path = _db_path(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        db_path=db_path,
    )
    mark_clip_rendered(
        "clip-1",
        render_dir=tmp_path / "renders" / "clip-1",
        metadata_path=tmp_path / "metadata" / "clip-1.json",
        db_path=db_path,
    )

    selected = mark_clip_selected(
        "clip-1",
        selected_render_layout="hybrid",
        selected_render_path=tmp_path / "renders" / "clip-1" / "hybrid.mp4",
        db_path=db_path,
    )

    assert selected.status == "selected"
    assert selected.selected_render_layout == "hybrid"
    assert selected.selected_render_path == str(
        tmp_path / "renders" / "clip-1" / "hybrid.mp4"
    )
    assert get_review_eligible_clips(db_path=db_path) == ()

    exported = mark_clip_exported(
        "clip-1",
        selected_render_layout="hybrid",
        selected_render_path=tmp_path / "renders" / "clip-1" / "hybrid.mp4",
        export_path=tmp_path / "exports" / "clip-1.mp4",
        db_path=db_path,
        exported_at="2026-05-01T00:00:00+00:00",
    )

    assert exported.status == "exported"
    assert exported.export_path == str(tmp_path / "exports" / "clip-1.mp4")
    assert exported.exported_at == "2026-05-01T00:00:00+00:00"
    assert get_review_eligible_clips(db_path=db_path) == ()


def test_reset_clip_to_discovered_clears_processing_artifact_fields(
    tmp_path: Path,
) -> None:
    db_path = _db_path(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        rank_score=0.75,
        rank_breakdown={"views": 0.75},
        db_path=db_path,
    )
    mark_clip_downloaded(
        "clip-1",
        download_path=tmp_path / "downloads" / "clip-1.mp4",
        metadata_path=tmp_path / "metadata" / "download.json",
        db_path=db_path,
    )
    mark_clip_rendered(
        "clip-1",
        render_dir=tmp_path / "renders" / "clip-1",
        metadata_path=tmp_path / "metadata" / "render.json",
        db_path=db_path,
    )
    mark_clip_exported(
        "clip-1",
        selected_render_layout="hybrid",
        selected_render_path=tmp_path / "renders" / "clip-1" / "hybrid.mp4",
        export_path=tmp_path / "exports" / "clip-1.mp4",
        db_path=db_path,
    )

    clip = reset_clip_to_discovered("clip-1", db_path=db_path)

    assert clip.status == "discovered"
    assert clip.rank_score == 0.75
    assert clip.rank_breakdown == {"views": 0.75}
    assert clip.download_path is None
    assert clip.metadata_path is None
    assert clip.render_dir is None
    assert clip.skip_reason is None
    assert clip.error_message is None
    assert clip.selected_render_layout is None
    assert clip.selected_render_path is None
    assert clip.export_path is None
    assert clip.exported_at is None
    assert [state.clip_id for state in get_unprocessed_clips(db_path=db_path)] == ["clip-1"]


def test_reset_all_clips_to_discovered_resets_every_clip(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    for clip_id in ("clip-rendered", "clip-failed"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            db_path=db_path,
        )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=db_path,
    )
    mark_clip_failed("clip-failed", error_message="render failed", db_path=db_path)

    count = reset_all_clips_to_discovered(db_path=db_path)

    assert count == 2
    states = get_unprocessed_clips(db_path=db_path)
    assert {state.clip_id for state in states} == {"clip-rendered", "clip-failed"}
    assert all(state.status == "discovered" for state in states)
    assert all(state.render_dir is None for state in states)
    assert all(state.error_message is None for state in states)


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


def test_unprocessed_query_filters_by_streamer_before_limit(tmp_path: Path) -> None:
    db_path = _db_path(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-other",
        url="https://clips.twitch.tv/clip-other",
        streamer_login="other",
        rank_score=1.0,
        db_path=db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-example-1",
        url="https://clips.twitch.tv/clip-example-1",
        streamer_login="example",
        rank_score=0.9,
        db_path=db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-example-2",
        url="https://clips.twitch.tv/clip-example-2",
        streamer_login="example",
        rank_score=0.8,
        db_path=db_path,
    )

    clips = get_unprocessed_clips(
        db_path=db_path,
        streamer_login="Example",
        limit=1,
    )

    assert [clip.clip_id for clip in clips] == ["clip-example-1"]


def test_get_clip_returns_none_for_unknown_clip(tmp_path: Path) -> None:
    assert get_clip("missing", db_path=_db_path(tmp_path)) is None


def test_marking_unknown_clip_raises(tmp_path: Path) -> None:
    with pytest.raises(ClipStateError, match="Clip not found"):
        mark_clip_failed("missing", error_message="boom", db_path=_db_path(tmp_path))


def test_resetting_unknown_clip_raises(tmp_path: Path) -> None:
    with pytest.raises(ClipStateError, match="Clip not found"):
        reset_clip_to_discovered("missing", db_path=_db_path(tmp_path))

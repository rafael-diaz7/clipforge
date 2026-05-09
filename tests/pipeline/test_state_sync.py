from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.state_sync import (
    record_discovered_clips,
    record_rendered_clip,
    rerank_persisted_clips,
)
from clipforge.storage.state import get_clip, upsert_discovered_clip
from clipforge.integrations.twitch import TwitchClip
from tests.constants import TWITCH_CLIP_SLUG, TWITCH_CLIP_URL


def test_record_discovered_clips_upserts_clip_state(tmp_path: Path) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    clips = (
        TwitchClip(
            id="clip-1",
            url="https://clips.twitch.tv/clip-1",
            broadcaster_name="Example",
            creator_name="Viewer",
            title="great clip",
            view_count=42,
            created_at="2026-05-01T00:00:00Z",
            duration=28.5,
            thumbnail_url="https://example.test/thumb.jpg",
        ),
    )

    record_discovered_clips(
        clips=clips,
        channel="https://twitch.tv/Example",
        config=config,
    )

    state = get_clip("clip-1", db_path=config.state_db_path)
    assert state is not None
    assert state.status == "discovered"
    assert state.streamer_login == "example"
    assert state.title == "great clip"
    assert state.view_count == 42
    assert state.created_at == "2026-05-01T00:00:00Z"
    assert state.rank_score is not None
    assert state.rank_breakdown is not None
    assert set(state.rank_breakdown) == {"views", "age", "duration", "title"}


def test_rerank_persisted_clips_updates_stale_scores(tmp_path: Path) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        title="great clip",
        view_count=100,
        created_at="2026-05-01T00:00:00Z",
        duration_seconds=30,
        rank_score=0.01,
        rank_breakdown={"age": 0.01},
        db_path=config.state_db_path,
    )

    count = rerank_persisted_clips(
        config=config,
        reference_time=datetime(2026, 5, 2, tzinfo=UTC),
    )

    state = get_clip("clip-1", db_path=config.state_db_path)
    assert count == 1
    assert state is not None
    assert state.rank_score != 0.01
    assert state.rank_breakdown is not None
    assert set(state.rank_breakdown) == {"views", "age", "duration", "title"}


def test_rerank_persisted_clips_channel_only_updates_matching_streamer_rows(
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-example",
        url="https://clips.twitch.tv/clip-example",
        streamer_login="example",
        title="example clip",
        view_count=100,
        created_at="2026-05-01T00:00:00Z",
        duration_seconds=30,
        rank_score=0.01,
        rank_breakdown={"age": 0.01},
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-other",
        url="https://clips.twitch.tv/clip-other",
        streamer_login="other",
        title="other clip",
        view_count=100,
        created_at="2026-05-01T00:00:00Z",
        duration_seconds=30,
        rank_score=0.02,
        rank_breakdown={"age": 0.02},
        db_path=config.state_db_path,
    )

    count = rerank_persisted_clips(
        config=config,
        channel="Example",
        reference_time=datetime(2026, 5, 2, tzinfo=UTC),
    )

    example = get_clip("clip-example", db_path=config.state_db_path)
    other = get_clip("clip-other", db_path=config.state_db_path)
    assert count == 1
    assert example is not None
    assert other is not None
    assert example.rank_score != 0.01
    assert other.rank_score == 0.02
    assert other.rank_breakdown == {"age": 0.02}


def test_rerank_persisted_clips_gives_older_clips_lower_age_contribution(
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    for clip_id, created_at in (
        ("older", "2026-04-20T00:00:00Z"),
        ("newer", "2026-05-01T00:00:00Z"),
    ):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            title="same title",
            view_count=100,
            created_at=created_at,
            duration_seconds=30,
            db_path=config.state_db_path,
        )

    rerank_persisted_clips(
        config=config,
        reference_time=datetime(2026, 5, 2, tzinfo=UTC),
    )

    older = get_clip("older", db_path=config.state_db_path)
    newer = get_clip("newer", db_path=config.state_db_path)
    assert older is not None
    assert newer is not None
    assert older.rank_breakdown is not None
    assert newer.rank_breakdown is not None
    assert older.rank_breakdown["age"] < newer.rank_breakdown["age"]


def test_record_rendered_clip_creates_missing_clip_state(tmp_path: Path) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    render_dir = tmp_path / "renders" / TWITCH_CLIP_SLUG / "clipr"
    metadata_path = tmp_path / "metadata" / f"{TWITCH_CLIP_SLUG}.json"

    record_rendered_clip(
        clip_id=TWITCH_CLIP_SLUG,
        twitch_clip_url=TWITCH_CLIP_URL,
        render_dir=render_dir,
        metadata_path=metadata_path,
        config=config,
    )

    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.status == "rendered"
    assert state.url == TWITCH_CLIP_URL
    assert state.metadata_path == str(metadata_path)
    assert state.render_dir == str(render_dir)


def test_record_rendered_clip_preserves_existing_discovery_fields(tmp_path: Path) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id=TWITCH_CLIP_SLUG,
        url=TWITCH_CLIP_URL,
        streamer_login="example",
        title="existing title",
        view_count=10,
        duration_seconds=12,
        db_path=config.state_db_path,
    )
    metadata_path = tmp_path / "metadata" / f"{TWITCH_CLIP_SLUG}.json"

    record_rendered_clip(
        clip_id=TWITCH_CLIP_SLUG,
        twitch_clip_url=TWITCH_CLIP_URL,
        render_dir=tmp_path / "renders",
        metadata_path=metadata_path,
        config=config,
    )

    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.status == "rendered"
    assert state.title == "existing title"
    assert state.metadata_path == str(metadata_path)

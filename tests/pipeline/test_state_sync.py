from __future__ import annotations

from pathlib import Path

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.state_sync import record_discovered_clips, record_rendered_clip
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
    assert state.rank_score is not None
    assert state.rank_breakdown is not None
    assert set(state.rank_breakdown) == {"views", "age", "duration", "title"}


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

"""Helpers that sync pipeline outcomes into persistent state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from clipforge.core.config import ClipforgeConfig
from clipforge.integrations.twitch import TwitchClip, twitch_channel_login_from_input
from clipforge.pipeline.ranking import rank_clips, score_clip
from clipforge.storage.state import (
    ClipState,
    get_clip,
    get_persisted_clips,
    mark_clip_rendered,
    update_clip_rank,
    upsert_discovered_clip,
)


def record_discovered_clips(
    *,
    clips: Sequence[TwitchClip],
    channel: str,
    config: ClipforgeConfig,
) -> None:
    """Persist Twitch clips after discovery has succeeded."""

    streamer_login = twitch_channel_login_from_input(channel)
    for ranked_clip in rank_clips(clips):
        clip = ranked_clip.clip
        upsert_discovered_clip(
            clip_id=clip.id,
            url=clip.url,
            streamer_login=streamer_login,
            title=clip.title,
            view_count=clip.view_count,
            created_at=clip.created_at,
            duration_seconds=clip.duration,
            rank_score=ranked_clip.score,
            rank_breakdown=ranked_clip.breakdown,
            db_path=config.state_db_path,
        )


def rerank_persisted_clips(
    *,
    config: ClipforgeConfig,
    channel: str | None = None,
    reference_time: datetime | None = None,
) -> int:
    """Refresh rank fields for clips already persisted in SQLite."""

    streamer_login = twitch_channel_login_from_input(channel) if channel else None
    clips = get_persisted_clips(
        db_path=config.state_db_path,
        streamer_login=streamer_login,
    )
    resolved_reference_time = reference_time or datetime.now(UTC)

    for state in clips:
        ranked_clip = score_clip(
            _twitch_clip_from_state(state),
            reference_time=resolved_reference_time,
        )
        update_clip_rank(
            state.clip_id,
            rank_score=ranked_clip.score,
            rank_breakdown=ranked_clip.breakdown,
            db_path=config.state_db_path,
        )

    return len(clips)


def record_rendered_clip(
    *,
    clip_id: str,
    twitch_clip_url: str,
    render_dir: Path,
    metadata_path: Path,
    config: ClipforgeConfig,
) -> None:
    """Persist rendered state after render outputs and metadata exist."""

    if get_clip(clip_id, db_path=config.state_db_path) is None:
        upsert_discovered_clip(
            clip_id=clip_id,
            url=twitch_clip_url,
            db_path=config.state_db_path,
        )
    mark_clip_rendered(
        clip_id,
        render_dir=render_dir,
        metadata_path=metadata_path,
        db_path=config.state_db_path,
    )


def _twitch_clip_from_state(clip: ClipState) -> TwitchClip:
    return TwitchClip(
        id=clip.clip_id,
        url=clip.url,
        broadcaster_name=clip.streamer_login or "",
        creator_name="",
        title=clip.title or "",
        view_count=clip.view_count or 0,
        created_at=clip.created_at or clip.discovered_at,
        duration=clip.duration_seconds or 0.0,
        thumbnail_url="",
    )

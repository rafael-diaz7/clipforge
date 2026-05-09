"""Helpers that sync pipeline outcomes into persistent state."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from clipforge.core.config import ClipforgeConfig
from clipforge.integrations.twitch import TwitchClip, twitch_channel_login_from_input
from clipforge.pipeline.ranking import rank_clips
from clipforge.storage.state import get_clip, mark_clip_rendered, upsert_discovered_clip


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
            duration_seconds=clip.duration,
            rank_score=ranked_clip.score,
            rank_breakdown=ranked_clip.breakdown,
            db_path=config.state_db_path,
        )


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

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from clipforge.integrations.twitch import TwitchClip
from clipforge.pipeline.ranking import rank_clips, score_clip


def _clip(
    clip_id: str,
    *,
    title: str = "Great clutch moment",
    view_count: int = 100,
    created_at: str = "2026-05-01T00:00:00Z",
    duration: float = 30,
) -> TwitchClip:
    return TwitchClip(
        id=clip_id,
        url=f"https://clips.twitch.tv/{clip_id}",
        broadcaster_name="Example",
        creator_name="Viewer",
        title=title,
        view_count=view_count,
        created_at=created_at,
        duration=duration,
        thumbnail_url="https://example.test/thumb.jpg",
    )


def test_score_clip_exposes_transparent_breakdown() -> None:
    ranked = score_clip(
        _clip("clip-1", view_count=999, created_at="2026-05-07T00:00:00Z"),
        reference_time=datetime(2026, 5, 8, tzinfo=UTC),
    )

    assert set(ranked.breakdown) == {"views", "age", "duration", "title"}
    assert ranked.breakdown["views"] == pytest.approx(0.6)
    assert ranked.breakdown["age"] == pytest.approx(0.9286)
    assert ranked.breakdown["duration"] == 1.0
    assert ranked.breakdown["title"] == 1.0
    assert ranked.score == pytest.approx(0.7622)


def test_rank_clips_orders_by_score_descending() -> None:
    clips = (
        _clip("low", view_count=10, created_at="2026-05-01T00:00:00Z"),
        _clip("high", view_count=1000, created_at="2026-05-01T00:00:00Z"),
        _clip("mid", view_count=100, created_at="2026-05-01T00:00:00Z"),
    )

    ranked = rank_clips(clips, reference_time=datetime(2026, 5, 2, tzinfo=UTC))

    assert [item.clip.id for item in ranked] == ["high", "mid", "low"]


def test_rank_clips_uses_deterministic_tie_breakers() -> None:
    clips = (
        _clip("older", created_at="2026-05-01T00:00:00Z"),
        _clip("newer", created_at="2026-05-02T00:00:00Z"),
    )

    ranked = rank_clips(clips, reference_time=datetime(2026, 5, 2, tzinfo=UTC))

    assert [item.clip.id for item in ranked] == ["newer", "older"]


def test_rank_clips_tie_breaks_by_id_after_score_created_at_and_views() -> None:
    clips = (
        _clip("clip-b", created_at="2026-05-01T00:00:00Z"),
        _clip("clip-a", created_at="2026-05-01T00:00:00Z"),
    )

    ranked = rank_clips(clips, reference_time=datetime(2026, 5, 2, tzinfo=UTC))

    assert [item.clip.id for item in ranked] == ["clip-a", "clip-b"]

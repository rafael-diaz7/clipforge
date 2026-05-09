"""Deterministic ranking for discovered Twitch clips."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Sequence

from clipforge.integrations.twitch import TwitchClip

_FALLBACK_REFERENCE_TIME = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class ClipRank:
    clip: TwitchClip
    score: float
    breakdown: dict[str, float]


def rank_clips(
    clips: Sequence[TwitchClip],
    *,
    reference_time: datetime | None = None,
) -> tuple[ClipRank, ...]:
    """Return clips scored and ordered for human review."""

    resolved_reference_time = reference_time or _latest_created_at(clips)
    scored = tuple(score_clip(clip, reference_time=resolved_reference_time) for clip in clips)
    return tuple(
        sorted(
            scored,
            key=lambda ranked: (
                -ranked.score,
                -_created_at_sort_value(ranked.clip.created_at),
                -ranked.clip.view_count,
                ranked.clip.id,
            ),
        )
    )


def score_clip(
    clip: TwitchClip,
    *,
    reference_time: datetime | None = None,
) -> ClipRank:
    """Score one clip using only available Twitch metadata."""

    now = (
        reference_time
        or _parse_twitch_timestamp(clip.created_at)
        or _FALLBACK_REFERENCE_TIME
    )
    breakdown = {
        "views": _view_score(clip.view_count),
        "age": _age_score(clip.created_at, reference_time=now),
        "duration": _duration_score(clip.duration),
        "title": _title_score(clip.title),
    }
    total = round(
        breakdown["views"] * 0.55
        + breakdown["age"] * 0.25
        + breakdown["duration"] * 0.15
        + breakdown["title"] * 0.05,
        4,
    )
    return ClipRank(clip=clip, score=total, breakdown=breakdown)


def _view_score(view_count: int) -> float:
    if view_count <= 0:
        return 0.0
    return round(min(math.log10(view_count + 1) / 5.0, 1.0), 4)


def _age_score(created_at: str, *, reference_time: datetime) -> float:
    created = _parse_twitch_timestamp(created_at)
    if created is None:
        return 0.0

    age_hours = max((reference_time - created).total_seconds() / 3600, 0.0)
    age_days = age_hours / 24
    return round(max(1.0 - (age_days / 14), 0.0), 4)


def _duration_score(duration: float) -> float:
    if duration <= 0:
        return 0.0
    if 18 <= duration <= 45:
        return 1.0
    if duration < 18:
        return round(max(duration / 18, 0.0), 4)
    return round(max(1.0 - ((duration - 45) / 45), 0.0), 4)


def _title_score(title: str) -> float:
    clean_title = " ".join(title.split())
    if not clean_title:
        return 0.0
    if len(clean_title) < 8:
        return 0.5
    if len(clean_title) > 90:
        return 0.7
    return 1.0


def _parse_twitch_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _latest_created_at(clips: Sequence[TwitchClip]) -> datetime:
    created_times = (
        created_at
        for clip in clips
        if (created_at := _parse_twitch_timestamp(clip.created_at)) is not None
    )
    return max(created_times, default=_FALLBACK_REFERENCE_TIME)


def _created_at_sort_value(value: str) -> float:
    created = _parse_twitch_timestamp(value)
    if created is None:
        return 0.0
    return created.timestamp()

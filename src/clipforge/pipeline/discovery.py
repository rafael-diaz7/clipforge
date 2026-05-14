"""Discovery helpers for Twitch clips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable, Sequence

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.integrations.twitch import TwitchClip, list_channel_clips


@dataclass(frozen=True)
class ClipDiscoveryRequest:
    limit: int
    started_at: str | None
    ended_at: str | None


@dataclass(frozen=True)
class ClipDiscoveryResult:
    clips: tuple[TwitchClip, ...]
    requests: tuple[ClipDiscoveryRequest, ...]


ListChannelClipsFn = Callable[..., tuple[TwitchClip, ...]]


def discover_channel_clips(
    channel: str,
    *,
    config: ClipforgeConfig | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    limit: int | None = None,
    reference_time: datetime | None = None,
    list_clips_fn: ListChannelClipsFn = list_channel_clips,
) -> ClipDiscoveryResult:
    """Discover clips for a channel, deduping overlapping configured windows."""

    if ended_at and not started_at:
        raise ValueError("ended_at requires started_at for Twitch clip discovery.")

    config = config or load_config()
    if started_at:
        request = ClipDiscoveryRequest(
            limit=limit or max((window.limit for window in config.discovery_windows), default=100),
            started_at=started_at,
            ended_at=ended_at,
        )
        clips = list_clips_fn(
            channel,
            limit=request.limit,
            started_at=request.started_at,
            ended_at=request.ended_at,
            config=config,
        )
        return ClipDiscoveryResult(clips=_dedupe_clips(clips), requests=(request,))

    now = (reference_time or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    resolved_ended_at = _format_twitch_timestamp(now)
    requests = tuple(
        ClipDiscoveryRequest(
            limit=limit or window.limit,
            started_at=_format_twitch_timestamp(now - timedelta(days=window.days)),
            ended_at=resolved_ended_at,
        )
        for window in config.discovery_windows
    )

    discovered: list[TwitchClip] = []
    for request in requests:
        discovered.extend(
            list_clips_fn(
                channel,
                limit=request.limit,
                started_at=request.started_at,
                ended_at=request.ended_at,
                config=config,
            )
        )

    return ClipDiscoveryResult(
        clips=_dedupe_clips(discovered),
        requests=requests,
    )


def _dedupe_clips(clips: Sequence[TwitchClip]) -> tuple[TwitchClip, ...]:
    seen: set[str] = set()
    deduped: list[TwitchClip] = []
    for clip in clips:
        if clip.id in seen:
            continue
        seen.add(clip.id)
        deduped.append(clip)
    return tuple(deduped)


def _format_twitch_timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")

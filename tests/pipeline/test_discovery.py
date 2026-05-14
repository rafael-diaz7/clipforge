from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from clipforge.core.config import ClipDiscoveryWindow, ClipforgeConfig
from clipforge.integrations.twitch import TwitchClip
from clipforge.pipeline.discovery import discover_channel_clips


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")


def _clip(clip_id: str, *, views: int = 100) -> TwitchClip:
    return TwitchClip(
        id=clip_id,
        url=f"https://clips.twitch.tv/{clip_id}",
        broadcaster_name="Example",
        creator_name="Viewer",
        title=clip_id,
        view_count=views,
        created_at="2026-05-01T00:00:00Z",
        duration=30,
        thumbnail_url="https://example.test/thumb.jpg",
    )


def test_discover_channel_clips_runs_configured_7d_and_31d_windows(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_list_channel_clips(channel: str, **kwargs) -> tuple[TwitchClip, ...]:
        calls.append({"channel": channel, **kwargs})
        return (_clip(f"clip-{len(calls)}"),)

    result = discover_channel_clips(
        "example",
        config=_config(tmp_path),
        reference_time=datetime(2026, 5, 14, 12, tzinfo=UTC),
        list_clips_fn=fake_list_channel_clips,
    )

    assert [call["limit"] for call in calls] == [100, 100]
    assert [call["started_at"] for call in calls] == [
        "2026-05-07T12:00:00Z",
        "2026-04-13T12:00:00Z",
    ]
    assert {call["ended_at"] for call in calls} == {"2026-05-14T12:00:00Z"}
    assert [clip.id for clip in result.clips] == ["clip-1", "clip-2"]


def test_discover_channel_clips_dedupes_overlapping_windows_by_clip_id(
    tmp_path: Path,
) -> None:
    responses = iter(
        (
            (_clip("duplicate", views=100), _clip("fresh", views=50)),
            (_clip("duplicate", views=999), _clip("older", views=10)),
        )
    )

    result = discover_channel_clips(
        "example",
        config=_config(tmp_path),
        reference_time=datetime(2026, 5, 14, tzinfo=UTC),
        list_clips_fn=lambda *args, **kwargs: next(responses),
    )

    assert [clip.id for clip in result.clips] == ["duplicate", "fresh", "older"]
    assert result.clips[0].view_count == 100


def test_discover_channel_clips_allows_configured_window_limits(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    config = ClipforgeConfig(
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
        discovery_windows=(
            ClipDiscoveryWindow(days=3, limit=12),
            ClipDiscoveryWindow(days=14, limit=25),
        ),
    )

    discover_channel_clips(
        "example",
        config=config,
        reference_time=datetime(2026, 5, 14, tzinfo=UTC),
        list_clips_fn=lambda *args, **kwargs: calls.append(kwargs["limit"]) or (),
    )

    assert calls == [12, 25]

"""Pipeline artifact writers."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from clipforge.core.config import ClipforgeConfig
from clipforge.utils.paths import ensure_directory, safe_filename, utc_timestamp
from clipforge.integrations.twitch import TwitchClip, twitch_channel_login_from_input
from clipforge.media.download import DownloadResult
from clipforge.media.layouts import Layout


def write_clip_discovery_export(
    *,
    clips: Sequence[TwitchClip],
    channel: str,
    limit: int,
    started_at: str | None,
    ended_at: str | None,
    config: ClipforgeConfig,
    output_path: Path | None = None,
    discovery_windows: Sequence[Any] | None = None,
) -> Path:
    """Persist discovered Twitch clips in a queue-friendly JSON shape."""

    normalized_channel = twitch_channel_login_from_input(channel)
    export_path = output_path or _default_clip_discovery_export_path(
        normalized_channel,
        started_at=started_at,
        config=config,
    )
    ensure_directory(export_path.parent)
    payload = {
        "type": "clipforge.twitch_clip_discovery",
        "version": 1,
        "channel": normalized_channel,
        "created_at": utc_timestamp(),
        "filters": {
            "limit": limit,
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "clips": [asdict(clip) for clip in clips],
    }
    if discovery_windows is not None:
        payload["filters"]["windows"] = [
            {
                "limit": window.limit,
                "started_at": window.started_at,
                "ended_at": window.ended_at,
            }
            for window in discovery_windows
        ]
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return export_path


def write_metadata(
    *,
    clip_id: str,
    twitch_clip_url: str,
    download_result: DownloadResult,
    source_path: Path,
    layouts: Sequence[Layout],
    outputs: Sequence[dict[str, Any]],
    config: ClipforgeConfig,
    caption_metadata_path: Path | None = None,
) -> Path:
    """Persist metadata for a full pipeline run."""

    metadata_dir = ensure_directory(config.metadata_dir)
    metadata_path = metadata_dir / f"{clip_id}.json"
    created_at = utc_timestamp()
    payload: dict[str, Any] = {
        "clip_id": clip_id,
        "twitch_clip_url": twitch_clip_url,
        "downloader_backend": download_result.backend,
        "download_media_url": download_result.media_url,
        "source_path": str(source_path),
        "outputs": list(outputs),
        "layouts": [_layout_metadata_payload(layout) for layout in layouts],
        "target_resolution": {
            "width": config.target_width,
            "height": config.target_height,
        },
        "created_at": created_at,
        "rendered_at": created_at,
    }
    if caption_metadata_path is not None:
        payload["caption_metadata_path"] = str(caption_metadata_path)

    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def _default_clip_discovery_export_path(
    channel: str,
    *,
    started_at: str | None,
    config: ClipforgeConfig,
) -> Path:
    safe_channel = safe_filename(channel)
    date_prefix = _date_prefix_from_timestamp(started_at)
    return (
        config.metadata_dir
        / "discovered_clips"
        / safe_channel
        / f"{date_prefix}-{safe_channel}.json"
    )


def _layout_metadata_payload(layout: Layout) -> dict[str, Any]:
    payload = asdict(layout)
    if layout.caption_region is None:
        payload.pop("caption_region", None)
    return payload


def _date_prefix_from_timestamp(value: str | None) -> str:
    if value:
        return safe_filename(value[:10], fallback=_utc_date_prefix())
    return _utc_date_prefix()


def _utc_date_prefix() -> str:
    return datetime.now(UTC).date().isoformat()

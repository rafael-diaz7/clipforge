from __future__ import annotations

import json
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, EXAMPLE_LAYOUTS_DIR
from clipforge.media.download import DownloadResult
from clipforge.media.layouts import load_example_layouts
from clipforge.pipeline.artifacts import write_clip_discovery_export, write_metadata
from clipforge.integrations.twitch import TwitchClip
from tests.constants import TWITCH_CLIP_SLUG, TWITCH_CLIP_URL


def test_write_clip_discovery_export_uses_default_path(tmp_path: Path) -> None:
    config = ClipforgeConfig(
        metadata_dir=tmp_path / "metadata",
        example_layouts_dir=EXAMPLE_LAYOUTS_DIR,
    )
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

    export_path = write_clip_discovery_export(
        clips=clips,
        channel="https://twitch.tv/Example",
        limit=5,
        started_at="2026-05-01T00:00:00Z",
        ended_at="2026-05-06T00:00:00Z",
        config=config,
    )

    assert export_path == (
        tmp_path
        / "metadata"
        / "discovered_clips"
        / "example"
        / "2026-05-01-example.json"
    )
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["type"] == "clipforge.twitch_clip_discovery"
    assert payload["version"] == 1
    assert payload["channel"] == "example"
    assert payload["filters"] == {
        "limit": 5,
        "started_at": "2026-05-01T00:00:00Z",
        "ended_at": "2026-05-06T00:00:00Z",
    }
    assert payload["clips"][0]["url"] == "https://clips.twitch.tv/clip-1"
    assert payload["clips"][0]["view_count"] == 42


def test_write_clip_discovery_export_accepts_custom_path(tmp_path: Path) -> None:
    output_path = tmp_path / "queue.json"

    export_path = write_clip_discovery_export(
        clips=(),
        channel="example",
        limit=10,
        started_at=None,
        ended_at=None,
        config=ClipforgeConfig(metadata_dir=tmp_path / "metadata"),
        output_path=output_path,
    )

    assert export_path == output_path
    assert output_path.exists()


def test_write_metadata_records_full_pipeline_artifacts(tmp_path: Path) -> None:
    config = ClipforgeConfig(
        metadata_dir=tmp_path / "metadata",
        example_layouts_dir=EXAMPLE_LAYOUTS_DIR,
    )
    source_path = tmp_path / "downloads" / f"{TWITCH_CLIP_SLUG}.mp4"
    layouts = load_example_layouts(("center_gameplay",), layouts_dir=EXAMPLE_LAYOUTS_DIR)
    outputs = ({"layout": "center_gameplay", "path": str(tmp_path / "render.mp4")},)

    metadata_path = write_metadata(
        clip_id=TWITCH_CLIP_SLUG,
        twitch_clip_url=TWITCH_CLIP_URL,
        download_result=DownloadResult(
            source_path=source_path,
            backend="clipr",
            media_url="https://cdn.example.test/source.mp4",
        ),
        source_path=source_path,
        layouts=layouts,
        outputs=outputs,
        config=config,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["clip_id"] == TWITCH_CLIP_SLUG
    assert payload["twitch_clip_url"] == TWITCH_CLIP_URL
    assert payload["downloader_backend"] == "clipr"
    assert payload["download_media_url"] == "https://cdn.example.test/source.mp4"
    assert payload["source_path"] == str(source_path)
    assert payload["outputs"] == list(outputs)
    assert [layout["name"] for layout in payload["layouts"]] == ["center_gameplay"]
    assert "caption_region" not in payload["layouts"][0]
    assert payload["target_resolution"] == {"width": 1080, "height": 1920}
    assert "caption_metadata_path" not in payload
    assert payload["created_at"].endswith("+00:00")
    assert payload["rendered_at"].endswith("+00:00")


def test_write_metadata_optionally_references_caption_metadata(tmp_path: Path) -> None:
    config = ClipforgeConfig(
        metadata_dir=tmp_path / "metadata",
        example_layouts_dir=EXAMPLE_LAYOUTS_DIR,
    )
    source_path = tmp_path / "downloads" / f"{TWITCH_CLIP_SLUG}.mp4"
    caption_path = tmp_path / "metadata" / "captions" / f"{TWITCH_CLIP_SLUG}.json"

    metadata_path = write_metadata(
        clip_id=TWITCH_CLIP_SLUG,
        twitch_clip_url=TWITCH_CLIP_URL,
        download_result=DownloadResult(
            source_path=source_path,
            backend="clipr",
            media_url="https://cdn.example.test/source.mp4",
        ),
        source_path=source_path,
        layouts=(),
        outputs=(),
        config=config,
        caption_metadata_path=caption_path,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["caption_metadata_path"] == str(caption_path)

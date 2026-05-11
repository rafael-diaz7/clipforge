from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig
from clipforge.media.captions import CaptionMetadata, CaptionSegment, save_captions
from clipforge.media.download import DownloadResult
from clipforge.pipeline.cli import main
from clipforge.integrations.twitch import TwitchClip
from clipforge.storage.state import (
    get_clip,
    mark_clip_failed,
    mark_clip_needs_rerender,
    mark_clip_rendered,
    upsert_discovered_clip,
)
from tests.constants import TWITCH_CLIP_URL


def test_main_supports_full_pipeline_url_shortcut(
    monkeypatch,
    capsys,
) -> None:
    calls: list[str] = []

    def fake_process(url: str) -> Path:
        calls.append(url)
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["--url", TWITCH_CLIP_URL])

    assert exit_code == 0
    assert calls == [TWITCH_CLIP_URL]
    assert capsys.readouterr().err == ""


def test_main_routes_url_shortcut_caption_flag(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_process(url: str, *, generate_captions: bool) -> Path:
        calls.append({"url": url, "generate_captions": generate_captions})
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["--url", TWITCH_CLIP_URL, "--generate-captions"])

    assert exit_code == 0
    assert calls == [{"url": TWITCH_CLIP_URL, "generate_captions": True}]


def test_main_routes_url_shortcut_force_caption_flag(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_process(
        url: str,
        *,
        generate_captions: bool,
        force_captions: bool,
    ) -> Path:
        calls.append(
            {
                "url": url,
                "generate_captions": generate_captions,
                "force_captions": force_captions,
            }
        )
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(
        ["--url", TWITCH_CLIP_URL, "--generate-captions", "--force-captions"]
    )

    assert exit_code == 0
    assert calls == [
        {
            "url": TWITCH_CLIP_URL,
            "generate_captions": True,
            "force_captions": True,
        }
    ]


def test_main_routes_process_caption_flag(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_process(url: str, *, generate_captions: bool) -> Path:
        calls.append({"url": url, "generate_captions": generate_captions})
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["process", "--url", TWITCH_CLIP_URL, "--generate-captions"])

    assert exit_code == 0
    assert calls == [{"url": TWITCH_CLIP_URL, "generate_captions": True}]


def test_main_routes_process_force_caption_flag(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_process(
        url: str,
        *,
        generate_captions: bool,
        force_captions: bool,
    ) -> Path:
        calls.append(
            {
                "url": url,
                "generate_captions": generate_captions,
                "force_captions": force_captions,
            }
        )
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(
        [
            "process",
            "--url",
            TWITCH_CLIP_URL,
            "--generate-captions",
            "--force-captions",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "url": TWITCH_CLIP_URL,
            "generate_captions": True,
            "force_captions": True,
        }
    ]


def test_main_routes_render_all_command(monkeypatch, capsys) -> None:
    def fake_render_all(source_path: Path, *, clip_id: str | None = None) -> tuple[Path, ...]:
        assert source_path == Path("source.mp4")
        assert clip_id == "test-clip"
        return (Path("one.mp4"), Path("two.mp4"))

    monkeypatch.setattr("clipforge.pipeline.cli.render_all_candidates", fake_render_all)

    exit_code = main(
        ["render-all", "--source", "source.mp4", "--clip-id", "test-clip"]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["one.mp4", "two.mp4"]


def test_main_routes_render_command_caption_path(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_render_candidate(
        source_path: Path,
        *,
        layout_ref: str,
        clip_id: str | None = None,
        caption_metadata_path: Path | None = None,
    ) -> Path:
        calls.append(
            {
                "source_path": source_path,
                "layout_ref": layout_ref,
                "clip_id": clip_id,
                "caption_metadata_path": caption_metadata_path,
            }
        )
        return Path("render.mp4")

    monkeypatch.setattr("clipforge.pipeline.cli.render_candidate", fake_render_candidate)

    exit_code = main(
        [
            "render",
            "--source",
            "source.mp4",
            "--layout",
            "center_gameplay",
            "--clip-id",
            "test-clip",
            "--captions",
            "captions.json",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "source_path": Path("source.mp4"),
            "layout_ref": "center_gameplay",
            "clip_id": "test-clip",
            "caption_metadata_path": Path("captions.json"),
        }
    ]
    assert capsys.readouterr().out.splitlines() == ["render.mp4"]


def test_main_routes_render_all_command_caption_path(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_render_all(
        source_path: Path,
        *,
        clip_id: str | None = None,
        caption_metadata_path: Path | None = None,
    ) -> tuple[Path, ...]:
        calls.append(
            {
                "source_path": source_path,
                "clip_id": clip_id,
                "caption_metadata_path": caption_metadata_path,
            }
        )
        return (Path("one.mp4"), Path("two.mp4"))

    monkeypatch.setattr("clipforge.pipeline.cli.render_all_candidates", fake_render_all)

    exit_code = main(
        [
            "render-all",
            "--source",
            "source.mp4",
            "--clip-id",
            "test-clip",
            "--captions",
            "captions.json",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "source_path": Path("source.mp4"),
            "clip_id": "test-clip",
            "caption_metadata_path": Path("captions.json"),
        }
    ]
    assert capsys.readouterr().out.splitlines() == ["one.mp4", "two.mp4"]


def test_main_routes_render_all_static_layouts_flag(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_render_all(
        source_path: Path,
        *,
        clip_id: str | None = None,
        use_generated_layouts: bool = True,
    ) -> tuple[Path, ...]:
        calls.append(
            {
                "source_path": source_path,
                "clip_id": clip_id,
                "use_generated_layouts": use_generated_layouts,
            }
        )
        return (Path("static-one.mp4"),)

    monkeypatch.setattr("clipforge.pipeline.cli.render_all_candidates", fake_render_all)

    exit_code = main(
        [
            "render-all",
            "--source",
            "source.mp4",
            "--clip-id",
            "test-clip",
            "--static-layouts",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "source_path": Path("source.mp4"),
            "clip_id": "test-clip",
            "use_generated_layouts": False,
        }
    ]
    assert capsys.readouterr().out.splitlines() == ["static-one.mp4"]


def test_main_routes_process_static_layouts_flag(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_process(url: str, *, use_generated_layouts: bool) -> Path:
        calls.append({"url": url, "use_generated_layouts": use_generated_layouts})
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["process", "--url", TWITCH_CLIP_URL, "--static-layouts"])

    assert exit_code == 0
    assert calls == [
        {"url": TWITCH_CLIP_URL, "use_generated_layouts": False},
    ]


def test_main_routes_captions_command(monkeypatch, capsys, tmp_path: Path) -> None:
    config = ClipforgeConfig(openai_api_key="test-key", metadata_dir=tmp_path / "metadata")
    caption_path = tmp_path / "metadata" / "captions" / "clip-123.json"
    calls: list[dict[str, object]] = []

    def fake_generate_caption_metadata(
        source_path: Path,
        *,
        clip_id: str,
        output_path: Path | None,
        config: ClipforgeConfig,
    ) -> Path:
        calls.append(
            {
                "source_path": source_path,
                "clip_id": clip_id,
                "output_path": output_path,
                "config": config,
            }
        )
        return caption_path

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr(
        "clipforge.pipeline.cli.generate_caption_metadata",
        fake_generate_caption_metadata,
    )

    exit_code = main(
        [
            "captions",
            "--source",
            "source.mp4",
            "--clip-id",
            "clip-123",
            "--output",
            str(caption_path),
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "source_path": Path("source.mp4"),
            "clip_id": "clip-123",
            "output_path": caption_path,
            "config": config,
        }
    ]
    assert capsys.readouterr().out.splitlines() == [str(caption_path)]


def test_main_routes_analyze_frames_command(monkeypatch, capsys, tmp_path: Path) -> None:
    metadata_path = tmp_path / "analysis" / "clip-123" / "frames.json"
    calls: list[dict[str, object]] = []

    def fake_sample_frames(
        source_path: Path,
        *,
        clip_id: str,
        count: int,
        interval_seconds: float | None,
    ) -> Path:
        calls.append(
            {
                "source_path": source_path,
                "clip_id": clip_id,
                "count": count,
                "interval_seconds": interval_seconds,
            }
        )
        return metadata_path

    monkeypatch.setattr("clipforge.pipeline.cli.sample_frames", fake_sample_frames)

    exit_code = main(
        [
            "analyze",
            "frames",
            "--source",
            "source.mp4",
            "--clip-id",
            "clip-123",
            "--count",
            "4",
            "--interval-seconds",
            "1.5",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "source_path": Path("source.mp4"),
            "clip_id": "clip-123",
            "count": 4,
            "interval_seconds": 1.5,
        }
    ]
    assert capsys.readouterr().out.splitlines() == [str(metadata_path)]


def test_main_routes_analyze_overlay_command(monkeypatch, capsys, tmp_path: Path) -> None:
    overlay_path = tmp_path / "analysis" / "clip-123" / "overlay.json"
    calls: list[str] = []

    def fake_analyze_overlay(*, clip_id: str) -> Path:
        calls.append(clip_id)
        return overlay_path

    monkeypatch.setattr("clipforge.pipeline.cli.analyze_overlay", fake_analyze_overlay)

    exit_code = main(["analyze", "overlay", "--clip-id", "clip-123"])

    assert exit_code == 0
    assert calls == ["clip-123"]
    assert capsys.readouterr().out.splitlines() == [str(overlay_path)]


def test_main_routes_analyze_overlay_debug_command(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    debug_dir = tmp_path / "analysis" / "clip-123" / "debug"
    calls: list[str] = []

    def fake_write_overlay_debug_images(*, clip_id: str) -> Path:
        calls.append(clip_id)
        return debug_dir

    monkeypatch.setattr(
        "clipforge.pipeline.cli.write_overlay_debug_images",
        fake_write_overlay_debug_images,
    )

    exit_code = main(["analyze", "overlay-debug", "--clip-id", "clip-123"])

    assert exit_code == 0
    assert calls == ["clip-123"]
    assert capsys.readouterr().out.splitlines() == [str(debug_dir)]


def test_main_routes_analyze_layouts_command(monkeypatch, capsys, tmp_path: Path) -> None:
    layout_paths = (
        tmp_path / "analysis" / "clip-123" / "layouts" / "detected_streamer_focus.json",
        tmp_path / "analysis" / "clip-123" / "layouts" / "detected_hybrid.json",
        tmp_path
        / "analysis"
        / "clip-123"
        / "layouts"
        / "detected_hybrid_full_game_bottom.json",
    )
    calls: list[str] = []

    def fake_generate_detected_layout_candidates(*, clip_id: str) -> tuple[Path, ...]:
        calls.append(clip_id)
        return layout_paths

    monkeypatch.setattr(
        "clipforge.pipeline.cli.generate_detected_layout_candidates",
        fake_generate_detected_layout_candidates,
    )

    exit_code = main(["analyze", "layouts", "--clip-id", "clip-123"])

    assert exit_code == 0
    assert calls == ["clip-123"]
    assert capsys.readouterr().out.splitlines() == [str(path) for path in layout_paths]


def test_main_routes_clips_command(monkeypatch, capsys, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []
    config = ClipforgeConfig(
        twitch_client_id="client-id",
        twitch_client_secret="client-secret",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
    )

    def fake_list_channel_clips(
        channel_login: str,
        *,
        limit: int,
        started_at: str | None,
        ended_at: str | None,
        config: ClipforgeConfig,
    ) -> tuple[TwitchClip, ...]:
        calls.append(
            {
                "channel_login": channel_login,
                "limit": limit,
                "started_at": started_at,
                "ended_at": ended_at,
                "config": config,
            }
        )
        return (
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

    def fake_record_discovered_clips(
        *,
        clips: tuple[TwitchClip, ...],
        channel: str,
        config: ClipforgeConfig,
    ) -> None:
        recorded.append({"clips": clips, "channel": channel, "config": config})

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.list_channel_clips", fake_list_channel_clips)
    monkeypatch.setattr(
        "clipforge.pipeline.cli.record_discovered_clips",
        fake_record_discovered_clips,
    )

    exit_code = main(
        [
            "clips",
            "--channel",
            "example",
            "--limit",
            "5",
            "--started-at",
            "2026-05-01T00:00:00Z",
            "--ended-at",
            "2026-05-06T00:00:00Z",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "channel_login": "example",
            "limit": 5,
            "started_at": "2026-05-01T00:00:00Z",
            "ended_at": "2026-05-06T00:00:00Z",
            "config": config,
        }
    ]
    assert recorded[0]["channel"] == "example"
    assert capsys.readouterr().out.splitlines() == [
        "2026-05-01T00:00:00Z\t42\t28.5s\thttps://clips.twitch.tv/clip-1\tgreat clip"
    ]


def test_main_exports_clips_command_as_json(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(
        twitch_client_id="client-id",
        twitch_client_secret="client-secret",
        metadata_dir=tmp_path / "metadata",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
    )
    export_path = tmp_path / "metadata" / "discovered_clips" / "example.json"

    def fake_list_channel_clips(
        channel_login: str,
        *,
        limit: int,
        started_at: str | None,
        ended_at: str | None,
        config: ClipforgeConfig,
    ) -> tuple[TwitchClip, ...]:
        assert channel_login == "https://twitch.tv/Example"
        assert limit == 5
        assert started_at == "2026-05-01T00:00:00Z"
        assert ended_at == "2026-05-06T00:00:00Z"
        return ()

    def fake_write_clip_discovery_export(**kwargs) -> Path:
        assert kwargs["channel"] == "https://twitch.tv/Example"
        assert kwargs["limit"] == 5
        assert kwargs["started_at"] == "2026-05-01T00:00:00Z"
        assert kwargs["ended_at"] == "2026-05-06T00:00:00Z"
        return export_path

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.list_channel_clips", fake_list_channel_clips)
    monkeypatch.setattr(
        "clipforge.pipeline.cli.record_discovered_clips",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "clipforge.pipeline.cli.write_clip_discovery_export",
        fake_write_clip_discovery_export,
    )

    exit_code = main(
        [
            "clips",
            "--channel",
            "https://twitch.tv/Example",
            "--limit",
            "5",
            "--started-at",
            "2026-05-01T00:00:00Z",
            "--ended-at",
            "2026-05-06T00:00:00Z",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [f"export: {export_path}"]


def test_main_exports_clips_command_to_custom_json_path(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(
        twitch_client_id="client-id",
        twitch_client_secret="client-secret",
        metadata_dir=tmp_path / "metadata",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
    )
    output_path = tmp_path / "queue.json"

    def fake_write_clip_discovery_export(**kwargs) -> Path:
        assert kwargs["output_path"] == output_path
        return output_path

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr(
        "clipforge.pipeline.cli.list_channel_clips",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.cli.record_discovered_clips",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "clipforge.pipeline.cli.write_clip_discovery_export",
        fake_write_clip_discovery_export,
    )

    exit_code = main(
        [
            "clips",
            "--channel",
            "example",
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [f"export: {output_path}"]


def test_main_lists_pending_clips_from_state(monkeypatch, capsys, tmp_path: Path) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        streamer_login="example",
        title="first",
        view_count=10,
        duration_seconds=12.5,
        rank_score=0.2,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-2",
        url="https://clips.twitch.tv/clip-2",
        streamer_login="example",
        title="second",
        view_count=20,
        duration_seconds=30,
        rank_score=0.9,
        db_path=config.state_db_path,
    )

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)

    exit_code = main(["clips", "pending"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "https://clips.twitch.tv/clip-1" not in output
    assert "https://clips.twitch.tv/clip-2" not in output
    assert output.splitlines() == [
        "rank  streamer  score  views  duration  status      clip_id  title",
        "----  --------  -----  -----  --------  ----------  -------  ------",
        "1     example   0.9    20     30s       discovered  clip-2   second",
        "2     example   0.2    10     12.5s     discovered  clip-1   first",
    ]


def test_main_lists_pending_clips_with_limit_channel_and_url(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-example-1",
        url="https://clips.twitch.tv/clip-example-1",
        streamer_login="example",
        title="example first",
        rank_score=0.2,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-other",
        url="https://clips.twitch.tv/clip-other",
        streamer_login="other",
        title="other channel",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-example-2",
        url="https://clips.twitch.tv/clip-example-2",
        streamer_login="example",
        title="example second",
        rank_score=0.9,
        db_path=config.state_db_path,
    )

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)

    exit_code = main(
        [
            "clips",
            "pending",
            "--channel",
            "Example",
            "--limit",
            "1",
            "--show-url",
        ]
    )

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0].split() == [
        "rank",
        "streamer",
        "score",
        "views",
        "duration",
        "status",
        "clip_id",
        "url",
        "title",
    ]
    assert len(lines) == 3
    assert "clip-example-2" in lines[2]
    assert "https://clips.twitch.tv/clip-example-2" in lines[2]
    assert "clip-example-1" not in "\n".join(lines)
    assert "clip-other" not in "\n".join(lines)


def test_main_processes_top_pending_clip_from_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        rank_score=0.2,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-2",
        url="https://clips.twitch.tv/clip-2",
        rank_score=0.9,
        db_path=config.state_db_path,
    )
    calls: list[dict[str, object]] = []

    def fake_process(url: str, *, config: ClipforgeConfig) -> Path:
        calls.append({"url": url, "config": config})
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--top", "1"])

    assert exit_code == 0
    assert calls == [{"url": "https://clips.twitch.tv/clip-2", "config": config}]


def test_main_processes_top_pending_clips_by_rank_score(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-low",
        url="https://clips.twitch.tv/clip-low",
        rank_score=0.2,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-high",
        url="https://clips.twitch.tv/clip-high",
        rank_score=0.9,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-mid",
        url="https://clips.twitch.tv/clip-mid",
        rank_score=0.5,
        db_path=config.state_db_path,
    )
    calls: list[str] = []

    def fake_process(url: str, *, config: ClipforgeConfig) -> Path:
        calls.append(url)
        return Path(f"{url.rsplit('/', 1)[-1]}.json")

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--top", "2"])

    assert exit_code == 0
    assert calls == [
        "https://clips.twitch.tv/clip-high",
        "https://clips.twitch.tv/clip-mid",
    ]
    assert capsys.readouterr().out.splitlines() == [
        "processed: clip-high: clip-high.json",
        "processed: clip-mid: clip-mid.json",
    ]


def test_main_processes_needs_rerender_clip_with_force(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rerender",
        url="https://clips.twitch.tv/clip-rerender",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_needs_rerender(
        "clip-rerender",
        skip_reason="layouts did not fit",
        db_path=config.state_db_path,
    )
    calls: list[dict[str, object]] = []

    def fake_process(url: str, **kwargs) -> Path:
        calls.append({"url": url, **kwargs})
        mark_clip_rendered(
            "clip-rerender",
            render_dir=tmp_path / "renders" / "clip-rerender",
            metadata_path=tmp_path / "metadata" / "clip-rerender.json",
            db_path=config.state_db_path,
        )
        return tmp_path / "metadata" / "clip-rerender.json"

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--top", "1"])

    state = get_clip("clip-rerender", db_path=config.state_db_path)
    assert exit_code == 0
    assert calls == [
        {
            "url": "https://clips.twitch.tv/clip-rerender",
            "config": config,
            "force": True,
        }
    ]
    assert state is not None
    assert state.status == "rendered"
    assert capsys.readouterr().out.splitlines() == [
        f"processed: clip-rerender: {tmp_path / 'metadata' / 'clip-rerender.json'}"
    ]


def test_main_processes_pending_clips_sequentially_and_preserves_rendered_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    for clip_id, score in (("clip-1", 0.9), ("clip-2", 0.8)):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            rank_score=score,
            db_path=config.state_db_path,
        )
    calls: list[str] = []

    def fake_process(url: str, *, config: ClipforgeConfig) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        mark_clip_rendered(
            clip_id,
            render_dir=tmp_path / "renders" / clip_id,
            metadata_path=tmp_path / "metadata" / f"{clip_id}.json",
            db_path=config.state_db_path,
        )
        return tmp_path / "metadata" / f"{clip_id}.json"

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--top", "2"])

    assert exit_code == 0
    assert calls == ["clip-1", "clip-2"]
    first_state = get_clip("clip-1", db_path=config.state_db_path)
    second_state = get_clip("clip-2", db_path=config.state_db_path)
    assert first_state is not None
    assert second_state is not None
    assert first_state.status == "rendered"
    assert second_state.status == "rendered"
    assert first_state.metadata_path == str(tmp_path / "metadata" / "clip-1.json")
    assert second_state.metadata_path == str(tmp_path / "metadata" / "clip-2.json")


def test_main_stops_on_first_clip_processing_failure_by_default(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    for clip_id, score in (("clip-1", 0.9), ("clip-2", 0.8), ("clip-3", 0.7)):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            rank_score=score,
            db_path=config.state_db_path,
        )
    calls: list[str] = []

    def fake_process(url: str, *, config: ClipforgeConfig) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        if clip_id == "clip-2":
            raise RuntimeError("render failed")
        return tmp_path / "metadata" / f"{clip_id}.json"

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--top", "3"])

    failed_state = get_clip("clip-2", db_path=config.state_db_path)
    unprocessed_state = get_clip("clip-3", db_path=config.state_db_path)
    assert exit_code == 1
    assert calls == ["clip-1", "clip-2"]
    assert failed_state is not None
    assert failed_state.status == "failed"
    assert failed_state.error_message == "render failed"
    assert unprocessed_state is not None
    assert unprocessed_state.status == "discovered"
    assert capsys.readouterr().out.splitlines() == [
        f"processed: clip-1: {tmp_path / 'metadata' / 'clip-1.json'}",
        "failed: clip-2: render failed",
    ]


def test_main_continue_on_error_processes_remaining_pending_clips(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    for clip_id, score in (("clip-1", 0.9), ("clip-2", 0.8), ("clip-3", 0.7)):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            rank_score=score,
            db_path=config.state_db_path,
        )
    calls: list[str] = []

    def fake_process(url: str, *, config: ClipforgeConfig) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        if clip_id == "clip-2":
            raise RuntimeError("render failed")
        return tmp_path / "metadata" / f"{clip_id}.json"

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--top", "3", "--continue-on-error"])

    failed_state = get_clip("clip-2", db_path=config.state_db_path)
    remaining_state = get_clip("clip-3", db_path=config.state_db_path)
    assert exit_code == 1
    assert calls == ["clip-1", "clip-2", "clip-3"]
    assert failed_state is not None
    assert failed_state.status == "failed"
    assert failed_state.error_message == "render failed"
    assert remaining_state is not None
    assert remaining_state.status == "discovered"
    assert capsys.readouterr().out.splitlines() == [
        f"processed: clip-1: {tmp_path / 'metadata' / 'clip-1.json'}",
        "failed: clip-2: render failed",
        f"processed: clip-3: {tmp_path / 'metadata' / 'clip-3.json'}",
    ]


def test_main_processes_specific_pending_clip_from_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        rank_score=0.9,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-2",
        url="https://clips.twitch.tv/clip-2",
        rank_score=0.2,
        db_path=config.state_db_path,
    )
    calls: list[str] = []

    def fake_process(url: str, *, config: ClipforgeConfig) -> Path:
        calls.append(url)
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--clip-id", "clip-2"])

    assert exit_code == 0
    assert calls == ["https://clips.twitch.tv/clip-2"]


def test_main_rejects_rendered_clip_without_force(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=config.state_db_path,
    )

    def fail_process(*args, **kwargs) -> Path:
        raise AssertionError("Rendered clips should require --force before processing.")

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fail_process)

    exit_code = main(["clips", "process", "--clip-id", "clip-rendered"])

    captured = capsys.readouterr()
    state = get_clip("clip-rendered", db_path=config.state_db_path)
    assert exit_code == 1
    assert "Clip is already rendered: clip-rendered" in captured.err
    assert "--force" in captured.err
    assert state is not None
    assert state.status == "rendered"


def test_main_reprocesses_rendered_clip_by_id_with_force(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "old",
        metadata_path=tmp_path / "metadata" / "old.json",
        db_path=config.state_db_path,
    )
    calls: list[str] = []

    def fake_process(url: str, *, force: bool, config: ClipforgeConfig) -> Path:
        assert force is True
        calls.append(url)
        metadata_path = tmp_path / "metadata" / "new.json"
        mark_clip_rendered(
            "clip-rendered",
            render_dir=tmp_path / "renders" / "new",
            metadata_path=metadata_path,
            db_path=config.state_db_path,
        )
        return metadata_path

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--clip-id", "clip-rendered", "--force"])

    state = get_clip("clip-rendered", db_path=config.state_db_path)
    assert exit_code == 0
    assert calls == ["https://clips.twitch.tv/clip-rendered"]
    assert state is not None
    assert state.status == "rendered"
    assert state.metadata_path == str(tmp_path / "metadata" / "new.json")
    assert state.render_dir == str(tmp_path / "renders" / "new")
    assert capsys.readouterr().out.splitlines() == [
        f"processed: clip-rendered: {tmp_path / 'metadata' / 'new.json'}"
    ]


def test_main_rerenders_rendered_clip_by_id_without_force(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "old",
        metadata_path=tmp_path / "metadata" / "old.json",
        db_path=config.state_db_path,
    )
    calls: list[dict[str, object]] = []

    def fake_process(
        url: str,
        *,
        rerender: bool,
        config: ClipforgeConfig,
    ) -> Path:
        calls.append({"url": url, "rerender": rerender, "config": config})
        return tmp_path / "metadata" / "new.json"

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "process", "--clip-id", "clip-rendered", "--rerender"])

    assert exit_code == 0
    assert calls == [
        {
            "url": "https://clips.twitch.tv/clip-rendered",
            "rerender": True,
            "config": config,
        }
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"processed: clip-rendered: {tmp_path / 'metadata' / 'new.json'}"
    ]


def test_main_routes_clips_rerender_command(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        db_path=config.state_db_path,
    )
    metadata_path = tmp_path / "metadata" / "clip-1.json"
    calls: list[dict[str, object]] = []

    def fake_process(
        url: str,
        *,
        rerender: bool,
        use_generated_layouts: bool,
        config: ClipforgeConfig,
    ) -> Path:
        calls.append(
            {
                "url": url,
                "rerender": rerender,
                "use_generated_layouts": use_generated_layouts,
                "config": config,
            }
        )
        return metadata_path

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["clips", "rerender", "--clip-id", "clip-1"])

    assert exit_code == 0
    assert calls == [
        {
            "url": "https://clips.twitch.tv/clip-1",
            "rerender": True,
            "use_generated_layouts": True,
            "config": config,
        }
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"rerendered: clip-1: {metadata_path}"
    ]


def test_main_reprocesses_rendered_clip_with_force_and_captions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=config.state_db_path,
    )
    calls: list[dict[str, object]] = []

    def fake_process(
        url: str,
        *,
        generate_captions: bool,
        force: bool,
        config: ClipforgeConfig,
    ) -> Path:
        calls.append(
            {
                "url": url,
                "generate_captions": generate_captions,
                "force": force,
                "config": config,
            }
        )
        return tmp_path / "metadata" / "captions.json"

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(
        [
            "clips",
            "process",
            "--clip-id",
            "clip-rendered",
            "--force",
            "--generate-captions",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "url": "https://clips.twitch.tv/clip-rendered",
            "generate_captions": True,
            "force": True,
            "config": config,
        }
    ]


def test_main_routes_clips_process_force_caption_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=config.state_db_path,
    )
    calls: list[dict[str, object]] = []

    def fake_process(
        url: str,
        *,
        generate_captions: bool,
        force_captions: bool,
        force: bool,
        config: ClipforgeConfig,
    ) -> Path:
        calls.append(
            {
                "url": url,
                "generate_captions": generate_captions,
                "force_captions": force_captions,
                "force": force,
                "config": config,
            }
        )
        return tmp_path / "metadata" / "captions.json"

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(
        [
            "clips",
            "process",
            "--clip-id",
            "clip-rendered",
            "--force",
            "--generate-captions",
            "--force-captions",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "url": "https://clips.twitch.tv/clip-rendered",
            "generate_captions": True,
            "force_captions": True,
            "force": True,
            "config": config,
        }
    ]


def test_main_reprocesses_rendered_clip_reuses_existing_captions_by_default(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(
        downloads_dir=tmp_path / "downloads",
        renders_dir=tmp_path / "renders",
        metadata_dir=tmp_path / "metadata",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
    )
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "old",
        db_path=config.state_db_path,
    )
    caption_path = save_captions(
        clip_id="clip-rendered",
        segments=(CaptionSegment(start_time=0, end_time=1, text="existing"),),
        config=config,
    )
    source_path = tmp_path / "downloads" / "clip-rendered" / "ytdlp" / "clip-rendered.mp4"
    events: list[str] = []

    def fake_download_twitch_clip(
        url: str,
        *,
        clip_id: str | None,
        config: ClipforgeConfig,
        on_media_url_resolved,
    ) -> DownloadResult:
        events.append("download")
        return DownloadResult(source_path=source_path, backend="ytdlp")

    def fail_generate_caption_metadata(*args, **kwargs) -> Path:
        raise AssertionError("Forced clip reprocessing should reuse captions by default.")

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        assert caption_metadata.clip_id == "clip-rendered"
        events.append(f"render:{layout.name}")
        return output

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        fake_download_twitch_clip,
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_caption_metadata",
        fail_generate_caption_metadata,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    exit_code = main(
        [
            "clips",
            "process",
            "--clip-id",
            "clip-rendered",
            "--force",
            "--generate-captions",
            "--static-layouts",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"captions: reusing existing {caption_path}" in output
    assert events == [
        "download",
        "render:center_gameplay",
        "render:fullscreen_downscaled_blur_bg",
        "render:facecam_focus",
        "render:hybrid",
        "render:hybrid_full_game_bottom",
    ]


def test_main_clips_pending_and_top_process_exclude_rendered_clips(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        title="rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-pending",
        url="https://clips.twitch.tv/clip-pending",
        title="pending",
        rank_score=0.5,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=config.state_db_path,
    )
    calls: list[str] = []

    def fake_process(url: str, *, config: ClipforgeConfig) -> Path:
        calls.append(url)
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    pending_exit_code = main(["clips", "pending"])
    process_exit_code = main(["clips", "process", "--top", "1"])

    assert pending_exit_code == 0
    assert process_exit_code == 0
    pending_output = capsys.readouterr().out
    assert "clip-pending" in pending_output
    assert "clip-rendered" not in pending_output
    assert "https://clips.twitch.tv/clip-pending" not in pending_output
    assert calls == ["https://clips.twitch.tv/clip-pending"]
    assert get_clip("clip-rendered", db_path=config.state_db_path).status == "rendered"


def test_main_reranks_clips_from_state_without_changing_status(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        streamer_login="example",
        title="rendered clip",
        view_count=100,
        created_at="2026-05-01T00:00:00Z",
        duration_seconds=30,
        rank_score=0.01,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=config.state_db_path,
    )

    def fail_twitch(*args, **kwargs) -> tuple[TwitchClip, ...]:
        raise AssertionError("Twitch should not be called during rerank.")

    def fail_process(*args, **kwargs) -> Path:
        raise AssertionError("Processing should not run during rerank.")

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.list_channel_clips", fail_twitch)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fail_process)

    exit_code = main(["clips", "rerank"])

    state = get_clip("clip-rendered", db_path=config.state_db_path)
    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["Reranked 1 clip"]
    assert state is not None
    assert state.status == "rendered"
    assert state.rank_score != 0.01


def test_main_resets_one_clip_to_discovered(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        metadata_path=tmp_path / "metadata" / "clip-rendered.json",
        db_path=config.state_db_path,
    )

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)

    exit_code = main(["clips", "reset", "--clip-id", "clip-rendered"])

    state = get_clip("clip-rendered", db_path=config.state_db_path)
    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["Reset 1 clip to discovered"]
    assert state is not None
    assert state.status == "discovered"
    assert state.render_dir is None
    assert state.metadata_path is None


def test_main_resets_all_clips_to_discovered(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    for clip_id in ("clip-rendered", "clip-failed"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            db_path=config.state_db_path,
        )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=config.state_db_path,
    )
    mark_clip_failed(
        "clip-failed",
        error_message="render failed",
        db_path=config.state_db_path,
    )

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)

    exit_code = main(["clips", "reset", "--all"])

    rendered = get_clip("clip-rendered", db_path=config.state_db_path)
    failed = get_clip("clip-failed", db_path=config.state_db_path)
    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["Reset 2 clips to discovered"]
    assert rendered is not None
    assert failed is not None
    assert rendered.status == "discovered"
    assert failed.status == "discovered"
    assert rendered.render_dir is None
    assert failed.error_message is None


def test_main_returns_non_zero_when_reset_clip_is_missing(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)

    exit_code = main(["clips", "reset", "--clip-id", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Clip not found: missing" in captured.err


def test_main_routes_clips_review_command(monkeypatch, capsys, tmp_path: Path) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    calls: list[dict[str, object]] = []
    export_path = tmp_path / "exports" / "ready" / "jynxzi" / "clip-1" / "hybrid.mp4"

    def fake_review_streamer_clips(**kwargs) -> tuple[Path, ...]:
        calls.append(kwargs)
        return (export_path,)

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr(
        "clipforge.pipeline.cli.review_streamer_clips",
        fake_review_streamer_clips,
    )

    exit_code = main(
        [
            "clips",
            "review",
            "--streamer",
            "jynxzi",
            "--count",
            "3",
            "--force",
            "--generate-captions",
            "--clip-id",
            "clip-1",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["streamer"] == "jynxzi"
    assert call["count"] == 3
    assert call["force"] is True
    assert call["rerender"] is False
    assert call["generate_captions"] is True
    assert call["force_captions"] is False
    assert call["clip_ids"] == ["clip-1"]
    assert call["started_at"] is not None
    assert call["ended_at"] is not None
    assert call["discovery_limit"] == 10
    assert call["use_generated_layouts"] is True
    assert call["config"] == config
    assert capsys.readouterr().out.splitlines() == [
        "ready exports:",
        str(export_path),
    ]


def test_main_routes_clips_review_rerender_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    calls: list[dict[str, object]] = []

    def fake_review_streamer_clips(**kwargs) -> tuple[Path, ...]:
        calls.append(kwargs)
        return ()

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr(
        "clipforge.pipeline.cli.review_streamer_clips",
        fake_review_streamer_clips,
    )

    exit_code = main(
        [
            "clips",
            "review",
            "--streamer",
            "doublelift",
            "--clip-id",
            "clip-1",
            "--rerender",
        ]
    )

    assert exit_code == 0
    assert calls[0]["rerender"] is True
    assert calls[0]["clip_ids"] == ["clip-1"]


def test_main_rejects_rerender_with_caption_generation(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config = ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        db_path=config.state_db_path,
    )

    def fail_process(*args, **kwargs) -> Path:
        raise AssertionError("Conflicting rerender caption flags should fail in CLI.")

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)
    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fail_process)

    exit_code = main(
        [
            "clips",
            "process",
            "--clip-id",
            "clip-1",
            "--rerender",
            "--generate-captions",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--rerender reuses existing captions" in captured.err


def test_main_rejects_clips_output_without_json_format(monkeypatch, capsys) -> None:
    def fake_list_channel_clips(*args, **kwargs) -> tuple[TwitchClip, ...]:
        raise AssertionError("Twitch should not be called for invalid CLI options.")

    monkeypatch.setattr("clipforge.pipeline.cli.list_channel_clips", fake_list_channel_clips)

    exit_code = main(["clips", "--channel", "example", "--output", "clips.json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--output requires --format json" in captured.err


def test_main_rejects_clips_ended_at_without_started_at(capsys) -> None:
    exit_code = main(["clips", "--channel", "example", "--ended-at", "2026-05-06T00:00:00Z"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--ended-at requires --started-at" in captured.err


def test_main_returns_non_zero_for_missing_clipr_api_key(monkeypatch, capsys) -> None:
    def fake_process(url: str) -> Path:
        raise RuntimeError("Missing required configuration: CLIPR_API_KEY")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["--url", TWITCH_CLIP_URL])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "CLIPR_API_KEY" in captured.err


def test_main_returns_non_zero_for_invalid_twitch_clip_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    calls: list[str] = []

    def fake_get(self, twitch_clip_url: str) -> str:
        calls.append(twitch_clip_url)
        raise AssertionError("network call should not be reached")

    monkeypatch.setenv("CLIPR_API_KEY", "test-key")
    monkeypatch.setattr("clipforge.integrations.clipr.CliprClient._get", fake_get)

    exit_code = main(["resolve-url", "--url", "https://example.com/not-a-clip"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Unsupported Twitch clip URL" in captured.err
    assert calls == []

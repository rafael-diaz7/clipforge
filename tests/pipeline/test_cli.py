from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.cli import main
from clipforge.integrations.twitch import TwitchClip
from clipforge.storage.state import get_clip, mark_clip_rendered, upsert_discovered_clip
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


def test_main_routes_process_caption_flag(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_process(url: str, *, generate_captions: bool) -> Path:
        calls.append({"url": url, "generate_captions": generate_captions})
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.pipeline.cli.process_clip", fake_process)

    exit_code = main(["process", "--url", TWITCH_CLIP_URL, "--generate-captions"])

    assert exit_code == 0
    assert calls == [{"url": TWITCH_CLIP_URL, "generate_captions": True}]


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
        title="first",
        view_count=10,
        duration_seconds=12.5,
        rank_score=0.2,
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-2",
        url="https://clips.twitch.tv/clip-2",
        title="second",
        view_count=20,
        duration_seconds=30,
        rank_score=0.9,
        db_path=config.state_db_path,
    )

    monkeypatch.setattr("clipforge.pipeline.cli.load_config", lambda: config)

    exit_code = main(["clips", "pending"])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "clip-2\t0.9\t20\t30s\thttps://clips.twitch.tv/clip-2\tsecond",
        "clip-1\t0.2\t10\t12.5s\thttps://clips.twitch.tv/clip-1\tfirst",
    ]


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
    assert capsys.readouterr().out.splitlines() == [
        "clip-pending\t0.5\t\t\thttps://clips.twitch.tv/clip-pending\tpending"
    ]
    assert calls == ["https://clips.twitch.tv/clip-pending"]
    assert get_clip("clip-rendered", db_path=config.state_db_path).status == "rendered"


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

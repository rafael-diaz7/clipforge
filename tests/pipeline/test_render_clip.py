from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.config import ClipforgeConfig, EXAMPLE_LAYOUTS_DIR
from clipforge.download import DownloadResult
from clipforge.layouts import load_example_layout
from clipforge.render_clip import (
    main,
    process_clip,
    render_all_candidates,
    render_candidate,
)
from clipforge.state import get_clip, upsert_discovered_clip
from clipforge.twitch import TwitchClip
from tests.constants import TWITCH_CLIP_SLUG, TWITCH_CLIP_URL


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(
        downloads_dir=tmp_path / "downloads",
        renders_dir=tmp_path / "renders",
        metadata_dir=tmp_path / "metadata",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
        example_layouts_dir=EXAMPLE_LAYOUTS_DIR,
    )


def test_main_supports_full_pipeline_url_shortcut(
    monkeypatch,
    capsys,
) -> None:
    calls: list[str] = []

    def fake_process(url: str) -> Path:
        calls.append(url)
        return Path("metadata.json")

    monkeypatch.setattr("clipforge.render_clip.process_clip", fake_process)

    exit_code = main(["--url", TWITCH_CLIP_URL])

    assert exit_code == 0
    assert calls == [TWITCH_CLIP_URL]
    assert capsys.readouterr().err == ""


def test_main_routes_render_all_command(monkeypatch, capsys) -> None:
    def fake_render_all(source_path: Path, *, clip_id: str | None = None) -> tuple[Path, ...]:
        assert source_path == Path("source.mp4")
        assert clip_id == "test-clip"
        return (Path("one.mp4"), Path("two.mp4"))

    monkeypatch.setattr("clipforge.render_clip.render_all_candidates", fake_render_all)

    exit_code = main(
        ["render-all", "--source", "source.mp4", "--clip-id", "test-clip"]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["one.mp4", "two.mp4"]


def test_main_routes_clips_command(tmp_path: Path, monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []
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

    monkeypatch.setattr("clipforge.render_clip.load_config", lambda: config)
    monkeypatch.setattr("clipforge.render_clip.list_channel_clips", fake_list_channel_clips)

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
    assert capsys.readouterr().out.splitlines() == [
        "2026-05-01T00:00:00Z\t42\t28.5s\thttps://clips.twitch.tv/clip-1\tgreat clip"
    ]
    state = get_clip("clip-1", db_path=config.state_db_path)
    assert state is not None
    assert state.status == "discovered"
    assert state.streamer_login == "example"
    assert state.title == "great clip"
    assert state.view_count == 42


def test_main_exports_clips_command_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = ClipforgeConfig(
        twitch_client_id="client-id",
        twitch_client_secret="client-secret",
        metadata_dir=tmp_path / "metadata",
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
        assert channel_login == "https://twitch.tv/Example"
        assert limit == 5
        assert started_at == "2026-05-01T00:00:00Z"
        assert ended_at == "2026-05-06T00:00:00Z"
        assert config.metadata_dir == tmp_path / "metadata"
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

    monkeypatch.setattr("clipforge.render_clip.load_config", lambda: config)
    monkeypatch.setattr("clipforge.render_clip.list_channel_clips", fake_list_channel_clips)

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

    expected_path = (
        tmp_path
        / "metadata"
        / "discovered_clips"
        / "example"
        / "2026-05-01-example.json"
    )
    payload = json.loads(expected_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [f"export: {expected_path}"]
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


def test_main_exports_clips_command_to_custom_json_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = ClipforgeConfig(
        twitch_client_id="client-id",
        twitch_client_secret="client-secret",
        metadata_dir=tmp_path / "metadata",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
    )
    output_path = tmp_path / "queue.json"

    monkeypatch.setattr("clipforge.render_clip.load_config", lambda: config)
    monkeypatch.setattr(
        "clipforge.render_clip.list_channel_clips",
        lambda *args, **kwargs: (),
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
    assert output_path.exists()
    assert capsys.readouterr().out.splitlines() == [f"export: {output_path}"]


def test_main_rejects_clips_output_without_json_format(monkeypatch, capsys) -> None:
    def fake_list_channel_clips(*args, **kwargs) -> tuple[TwitchClip, ...]:
        raise AssertionError("Twitch should not be called for invalid CLI options.")

    monkeypatch.setattr("clipforge.render_clip.list_channel_clips", fake_list_channel_clips)

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

    monkeypatch.setattr("clipforge.render_clip.process_clip", fake_process)

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
    monkeypatch.setattr("clipforge.clipr.CliprClient._get", fake_get)

    exit_code = main(["resolve-url", "--url", "https://example.com/not-a-clip"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Unsupported Twitch clip URL" in captured.err
    assert calls == []


def test_render_candidate_uses_layout_name_for_output_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[Path, Path, str]] = []
    config = _config(tmp_path)

    def fake_render(source_path: Path, output_path: Path, layout) -> Path:
        calls.append((source_path, output_path, layout.name))
        return output_path

    monkeypatch.setattr("clipforge.render_clip.render_layout", fake_render)

    output_path = render_candidate(
        tmp_path / "source.mp4",
        layout_ref="center_gameplay",
        clip_id="clip-123",
        config=config,
    )

    assert output_path == tmp_path / "renders" / "clip-123_center_gameplay.mp4"
    assert calls == [
        (
            tmp_path / "source.mp4",
            tmp_path / "renders" / "clip-123_center_gameplay.mp4",
            "center_gameplay",
        )
    ]


def test_render_all_candidates_renders_default_layouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rendered_layouts: list[str] = []
    config = _config(tmp_path)

    def fake_render(source_path: Path, output_path: Path, layout) -> Path:
        rendered_layouts.append(layout.name)
        return output_path

    monkeypatch.setattr("clipforge.render_clip.render_layout", fake_render)

    output_paths = render_all_candidates(
        tmp_path / "source.mp4",
        clip_id="clip-123",
        config=config,
    )

    assert rendered_layouts == ["center_gameplay", "facecam_focus", "hybrid"]
    assert output_paths == (
        tmp_path / "renders" / "clip-123_center_gameplay.mp4",
        tmp_path / "renders" / "clip-123_facecam_focus.mp4",
        tmp_path / "renders" / "clip-123_hybrid.mp4",
    )


def test_process_clip_writes_metadata(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = _config(tmp_path)
    source_path = (
        tmp_path
        / "downloads"
        / TWITCH_CLIP_SLUG
        / "clipr"
        / f"{TWITCH_CLIP_SLUG}.mp4"
    )

    def fake_download_twitch_clip(
        url: str,
        *,
        clip_id: str | None,
        config: ClipforgeConfig,
        on_media_url_resolved,
    ) -> DownloadResult:
        assert clip_id == TWITCH_CLIP_SLUG
        on_media_url_resolved("https://cdn.example.test/source.mp4")
        return DownloadResult(
            source_path=source_path,
            backend="clipr",
            media_url="https://cdn.example.test/source.mp4",
        )

    monkeypatch.setattr(
        "clipforge.render_clip.download_twitch_clip",
        fake_download_twitch_clip,
    )

    def fake_render(source: Path, output: Path, layout) -> Path:
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.render_clip.render_layout", fake_render)

    metadata_path = process_clip(
        TWITCH_CLIP_URL,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["clip_id"] == TWITCH_CLIP_SLUG
    assert metadata["downloader_backend"] == "clipr"
    assert metadata["download_media_url"] == "https://cdn.example.test/source.mp4"
    assert "clipr_download_url" not in metadata
    assert metadata["source_path"] == str(source_path)
    assert [output["layout"] for output in metadata["outputs"]] == [
        "center_gameplay",
        "facecam_focus",
        "hybrid",
    ]
    assert [output["path"] for output in metadata["outputs"]] == [
        str(
            tmp_path
            / "renders"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "center_gameplay.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "facecam_focus.mp4"
        ),
        str(tmp_path / "renders" / TWITCH_CLIP_SLUG / "clipr" / "hybrid.mp4"),
    ]
    assert [layout["name"] for layout in metadata["layouts"]] == [
        "center_gameplay",
        "facecam_focus",
        "hybrid",
    ]
    assert metadata["target_resolution"] == {"width": 1080, "height": 1920}
    assert metadata["created_at"].endswith("+00:00")
    assert metadata["rendered_at"].endswith("+00:00")
    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.status == "rendered"
    assert state.url == TWITCH_CLIP_URL
    assert state.metadata_path == str(metadata_path)
    assert state.render_dir == str(tmp_path / "renders" / TWITCH_CLIP_SLUG / "clipr")
    output = capsys.readouterr().out
    assert "download_url: https://cdn.example.test/source.mp4" in output
    assert "metadata:" in output


def test_process_clip_marks_existing_state_as_rendered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    upsert_discovered_clip(
        clip_id=TWITCH_CLIP_SLUG,
        url=TWITCH_CLIP_URL,
        streamer_login="example",
        title="existing title",
        view_count=10,
        duration_seconds=12,
        db_path=config.state_db_path,
    )
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "clipr" / f"{TWITCH_CLIP_SLUG}.mp4"

    def fake_download_twitch_clip(
        url: str,
        *,
        clip_id: str | None,
        config: ClipforgeConfig,
        on_media_url_resolved,
    ) -> DownloadResult:
        return DownloadResult(
            source_path=source_path,
            backend="clipr",
            media_url="https://cdn.example.test/source.mp4",
        )

    monkeypatch.setattr(
        "clipforge.render_clip.download_twitch_clip",
        fake_download_twitch_clip,
    )
    monkeypatch.setattr(
        "clipforge.render_clip.render_layout",
        lambda source, output, layout: output,
    )

    metadata_path = process_clip(TWITCH_CLIP_URL, config=config)

    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.status == "rendered"
    assert state.title == "existing title"
    assert state.metadata_path == str(metadata_path)


def test_render_candidate_accepts_layout_file_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    layout = load_example_layout("center_gameplay")
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "name": layout.name,
                "description": layout.description,
                "output": {
                    "width": layout.output.width,
                    "height": layout.output.height,
                },
                "regions": [
                    {
                        "name": region.name,
                        "source_region": {
                            "x": region.source_region.x,
                            "y": region.source_region.y,
                            "width": region.source_region.width,
                            "height": region.source_region.height,
                        },
                        "output_region": {
                            "x": region.output_region.x,
                            "y": region.output_region.y,
                            "width": region.output_region.width,
                            "height": region.output_region.height,
                        },
                    }
                    for region in layout.regions
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "clipforge.render_clip.render_layout",
        lambda source, output, layout: output,
    )

    output_path = render_candidate(
        tmp_path / "source.mp4",
        layout_ref=str(layout_path),
        config=config,
    )

    assert output_path == tmp_path / "renders" / "source_center_gameplay.mp4"

from __future__ import annotations

import json
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, EXAMPLE_LAYOUTS_DIR
from clipforge.media.download import DownloadResult
from clipforge.media.layouts import load_example_layout
from clipforge.pipeline.workflows import (
    process_clip,
    render_all_candidates,
    render_candidate,
)
from clipforge.storage.state import get_clip, upsert_discovered_clip
from tests.constants import TWITCH_CLIP_SLUG, TWITCH_CLIP_URL


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(
        downloads_dir=tmp_path / "downloads",
        renders_dir=tmp_path / "renders",
        metadata_dir=tmp_path / "metadata",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
        example_layouts_dir=EXAMPLE_LAYOUTS_DIR,
    )


def test_render_candidate_uses_layout_name_for_output_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[Path, Path, str]] = []
    config = _config(tmp_path)

    def fake_render(source_path: Path, output_path: Path, layout) -> Path:
        calls.append((source_path, output_path, layout.name))
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

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

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

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
        "clipforge.pipeline.workflows.download_twitch_clip",
        fake_download_twitch_clip,
    )

    def fake_render(source: Path, output: Path, layout) -> Path:
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

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
    assert "caption_metadata_path" not in metadata
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


def test_process_clip_can_generate_captions_before_rendering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    caption_path = tmp_path / "metadata" / "captions" / f"{TWITCH_CLIP_SLUG}.json"
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

    def fake_generate_caption_metadata(
        path: Path,
        *,
        clip_id: str,
        config: ClipforgeConfig,
    ) -> Path:
        assert path == source_path
        assert clip_id == TWITCH_CLIP_SLUG
        events.append("captions")
        return caption_path

    def fake_render(source: Path, output: Path, layout) -> Path:
        events.append(f"render:{layout.name}")
        return output

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        fake_download_twitch_clip,
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_caption_metadata",
        fake_generate_caption_metadata,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    metadata_path = process_clip(
        TWITCH_CLIP_URL,
        generate_captions=True,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["caption_metadata_path"] == str(caption_path)
    assert events[:2] == ["download", "captions"]
    assert events[2:] == [
        "render:center_gameplay",
        "render:facecam_focus",
        "render:hybrid",
    ]


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
        "clipforge.pipeline.workflows.download_twitch_clip",
        fake_download_twitch_clip,
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.render_layout",
        lambda source, output, layout: output,
    )

    metadata_path = process_clip(TWITCH_CLIP_URL, config=config)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.status == "rendered"
    assert state.title == "existing title"
    assert state.metadata_path == str(metadata_path)
    assert state.render_dir == str(
        tmp_path / "renders" / "example" / TWITCH_CLIP_SLUG / "clipr"
    )
    assert [output["path"] for output in metadata["outputs"]] == [
        str(
            tmp_path
            / "renders"
            / "example"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "center_gameplay.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "example"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "facecam_focus.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "example"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "hybrid.mp4"
        ),
    ]


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
        "clipforge.pipeline.workflows.render_layout",
        lambda source, output, layout: output,
    )

    output_path = render_candidate(
        tmp_path / "source.mp4",
        layout_ref=str(layout_path),
        config=config,
    )

    assert output_path == tmp_path / "renders" / "source_center_gameplay.mp4"

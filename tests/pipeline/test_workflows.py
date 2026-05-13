from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig, EXAMPLE_LAYOUTS_DIR
from clipforge.media.captions import CaptionMetadata, CaptionSegment, save_captions
from clipforge.media.caption_rendering import CaptionStyle
from clipforge.media.download import DownloadResult
from clipforge.media.layouts import load_example_layout
from clipforge.media.render import Watermark
from clipforge.media.render_settings import FFmpegRenderSettings
from clipforge.pipeline.workflows import (
    ClipProcessingError,
    process_clip,
    render_all_candidates,
    render_candidate,
    render_selected_layout_from_metadata,
)
from clipforge.storage.state import get_clip, upsert_discovered_clip
from tests.constants import TWITCH_CLIP_SLUG, TWITCH_CLIP_URL

STATIC_LAYOUT_NAMES = [
    "center_gameplay",
    "fullscreen_downscaled_blur_bg",
    "facecam_focus",
    "hybrid",
    "hybrid_full_game_bottom",
]
GENERATED_LAYOUT_CANDIDATE_NAMES = [
    "center_gameplay",
    "fullscreen_downscaled_blur_bg",
    "detected_streamer_focus",
    "detected_hybrid",
    "detected_hybrid_full_game_bottom",
]


@pytest.fixture(autouse=True)
def _isolate_streamer_watermark_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIMA_WALLHACKS_WATERMARK", raising=False)
    monkeypatch.delenv("OHNEPIXEL_WATERMARK", raising=False)


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(
        project_root=tmp_path,
        downloads_dir=tmp_path / "downloads",
        renders_dir=tmp_path / "renders",
        metadata_dir=tmp_path / "metadata",
        analysis_dir=tmp_path / "analysis",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
        example_layouts_dir=EXAMPLE_LAYOUTS_DIR,
    )


def _write_png_header(path: Path, *, width: int = 400, height: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, byteorder="big")
        + b"IHDR"
        + width.to_bytes(4, byteorder="big")
        + height.to_bytes(4, byteorder="big")
    )
    return path


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


def test_render_candidate_can_burn_caption_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    config = _config(tmp_path)
    caption_path = save_captions(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
        config=config,
    )

    def fake_render(
        source_path: Path,
        output_path: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        calls.append(
            {
                "source_path": source_path,
                "output_path": output_path,
                "layout": layout.name,
                "caption_metadata": caption_metadata,
                "caption_renderer_backend": caption_renderer_backend,
                "ass_temp_dir": ass_temp_dir,
            }
        )
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    output_path = render_candidate(
        tmp_path / "source.mp4",
        layout_ref="center_gameplay",
        clip_id="clip-123",
        caption_metadata_path=caption_path,
        config=config,
    )

    assert output_path == tmp_path / "renders" / "clip-123_center_gameplay.mp4"
    assert calls[0]["caption_metadata"] == CaptionMetadata(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
    )
    assert calls[0]["caption_renderer_backend"] == "drawtext"
    assert calls[0]["ass_temp_dir"] == config.ass_temp_dir


def test_render_candidate_passes_review_ffmpeg_settings_when_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    config = replace(
        _config(tmp_path),
        review_fast_render=True,
        review_ffmpeg_render_settings=FFmpegRenderSettings(preset="veryfast", crf=23),
    )

    def fake_render(source_path: Path, output_path: Path, layout, **kwargs) -> Path:
        calls.append(kwargs)
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    render_candidate(
        tmp_path / "source.mp4",
        layout_ref="center_gameplay",
        clip_id="clip-123",
        config=config,
    )

    assert calls[0]["render_settings"] == FFmpegRenderSettings(
        preset="veryfast",
        crf=23,
    )


def test_render_candidate_uses_review_output_width_for_preview_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    config = replace(_config(tmp_path), review_output_width=720)

    def fake_render(source_path: Path, output_path: Path, layout, **kwargs) -> Path:
        calls.append(
            {
                "output_path": output_path,
                "width": layout.output.width,
                "height": layout.output.height,
            }
        )
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    output_path = render_candidate(
        tmp_path / "source.mp4",
        layout_ref="center_gameplay",
        clip_id="clip-123",
        config=config,
    )

    assert output_path == tmp_path / "renders" / "clip-123_center_gameplay_720x1280.mp4"
    assert calls == [
        {
            "output_path": output_path,
            "width": 720,
            "height": 1280,
        }
    ]


def test_render_candidate_scales_caption_style_for_review_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[CaptionStyle] = []
    config = replace(_config(tmp_path), review_output_width=720)
    caption_path = save_captions(
        clip_id="clip-123",
        segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
        config=config,
    )

    def fake_render(
        source_path: Path,
        output_path: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_style: CaptionStyle,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        del source_path, layout, caption_metadata, caption_renderer_backend, ass_temp_dir
        calls.append(caption_style)
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    render_candidate(
        tmp_path / "source.mp4",
        layout_ref="center_gameplay",
        clip_id="clip-123",
        caption_metadata_path=caption_path,
        config=config,
    )

    assert calls[0].font_size == 37
    assert calls[0].safe_margin_x == 64
    assert calls[0].safe_margin_bottom == 147
    assert calls[0].box_border_width == 13


def test_render_selected_layout_from_metadata_uses_final_settings_and_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    config = replace(
        _config(tmp_path),
        review_output_width=720,
        review_fast_render=True,
        review_ffmpeg_render_settings=FFmpegRenderSettings(preset="veryfast", crf=28),
        ffmpeg_render_settings=FFmpegRenderSettings(preset="slow", crf=18),
    )
    source_path = tmp_path / "downloads" / "clip-123.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source")
    layout = load_example_layout("center_gameplay")
    metadata_path = tmp_path / "metadata" / "clip-123.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "source_path": str(source_path),
                "layouts": [
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
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_render(source_path: Path, output_path: Path, layout, **kwargs) -> Path:
        calls.append(
            {
                "source_path": source_path,
                "output_path": output_path,
                "width": layout.output.width,
                "height": layout.output.height,
                "render_settings": kwargs["render_settings"],
            }
        )
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    output_path = tmp_path / "exports" / "ready" / "clip-123.mp4"
    render_selected_layout_from_metadata(
        metadata_path,
        selected_layout="center_gameplay",
        output_path=output_path,
        channel=None,
        config=config,
    )

    assert calls == [
        {
            "source_path": source_path,
            "output_path": output_path,
            "width": 1080,
            "height": 1920,
            "render_settings": FFmpegRenderSettings(preset="slow", crf=18),
        }
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

    assert rendered_layouts == STATIC_LAYOUT_NAMES
    assert output_paths == (
        tmp_path / "renders" / "clip-123_center_gameplay.mp4",
        tmp_path / "renders" / "clip-123_fullscreen_downscaled_blur_bg.mp4",
        tmp_path / "renders" / "clip-123_facecam_focus.mp4",
        tmp_path / "renders" / "clip-123_hybrid.mp4",
        tmp_path / "renders" / "clip-123_hybrid_full_game_bottom.mp4",
    )


def test_render_all_candidates_prefers_generated_analysis_layouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rendered_layouts: list[str] = []
    config = _config(tmp_path)
    _write_detected_layout(
        config.analysis_dir,
        clip_id="clip-123",
        source_template="facecam_focus",
        layout_name="detected_streamer_focus",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id="clip-123",
        source_template="hybrid",
        layout_name="detected_hybrid",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id="clip-123",
        source_template="hybrid_full_game_bottom",
        layout_name="detected_hybrid_full_game_bottom",
    )

    def fake_render(source_path: Path, output_path: Path, layout) -> Path:
        rendered_layouts.append(layout.name)
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    output_paths = render_all_candidates(
        tmp_path / "source.mp4",
        clip_id="clip-123",
        config=config,
    )

    assert rendered_layouts == GENERATED_LAYOUT_CANDIDATE_NAMES
    assert output_paths == (
        tmp_path / "renders" / "clip-123_center_gameplay.mp4",
        tmp_path / "renders" / "clip-123_fullscreen_downscaled_blur_bg.mp4",
        tmp_path / "renders" / "clip-123_detected_streamer_focus.mp4",
        tmp_path / "renders" / "clip-123_detected_hybrid.mp4",
        tmp_path / "renders" / "clip-123_detected_hybrid_full_game_bottom.mp4",
    )


def test_render_all_candidates_falls_back_when_generated_layout_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rendered_layouts: list[str] = []
    config = _config(tmp_path)
    _write_detected_layout(
        config.analysis_dir,
        clip_id="clip-123",
        source_template="facecam_focus",
        layout_name="detected_streamer_focus",
    )

    def fake_render(source_path: Path, output_path: Path, layout) -> Path:
        rendered_layouts.append(layout.name)
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    output_paths = render_all_candidates(
        tmp_path / "source.mp4",
        clip_id="clip-123",
        config=config,
    )

    assert rendered_layouts == [
        "center_gameplay",
        "fullscreen_downscaled_blur_bg",
        "detected_streamer_focus",
        "hybrid",
        "hybrid_full_game_bottom",
    ]
    assert output_paths == (
        tmp_path / "renders" / "clip-123_center_gameplay.mp4",
        tmp_path / "renders" / "clip-123_fullscreen_downscaled_blur_bg.mp4",
        tmp_path / "renders" / "clip-123_detected_streamer_focus.mp4",
        tmp_path / "renders" / "clip-123_hybrid.mp4",
        tmp_path / "renders" / "clip-123_hybrid_full_game_bottom.mp4",
    )


def test_render_all_candidates_static_layouts_opt_out(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rendered_layouts: list[str] = []
    config = _config(tmp_path)
    _write_detected_layout(
        config.analysis_dir,
        clip_id="clip-123",
        source_template="facecam_focus",
        layout_name="detected_streamer_focus",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id="clip-123",
        source_template="hybrid",
        layout_name="detected_hybrid",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id="clip-123",
        source_template="hybrid_full_game_bottom",
        layout_name="detected_hybrid_full_game_bottom",
    )

    def fake_render(source_path: Path, output_path: Path, layout) -> Path:
        rendered_layouts.append(layout.name)
        return output_path

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    output_paths = render_all_candidates(
        tmp_path / "source.mp4",
        clip_id="clip-123",
        use_generated_layouts=False,
        config=config,
    )

    assert rendered_layouts == STATIC_LAYOUT_NAMES
    assert output_paths == (
        tmp_path / "renders" / "clip-123_center_gameplay.mp4",
        tmp_path / "renders" / "clip-123_fullscreen_downscaled_blur_bg.mp4",
        tmp_path / "renders" / "clip-123_facecam_focus.mp4",
        tmp_path / "renders" / "clip-123_hybrid.mp4",
        tmp_path / "renders" / "clip-123_hybrid_full_game_bottom.mp4",
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
        use_generated_layouts=False,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["clip_id"] == TWITCH_CLIP_SLUG
    assert metadata["downloader_backend"] == "clipr"
    assert metadata["download_media_url"] == "https://cdn.example.test/source.mp4"
    assert "clipr_download_url" not in metadata
    assert metadata["source_path"] == str(source_path)
    assert [output["layout"] for output in metadata["outputs"]] == STATIC_LAYOUT_NAMES
    assert [output["path"] for output in metadata["outputs"]] == [
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "center_gameplay.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "fullscreen_downscaled_blur_bg.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "facecam_focus.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "hybrid.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "hybrid_full_game_bottom.mp4"
        ),
    ]
    assert [layout["name"] for layout in metadata["layouts"]] == STATIC_LAYOUT_NAMES
    assert "caption_metadata_path" not in metadata
    assert metadata["target_resolution"] == {"width": 1080, "height": 1920}
    assert metadata["created_at"].endswith("+00:00")
    assert metadata["rendered_at"].endswith("+00:00")
    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.status == "rendered"
    assert state.url == TWITCH_CLIP_URL
    assert state.metadata_path == str(metadata_path)
    assert state.render_dir == str(
        tmp_path / "renders" / "unknown_streamer" / TWITCH_CLIP_SLUG / "clipr"
    )
    output = capsys.readouterr().out
    assert "download_url: https://cdn.example.test/source.mp4" in output
    assert "metadata:" in output


def test_process_clip_records_lower_resolution_review_preview_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = replace(_config(tmp_path), review_output_width=720)
    source_path = (
        tmp_path
        / "downloads"
        / TWITCH_CLIP_SLUG
        / "clipr"
        / f"{TWITCH_CLIP_SLUG}.mp4"
    )
    calls: list[tuple[str, int, int]] = []

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="clipr"),
    )

    def fake_render(source: Path, output: Path, layout) -> Path:
        del source
        calls.append((layout.name, layout.output.width, layout.output.height))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    metadata_path = process_clip(
        TWITCH_CLIP_URL,
        use_generated_layouts=False,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert calls[0] == ("center_gameplay", 720, 1280)
    assert metadata["outputs"][0]["resolution"] == {"width": 720, "height": 1280}
    assert metadata["outputs"][0]["render_profile"] == "review"
    assert Path(metadata["outputs"][0]["path"]).parts[-3:] == (
        "clipr",
        "preview_720x1280",
        "center_gameplay.mp4",
    )
    assert metadata["layouts"][0]["output"] == {"width": 1080, "height": 1920}


def test_process_clip_uses_generated_layouts_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = (
        tmp_path
        / "downloads"
        / TWITCH_CLIP_SLUG
        / "clipr"
        / f"{TWITCH_CLIP_SLUG}.mp4"
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="facecam_focus",
        layout_name="detected_streamer_focus",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid",
        layout_name="detected_hybrid",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid_full_game_bottom",
        layout_name="detected_hybrid_full_game_bottom",
    )
    _write_frame_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)
    _write_overlay_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)

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
    assert [output["layout"] for output in metadata["outputs"]] == (
        GENERATED_LAYOUT_CANDIDATE_NAMES
    )
    assert [output["path"] for output in metadata["outputs"]] == [
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "center_gameplay.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "fullscreen_downscaled_blur_bg.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "detected_streamer_focus.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "detected_hybrid.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "detected_hybrid_full_game_bottom.mp4"
        ),
    ]
    assert [layout["name"] for layout in metadata["layouts"]] == (
        GENERATED_LAYOUT_CANDIDATE_NAMES
    )


def test_process_clip_can_generate_captions_before_rendering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = (
        tmp_path
        / "downloads"
        / TWITCH_CLIP_SLUG
        / "ytdlp"
        / f"{TWITCH_CLIP_SLUG}.mp4"
    )
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
        return save_captions(
            clip_id=clip_id,
            segments=(CaptionSegment(start_time=0, end_time=1, text="hello"),),
            output_path=caption_path,
            config=config,
        )

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        assert caption_metadata.clip_id == TWITCH_CLIP_SLUG
        assert caption_renderer_backend == "drawtext"
        assert ass_temp_dir == config.ass_temp_dir
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
        use_generated_layouts=False,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["caption_metadata_path"] == str(caption_path)
    assert events[:2] == ["download", "captions"]
    assert events[2:] == [
        "render:center_gameplay",
        "render:fullscreen_downscaled_blur_bg",
        "render:facecam_focus",
        "render:hybrid",
        "render:hybrid_full_game_bottom",
    ]


def test_process_clip_reuses_existing_caption_metadata(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = _config(tmp_path)
    source_path = (
        tmp_path
        / "downloads"
        / TWITCH_CLIP_SLUG
        / "ytdlp"
        / f"{TWITCH_CLIP_SLUG}.mp4"
    )
    caption_path = save_captions(
        clip_id=TWITCH_CLIP_SLUG,
        segments=(CaptionSegment(start_time=0, end_time=1, text="existing"),),
        config=config,
    )
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
        raise AssertionError("Existing caption metadata should be reused.")

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        assert caption_metadata == CaptionMetadata(
            clip_id=TWITCH_CLIP_SLUG,
            segments=(CaptionSegment(start_time=0, end_time=1, text="existing"),),
        )
        events.append(f"render:{layout.name}")
        return output

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        fake_download_twitch_clip,
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_caption_metadata",
        fail_generate_caption_metadata,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    metadata_path = process_clip(
        TWITCH_CLIP_URL,
        generate_captions=True,
        use_generated_layouts=False,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["caption_metadata_path"] == str(caption_path)
    assert events == [
        "download",
        "render:center_gameplay",
        "render:fullscreen_downscaled_blur_bg",
        "render:facecam_focus",
        "render:hybrid",
        "render:hybrid_full_game_bottom",
    ]
    assert f"captions: reusing existing {caption_path}" in capsys.readouterr().out


def test_process_clip_force_captions_regenerates_existing_caption_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    caption_path = save_captions(
        clip_id=TWITCH_CLIP_SLUG,
        segments=(CaptionSegment(start_time=0, end_time=1, text="stale"),),
        config=config,
    )
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
        events.append("captions")
        return save_captions(
            clip_id=clip_id,
            segments=(CaptionSegment(start_time=0, end_time=1, text="fresh"),),
            config=config,
        )

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        assert caption_metadata == CaptionMetadata(
            clip_id=TWITCH_CLIP_SLUG,
            segments=(CaptionSegment(start_time=0, end_time=1, text="fresh"),),
        )
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
        force_captions=True,
        use_generated_layouts=False,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["caption_metadata_path"] == str(caption_path)
    assert events[:2] == ["download", "captions"]


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

    metadata_path = process_clip(TWITCH_CLIP_URL, use_generated_layouts=False, config=config)

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
            / "fullscreen_downscaled_blur_bg.mp4"
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
        str(
            tmp_path
            / "renders"
            / "example"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "hybrid_full_game_bottom.mp4"
        ),
    ]


def test_process_clip_explicit_channel_scopes_render_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    upsert_discovered_clip(
        clip_id=TWITCH_CLIP_SLUG,
        url=TWITCH_CLIP_URL,
        streamer_login="stored_channel",
        db_path=config.state_db_path,
    )
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "clipr" / f"{TWITCH_CLIP_SLUG}.mp4"

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="clipr"),
    )

    def fake_render(source: Path, output: Path, layout) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    metadata_path = process_clip(
        TWITCH_CLIP_URL,
        channel="dima_wallhacks",
        use_generated_layouts=False,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert [output["path"] for output in metadata["outputs"]] == [
        str(
            tmp_path
            / "renders"
            / "dima_wallhacks"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "center_gameplay.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "dima_wallhacks"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "fullscreen_downscaled_blur_bg.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "dima_wallhacks"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "facecam_focus.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "dima_wallhacks"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "hybrid.mp4"
        ),
        str(
            tmp_path
            / "renders"
            / "dima_wallhacks"
            / TWITCH_CLIP_SLUG
            / "clipr"
            / "hybrid_full_game_bottom.mp4"
        ),
    ]
    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.render_dir == str(
        tmp_path / "renders" / "dima_wallhacks" / TWITCH_CLIP_SLUG / "clipr"
    )


def test_process_clip_applies_streamer_watermark_from_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    watermark_path = _write_png_header(
        tmp_path / "assets" / "watermarks" / "ohnepixel.png",
        width=300,
        height=80,
    )
    monkeypatch.setenv(
        "OHNEPIXEL_WATERMARK",
        "assets/watermarks/ohnepixel.png",
    )
    upsert_discovered_clip(
        clip_id=TWITCH_CLIP_SLUG,
        url=TWITCH_CLIP_URL,
        streamer_login="ohnepixel",
        db_path=config.state_db_path,
    )
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    watermarks: list[Watermark] = []

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="ytdlp"),
    )

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        watermark: Watermark,
    ) -> Path:
        watermarks.append(watermark)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    process_clip(TWITCH_CLIP_URL, use_generated_layouts=False, config=config)

    assert len(watermarks) == len(STATIC_LAYOUT_NAMES)
    assert all(watermark.path == watermark_path for watermark in watermarks)
    assert all(watermark.native_width == 300 for watermark in watermarks)
    assert all(watermark.native_height == 80 for watermark in watermarks)


def test_process_clip_explicit_channel_applies_streamer_watermark_from_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    watermark_path = _write_png_header(
        tmp_path / "assets" / "dima_watermark.png",
        width=640,
        height=180,
    )
    monkeypatch.setenv("DIMA_WALLHACKS_WATERMARK", "assets/dima_watermark.png")
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "clipr" / f"{TWITCH_CLIP_SLUG}.mp4"
    watermarks: list[Watermark] = []

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="clipr"),
    )

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        watermark: Watermark,
    ) -> Path:
        watermarks.append(watermark)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    process_clip(
        TWITCH_CLIP_URL,
        channel="dima_wallhacks",
        use_generated_layouts=False,
        config=config,
    )

    assert len(watermarks) == len(STATIC_LAYOUT_NAMES)
    assert all(watermark.path == watermark_path for watermark in watermarks)
    assert all(watermark.native_width == 640 for watermark in watermarks)
    assert all(watermark.native_height == 180 for watermark in watermarks)


def test_process_clip_keeps_render_calls_unchanged_without_watermark_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    upsert_discovered_clip(
        clip_id=TWITCH_CLIP_SLUG,
        url=TWITCH_CLIP_URL,
        streamer_login="ohnepixel",
        db_path=config.state_db_path,
    )
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    events: list[str] = []

    monkeypatch.delenv("OHNEPIXEL_WATERMARK", raising=False)
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="ytdlp"),
    )

    def fake_render(source: Path, output: Path, layout) -> Path:
        events.append(layout.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    process_clip(TWITCH_CLIP_URL, use_generated_layouts=False, config=config)

    assert events == STATIC_LAYOUT_NAMES


def test_process_clip_generates_analysis_artifacts_and_renders_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    events: list[str] = []

    def fake_download_twitch_clip(
        url: str,
        *,
        clip_id: str | None,
        config: ClipforgeConfig,
        on_media_url_resolved,
    ) -> DownloadResult:
        events.append("download")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"video")
        return DownloadResult(source_path=source_path, backend="ytdlp")

    def fake_sample_frames(
        source: Path,
        *,
        clip_id: str,
        analysis_dir: Path,
        duration_seconds: float | None,
    ) -> Path:
        assert source == source_path
        assert duration_seconds is None
        events.append("frames")
        return _write_frame_analysis(analysis_dir, clip_id=clip_id)

    def fake_analyze_overlay(*, clip_id: str, analysis_dir: Path) -> Path:
        events.append("overlay")
        return _write_overlay_analysis(analysis_dir, clip_id=clip_id)

    def fake_generate_layouts(
        *,
        clip_id: str,
        analysis_dir: Path,
        example_layouts_dir: Path,
    ) -> tuple[Path, ...]:
        del example_layouts_dir
        events.append("layouts")
        return (
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="facecam_focus",
                layout_name="detected_streamer_focus",
            ),
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="hybrid",
                layout_name="detected_hybrid",
            ),
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="hybrid_full_game_bottom",
                layout_name="detected_hybrid_full_game_bottom",
            ),
        )

    def fake_render(source: Path, output: Path, layout) -> Path:
        events.append(f"render:{layout.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(layout.name, encoding="utf-8")
        return output

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        fake_download_twitch_clip,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.sample_frames", fake_sample_frames)
    monkeypatch.setattr("clipforge.pipeline.workflows.analyze_overlay", fake_analyze_overlay)
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_detected_layout_candidates",
        fake_generate_layouts,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    metadata_path = process_clip(TWITCH_CLIP_URL, config=config)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert events == [
        "download",
        "frames",
        "overlay",
        "layouts",
        "render:center_gameplay",
        "render:fullscreen_downscaled_blur_bg",
        "render:detected_streamer_focus",
        "render:detected_hybrid",
        "render:detected_hybrid_full_game_bottom",
    ]
    assert (config.analysis_dir / TWITCH_CLIP_SLUG / "frames.json").is_file()
    assert (config.analysis_dir / TWITCH_CLIP_SLUG / "overlay.json").is_file()
    assert (
        config.analysis_dir
        / TWITCH_CLIP_SLUG
        / "layouts"
        / "detected_streamer_focus.json"
    ).is_file()
    assert [output["layout"] for output in metadata["outputs"]] == (
        GENERATED_LAYOUT_CANDIDATE_NAMES
    )
    assert all(Path(output["path"]).is_file() for output in metadata["outputs"])


def test_process_clip_reuses_existing_analysis_and_render_artifacts_without_force(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"video")
    _write_frame_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)
    _write_overlay_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="facecam_focus",
        layout_name="detected_streamer_focus",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid",
        layout_name="detected_hybrid",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid_full_game_bottom",
        layout_name="detected_hybrid_full_game_bottom",
    )
    for name in GENERATED_LAYOUT_CANDIDATE_NAMES:
        render_path = (
            config.renders_dir
            / "unknown_streamer"
            / TWITCH_CLIP_SLUG
            / "ytdlp"
            / f"{name}.mp4"
        )
        render_path.parent.mkdir(parents=True, exist_ok=True)
        render_path.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="ytdlp"),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.sample_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("frames should be reused")
        ),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.analyze_overlay",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("overlay should be reused")
        ),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_detected_layout_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("layouts should be reused")
        ),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.render_layout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("renders should be reused")
        ),
    )

    metadata_path = process_clip(TWITCH_CLIP_URL, config=config)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert [Path(output["path"]).read_text(encoding="utf-8") for output in metadata["outputs"]] == [
        "existing",
        "existing",
        "existing",
        "existing",
        "existing",
    ]


def test_process_clip_force_regenerates_analysis_and_render_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"video")
    _write_frame_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)
    _write_overlay_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="facecam_focus",
        layout_name="detected_streamer_focus",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid",
        layout_name="detected_hybrid",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid_full_game_bottom",
        layout_name="detected_hybrid_full_game_bottom",
    )
    events: list[str] = []

    def fake_sample_frames(
        source: Path,
        *,
        clip_id: str,
        analysis_dir: Path,
        duration_seconds: float | None,
    ) -> Path:
        assert duration_seconds is None
        events.append("frames")
        return _write_frame_analysis(analysis_dir, clip_id=clip_id)

    def fake_analyze_overlay(*, clip_id: str, analysis_dir: Path) -> Path:
        events.append("overlay")
        return _write_overlay_analysis(analysis_dir, clip_id=clip_id)

    def fake_generate_layouts(
        *,
        clip_id: str,
        analysis_dir: Path,
        example_layouts_dir: Path,
    ) -> tuple[Path, ...]:
        del example_layouts_dir
        events.append("layouts")
        return (
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="facecam_focus",
                layout_name="detected_streamer_focus",
            ),
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="hybrid",
                layout_name="detected_hybrid",
            ),
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="hybrid_full_game_bottom",
                layout_name="detected_hybrid_full_game_bottom",
            ),
        )

    def fake_render(source: Path, output: Path, layout) -> Path:
        events.append(f"render:{layout.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("fresh", encoding="utf-8")
        return output

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="ytdlp"),
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.sample_frames", fake_sample_frames)
    monkeypatch.setattr("clipforge.pipeline.workflows.analyze_overlay", fake_analyze_overlay)
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_detected_layout_candidates",
        fake_generate_layouts,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    process_clip(TWITCH_CLIP_URL, force=True, config=config)

    assert events == [
        "frames",
        "overlay",
        "layouts",
        "render:center_gameplay",
        "render:fullscreen_downscaled_blur_bg",
        "render:detected_streamer_focus",
        "render:detected_hybrid",
        "render:detected_hybrid_full_game_bottom",
    ]


def test_process_clip_rerender_reuses_source_and_captions_without_transcription(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"video")
    caption_path = save_captions(
        clip_id=TWITCH_CLIP_SLUG,
        segments=(CaptionSegment(start_time=0, end_time=1, text="existing"),),
        config=config,
    )
    _write_frame_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)
    _write_overlay_analysis(config.analysis_dir, clip_id=TWITCH_CLIP_SLUG)
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="facecam_focus",
        layout_name="detected_streamer_focus",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid",
        layout_name="detected_hybrid",
    )
    _write_detected_layout(
        config.analysis_dir,
        clip_id=TWITCH_CLIP_SLUG,
        source_template="hybrid_full_game_bottom",
        layout_name="detected_hybrid_full_game_bottom",
    )
    events: list[str] = []

    def fail_download(*args, **kwargs) -> DownloadResult:
        raise AssertionError("Rerender should reuse the existing source video.")

    def fail_generate_caption_metadata(*args, **kwargs) -> Path:
        raise AssertionError("Rerender should not call transcription.")

    def fake_sample_frames(
        source: Path,
        *,
        clip_id: str,
        analysis_dir: Path,
        duration_seconds: float | None,
    ) -> Path:
        assert duration_seconds is None
        assert source == source_path
        events.append("frames")
        return _write_frame_analysis(analysis_dir, clip_id=clip_id)

    def fake_analyze_overlay(*, clip_id: str, analysis_dir: Path) -> Path:
        events.append("overlay")
        return _write_overlay_analysis(analysis_dir, clip_id=clip_id)

    def fake_generate_layouts(
        *,
        clip_id: str,
        analysis_dir: Path,
        example_layouts_dir: Path,
    ) -> tuple[Path, ...]:
        del example_layouts_dir
        events.append("layouts")
        return (
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="facecam_focus",
                layout_name="detected_streamer_focus",
            ),
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="hybrid",
                layout_name="detected_hybrid",
            ),
            _write_detected_layout(
                analysis_dir,
                clip_id=clip_id,
                source_template="hybrid_full_game_bottom",
                layout_name="detected_hybrid_full_game_bottom",
            ),
        )

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        assert caption_metadata.clip_id == TWITCH_CLIP_SLUG
        events.append(f"render:{layout.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("fresh", encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.download_twitch_clip", fail_download)
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_caption_metadata",
        fail_generate_caption_metadata,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.sample_frames", fake_sample_frames)
    monkeypatch.setattr("clipforge.pipeline.workflows.analyze_overlay", fake_analyze_overlay)
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_detected_layout_candidates",
        fake_generate_layouts,
    )
    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    metadata_path = process_clip(TWITCH_CLIP_URL, rerender=True, config=config)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert metadata["caption_metadata_path"] == str(caption_path)
    assert "rerender: regenerating visual artifacts" in output
    assert f"source: reusing existing {source_path}" in output
    assert f"captions: reusing existing {caption_path}" in output
    assert events == [
        "frames",
        "overlay",
        "layouts",
        "render:center_gameplay",
        "render:fullscreen_downscaled_blur_bg",
        "render:detected_streamer_focus",
        "render:detected_hybrid",
        "render:detected_hybrid_full_game_bottom",
    ]


def test_process_clip_rerender_preserves_explicit_channel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"video")
    save_captions(
        clip_id=TWITCH_CLIP_SLUG,
        segments=(CaptionSegment(start_time=0, end_time=1, text="existing"),),
        config=config,
    )
    rendered_paths: list[Path] = []

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Rerender should reuse the existing source video.")
        ),
    )

    def fake_render(
        source: Path,
        output: Path,
        layout,
        *,
        caption_metadata: CaptionMetadata,
        caption_renderer_backend: str,
        ass_temp_dir: Path,
    ) -> Path:
        del source, layout, caption_metadata, caption_renderer_backend, ass_temp_dir
        rendered_paths.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("fresh", encoding="utf-8")
        return output

    monkeypatch.setattr("clipforge.pipeline.workflows.render_layout", fake_render)

    metadata_path = process_clip(
        TWITCH_CLIP_URL,
        rerender=True,
        channel="dima_wallhacks",
        use_generated_layouts=False,
        config=config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_dir = tmp_path / "renders" / "dima_wallhacks" / TWITCH_CLIP_SLUG / "ytdlp"
    assert rendered_paths[0] == expected_dir / "center_gameplay.mp4"
    assert Path(metadata["outputs"][0]["path"]) == expected_dir / "center_gameplay.mp4"
    state = get_clip(TWITCH_CLIP_SLUG, db_path=config.state_db_path)
    assert state is not None
    assert state.render_dir == str(expected_dir)


def test_process_clip_rerender_fails_when_captions_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)

    def fail_generate_caption_metadata(*args, **kwargs) -> Path:
        raise AssertionError("Rerender should not generate missing captions.")

    def fail_download(*args, **kwargs) -> DownloadResult:
        raise AssertionError("Rerender should fail on missing captions before download.")

    monkeypatch.setattr("clipforge.pipeline.workflows.download_twitch_clip", fail_download)
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.generate_caption_metadata",
        fail_generate_caption_metadata,
    )

    with pytest.raises(
        ClipProcessingError,
        match="Captions are missing and rerender mode does not regenerate transcriptions",
    ):
        process_clip(TWITCH_CLIP_URL, rerender=True, config=config)


def test_process_clip_surfaces_failing_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "downloads" / TWITCH_CLIP_SLUG / "ytdlp" / f"{TWITCH_CLIP_SLUG}.mp4"

    monkeypatch.setattr(
        "clipforge.pipeline.workflows.download_twitch_clip",
        lambda *args, **kwargs: DownloadResult(source_path=source_path, backend="ytdlp"),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.workflows.sample_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ffmpeg exploded")),
    )

    with pytest.raises(ClipProcessingError, match="frames stage failed: ffmpeg exploded"):
        process_clip(TWITCH_CLIP_URL, config=config)


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


def _write_detected_layout(
    analysis_dir: Path,
    *,
    clip_id: str,
    source_template: str,
    layout_name: str,
) -> Path:
    layout = load_example_layout(source_template)
    path = analysis_dir / clip_id / "layouts" / f"{layout_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": layout_name,
                "description": f"Detected layout generated from {source_template}.",
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
    return path


def _write_frame_analysis(analysis_dir: Path, *, clip_id: str) -> Path:
    clip_dir = analysis_dir / clip_id
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = tuple(frames_dir / f"frame_{index:04d}.jpg" for index in range(1, 13))
    for frame_path in frame_paths:
        frame_path.write_bytes(b"jpeg")
    metadata_path = clip_dir / "frames.json"
    metadata_path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "source_path": "source.mp4",
                "sampled_timestamps": list(range(0, 24, 2)),
                "frame_paths": [str(frame_path) for frame_path in frame_paths],
                "sampling_mode": {
                    "type": "test",
                    "count": 12,
                    "interval_seconds": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


def _write_overlay_analysis(analysis_dir: Path, *, clip_id: str) -> Path:
    path = analysis_dir / clip_id / "overlay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "selected_rect": None,
                "selected_face_rect": None,
                "selected_overlay_rect": None,
                "confidence": 0.0,
                "fallback": True,
                "reason": "test fallback",
                "candidate_clusters": [],
            }
        ),
        encoding="utf-8",
    )
    return path

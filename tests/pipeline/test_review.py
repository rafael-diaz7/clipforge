from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig
from clipforge.integrations.twitch import TwitchClip
from clipforge.pipeline.review import ClipReviewError, review_streamer_clips
from clipforge.storage.state import (
    get_clip,
    mark_clip_exported,
    mark_clip_needs_rerender,
    mark_clip_rendered,
    mark_clip_skipped,
    upsert_discovered_clip,
)


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(
        downloads_dir=tmp_path / "downloads",
        renders_dir=tmp_path / "renders",
        metadata_dir=tmp_path / "metadata",
        analysis_dir=tmp_path / "analysis",
        exports_dir=tmp_path / "exports",
        state_db_path=tmp_path / "state" / "clipforge.sqlite",
    )


def _clip(clip_id: str, *, views: int, title: str | None = None) -> TwitchClip:
    return TwitchClip(
        id=clip_id,
        url=f"https://clips.twitch.tv/{clip_id}",
        broadcaster_name="Example",
        creator_name="Viewer",
        title=title or clip_id,
        view_count=views,
        created_at="2026-05-01T00:00:00Z",
        duration=30,
        thumbnail_url="https://example.test/thumb.jpg",
    )


def _write_metadata(config: ClipforgeConfig, clip_id: str, *, content: bytes = b"video") -> Path:
    render_dir = config.renders_dir / "example" / clip_id / "ytdlp"
    render_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for layout in ("center_gameplay", "hybrid"):
        path = render_dir / f"{layout}.mp4"
        path.write_bytes(content + layout.encode("utf-8"))
        outputs.append({"layout": layout, "path": str(path)})

    metadata_path = config.metadata_dir / f"{clip_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
    return metadata_path


def _write_review_metadata(
    config: ClipforgeConfig,
    clip_id: str,
    *,
    preview_resolution: dict[str, int],
    render_settings: dict[str, object] | None = None,
) -> Path:
    source_path = config.downloads_dir / clip_id / "ytdlp" / f"{clip_id}.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source")
    render_dir = config.renders_dir / "example" / clip_id / "ytdlp"
    render_dir.mkdir(parents=True, exist_ok=True)
    preview_path = render_dir / "hybrid.mp4"
    preview_path.write_bytes(b"preview")
    metadata_path = config.metadata_dir / f"{clip_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "source_path": str(source_path),
                "target_resolution": {"width": 1080, "height": 1920},
                "outputs": [
                    {
                        "layout": "hybrid",
                        "path": str(preview_path),
                        "resolution": preview_resolution,
                        "render_profile": "review",
                        "render_settings": render_settings
                        or {
                            "encoder": "libx264",
                            "preset": "medium",
                            "crf": 23,
                            "quality": None,
                            "threads": 0,
                            "fallback_to_software": True,
                        },
                    }
                ],
                "layouts": [
                    {
                        "name": "hybrid",
                        "description": "Hybrid test layout.",
                        "output": {"width": 1080, "height": 1920},
                        "regions": [
                            {
                                "name": "gameplay",
                                "source_region": {
                                    "x": 0.0,
                                    "y": 0.0,
                                    "width": 1.0,
                                    "height": 1.0,
                                },
                                "output_region": {
                                    "x": 0.0,
                                    "y": 0.0,
                                    "width": 1.0,
                                    "height": 1.0,
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


def test_review_discovers_upserts_selects_top_ranked_and_exports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    events: list[str] = []

    def fake_list_channel_clips(channel_login: str, **kwargs) -> tuple[TwitchClip, ...]:
        events.append("discover")
        assert channel_login == "example"
        return (
            _clip("clip-low", views=10, title="low"),
            _clip("clip-high", views=1000, title="high"),
        )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        events.append(f"process:{clip_id}")
        assert kwargs["config"] == config
        assert "candidate_output_size" not in kwargs
        return _write_metadata(config, clip_id)

    monkeypatch.setattr("clipforge.pipeline.review.list_channel_clips", fake_list_channel_clips)
    monkeypatch.setattr("clipforge.pipeline.review.process_clip", fake_process)

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "2",
        output_fn=lambda line: None,
    )

    assert events == ["discover", "discover", "process:clip-high"]
    assert exported == (
        tmp_path / "exports" / "example" / "high__clip-high" / "hybrid.mp4",
    )
    assert exported[0].read_bytes().startswith(b"videohybrid")
    assert get_clip("clip-low", db_path=config.state_db_path) is not None
    exported_state = get_clip("clip-high", db_path=config.state_db_path)
    assert exported_state is not None
    assert exported_state.status == "exported"
    assert exported_state.selected_render_layout == "hybrid"
    assert exported_state.selected_render_path.endswith("hybrid.mp4")
    assert exported_state.export_path == str(exported[0])


def test_review_reuses_rendered_candidates_without_reprocessing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    metadata_path = _write_metadata(config, "clip-1")
    upsert_discovered_clip(
        clip_id="clip-1",
        url="https://clips.twitch.tv/clip-1",
        streamer_login="example",
        title="ready",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-1",
        render_dir=metadata_path.parent,
        metadata_path=metadata_path,
        db_path=config.state_db_path,
    )

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100, title="ready"),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Rendered clips should reuse existing metadata.")
        ),
    )

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    assert exported == (
        tmp_path / "exports" / "example" / "ready__clip-1" / "center_gameplay.mp4",
    )


def test_review_reuses_candidate_when_preview_matches_final_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: _write_review_metadata(
            config,
            "clip-1",
            preview_resolution={"width": 1080, "height": 1920},
        ),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.render_selected_layout_from_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("matching preview should be copied")
        ),
    )

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    assert exported[0].read_bytes() == b"preview"
    metadata = json.loads((config.metadata_dir / "clip-1.json").read_text(encoding="utf-8"))
    assert metadata["selected_export"]["layout"] == "hybrid"
    assert metadata["selected_export"]["preview_candidate"]["resolution"] == {
        "width": 1080,
        "height": 1920,
    }
    assert metadata["selected_export"]["export"]["resolution"] == {
        "width": 1080,
        "height": 1920,
    }
    assert metadata["selected_export"]["reused_preview"] is True


def test_review_rerenders_selected_layout_when_preview_differs_from_final(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: _write_review_metadata(
            config,
            "clip-1",
            preview_resolution={"width": 720, "height": 1280},
        ),
    )

    def fake_render_selected(
        metadata_path: Path,
        *,
        selected_layout: str,
        output_path: Path,
        channel: str | None,
        config: ClipforgeConfig,
    ) -> Path:
        calls.append(
            {
                "metadata_path": metadata_path,
                "selected_layout": selected_layout,
                "output_path": output_path,
                "channel": channel,
                "config": config,
            }
        )
        output_path.write_bytes(b"final")
        return output_path

    monkeypatch.setattr(
        "clipforge.pipeline.review.render_selected_layout_from_metadata",
        fake_render_selected,
    )

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    assert len(calls) == 1
    assert calls[0]["selected_layout"] == "hybrid"
    assert calls[0]["channel"] == "example"
    assert exported[0].read_bytes() == b"final"
    metadata = json.loads((config.metadata_dir / "clip-1.json").read_text(encoding="utf-8"))
    assert metadata["selected_export"]["preview_candidate"]["resolution"] == {
        "width": 720,
        "height": 1280,
    }
    assert metadata["selected_export"]["export"]["resolution"] == {
        "width": 1080,
        "height": 1920,
    }
    assert metadata["selected_export"]["reused_preview"] is False


def test_review_rerenders_selected_layout_when_review_settings_differ(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[Path] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: _write_review_metadata(
            config,
            "clip-1",
            preview_resolution={"width": 1080, "height": 1920},
            render_settings={
                "encoder": "libx264",
                "preset": "veryfast",
                "crf": 28,
                "quality": None,
                "threads": 0,
                "fallback_to_software": True,
            },
        ),
    )

    def fake_render_selected(
        metadata_path: Path,
        *,
        selected_layout: str,
        output_path: Path,
        channel: str | None,
        config: ClipforgeConfig,
    ) -> Path:
        del metadata_path, selected_layout, channel, config
        calls.append(output_path)
        output_path.write_bytes(b"final-settings")
        return output_path

    monkeypatch.setattr(
        "clipforge.pipeline.review.render_selected_layout_from_metadata",
        fake_render_selected,
    )

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    assert calls == [exported[0]]
    assert exported[0].read_bytes() == b"final-settings"


def test_review_excludes_exported_clips(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for clip_id in ("clip-exported", "clip-eligible"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            streamer_login="example",
            rank_score=1.0 if clip_id == "clip-exported" else 0.1,
            db_path=config.state_db_path,
        )
    mark_clip_exported(
        "clip-exported",
        selected_render_layout="hybrid",
        selected_render_path=tmp_path / "renders" / "clip-exported.mp4",
        export_path=tmp_path / "exports" / "clip-exported.mp4",
        db_path=config.state_db_path,
    )

    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (
            _clip("clip-exported", views=1000),
            _clip("clip-eligible", views=10),
        ),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        return _write_metadata(config, clip_id)

    monkeypatch.setattr("clipforge.pipeline.review.process_clip", fake_process)

    review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    assert calls == ["clip-eligible"]


def test_review_rejects_invalid_selection_and_reprompts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    responses = iter(("x", "3", "1"))
    output: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: _write_metadata(config, "clip-1"),
    )

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: next(responses),
        output_fn=output.append,
    )

    assert output.count("Invalid selection. Enter 1-2, s to skip, or r to rerender.") == 2
    assert exported == (
        tmp_path / "exports" / "example" / "clip-1__clip-1" / "center_gameplay.mp4",
    )


def test_review_skip_does_not_export_or_mark_selected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: _write_metadata(config, "clip-1"),
    )

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "skip",
        output_fn=lambda line: None,
    )

    state = get_clip("clip-1", db_path=config.state_db_path)
    assert exported == ()
    assert state is not None
    assert state.status == "skipped"
    assert state.skip_reason == "review skipped after candidates generated"
    assert state.export_path is None
    assert not (tmp_path / "exports").exists()


def test_review_rerender_choice_marks_clip_needs_rerender(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    output: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: _write_metadata(config, "clip-1"),
    )

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        input_fn=lambda prompt: "r",
        output_fn=output.append,
    )

    state = get_clip("clip-1", db_path=config.state_db_path)
    assert exported == ()
    assert state is not None
    assert state.status == "needs_rerender"
    assert state.skip_reason == "review requested rerender after candidates generated"
    assert state.export_path is None
    assert output[-1] == (
        "rerender requested: clip-1 "
        "(will be picked up by processing or review --rerender)"
    )


def test_review_marks_clip_failed_when_processing_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        review_streamer_clips(
            streamer="example",
            count=1,
            config=config,
            input_fn=lambda prompt: "1",
            output_fn=lambda line: None,
        )

    state = get_clip("clip-1", db_path=config.state_db_path)
    assert state is not None
    assert state.status == "failed"
    assert state.error_message == "render failed"
    assert state.skip_reason is None


def test_review_force_controls_export_overwrite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    export_path = tmp_path / "exports" / "example" / "clip-1__clip-1" / "hybrid.mp4"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_bytes(b"old")

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.process_clip",
        lambda url, **kwargs: _write_metadata(config, "clip-1", content=b"new"),
    )

    with pytest.raises(ClipReviewError, match="--force"):
        review_streamer_clips(
            streamer="example",
            count=1,
            config=config,
            input_fn=lambda prompt: "2",
            output_fn=lambda line: None,
        )
    assert export_path.read_bytes() == b"old"

    exported = review_streamer_clips(
        streamer="example",
        count=1,
        force=True,
        config=config,
        input_fn=lambda prompt: "2",
        output_fn=lambda line: None,
    )

    assert exported == (export_path,)
    assert export_path.read_bytes().startswith(b"newhybrid")


def test_review_clip_id_override_processes_named_clip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-high", views=1000), _clip("clip-low", views=10)),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        return _write_metadata(config, clip_id)

    monkeypatch.setattr("clipforge.pipeline.review.process_clip", fake_process)

    review_streamer_clips(
        streamer="example",
        count=1,
        clip_ids=("clip-low",),
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    assert calls == ["clip-low"]


def test_review_clip_id_override_requires_force_for_skipped_clip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    upsert_discovered_clip(
        clip_id="clip-skipped",
        url="https://clips.twitch.tv/clip-skipped",
        streamer_login="example",
        db_path=config.state_db_path,
    )
    mark_clip_skipped(
        "clip-skipped",
        skip_reason="already reviewed",
        db_path=config.state_db_path,
    )

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-skipped", views=100),),
    )

    def fake_process(url: str, **kwargs) -> Path:
        raise AssertionError("Skipped clip should require --force.")

    monkeypatch.setattr("clipforge.pipeline.review.process_clip", fake_process)

    with pytest.raises(ClipReviewError, match="--force"):
        review_streamer_clips(
            streamer="example",
            count=1,
            clip_ids=("clip-skipped",),
            config=config,
            input_fn=lambda prompt: "1",
            output_fn=lambda line: None,
        )

    state = get_clip("clip-skipped", db_path=config.state_db_path)
    assert state is not None
    assert state.status == "skipped"


def test_review_clip_id_override_force_processes_skipped_clip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    upsert_discovered_clip(
        clip_id="clip-skipped",
        url="https://clips.twitch.tv/clip-skipped",
        streamer_login="example",
        db_path=config.state_db_path,
    )
    mark_clip_skipped(
        "clip-skipped",
        skip_reason="already reviewed",
        db_path=config.state_db_path,
    )

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-skipped", views=100),),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        return _write_metadata(config, clip_id)

    monkeypatch.setattr("clipforge.pipeline.review.process_clip", fake_process)

    review_streamer_clips(
        streamer="example",
        count=1,
        force=True,
        clip_ids=("clip-skipped",),
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    state = get_clip("clip-skipped", db_path=config.state_db_path)
    assert calls == ["clip-skipped"]
    assert state is not None
    assert state.status == "exported"


def test_review_without_rerender_rejects_clip_id_that_needs_rerender(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-rerender",
        url="https://clips.twitch.tv/clip-rerender",
        streamer_login="example",
        db_path=config.state_db_path,
    )
    mark_clip_needs_rerender(
        "clip-rerender",
        skip_reason="layouts did not fit",
        db_path=config.state_db_path,
    )
    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-rerender", views=100),),
    )

    with pytest.raises(ClipReviewError, match="Re-run with --rerender"):
        review_streamer_clips(
            streamer="example",
            count=1,
            clip_ids=("clip-rerender",),
            config=config,
            input_fn=lambda prompt: "1",
            output_fn=lambda line: None,
        )


def test_review_rerender_includes_clip_that_needs_rerender(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[dict[str, object]] = []
    upsert_discovered_clip(
        clip_id="clip-rerender",
        url="https://clips.twitch.tv/clip-rerender",
        streamer_login="example",
        rank_score=1.0,
        db_path=config.state_db_path,
    )
    mark_clip_needs_rerender(
        "clip-rerender",
        skip_reason="layouts did not fit",
        db_path=config.state_db_path,
    )

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-rerender", views=100),),
    )

    def fake_process(url: str, **kwargs) -> Path:
        calls.append({"url": url, **kwargs})
        return _write_metadata(config, "clip-rerender")

    monkeypatch.setattr("clipforge.pipeline.review.process_clip", fake_process)

    review_streamer_clips(
        streamer="example",
        count=1,
        rerender=True,
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    state = get_clip("clip-rerender", db_path=config.state_db_path)
    assert calls[0]["url"] == "https://clips.twitch.tv/clip-rerender"
    assert calls[0]["rerender"] is True
    assert calls[0]["channel"] == "example"
    assert state is not None
    assert state.status == "exported"


def test_review_rerender_passes_visual_only_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-1", views=100),),
    )

    def fake_process(url: str, **kwargs) -> Path:
        calls.append({"url": url, **kwargs})
        return _write_metadata(config, "clip-1")

    monkeypatch.setattr("clipforge.pipeline.review.process_clip", fake_process)

    review_streamer_clips(
        streamer="example",
        count=1,
        rerender=True,
        config=config,
        input_fn=lambda prompt: "1",
        output_fn=lambda line: None,
    )

    assert calls[0]["url"] == "https://clips.twitch.tv/clip-1"
    assert calls[0]["rerender"] is True
    assert calls[0]["channel"] == "example"
    assert calls[0]["config"] == config

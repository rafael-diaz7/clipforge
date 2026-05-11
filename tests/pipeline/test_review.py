from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig
from clipforge.integrations.twitch import TwitchClip
from clipforge.pipeline.review import ClipReviewError, review_streamer_clips
from clipforge.storage.state import (
    get_clip,
    mark_clip_exported,
    mark_clip_needs_rerender,
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

    assert events == ["discover", "process:clip-high"]
    assert exported == (
        tmp_path / "exports" / "ready" / "example" / "clip-high" / "hybrid.mp4",
    )
    assert exported[0].read_bytes().startswith(b"videohybrid")
    assert get_clip("clip-low", db_path=config.state_db_path) is not None
    exported_state = get_clip("clip-high", db_path=config.state_db_path)
    assert exported_state is not None
    assert exported_state.status == "exported"
    assert exported_state.selected_render_layout == "hybrid"
    assert exported_state.selected_render_path.endswith("hybrid.mp4")
    assert exported_state.export_path == str(exported[0])


def test_review_excludes_exported_and_posted_clips(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for clip_id in ("clip-exported", "clip-posted", "clip-eligible"):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            streamer_login="example",
            rank_score=1.0 if clip_id != "clip-eligible" else 0.1,
            db_path=config.state_db_path,
        )
    mark_clip_exported(
        "clip-exported",
        selected_render_layout="hybrid",
        selected_render_path=tmp_path / "renders" / "clip-exported.mp4",
        export_path=tmp_path / "exports" / "clip-exported.mp4",
        db_path=config.state_db_path,
    )
    with sqlite3.connect(config.state_db_path) as connection:
        connection.execute("UPDATE clips SET status = 'posted' WHERE clip_id = ?", ("clip-posted",))

    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.review.list_channel_clips",
        lambda *args, **kwargs: (
            _clip("clip-exported", views=1000),
            _clip("clip-posted", views=900),
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
        tmp_path / "exports" / "ready" / "example" / "clip-1" / "center_gameplay.mp4",
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
    export_path = tmp_path / "exports" / "ready" / "example" / "clip-1" / "hybrid.mp4"
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


def test_review_clip_id_override_processes_skipped_clip(
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
    assert calls[0]["config"] == config

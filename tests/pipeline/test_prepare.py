from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig
from clipforge.integrations.twitch import TwitchClip
from clipforge.pipeline.prepare import ClipPrepareError, prepare_streamer_clips
from clipforge.storage.state import (
    get_clip,
    get_review_eligible_clips,
    mark_clip_exported,
    mark_clip_failed,
    mark_clip_rendered,
    mark_clip_selected,
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


def _write_metadata(config: ClipforgeConfig, clip_id: str) -> Path:
    render_dir = config.renders_dir / "example" / clip_id / "ytdlp"
    render_dir.mkdir(parents=True, exist_ok=True)
    output_path = render_dir / "hybrid.mp4"
    output_path.write_bytes(b"video")
    metadata_path = config.metadata_dir / f"{clip_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "layout": "hybrid",
                        "path": str(output_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


def test_prepare_discovers_upserts_reranks_and_processes_top_ranked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    events: list[str] = []

    def fake_list_channel_clips(channel_login: str, **kwargs) -> tuple[TwitchClip, ...]:
        events.append(f"discover:{channel_login}:{kwargs['limit']}")
        return (
            _clip("clip-low", views=10, title="low"),
            _clip("clip-high", views=1000, title="high"),
        )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        events.append(f"process:{clip_id}")
        assert kwargs["config"] == config
        assert kwargs["channel"] == "example"
        return _write_metadata(config, clip_id)

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        fake_list_channel_clips,
    )

    result = prepare_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        process_clip_fn=fake_process,
    )

    prepared_state = get_clip("clip-high", db_path=config.state_db_path)
    assert events == ["discover:example:10", "process:clip-high"]
    assert result.discovered_count == 2
    assert result.reranked_count == 2
    assert result.selected_count == 1
    assert result.rendered_count == 1
    assert result.failed == ()
    assert result.prepared[0].clip_id == "clip-high"
    assert prepared_state is not None
    assert prepared_state.status == "rendered"
    assert prepared_state.render_dir is not None
    assert prepared_state.metadata_path == str(config.metadata_dir / "clip-high.json")
    assert prepared_state.selected_render_layout is None
    assert prepared_state.export_path is None


def test_prepare_excludes_already_rendered_clips_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    rendered_metadata = _write_metadata(config, "clip-rendered")
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        streamer_login="example",
        rank_score=100,
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=rendered_metadata.parent,
        metadata_path=rendered_metadata,
        db_path=config.state_db_path,
    )
    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (
            _clip("clip-rendered", views=1000),
            _clip("clip-fresh", views=10),
        ),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        return _write_metadata(config, clip_id)

    result = prepare_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        process_clip_fn=fake_process,
    )

    assert calls == ["clip-fresh"]
    assert [prepared.clip_id for prepared in result.prepared] == ["clip-fresh"]
    assert get_clip("clip-rendered", db_path=config.state_db_path).status == "rendered"


def test_prepare_excludes_review_terminal_states_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for clip_id in (
        "clip-skipped",
        "clip-exported",
        "clip-failed",
        "clip-selected",
        "clip-rendered",
        "clip-eligible",
    ):
        upsert_discovered_clip(
            clip_id=clip_id,
            url=f"https://clips.twitch.tv/{clip_id}",
            streamer_login="example",
            rank_score=10 if clip_id != "clip-eligible" else 1,
            db_path=config.state_db_path,
        )

    mark_clip_skipped(
        "clip-skipped",
        skip_reason="already reviewed",
        db_path=config.state_db_path,
    )
    mark_clip_failed(
        "clip-failed",
        error_message="previous failure",
        db_path=config.state_db_path,
    )
    for clip_id in ("clip-exported", "clip-selected", "clip-rendered"):
        metadata_path = _write_metadata(config, clip_id)
        mark_clip_rendered(
            clip_id,
            render_dir=metadata_path.parent,
            metadata_path=metadata_path,
            db_path=config.state_db_path,
        )
    mark_clip_exported(
        "clip-exported",
        selected_render_layout="hybrid",
        selected_render_path=config.renders_dir / "clip-exported.mp4",
        export_path=config.exports_dir / "clip-exported.mp4",
        db_path=config.state_db_path,
    )
    mark_clip_selected(
        "clip-selected",
        selected_render_layout="hybrid",
        selected_render_path=config.renders_dir / "clip-selected.mp4",
        db_path=config.state_db_path,
    )

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: tuple(
            _clip(clip_id, views=100)
            for clip_id in (
                "clip-skipped",
                "clip-exported",
                "clip-failed",
                "clip-selected",
                "clip-rendered",
                "clip-eligible",
            )
        ),
    )
    calls: list[str] = []

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        return _write_metadata(config, clip_id)

    result = prepare_streamer_clips(
        streamer="example",
        count=3,
        config=config,
        process_clip_fn=fake_process,
    )

    assert calls == ["clip-eligible"]
    assert [prepared.clip_id for prepared in result.prepared] == ["clip-eligible"]


def test_prepare_marks_failures_and_continues_when_practical(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (
            _clip("clip-fails", views=1000),
            _clip("clip-succeeds", views=100),
        ),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        if clip_id == "clip-fails":
            raise RuntimeError("render failed")
        return _write_metadata(config, clip_id)

    result = prepare_streamer_clips(
        streamer="example",
        count=2,
        config=config,
        process_clip_fn=fake_process,
    )

    failed_state = get_clip("clip-fails", db_path=config.state_db_path)
    prepared_state = get_clip("clip-succeeds", db_path=config.state_db_path)
    assert [failed.clip_id for failed in result.failed] == ["clip-fails"]
    assert result.failed[0].error_message == "render failed"
    assert [prepared.clip_id for prepared in result.prepared] == ["clip-succeeds"]
    assert failed_state is not None
    assert failed_state.status == "failed"
    assert failed_state.error_message == "render failed"
    assert prepared_state is not None
    assert prepared_state.status == "rendered"


def test_prepared_clips_are_visible_to_normal_review_queue(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-ready", views=100),),
    )

    prepare_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        process_clip_fn=lambda url, **kwargs: _write_metadata(config, "clip-ready"),
    )

    eligible = get_review_eligible_clips(
        db_path=config.state_db_path,
        streamer_login="example",
    )
    assert [clip.clip_id for clip in eligible] == ["clip-ready"]
    assert eligible[0].selected_render_layout is None
    assert eligible[0].export_path is None


def test_prepare_does_not_prompt_or_export(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-ready", views=100),),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare should not prompt")
        ),
    )
    monkeypatch.setattr(
        "clipforge.pipeline.exports.export_review_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare should not export")
        ),
    )

    result = prepare_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        process_clip_fn=lambda url, **kwargs: _write_metadata(config, "clip-ready"),
    )

    assert [prepared.clip_id for prepared in result.prepared] == ["clip-ready"]
    assert not config.exports_dir.exists()


def test_prepare_rejects_non_positive_count(tmp_path: Path) -> None:
    with pytest.raises(ClipPrepareError, match="--count"):
        prepare_streamer_clips(streamer="example", count=0, config=_config(tmp_path))

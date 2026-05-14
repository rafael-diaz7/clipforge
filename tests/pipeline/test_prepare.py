from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig
from clipforge.integrations.twitch import TwitchClip
from clipforge.media.layouts import OutputSize
from clipforge.pipeline.prepare import (
    ClipPrepareError,
    prepare_streamer_clips,
    prepare_until_count,
)
from clipforge.storage.state import (
    get_clip,
    get_mobile_review_clips,
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
        assert kwargs["candidate_output_size"] == OutputSize(width=1080, height=1920)
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
    assert events == [
        "discover:example:100",
        "discover:example:100",
        "process:clip-high",
    ]
    assert result.discovered_count == 2
    assert result.reranked_count == 2
    assert result.selected_count == 1
    assert result.rendered_count == 1
    assert result.failed == ()
    assert result.prepared[0].clip_id == "clip-high"
    assert prepared_state is not None
    assert prepared_state.status == "mobile_review"
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
    assert prepared_state.status == "mobile_review"


def test_prepare_walks_past_first_failed_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (
            _clip("clip-fails", views=1000),
            _clip("clip-succeeds", views=100),
        ),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        if clip_id == "clip-fails":
            raise RuntimeError("download failed")
        return _write_metadata(config, clip_id)

    result = prepare_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        process_clip_fn=fake_process,
    )

    assert calls == ["clip-fails", "clip-succeeds"]
    assert [prepared.clip_id for prepared in result.prepared] == ["clip-succeeds"]
    assert [failed.clip_id for failed in result.failed] == ["clip-fails"]
    assert result.rendered_count == 1
    assert result.exhausted is False
    assert result.max_failures_reached is False


def test_prepare_allows_failures_before_enough_successes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (
            _clip("clip-fails-1", views=1000),
            _clip("clip-fails-2", views=900),
            _clip("clip-succeeds-1", views=800),
            _clip("clip-succeeds-2", views=700),
        ),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        if clip_id.startswith("clip-fails"):
            raise RuntimeError(f"{clip_id} failed")
        return _write_metadata(config, clip_id)

    result = prepare_streamer_clips(
        streamer="example",
        count=2,
        max_failures=10,
        config=config,
        process_clip_fn=fake_process,
    )

    assert calls == [
        "clip-fails-1",
        "clip-fails-2",
        "clip-succeeds-1",
        "clip-succeeds-2",
    ]
    assert [prepared.clip_id for prepared in result.prepared] == [
        "clip-succeeds-1",
        "clip-succeeds-2",
    ]
    assert [failed.clip_id for failed in result.failed] == [
        "clip-fails-1",
        "clip-fails-2",
    ]
    assert result.attempted_count == 4
    assert result.rendered_count == 2


def test_prepare_does_not_retry_same_failed_clip_within_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    clip = upsert_discovered_clip(
        clip_id="clip-fails",
        url="https://clips.twitch.tv/clip-fails",
        streamer_login="example",
        db_path=config.state_db_path,
    )
    calls: list[str] = []

    def fake_process(url: str, **kwargs) -> Path:
        calls.append(url.rsplit("/", 1)[-1])
        raise RuntimeError("download failed")

    prepared, failed = prepare_until_count(
        (clip, clip),
        count=1,
        max_failures=10,
        streamer_login="example",
        generate_captions=None,
        force_captions=False,
        use_generated_layouts=True,
        config=config,
        process_clip_fn=fake_process,
    )

    assert prepared == []
    assert [failure.clip_id for failure in failed] == ["clip-fails"]
    assert calls == ["clip-fails"]


def test_prepare_retries_failed_clip_after_cooldown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-old-failure",
        url="https://clips.twitch.tv/clip-old-failure",
        streamer_login="example",
        rank_score=100,
        db_path=config.state_db_path,
    )
    mark_clip_failed(
        "clip-old-failure",
        error_message="old transient failure",
        failed_at="2000-01-01T00:00:00+00:00",
        db_path=config.state_db_path,
    )
    upsert_discovered_clip(
        clip_id="clip-recent-failure",
        url="https://clips.twitch.tv/clip-recent-failure",
        streamer_login="example",
        rank_score=90,
        db_path=config.state_db_path,
    )
    mark_clip_failed(
        "clip-recent-failure",
        error_message="recent transient failure",
        failed_at="2999-01-01T00:00:00+00:00",
        db_path=config.state_db_path,
    )
    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        return _write_metadata(config, clip_id)

    result = prepare_streamer_clips(
        streamer="example",
        count=1,
        failed_retry_cooldown_minutes=60,
        config=config,
        process_clip_fn=fake_process,
    )

    assert calls == ["clip-old-failure"]
    assert [prepared.clip_id for prepared in result.prepared] == ["clip-old-failure"]
    assert get_clip("clip-old-failure", db_path=config.state_db_path).status == (
        "mobile_review"
    )
    assert get_clip("clip-recent-failure", db_path=config.state_db_path).status == (
        "failed"
    )


def test_prepare_stops_at_max_failure_cap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (
            _clip("clip-fails-1", views=1000),
            _clip("clip-fails-2", views=900),
            _clip("clip-succeeds", views=800),
        ),
    )

    def fake_process(url: str, **kwargs) -> Path:
        clip_id = url.rsplit("/", 1)[-1]
        calls.append(clip_id)
        if clip_id.startswith("clip-fails"):
            raise RuntimeError(f"{clip_id} failed")
        return _write_metadata(config, clip_id)

    result = prepare_streamer_clips(
        streamer="example",
        count=1,
        max_failures=2,
        config=config,
        process_clip_fn=fake_process,
    )

    assert calls == ["clip-fails-1", "clip-fails-2"]
    assert result.prepared == ()
    assert [failed.clip_id for failed in result.failed] == [
        "clip-fails-1",
        "clip-fails-2",
    ]
    assert result.max_failures_reached is True
    assert result.exhausted is False


def test_prepare_reports_exhausted_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-only", views=100),),
    )

    result = prepare_streamer_clips(
        streamer="example",
        count=2,
        config=config,
        process_clip_fn=lambda url, **kwargs: _write_metadata(config, "clip-only"),
    )

    assert [prepared.clip_id for prepared in result.prepared] == ["clip-only"]
    assert result.failed == ()
    assert result.exhausted is True
    assert result.max_failures_reached is False


def test_prepared_clips_are_visible_to_mobile_review_queue_only(
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

    eligible = get_mobile_review_clips(db_path=config.state_db_path)
    assert [clip.clip_id for clip in eligible] == ["clip-ready"]
    assert eligible[0].selected_render_layout is None
    assert eligible[0].export_path is None
    assert (
        get_review_eligible_clips(
            db_path=config.state_db_path,
            streamer_login="example",
        )
        == ()
    )


def test_prepare_output_size_is_not_overridden_by_review_output_width(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path), review_output_width=720)
    candidate_sizes: list[OutputSize] = []

    monkeypatch.setattr(
        "clipforge.pipeline.prepare.list_channel_clips",
        lambda *args, **kwargs: (_clip("clip-ready", views=100),),
    )

    def fake_process(url: str, **kwargs) -> Path:
        candidate_sizes.append(kwargs["candidate_output_size"])
        return _write_metadata(config, "clip-ready")

    prepare_streamer_clips(
        streamer="example",
        count=1,
        config=config,
        process_clip_fn=fake_process,
    )

    assert candidate_sizes == [OutputSize(width=1080, height=1920)]


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

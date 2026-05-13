from __future__ import annotations

import json
from pathlib import Path

from clipforge.core.config import ClipforgeConfig
from clipforge.server.http import ReviewApplication
from clipforge.server.review import ReviewQueueService
from clipforge.storage.state import (
    get_clip,
    get_mobile_review_clips,
    mark_clip_rendered,
    mark_clip_mobile_review,
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


def _app(config: ClipforgeConfig) -> ReviewApplication:
    return ReviewApplication(ReviewQueueService(config=config))


def _write_rendered_clip(
    config: ClipforgeConfig,
    clip_id: str,
    *,
    rank_score: float = 1.0,
    title: str | None = None,
    streamer: str = "example",
    layouts: tuple[str, ...] = ("center_gameplay", "hybrid"),
) -> Path:
    upsert_discovered_clip(
        clip_id=clip_id,
        url=f"https://clips.twitch.tv/{clip_id}",
        streamer_login=streamer,
        title=title or clip_id,
        view_count=42,
        created_at="2026-05-01T00:00:00Z",
        duration_seconds=30,
        rank_score=rank_score,
        db_path=config.state_db_path,
    )
    render_dir = config.renders_dir / streamer / clip_id / "ytdlp"
    render_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for layout in layouts:
        output_path = render_dir / f"{layout}.mp4"
        output_path.write_bytes(f"video:{clip_id}:{layout}".encode("utf-8"))
        outputs.append(
            {
                "layout": layout,
                "path": str(output_path),
                "resolution": {"width": 1080, "height": 1920},
            }
        )
    metadata_path = config.metadata_dir / f"{clip_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "target_resolution": {"width": 1080, "height": 1920},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    mark_clip_mobile_review(
        clip_id,
        render_dir=render_dir,
        metadata_path=metadata_path,
        db_path=config.state_db_path,
    )
    return metadata_path


def test_empty_queue_page_works(tmp_path: Path) -> None:
    response = _app(_config(tmp_path)).handle("GET", "/")

    assert response.status == 200
    assert b"Review queue is empty" in response.body


def test_review_page_returns_current_review_item(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready", title="Phone Review Clip")

    response = _app(config).handle("GET", "/")

    assert response.status == 200
    assert b"Phone Review Clip" in response.body
    assert b"clip-ready" in response.body
    assert b"center_gameplay" in response.body
    assert b"hybrid" in response.body


def test_candidate_video_route_serves_allowed_render_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready")

    response = _app(config).handle(
        "GET",
        "/clips/clip-ready/renders/hybrid.mp4",
    )

    assert response.status == 200
    assert response.body == b"video:clip-ready:hybrid"
    assert ("Content-Type", "video/mp4") in response.headers


def test_candidate_video_route_rejects_path_traversal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready")

    response = _app(config).handle(
        "GET",
        "/clips/clip-ready/renders/..%2Fsecret.mp4",
    )

    assert response.status == 404


def test_candidate_video_route_rejects_metadata_outside_render_dir(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    metadata_path = _write_rendered_clip(config, "clip-ready", layouts=("hybrid",))
    secret_path = tmp_path / "secret.mp4"
    secret_path.write_bytes(b"secret")
    metadata_path.write_text(
        json.dumps({"outputs": [{"layout": "hybrid", "path": str(secret_path)}]}),
        encoding="utf-8",
    )

    response = _app(config).handle(
        "GET",
        "/clips/clip-ready/renders/hybrid.mp4",
    )

    assert response.status == 403


def test_approve_action_exports_selected_render_and_removes_from_queue(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready", title="Ready")

    response = _app(config).handle(
        "POST",
        "/approve",
        body=b"clip_id=clip-ready&layout=hybrid",
    )

    state = get_clip("clip-ready", db_path=config.state_db_path)
    expected_export = (
        config.exports_dir / "ready" / "example" / "clip-ready" / "hybrid.mp4"
    )
    assert response.status == 200
    assert b"Download MP4" in response.body
    assert b"/exports/ready/example/clip-ready/hybrid.mp4" in response.body
    assert state is not None
    assert state.status == "exported"
    assert state.selected_render_layout == "hybrid"
    assert state.export_path == str(expected_export)
    assert expected_export.read_bytes() == b"video:clip-ready:hybrid"
    assert get_mobile_review_clips(db_path=config.state_db_path) == ()


def test_approve_action_does_not_render_selected_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready", layouts=("hybrid",))

    def fail(*args, **kwargs):
        raise AssertionError("Mobile approval should copy the prepared candidate.")

    monkeypatch.setattr(
        "clipforge.pipeline.exports.render_selected_layout_from_metadata",
        fail,
    )

    response = _app(config).handle(
        "POST",
        "/approve",
        body=b"clip_id=clip-ready&layout=hybrid",
    )

    assert response.status == 200
    assert (
        config.exports_dir / "ready" / "example" / "clip-ready" / "hybrid.mp4"
    ).read_bytes() == b"video:clip-ready:hybrid"


def test_approve_action_is_idempotent_for_existing_export(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready", layouts=("hybrid",))
    app = _app(config)

    first = app.handle(
        "POST",
        "/approve",
        body=b"clip_id=clip-ready&layout=hybrid",
    )
    second = app.handle(
        "POST",
        "/approve",
        body=b"clip_id=clip-ready&layout=hybrid",
    )

    assert first.status == 200
    assert second.status == 200
    assert b"/exports/ready/example/clip-ready/hybrid.mp4" in second.body
    assert (
        config.exports_dir / "ready" / "example" / "clip-ready" / "hybrid.mp4"
    ).read_bytes() == b"video:clip-ready:hybrid"


def test_export_download_route_serves_ready_mp4(tmp_path: Path) -> None:
    config = _config(tmp_path)
    export_path = config.exports_dir / "ready" / "example" / "clip-1" / "hybrid.mp4"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_bytes(b"exported")

    response = _app(config).handle(
        "GET",
        "/exports/ready/example/clip-1/hybrid.mp4",
    )

    assert response.status == 200
    assert response.body == b"exported"
    assert ("Content-Type", "video/mp4") in response.headers
    assert (
        "Content-Disposition",
        'attachment; filename="hybrid.mp4"',
    ) in response.headers


def test_export_download_route_rejects_path_traversal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    secret_path = tmp_path / "secret.mp4"
    secret_path.write_bytes(b"secret")

    response = _app(config).handle(
        "GET",
        "/exports/ready/example/clip-1/..%2F..%2F..%2F..%2Fsecret.mp4",
    )

    assert response.status == 403


def test_skip_action_marks_skipped_and_advances(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-first", rank_score=2.0)
    _write_rendered_clip(config, "clip-next", rank_score=1.0)

    response = _app(config).handle("POST", "/skip", body=b"clip_id=clip-first")
    page = _app(config).handle("GET", "/")

    first = get_clip("clip-first", db_path=config.state_db_path)
    assert response.status == 303
    assert first is not None
    assert first.status == "skipped"
    assert first.export_path is None
    assert not config.exports_dir.exists()
    assert b"clip-next" in page.body
    assert b"clip-first" not in page.body


def test_rerender_action_marks_needs_rerender(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready")

    response = _app(config).handle("POST", "/rerender", body=b"clip_id=clip-ready")

    state = get_clip("clip-ready", db_path=config.state_db_path)
    assert response.status == 303
    assert state is not None
    assert state.status == "needs_rerender"
    assert state.skip_reason == "web review requested rerender after candidates generated"
    assert get_mobile_review_clips(db_path=config.state_db_path) == ()


def test_review_page_only_returns_mobile_review_items(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-normal", rank_score=2.0, title="Normal Render")
    normal_metadata = config.metadata_dir / "clip-normal.json"
    mark_clip_rendered(
        "clip-normal",
        render_dir=config.renders_dir / "example" / "clip-normal" / "ytdlp",
        metadata_path=normal_metadata,
        db_path=config.state_db_path,
    )
    _write_rendered_clip(config, "clip-mobile", rank_score=1.0, title="Mobile Render")

    response = _app(config).handle("GET", "/")

    assert response.status == 200
    assert b"clip-mobile" in response.body
    assert b"Mobile Render" in response.body
    assert b"clip-normal" not in response.body
    assert b"Normal Render" not in response.body


def test_server_routes_do_not_call_discovery_render_or_prepare_logic(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_rendered_clip(config, "clip-ready", layouts=("hybrid",))
    _write_rendered_clip(config, "clip-skip", rank_score=0.5, layouts=("hybrid",))
    app = _app(config)

    def fail(*args, **kwargs):
        raise AssertionError("Review server should only drain prepared rendered clips.")

    monkeypatch.setattr("clipforge.pipeline.prepare.prepare_streamer_clips", fail)
    monkeypatch.setattr("clipforge.pipeline.workflows.process_clip", fail)
    monkeypatch.setattr("clipforge.pipeline.workflows.render_all_candidates", fail)
    monkeypatch.setattr(
        "clipforge.pipeline.exports.render_selected_layout_from_metadata",
        fail,
    )

    assert app.handle("GET", "/").status == 200
    assert app.handle("GET", "/clips/clip-ready/renders/hybrid.mp4").status == 200
    assert (
        app.handle("POST", "/approve", body=b"clip_id=clip-ready&layout=hybrid").status
        == 200
    )
    assert app.handle("POST", "/skip", body=b"clip_id=clip-skip").status == 303

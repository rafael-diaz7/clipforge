from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.core.config import ClipforgeConfig
from clipforge.pipeline.processing import (
    SavedClipProcessingError,
    process_saved_clips,
    select_saved_clips_for_processing,
)
from clipforge.storage.state import (
    get_clip,
    mark_clip_needs_rerender,
    mark_clip_rendered,
    upsert_discovered_clip,
)


def _config(tmp_path: Path) -> ClipforgeConfig:
    return ClipforgeConfig(state_db_path=tmp_path / "state" / "clipforge.sqlite")


def test_select_saved_clips_rejects_rendered_clip_without_force_or_rerender(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    upsert_discovered_clip(
        clip_id="clip-rendered",
        url="https://clips.twitch.tv/clip-rendered",
        db_path=config.state_db_path,
    )
    mark_clip_rendered(
        "clip-rendered",
        render_dir=tmp_path / "renders" / "clip-rendered",
        db_path=config.state_db_path,
    )

    with pytest.raises(SavedClipProcessingError, match="already rendered"):
        select_saved_clips_for_processing(
            top=None,
            clip_id="clip-rendered",
            force=False,
            rerender=False,
            config=config,
        )


def test_process_saved_clips_forces_needs_rerender_and_records_failures(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    clip = upsert_discovered_clip(
        clip_id="clip-rerender",
        url="https://clips.twitch.tv/clip-rerender",
        db_path=config.state_db_path,
    )
    clip = mark_clip_needs_rerender(
        clip.clip_id,
        skip_reason="review requested rerender",
        db_path=config.state_db_path,
    )
    calls: list[dict[str, object]] = []

    def fail_process(url: str, **kwargs) -> Path:
        calls.append({"url": url, **kwargs})
        raise RuntimeError("render failed")

    results = process_saved_clips(
        (clip,),
        config=config,
        process_kwargs={"config": config},
        continue_on_error=True,
        process_clip_fn=fail_process,
    )

    assert calls == [
        {
            "url": "https://clips.twitch.tv/clip-rerender",
            "config": config,
            "force": True,
        }
    ]
    assert results[0].error_message == "render failed"
    assert get_clip("clip-rerender", db_path=config.state_db_path).status == "failed"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.config import ClipforgeConfig
from clipforge.media.captions import (
    CAPTION_METADATA_TYPE,
    CAPTION_METADATA_VERSION,
    CaptionError,
    CaptionMetadata,
    CaptionSegment,
    caption_metadata_path,
    load_caption_metadata,
    parse_caption_metadata,
    save_caption_metadata,
    save_captions,
)
from tests.constants import TWITCH_CLIP_SLUG


CAPTION_CLIP_ID = TWITCH_CLIP_SLUG
UNSAFE_CAPTION_CLIP_ID = "Funny Clip!"
UNSAFE_CAPTION_FILENAME = "Funny_Clip.json"
UNSAFE_PATH_CAPTION_CLIP_ID = "clip with spaces/and/slashes"
UNSAFE_PATH_CAPTION_FILENAME = "clip_with_spaces_and_slashes.json"


def test_save_caption_metadata_uses_deterministic_clip_path(tmp_path: Path) -> None:
    config = ClipforgeConfig(metadata_dir=tmp_path / "metadata")
    metadata = CaptionMetadata(
        clip_id=UNSAFE_CAPTION_CLIP_ID,
        segments=(
            CaptionSegment(start_time=2.5, end_time=3.0, text="second"),
            CaptionSegment(start_time=0.0, end_time=1.25, text=" first "),
        ),
    )

    output_path = save_caption_metadata(metadata, config=config)

    assert output_path == tmp_path / "metadata" / "captions" / UNSAFE_CAPTION_FILENAME
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "type": CAPTION_METADATA_TYPE,
        "version": CAPTION_METADATA_VERSION,
        "clip_id": UNSAFE_CAPTION_CLIP_ID,
        "segments": [
            {
                "start_time": 0.0,
                "end_time": 1.25,
                "text": "first",
            },
            {
                "start_time": 2.5,
                "end_time": 3.0,
                "text": "second",
            },
        ],
    }


def test_caption_metadata_round_trips_through_save_and_load(tmp_path: Path) -> None:
    config = ClipforgeConfig(metadata_dir=tmp_path / "metadata")
    metadata = CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(
            CaptionSegment(start_time=1, end_time=2, text="middle"),
            CaptionSegment(start_time=0, end_time=0.5, text="start"),
        ),
    )

    output_path = save_caption_metadata(metadata, config=config)

    assert load_caption_metadata(output_path) == metadata
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_empty_caption_segments_are_valid_metadata(tmp_path: Path) -> None:
    config = ClipforgeConfig(metadata_dir=tmp_path / "metadata")

    output_path = save_captions(clip_id=CAPTION_CLIP_ID, segments=(), config=config)

    assert load_caption_metadata(output_path) == CaptionMetadata(
        clip_id=CAPTION_CLIP_ID,
        segments=(),
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["segments"] == []


def test_caption_metadata_path_sanitizes_clip_id(tmp_path: Path) -> None:
    config = ClipforgeConfig(metadata_dir=tmp_path / "metadata")

    path = caption_metadata_path(UNSAFE_PATH_CAPTION_CLIP_ID, config=config)

    assert path == tmp_path / "metadata" / "captions" / UNSAFE_PATH_CAPTION_FILENAME


@pytest.mark.parametrize(
    "payload,error",
    (
        ({}, "caption metadata.type is required"),
        (
            {
                "type": "wrong",
                "version": CAPTION_METADATA_VERSION,
                "clip_id": CAPTION_CLIP_ID,
                "segments": [],
            },
            "caption metadata type",
        ),
        (
            {
                "type": CAPTION_METADATA_TYPE,
                "version": 2,
                "clip_id": CAPTION_CLIP_ID,
                "segments": [],
            },
            "unsupported caption metadata version",
        ),
        (
            {
                "type": CAPTION_METADATA_TYPE,
                "version": CAPTION_METADATA_VERSION,
                "clip_id": CAPTION_CLIP_ID,
                "segments": [{"start_time": 2, "end_time": 1, "text": "bad"}],
            },
            "end_time must be greater than start_time",
        ),
    ),
)
def test_parse_caption_metadata_rejects_invalid_payloads(payload, error: str) -> None:
    with pytest.raises(CaptionError, match=error):
        parse_caption_metadata(payload)
